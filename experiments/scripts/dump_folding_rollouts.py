"""Whole-episode PGND rollouts on the folding evaluation cases, saved in the frozen scorer's layout.

Protocol (gpwm's 10-episode folding evaluation, folding_8mm_v3 rollouts_full positions 0-9): every case starts at frame
0; frames 0-3 are given (PGND: frames 0, 1, 2 = history, frame 3 = current), frames 4 .. 901 (898) are predicted one
frame at a time from PGND's own output (positions and history), with the RECORDED gripper points of both hands given
every frame, exactly as in training (gripper-point mode; at the step predicting frame t+1 the network sees the gripper
at t+1 and its forward-difference velocity). The trained checkpoint's own config (hydra.yaml) sets scale, radius,
dt, grid; the model runs in eval mode (BatchNorm running statistics, no augmentation). The simulator is rebuilt every
16 steps (it keeps buffers per step); the rollout state is carried in tensors.

Output: <out>/<position:05d>.npz per case with `pred` [898, N, 3] in metres (world frame of the release), `rows`
(0 .. N-1: retained cloth vertices, then the 2,000 gripper points, the loader's order), `predicted` (cloth rows), and
`id` / `start` / `horizon` -- the layout scripts/unified/score_predictions.py reads. Gripper rows hold the recorded
positions (an input, not scored). Plus META.json.

    python experiments/scripts/dump_folding_rollouts.py --ckpt <final.pt> --config <hydra.yaml> --cases <folding_8mm_v3.json> \
        --first 0 --stop 10 --out <dir>
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import warp as wp                                                                    # noqa: E402
from pgnd.material import PGNDModel                                                  # noqa: E402
from pgnd.sim import CacheDiffSimWithFrictionBatch, StaticsBatch, CollidersBatch, Friction   # noqa: E402
from pgnd.data.dataset import preprocess_arrays                                      # noqa: E402
from pgnd.data.pwm_folding_source import load_index, read_clip, FLIP, N_GRIP, P_GRIP  # noqa: E402
from train.train_eval import prepare_gripper_points                                  # noqa: E402

R_AXES = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)   # preprocess_arrays' world->sim axis map


def to_world(sim, info):
    """Invert preprocess_arrays: sim = (raw @ R.T) * scale + T, raw = world * FLIP."""
    raw = ((np.asarray(sim, np.float64) - np.asarray(info['global_translation'])) / info['preprocess_scale']) @ R_AXES
    return (raw * FLIP).astype(np.float32)


@torch.no_grad()
def rollout(cfg, material, friction, rec, horizon, device, wp_device, chunk=16):
    cloth, grip = read_clip(rec)
    dt = float(cfg.sim.dt)
    info, frames = preprocess_arrays(cfg, rec['episode_id'], (cloth * FLIP).astype(np.float32),
                                     eef_xyz=(grip.mean(2) * FLIP).reshape(len(grip), -1).astype(np.float32),
                                     eef_rot=None, eef_gripper=None, gripper_points=(grip * FLIP).astype(np.float32),
                                     dt_frame=dt, dx=cfg.sim.num_grids[-1])
    if info['num_particles'] != cloth.shape[1]:
        raise ValueError(f'{rec["episode_id"]}: cloth points left the box')
    X = frames['x'].to(device); G = frames['gripper_points'].to(device)               # [T, Nc, 3], [T, 2, P, 3]
    V = torch.empty_like(X); V[1:] = (X[1:] - X[:-1]) / dt; V[0] = V[1]               # backward (as training)
    GV = torch.empty_like(G); GV[:-1] = (G[1:] - G[:-1]) / dt; GV[-1] = GV[-2]         # forward (as training)
    H = int(cfg.sim.n_history); f = H                                                 # current frame = 3
    if f + horizon >= len(X):
        raise ValueError(f'{rec["episode_id"]}: {len(X)} frames, cannot predict {horizon} after frame {f}')
    Nc = X.shape[1]
    x = X[f][None].clone(); v = V[f][None].clone()
    x_his = torch.cat([X[f + h] for h in range(-H, 0)], -1)[None]; v_his = torch.cat([V[f + h] for h in range(-H, 0)], -1)[None]
    clip_bound = torch.tensor([float(info['clip_bound'])])
    mu = friction.mu.clone()[None].repeat(1, 1)
    out = np.empty((horizon, Nc, 3), np.float32)
    for c0 in range(0, horizon, chunk):
        n = min(chunk, horizon - c0)
        sl = slice(f + 1 + c0, f + 1 + c0 + n)
        gx, gv, gmask = prepare_gripper_points(cfg, G[sl][None], GV[sl][None])       # [1, n, 2P, 3]
        Ntot = Nc + gx.shape[2]
        sim = CacheDiffSimWithFrictionBatch(cfg, n, 1, wp_device, requires_grad=False)
        statics = StaticsBatch(); statics.init(shape=(1, Ntot), device=wp_device); statics.update_clip_bound(clip_bound)
        colliders = CollidersBatch(); colliders.init(shape=(1, 0), device=wp_device)
        for s in range(n):
            enabled = torch.cat([torch.ones(1, Nc, dtype=torch.bool, device=device), gmask[:, s]], 1)
            statics.update_enabled(enabled.cpu())
            zh = torch.zeros(1, gx.shape[2], H * 3, device=device)
            xi = torch.cat([x, gx[:, s]], 1); vi = torch.cat([v, gv[:, s]], 1)
            pred = material(xi, vi, torch.cat([x_his, zh], 1), torch.cat([v_his, zh], 1), enabled)
            xo, vo = sim(statics, colliders, s, xi, vi, mu, pred)
            x, v = xo[:, :Nc], vo[:, :Nc]
            if H > 0:
                x_his = torch.cat([x_his.reshape(1, Nc, H, 3)[:, :, 1:], x[:, :, None]], 2).reshape(1, Nc, -1)
                v_his = torch.cat([v_his.reshape(1, Nc, H, 3)[:, :, 1:], v[:, :, None]], 2).reshape(1, Nc, -1)
            out[c0 + s] = to_world(x[0].cpu().numpy(), info)
        if not np.isfinite(out[c0:c0 + n]).all():
            raise ValueError(f'{rec["episode_id"]}: non-finite prediction at step {c0}')
    grip_world = grip[f + 1:f + 1 + horizon].reshape(horizon, N_GRIP * P_GRIP, 3)       # recorded, unscored
    return np.concatenate([out, grip_world], 1), Nc


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--ckpt', required=True); a.add_argument('--config', required=True); a.add_argument('--cases', required=True)
    a.add_argument('--case_set', default='rollouts_full'); a.add_argument('--first', type=int, default=0)
    a.add_argument('--stop', type=int, default=10); a.add_argument('--out', required=True)
    a.add_argument('--release', default='/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/folding_8mm/test')
    args = a.parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device('cuda:0'); wp.init(); wp_device = wp.get_device('cuda:0'); wp.set_device(wp_device)
    material = PGNDModel(cfg).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    material.load_state_dict(ck['material']); material.eval()
    friction = Friction(np.array([cfg.model.friction.value])).to(device).eval()
    manifest = json.loads(Path(args.cases).read_text())
    fields = manifest['sets'][args.case_set]['fields']
    cases = [dict(zip(fields, c)) for c in manifest['sets'][args.case_set]['cases'][args.first:args.stop]]
    recs = {r['episode_id']: r for r in load_index(args.release)}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for k, case in enumerate(cases, start=args.first):
        path = out / f'{k:05d}.npz'
        if path.exists():
            print(f'{path.name} exists, skipped'); continue
        if int(case['start']) != 0:
            raise SystemExit(f'case {k} starts at {case["start"]}; this dump assumes start 0')
        t0 = time.time()
        pred, Nc = rollout(cfg, material, friction, recs[case['id']], int(case['horizon']), device, wp_device)
        predicted = np.zeros(pred.shape[1], bool); predicted[:Nc] = True
        tmp = path.with_suffix('.tmp.npz')
        np.savez(tmp, pred=pred, rows=np.arange(pred.shape[1], dtype=np.int32), predicted=predicted,
                 id=np.array(case['id']), start=np.int64(case['start']), horizon=np.int64(case['horizon']))
        tmp.rename(path)
        print(f'{k:3d} {case["id"]}: {pred.shape[1]} rows ({Nc} cloth), {time.time() - t0:.0f} s', flush=True)
    meta = dict(method='PGND (gpwm baseline)', checkpoint=str(Path(args.ckpt).resolve()), iteration=int(ck.get('iteration', -1)),
                config=str(Path(args.config).resolve()), case_set=args.case_set, case_manifest=str(Path(args.cases).resolve()),
                case_manifest_sha256=hashlib.sha256(Path(args.cases).read_bytes()).hexdigest(), first=args.first, stop=args.stop,
                rollout='frames 0-3 given (history 3 + current), 898 frames predicted one at a time from its own output; '
                        'recorded gripper points every frame (gripper rows of pred = recorded, unscored); eval mode',
                written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (out / 'META.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
