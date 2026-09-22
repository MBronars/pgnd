#!/usr/bin/env python3
"""Full-episode rollouts of a PGND mpm_drag checkpoint on the held-out test split.

For each chosen test episode: start from its first usable frame (n_history context frames),
then step the model to the end of the episode, with PGND's stock sphere forcing driven by the
recorded patch motion each step (exactly the training-time gripper). Writes one npz per episode
with the predicted and ground-truth points in world metres, plus per-frame mean position error.

    python experiments/scripts/rollout_drag.py --ckpt <ckpt.pt> --out <dir> [--n 6] [--seed 0]
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'experiments'))

from pgnd.sim import Friction, CacheDiffSimWithFrictionBatch, StaticsBatch, CollidersBatch  # noqa: E402
from pgnd.material import PGNDModel  # noqa: E402
from pgnd.data import RealTeleopBatchDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cfg', default=None, help='default: <ckpt>/../../hydra.yaml')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=6, help='test episodes to roll out')
    ap.add_argument('--max_frames', type=int, default=0, help='cap rollout length (0 = to episode end)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()

    ckpt = Path(a.ckpt)
    cfg = OmegaConf.load(a.cfg or ckpt.parent.parent / 'hydra.yaml')
    assert bool(cfg.sim.gripper_forcing) and not bool(cfg.sim.gripper_points), 'mpm_drag runs sphere forcing'
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    wp.init()
    wp_device, device = wp.get_device(f'cuda:{a.gpu}'), torch.device(f'cuda:{a.gpu}')

    material = PGNDModel(cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    material.load_state_dict(state['material'])
    material.eval()
    friction = Friction(np.array([cfg.model.friction.value])).to(device)
    friction.eval()
    print(f'checkpoint {ckpt} (iteration {state.get("iteration")})')

    n_history, skip = cfg.sim.n_history, cfg.train.dataset_skip_frame
    dt = eval(cfg.sim.dt) if isinstance(cfg.sim.dt, str) else cfg.sim.dt
    grip_radius = float(cfg.model.gripper_radius)
    scale = float(cfg.sim.preprocess_scale)

    ds = RealTeleopBatchDataset(cfg, dataset_root=root / 'experiments/log' / cfg.train.dataset_name / 'state',
                                source_data_root=root / 'experiments/log' / str(cfg.train.source_dataset_name),
                                device=device, num_steps=cfg.sim.num_steps_train, train=False,
                                dataset_non_overwrite=True, source_split='test')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    picks = sorted(random.sample(range(len(ds.episode_xs)), min(a.n, len(ds.episode_xs))))
    print(f'test episodes: {len(ds.episode_xs)}, rolling out {len(picks)}')

    for k, ep in enumerate(picks):
        xs, grippers = ds.episode_xs[ep], ds.episode_grippers[ep]
        vs = ds.episode_vs[ep]
        T, N = xs.shape[0], xs.shape[1]
        f0 = n_history * skip
        steps = (T - 1 - f0) // skip
        if a.max_frames:
            steps = min(steps, a.max_frames)
        x = xs[f0][None].to(device)
        v = vs[f0][None].to(device)
        x_his = torch.cat([xs[f0 - (n_history - i) * skip] for i in range(n_history)], dim=-1)[None].to(device)
        v_his = torch.cat([vs[f0 - (n_history - i) * skip] for i in range(n_history)], dim=-1)[None].to(device)
        enabled = torch.ones(1, N, dtype=torch.bool)
        actions = grippers[f0 + skip:f0 + (steps + 1) * skip + 1:skip][None].to(device)   # [1, steps, 1, 15]

        sim = CacheDiffSimWithFrictionBatch(cfg, steps, 1, wp_device, requires_grad=False)
        statics = StaticsBatch(); statics.init(shape=(1, N), device=wp_device)
        statics.update_clip_bound(ds.episode_clip_bound[ep][None])
        statics.update_enabled(enabled)
        colliders = CollidersBatch(); colliders.init(shape=(1, 1), device=wp_device)
        colliders.initialize_grippers(actions[:, 0])
        mu = friction.mu[None].repeat(1, 1)
        enabled_d = enabled.to(device)

        preds = []
        with torch.no_grad():
            for s in range(steps):
                g = actions[:, s]
                colliders.update_grippers(g)
                x_in = x.clone()
                pred = material(x, v, x_his, v_his, enabled_d)
                x, v = sim(statics, colliders, s, x, v, mu, pred)
                # stock sphere forcing: inside the radius while closed, move with the gripper
                gxyz, gv = g[:, :, :3], g[:, :, 3:6]
                from_g = x_in[:, None] - gxyz[:, :, None]
                m = (torch.norm(from_g, dim=-1) < grip_radius).unsqueeze(-1).repeat(1, 1, 1, 3)
                m = torch.logical_and(m, (g[:, :, -1] < .5)[:, :, None, None].repeat(1, 1, N, 3))
                gve = gv[:, :, None].repeat(1, 1, N, 1)
                gqv = g[:, :, 10:13]
                gax = gqv / (torch.linalg.norm(gqv, dim=-1, keepdims=True) + 1e-10)
                gve = torch.cross(gqv[:, :, None], from_g - (gax[:, :, None] * from_g).sum(-1, keepdims=True) * gax[:, :, None],
                                  dim=-1) + gve
                xn, vn = x.clone(), v.clone()
                mi = m[:, 0]
                xn[mi] = x_in[mi] + dt * gve[:, 0][mi]
                vn[mi] = gve[:, 0][mi]
                x, v = xn, vn
                preds.append(x[0].cpu().clone())
                if n_history > 0:
                    x_his = torch.cat([x_his.reshape(1, N, -1, 3)[:, :, 1:], x[:, :, None]], 2).reshape(1, N, -1)
                    v_his = torch.cat([v_his.reshape(1, N, -1, 3)[:, :, 1:], v[:, :, None]], 2).reshape(1, N, -1)

        pred = torch.stack(preds)                                        # [steps, N, 3] sim frame
        gt = xs[f0 + skip:f0 + (steps + 1) * skip:skip]                  # [steps, N, 3]
        ctx = xs[f0 - (n_history - 1) * skip:f0 + 1:skip]                # context frames, for the gif
        err_mm = (pred - gt).norm(dim=-1).mean(-1).numpy() / scale * 1000.0
        name = ds.traj_list[ep] if isinstance(ds.traj_list[ep], str) else str(ds.traj_list[ep])
        np.savez(out / f'ep{k:02d}.npz', pred=pred.numpy(), gt=gt.numpy(), context=ctx.numpy(),
                 gripper=actions[0, :, 0].cpu().numpy(), err_mm=err_mm, scale=scale,
                 episode=name, domain=ds.episode_domain[ep] if ds.episode_domain else '',
                 iteration=state.get('iteration', -1))
        print(f'  ep{k:02d} {name} ({ds.episode_domain[ep] if ds.episode_domain else "?"}): {steps} steps, '
              f'{N} points, mean error {err_mm.mean():.1f} mm, last frame {err_mm[-1]:.1f} mm', flush=True)
    print(f'wrote {len(picks)} rollouts -> {out}')


if __name__ == '__main__':
    main()
