"""Long-horizon OOD rollout for the PGND cloth-folding baseline (DATASET_SPEC sec. 6b).

Loads a trained checkpoint and autoregressively rolls out up to 1000 frames on a
trajectory. Supports both gripper models:
  * gripper_points=True  -- gripper fed to the model as point clouds (Path B)
  * gripper_forcing=True -- gripper is a kinematic collider; cloth particles
                            within gripper_radius of a closed gripper are
                            kinematically forced (matches train_eval.py:308-332)
The mode is read from the checkpoint's cfg, so the rollout always matches how
the model was trained.

Output: pred .npy, shape [CONTEXT+frames, 9021, 3], float32, WORLD metres:
    9021 = 7021 cloth + 1000 left_ee + 1000 right_ee   (in that order)
The gripper rows are the recorded (GT) gripper points, echoed for visualization
-- they are an input, not a prediction, in both modes.

Usage:
    python experiments/train/rollout_cloth.py \
        --ckpt experiments/log/cloth_fold/train/ckpt/100000.pt \
        --out  experiments/log/cloth_fold/pred_1000frames.npy
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp
import yaml
from omegaconf import OmegaConf

root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'experiments'))

from pgnd.sim import Friction, CacheDiffSimWithFrictionBatch, StaticsBatch, CollidersBatch
from pgnd.material import PGNDModel
from pgnd.data import RealTeleopBatchDataset
from pgnd.data.dataset import fps
from train.train_eval import prepare_gripper_points

# raw-metres <-> sim-frame rotation used by preprocess(): xyz_sim = (xyz_raw @ R.T) ...
R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)

CONTEXT_LEN = 2     # context frames echoed into the output (spec sec. 6b)
NUM_FRAMES = 1000   # predicted frames


def apply_gripper_forcing(x_in, x_out, v_out, g, grip_radius, dt):
    """Post-step kinematic gripper forcing -- a line-for-line port of the
    training loop's forcing block (train_eval.py:308-332).

    Cloth particles within `grip_radius` of a CLOSED gripper are overwritten to
    rigidly follow it (translation + rotation about the gripper's axis).

    x_in       : (1, n_cloth, 3) cloth positions BEFORE the step
    x_out/v_out: (1, n_cloth, 3) post-sim cloth state
    g          : (1, n_grip, 15) gripper poses at the target frame
    returns    : (x, v) with grabbed particles overwritten
    """
    x = x_out.clone()
    v = v_out.clone()
    n_cloth = x.shape[1]
    gripper_xyz = g[:, :, :3]                                   # (1, n_grip, 3)
    gripper_v = g[:, :, 3:6]
    x_from_gripper = x_in[:, None] - gripper_xyz[:, :, None]     # (1, n_grip, n_cloth, 3)
    dist = torch.norm(x_from_gripper, dim=-1)                   # (1, n_grip, n_cloth)
    mask = (dist < grip_radius).unsqueeze(-1).repeat(1, 1, 1, 3)
    gripper_closed = g[:, :, -1] < 0.5                          # (1, n_grip)  1=open 0=closed
    mask = torch.logical_and(mask, gripper_closed[:, :, None, None].repeat(1, 1, n_cloth, 3))
    gripper_v_expand = gripper_v[:, :, None].repeat(1, 1, n_cloth, 1)
    gripper_quat_vel = g[:, :, 10:13]
    gripper_angular_vel = torch.linalg.norm(gripper_quat_vel, dim=-1, keepdims=True)
    gripper_quat_axis = gripper_quat_vel / (gripper_angular_vel + 1e-10)
    grid_from_gripper_axis = x_from_gripper - \
        (gripper_quat_axis[:, :, None] * x_from_gripper).sum(dim=-1, keepdims=True) \
        * gripper_quat_axis[:, :, None]
    gripper_v_expand = torch.cross(gripper_quat_vel[:, :, None], grid_from_gripper_axis, dim=-1) \
        + gripper_v_expand
    for i in range(gripper_xyz.shape[1]):
        m = mask[:, i]
        x[m] = x_in[m] + dt * gripper_v_expand[:, i][m]
        v[m] = gripper_v_expand[:, i][m]
    return x, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='trained material checkpoint .pt')
    ap.add_argument('--cfg', default=None,
                    help='hydra.yaml of the training run (default: alongside ckpt)')
    ap.add_argument('--ood_source', default='experiments/log/data/cloth_ood',
                    help='converted source dataset dir (point at cloth_diverse_v1 for a training episode)')
    ap.add_argument('--episode', default='episode_0000',
                    help='episode name within --ood_source to roll out')
    ap.add_argument('--cache_root', default='experiments/log/cloth_fold/ood_dataset/state',
                    help='preprocessed-cache dir; point at the training cache to reuse frames.pt')
    ap.add_argument('--out', default='experiments/log/cloth_fold/pred_1000frames.npy')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--subsample_method', choices=['fps', 'random'], default='fps',
                    help='cloth subsampling at inference: "fps" matches PGND eval (uniform=True branch); '
                         '"random" matches training-time augmentation (randperm).')
    ap.add_argument('--subsample_seed', type=int, default=0,
                    help='seed for the randperm cloth subsample (only used when --subsample_method=random)')
    args = ap.parse_args()

    ckpt_path = (root / args.ckpt) if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    cfg_path = Path(args.cfg) if args.cfg else ckpt_path.parent.parent / 'hydra.yaml'
    cfg = OmegaConf.load(cfg_path)

    wp.init()
    wp_device = wp.get_device(f'cuda:{args.gpu}')
    device = torch.device(f'cuda:{args.gpu}')

    # --- model ---
    material = PGNDModel(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    material.load_state_dict(state['material'])
    material.eval()
    friction = Friction(np.array([cfg.model.friction.value])).to(device)
    friction.eval()

    # --- trajectory (preprocessed to sim frame) ---
    ds = RealTeleopBatchDataset(
        cfg,
        dataset_root=(root / args.cache_root) if not Path(args.cache_root).is_absolute() else Path(args.cache_root),
        source_data_root=root / args.ood_source,
        device=device,
        num_steps=1,  # only used to load the episode; rollout length set below
        eval_episode_name=args.episode,
        dataset_non_overwrite=True,
    )
    xs_full = ds.episode_xs[0]            # (T, N_cloth_full, 3) sim frame -- BEFORE subsample
    vs_full = ds.episode_vs[0]            # (T, N_cloth_full, 3)
    gp = ds.episode_gripper_points[0]     # (T, n_grippers, P, 3) -- echoed for viz (full)
    gp_v_full = ds.episode_gripper_vs[0]  # (T, n_grippers, P, 3)
    clip_bound = ds.episode_clip_bound[0:1]
    grippers = ds.episode_grippers[0]     # (T, n_grippers, 15) -- forcing-mode gripper poses
    T, n_cloth_full, _ = xs_full.shape
    n_grip, n_grip_pts = gp.shape[1], gp.shape[2]
    num_frames = min(NUM_FRAMES, T - CONTEXT_LEN)   # clamp to episode length
    assert num_frames > 0, f'traj too short: {T} frames'

    # ---- Subsample gripper points (for model input only; full gp kept for viz) ----
    # Match cfg.sim.n_gripper_particles (training uses the same; PGND box=500/gripper).
    # Use FPS per gripper (deterministic), picking from frame 0.
    n_gp_sub_cfg = cfg.sim.n_gripper_particles if 'n_gripper_particles' in cfg.sim else None
    if n_gp_sub_cfg is not None and n_gp_sub_cfg and n_grip_pts > n_gp_sub_cfg:
        enabled_g = torch.ones(n_grip_pts, dtype=torch.bool)
        sub_x_list, sub_v_list = [], []
        for g in range(n_grip):
            idx_g = fps(gp[0, g], enabled_g, int(n_gp_sub_cfg), device='cpu', random_start=False).cpu()
            sub_x_list.append(gp[:, g, idx_g])
            sub_v_list.append(gp_v_full[:, g, idx_g])
        gp_model = torch.stack(sub_x_list, dim=1)            # (T, n_grippers, n_gp_sub, 3)
        gp_v_model = torch.stack(sub_v_list, dim=1)
        print(f'  gripper subsampled (model input): {n_grip_pts} -> {n_gp_sub_cfg} per gripper via FPS')
    else:
        gp_model = gp
        gp_v_model = gp_v_full

    # ---- Subsample cloth to n_particles, matching PGND's data pipeline ----
    # PGND dispatches in dataset.__getitem__:
    #   train=True  (uniform=False)  -> torch.randperm  (training-time aug)
    #   train=False (uniform=True)   -> FPS              (eval -- deterministic, spatial)
    # The rollout previously fed the FULL ~7021 cloth to the model -- a 7x
    # mismatch with training that diluted the per-grid-cell aggregation.
    # We pick ONE subsample (FPS by default, matching PGND eval) and reuse the
    # same indices across the whole rollout so point identity is consistent.
    n_particles = int(cfg.sim.n_particles)
    if n_cloth_full > n_particles:
        if args.subsample_method == 'fps':
            enabled_full = torch.ones(n_cloth_full, dtype=torch.bool)
            cloth_indices = fps(xs_full[CONTEXT_LEN - 1], enabled_full, n_particles,
                                device='cpu', random_start=False).cpu()
            method_note = 'FPS (matches PGND eval)'
        else:  # 'random'
            gen = torch.Generator().manual_seed(args.subsample_seed)
            cloth_indices = torch.randperm(n_cloth_full, generator=gen)[:n_particles]
            method_note = f'randperm (matches training, seed={args.subsample_seed})'
        xs = xs_full[:, cloth_indices]          # (T, n_particles, 3)
        vs = vs_full[:, cloth_indices]
        n_cloth = n_particles
        print(f'  cloth subsampled: {n_cloth_full} -> {n_particles} via {method_note}')
    else:
        cloth_indices = torch.arange(n_cloth_full)
        xs, vs = xs_full, vs_full
        n_cloth = n_cloth_full
        print(f'  cloth kept at {n_cloth_full} pts (no subsample: <= n_particles)')

    forcing = bool(cfg.sim.gripper_forcing)
    use_points = bool(cfg.sim.gripper_points)
    assert forcing != use_points, 'expect exactly one of gripper_forcing / gripper_points'
    mode = 'gripper_forcing' if forcing else 'gripper_points'
    print(f'{args.episode}: {T} frames -> rolling out {num_frames}; mode={mode}; '
          f'{n_cloth} cloth pts, {n_grip}x{n_grip_pts} gripper pts')

    # inverse preprocess transform (sim frame -> world metres)
    with open(ds.root / args.episode / 'info.yaml') as f:
        info = yaml.safe_load(f)
    scale = float(info['preprocess_scale'])
    gtrans = torch.tensor(info['global_translation'], dtype=torch.float32)

    n_history = cfg.sim.n_history
    dt = eval(cfg.sim.dt) if isinstance(cfg.sim.dt, str) else cfg.sim.dt
    grip_radius = float(cfg.model.gripper_radius)

    # --- initial state: current = last context frame; history padded with frame 0 ---
    cur = CONTEXT_LEN - 1
    x = xs[cur:cur + 1].to(device)                          # (1, n_cloth, 3) -- subsampled
    v = vs[cur:cur + 1].to(device)                          # (1, n_cloth, 3) -- same subsample
    # history frames before `cur`, clamped at 0 then padded -> (1, n_cloth, n_history*3)
    his_idx = [max(0, cur - k) for k in range(n_history, 0, -1)]
    x_his = torch.cat([xs[i] for i in his_idx], dim=-1)[None].to(device)
    v_his = torch.cat([vs[i] for i in his_idx], dim=-1)[None].to(device)

    sim = CacheDiffSimWithFrictionBatch(cfg, 1, 1, wp_device, requires_grad=False)
    mu = friction.mu[None].repeat(1, 1)

    # forcing mode: the sim's in-grid gripper collider is active (num_grippers>0);
    # gripper_points mode: no collider (gripper enters as model-input particles).
    n_grip_coll = n_grip if forcing else 0
    colliders = CollidersBatch()
    colliders.init(shape=(1, n_grip_coll), device=wp_device)
    if forcing:
        colliders.initialize_grippers(grippers[cur][None].to(device))
    enabled_cloth = torch.ones(1, n_cloth, dtype=torch.bool, device=device)

    pred_cloth = [xs[0], xs[1]]   # echo the 2 context cloth frames

    with torch.no_grad():
        for step in range(num_frames):
            tgt = CONTEXT_LEN + step                         # frame being predicted (2..1001)

            if forcing:
                # --- gripper_forcing: model sees cloth only; gripper is a
                #     kinematic collider (in sim) + post-step particle forcing ---
                g = grippers[tgt][None].to(device)           # (1, n_grip, 15)
                colliders.update_grippers(g)

                statics = StaticsBatch()
                statics.init(shape=(1, n_cloth), device=wp_device)
                statics.update_clip_bound(clip_bound)
                statics.update_enabled(enabled_cloth.cpu())

                x_in = x.clone()                         # pre-step cloth (forcing ref)
                pred = material(x, v, x_his, v_his, enabled_cloth)
                x_out, v_out = sim(statics, colliders, 0, x, v, mu, pred)
                x, v = apply_gripper_forcing(x_in, x_out, v_out, g, grip_radius, dt)
            else:
                # --- gripper_points: cloth + gripper point clouds fed to model ---
                g_x = gp_model[tgt][None].to(device)         # (1, n_grip, n_gp_sub, 3) -- subsampled
                g_v = gp_v_model[tgt][None].to(device)
                g_x, g_v, g_mask = prepare_gripper_points(cfg, g_x[:, None], g_v[:, None])
                g_x, g_v, g_mask = g_x[:, 0], g_v[:, 0], g_mask[:, 0]   # drop the step axis

                x_in = torch.cat([x, g_x], dim=1)
                v_in = torch.cat([v, g_v], dim=1)
                zhist = torch.zeros((1, g_x.shape[1], n_history * 3), device=device)
                x_his_in = torch.cat([x_his, zhist], dim=1)
                v_his_in = torch.cat([v_his, zhist.clone()], dim=1)
                enabled = torch.cat([enabled_cloth, g_mask], dim=1)

                n_total = x_in.shape[1]
                statics = StaticsBatch()
                statics.init(shape=(1, n_total), device=wp_device)
                statics.update_clip_bound(clip_bound)
                statics.update_enabled(enabled.cpu())

                pred = material(x_in, v_in, x_his_in, v_his_in, enabled)
                x_out, v_out = sim(statics, colliders, 0, x_in, v_in, mu, pred)
                x, v = x_out[:, :n_cloth], v_out[:, :n_cloth]    # keep cloth, drop gripper

            pred_cloth.append(x[0].cpu())

            # advance cloth history (identical for both modes)
            if n_history > 0:
                x_his = torch.cat([x_his.reshape(1, n_cloth, -1, 3)[:, :, 1:],
                                   x[:, :, None]], dim=2).reshape(1, n_cloth, -1)
                v_his = torch.cat([v_his.reshape(1, n_cloth, -1, 3)[:, :, 1:],
                                   v[:, :, None]], dim=2).reshape(1, n_cloth, -1)

            if (step + 1) % 100 == 0:
                print(f'  rollout step {step + 1}/{num_frames}')

    # --- assemble [CONTEXT+frames, 9021, 3]: cloth, left_ee, right_ee ; sim -> world ---
    cloth = torch.stack(pred_cloth, dim=0)                   # (CONTEXT+num_frames, N_cloth, 3)
    grip = gp[:CONTEXT_LEN + num_frames]                     # echoed recorded gripper points
    grip = grip.reshape(grip.shape[0], n_grip * n_grip_pts, 3)
    full_sim = torch.cat([cloth, grip], dim=1)

    R_t = torch.tensor(R)
    full_world = ((full_sim - gtrans) / scale) @ R_t         # invert preprocess

    out_path = (root / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, full_world.numpy().astype(np.float32))
    # Save the cloth indices used so viz can subsample GT cloth identically.
    idx_path = out_path.with_name(out_path.stem + '_cloth_indices.npy')
    np.save(idx_path, cloth_indices.numpy().astype(np.int64))
    print(f'saved {tuple(full_world.shape)} -> {out_path}')
    print(f'saved cloth indices ({len(cloth_indices)}) -> {idx_path}')


if __name__ == '__main__':
    main()
