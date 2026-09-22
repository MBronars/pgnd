"""Visualize random training clips for a PGND checkpoint.

For N random training windows -- exactly what the model sees during training --
runs the model forward and renders a 7-frame side-by-side PRED|GT GIF per clip:
  2 context (history) frames + 5 predicted frames  (= cfg.sim.num_steps_train).
Mode (gripper_points vs gripper_forcing) is read from the checkpoint's cfg.

Each clip's GIF plays slowly (fps=3) so every frame is visible. The cloth is
shown at the model's training resolution (cfg.sim.n_particles random points).

Usage:
    python experiments/train/viz_train_clips.py \
        --ckpt experiments/log/cloth_fold/train/ckpt/100000.pt \
        --source experiments/log/data/cloth_diverse_v1 \
        --cache_root experiments/log/cloth_fold/dataset/state \
        --out experiments/log/cloth_fold/train_clips \
        --n_clips 5 --ep_range 0 743
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf

root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'experiments'))

from pgnd.sim import Friction, CacheDiffSimWithFrictionBatch, StaticsBatch, CollidersBatch
from pgnd.material import PGNDModel
from pgnd.data import RealTeleopBatchDataset
from train.train_eval import prepare_gripper_points

R_RAW = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
R_T = torch.tensor(R_RAW)


def draw(ax, cloth, left, right, lims, title):
    ax.clear()
    ax.scatter(cloth[:, 0], cloth[:, 1], cloth[:, 2], s=2.0, c='#1f77b4', alpha=0.6)
    ax.scatter(left[:, 0], left[:, 1], left[:, 2], s=1.5, c='#d62728', alpha=0.6)
    ax.scatter(right[:, 0], right[:, 1], right[:, 2], s=1.5, c='#2ca02c', alpha=0.6)
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ax.set_box_aspect((lims[0][1] - lims[0][0],
                       lims[1][1] - lims[1][0],
                       lims[2][1] - lims[2][0]))
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=25, azim=-60)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])


def sim_to_world(x_sim, scale, gtrans):
    return ((x_sim - gtrans) / scale) @ R_T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cfg', default=None)
    ap.add_argument('--source', required=True, help='converted source dir (cloth_diverse_v1)')
    ap.add_argument('--cache_root', required=True, help='preprocessed cache (.../dataset/state)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_clips', type=int, default=5)
    ap.add_argument('--ep_range', nargs=2, type=int, default=[0, 743])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fps', type=int, default=3, help='gif fps (slow, ~3 fps = each frame visible)')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt) if Path(args.ckpt).is_absolute() else root / args.ckpt
    cfg_path = Path(args.cfg) if args.cfg else ckpt_path.parent.parent / 'hydra.yaml'
    cfg = OmegaConf.load(cfg_path)

    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    wp.init()
    wp_device = wp.get_device(f'cuda:{args.gpu}')
    device = torch.device(f'cuda:{args.gpu}')

    material = PGNDModel(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    material.load_state_dict(state['material'])
    material.eval()
    friction = Friction(np.array([cfg.model.friction.value])).to(device)
    friction.eval()

    num_steps = cfg.sim.num_steps_train
    n_history = cfg.sim.n_history
    skip = cfg.train.dataset_skip_frame
    forcing = bool(cfg.sim.gripper_forcing)
    use_points = bool(cfg.sim.gripper_points)
    assert forcing != use_points
    grip_radius = float(cfg.model.gripper_radius)
    dt = eval(cfg.sim.dt) if isinstance(cfg.sim.dt, str) else cfg.sim.dt

    # Build training-mode dataset across requested episode range
    cfg_local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))  # mutable copy
    cfg_local.train.training_start_episode = args.ep_range[0]
    cfg_local.train.training_end_episode = args.ep_range[1]
    ds = RealTeleopBatchDataset(
        cfg_local,
        dataset_root=(root / args.cache_root) if not Path(args.cache_root).is_absolute() else Path(args.cache_root),
        source_data_root=(root / args.source) if not Path(args.source).is_absolute() else Path(args.source),
        device=device,
        num_steps=num_steps,
        train=True,
        dataset_non_overwrite=True,
    )
    print(f'dataset: {ds.total_len_sim} training windows across {len(ds.episode_xs)} episodes; mode={"forcing" if forcing else "points"}')

    out = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    out.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(ds.total_len_sim), min(args.n_clips, ds.total_len_sim))

    for clip_idx, win_idx in enumerate(indices):
        init_state, actions, gt_states, gripper_states = ds[win_idx]
        x, v, x_his, v_his, clip_bound, enabled, episode_vec = init_state
        ep_idx = int(episode_vec[0]); frame = int(episode_vec[1])
        ep_name = ds.traj_list[ep_idx].name

        with open(ds.traj_list[ep_idx] / 'info.yaml') as info_f:
            info = yaml.safe_load(info_f)
        scale = float(info['preprocess_scale'])
        gtrans = torch.tensor(info['global_translation'], dtype=torch.float32)

        # Add batch dim, move to device
        x_b = x[None].to(device); v_b = v[None].to(device)
        x_his_b = x_his[None].to(device); v_his_b = v_his[None].to(device)
        enabled_b = enabled[None].to(device)
        actions_b = actions[None].to(device)
        gt_x, gt_v = gt_states
        gt_x = gt_x[None].to(device)

        gripper_x, gripper_v = gripper_states
        gripper_x = gripper_x[None].to(device)
        gripper_v = gripper_v[None].to(device)

        n_cloth = x_b.shape[1]
        if use_points:
            gx, gvv, gmask = prepare_gripper_points(cfg, gripper_x, gripper_v)
        n_total = n_cloth if forcing else (n_cloth + gx.shape[2])

        sim_mod = CacheDiffSimWithFrictionBatch(cfg, num_steps, 1, wp_device, requires_grad=False)
        mu = friction.mu[None].repeat(1, 1)
        n_grip_coll = actions_b.shape[2] if forcing else 0
        colliders = CollidersBatch()
        colliders.init(shape=(1, n_grip_coll), device=wp_device)
        if forcing:
            colliders.initialize_grippers(actions_b[:, 0])

        statics = StaticsBatch()
        statics.init(shape=(1, n_total), device=wp_device)
        cb = clip_bound if isinstance(clip_bound, torch.Tensor) else torch.tensor([clip_bound])
        if cb.ndim == 0:
            cb = cb[None]
        statics.update_clip_bound(cb)

        pred_seq = []
        x_cur, v_cur = x_b, v_b
        x_his_cur, v_his_cur = x_his_b, v_his_b

        with torch.no_grad():
            for step in range(num_steps):
                if forcing:
                    colliders.update_grippers(actions_b[:, step])
                    statics.update_enabled(enabled_b.cpu())
                    x_in = x_cur.clone()
                    pred = material(x_cur, v_cur, x_his_cur, v_his_cur, enabled_b)
                    x_out, v_out = sim_mod(statics, colliders, step, x_cur, v_cur, mu, pred)
                    g = actions_b[:, step]
                    gxyz = g[:, :, :3]; gv_lin = g[:, :, 3:6]
                    xfg = x_in[:, None] - gxyz[:, :, None]
                    m = (torch.norm(xfg, dim=-1) < grip_radius).unsqueeze(-1).repeat(1, 1, 1, 3)
                    closed = g[:, :, -1] < 0.5
                    m = torch.logical_and(m, closed[:, :, None, None].repeat(1, 1, n_cloth, 3))
                    gve = gv_lin[:, :, None].repeat(1, 1, n_cloth, 1)
                    gqv = g[:, :, 10:13]
                    gqax = gqv / (torch.linalg.norm(gqv, dim=-1, keepdims=True) + 1e-10)
                    gfa = xfg - (gqax[:, :, None] * xfg).sum(-1, keepdims=True) * gqax[:, :, None]
                    gve = torch.cross(gqv[:, :, None], gfa, dim=-1) + gve
                    xn, vn = x_out.clone(), v_out.clone()
                    for i in range(gxyz.shape[1]):
                        mi = m[:, i]
                        xn[mi] = x_in[mi] + dt * gve[:, i][mi]
                        vn[mi] = gve[:, i][mi]
                    x_cur, v_cur = xn, vn
                else:
                    x_cat = torch.cat([x_cur, gx[:, step]], dim=1)
                    v_cat = torch.cat([v_cur, gvv[:, step]], dim=1)
                    zhist = torch.zeros((1, gx.shape[2], n_history * 3), device=device)
                    x_his_cat = torch.cat([x_his_cur, zhist], dim=1)
                    v_his_cat = torch.cat([v_his_cur, zhist.clone()], dim=1)
                    enabled_cat = torch.cat([enabled_b, gmask[:, step]], dim=1)
                    statics.update_enabled(enabled_cat.cpu())
                    pred = material(x_cat, v_cat, x_his_cat, v_his_cat, enabled_cat)
                    x_out, v_out = sim_mod(statics, colliders, step, x_cat, v_cat, mu, pred)
                    x_cur, v_cur = x_out[:, :n_cloth], v_out[:, :n_cloth]
                pred_seq.append(x_cur[0].detach().cpu().clone())

                if n_history > 0:
                    x_his_cur = torch.cat([x_his_cur.reshape(1, n_cloth, -1, 3)[:, :, 1:],
                                           x_cur[:, :, None]], dim=2).reshape(1, n_cloth, -1)
                    v_his_cur = torch.cat([v_his_cur.reshape(1, n_cloth, -1, 3)[:, :, 1:],
                                           v_cur[:, :, None]], dim=2).reshape(1, n_cloth, -1)

        # Reconstruct 7-frame clip in sim frame
        x_his_frames = x_his.reshape(n_cloth, n_history, 3).permute(1, 0, 2)  # (n_history, N, 3) oldest->newest
        gt_5 = gt_x[0].cpu()                                                   # (5, N, 3) GT predictions
        pred_5 = torch.stack(pred_seq, dim=0)                                  # (5, N, 3) model predictions
        gt_seq_sim = torch.cat([x_his_frames, gt_5], dim=0)                    # (7, N, 3)
        pred_seq_sim = torch.cat([x_his_frames, pred_5], dim=0)                # (7, N, 3) -- context echoed

        # Recorded gripper points for the same 7 frames (visualization only)
        full_gp = ds.episode_gripper_points[ep_idx]
        his_indices = [frame - k * skip for k in range(n_history, 0, -1)]
        pred_indices = [frame + (k + 1) * skip for k in range(num_steps)]
        clip_indices = his_indices + pred_indices
        grip_seq_sim = torch.stack([full_gp[i] for i in clip_indices], dim=0)  # (7, n_grip, P, 3)

        # Sim -> world
        gt_world = sim_to_world(gt_seq_sim, scale, gtrans)
        pred_world = sim_to_world(pred_seq_sim, scale, gtrans)
        grip_world = sim_to_world(grip_seq_sim.reshape(-1, 3), scale, gtrans).reshape(grip_seq_sim.shape)

        # Axis limits from GT cloth + grippers
        allpts = torch.cat([gt_world.reshape(-1, 3), grip_world.reshape(-1, 3)], dim=0)
        lo = allpts.min(0).values.numpy()
        hi = allpts.max(0).values.numpy()
        pad = 0.05 * (hi - lo + 1e-6)
        lims = list(zip(lo - pad, hi + pad))

        # Render
        clip_label = f'clip_{clip_idx:02d}_{ep_name}_f{frame}'
        clip_dir = out / clip_label
        clip_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        for j in range(n_history + num_steps):
            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
            left = grip_world[j, 0]
            right = grip_world[j, 1]
            if j < n_history:
                tag = f'CONTEXT {j + 1}/{n_history}'
            else:
                tag = f'PRED step {j - n_history + 1}/{num_steps}'
            err_mm = (torch.norm(pred_world[j] - gt_world[j], dim=-1).mean()).item() * 1000.0
            draw(ax1, pred_world[j].numpy(), left.numpy(), right.numpy(), lims,
                 f'PRED  {tag}   cloth L2 {err_mm:.1f} mm')
            draw(ax2, gt_world[j].numpy(), left.numpy(), right.numpy(), lims,
                 f'GT  {tag}')
            fig.suptitle(f'{ep_name}   src frame {clip_indices[j]}', fontsize=9)
            fig.tight_layout()
            p = clip_dir / f'{j:02d}.png'
            fig.savefig(p, dpi=90)
            plt.close(fig)
            frame_paths.append(p)

        gif_path = out / f'{clip_label}.gif'
        frames_pil = [Image.open(p) for p in frame_paths]
        dur_ms = int(1000 / args.fps)
        frames_pil[0].save(gif_path, save_all=True, append_images=frames_pil[1:],
                           duration=dur_ms, loop=0)
        print(f'clip {clip_idx}: {ep_name} f{frame} -> {gif_path}')


if __name__ == '__main__':
    main()
