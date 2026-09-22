"""Visualize short TRAINING clips: GT vs the model's num_steps_train-step prediction.

For debugging -- shows the actual short windows the model is trained on (length
num_steps_train, default 5), NOT the 1000-step rollout. Picks a few windows from
an episode, runs the model exactly as training does (teacher-forced GT initial
state + n_history history frames, then autoregressive for clip_len steps), and
renders GT vs predicted cloth for every frame of the clip.

Usage:
    python experiments/train/viz_train_clip.py \
        --ckpt    experiments/log/cloth_fold/train/ckpt/042000.pt \
        --episode episode_0000 \
        --out     experiments/log/cloth_fold/viz_clip_ep0000
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from train.train_eval import prepare_gripper_points

R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)


def draw(ax, cloth, left, right, lims, title):
    ax.clear()
    ax.scatter(cloth[:, 0], cloth[:, 1], cloth[:, 2], s=1.0, c='#1f77b4', alpha=0.5)
    ax.scatter(left[:, 0], left[:, 1], left[:, 2], s=1.5, c='#d62728', alpha=0.6)
    ax.scatter(right[:, 0], right[:, 1], right[:, 2], s=1.5, c='#2ca02c', alpha=0.6)
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ax.set_box_aspect((lims[0][1] - lims[0][0], lims[1][1] - lims[1][0], lims[2][1] - lims[2][0]))
    ax.set_title(title, fontsize=8)
    ax.view_init(elev=25, azim=-60)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cfg', default=None, help='hydra.yaml (default: alongside ckpt)')
    ap.add_argument('--source', default='experiments/log/data/cloth_diverse_v1',
                    help='converted source dataset dir')
    ap.add_argument('--episode', default='episode_0000')
    ap.add_argument('--cache_root', default='experiments/log/cloth_fold/dataset/state',
                    help='preprocessed-cache dir (training cache reuses frames.pt)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--num_clips', type=int, default=4)
    ap.add_argument('--clip_len', type=int, default=None,
                    help='steps per clip (default: cfg.sim.num_steps_train)')
    args = ap.parse_args()

    ckpt_path = (root / args.ckpt) if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    cfg = OmegaConf.load(Path(args.cfg) if args.cfg else ckpt_path.parent.parent / 'hydra.yaml')
    clip_len = args.clip_len or cfg.sim.num_steps_train
    n_history = cfg.sim.n_history

    wp.init()
    wp_device = wp.get_device(f'cuda:{args.gpu}')
    device = torch.device(f'cuda:{args.gpu}')

    material = PGNDModel(cfg).to(device)
    material.load_state_dict(torch.load(ckpt_path, map_location=device)['material'])
    material.eval()
    friction = Friction(np.array([cfg.model.friction.value])).to(device)
    friction.eval()

    ds = RealTeleopBatchDataset(
        cfg,
        dataset_root=(root / args.cache_root) if not Path(args.cache_root).is_absolute() else Path(args.cache_root),
        source_data_root=root / args.source,
        device=device,
        num_steps=1,
        eval_episode_name=args.episode,
        dataset_non_overwrite=True,
    )
    xs = ds.episode_xs[0]                  # (T, N_cloth, 3) sim frame
    vs = ds.episode_vs[0]
    gp = ds.episode_gripper_points[0]      # (T, n_grip, P, 3)
    gp_v = ds.episode_gripper_vs[0]
    clip_bound = ds.episode_clip_bound[0:1]
    T, n_cloth, _ = xs.shape

    with open(ds.root / args.episode / 'info.yaml') as f:
        info = yaml.safe_load(f)
    scale = float(info['preprocess_scale'])
    gtrans = torch.tensor(info['global_translation'], dtype=torch.float32)
    R_t = torch.tensor(R)

    def to_world(p):  # (.., 3) sim frame -> world metres
        return ((p - gtrans) / scale) @ R_t

    # clip start frames: evenly spaced, leaving room for history + clip_len
    lo, hi = n_history, T - clip_len - 1
    assert hi > lo, f'episode too short ({T} frames) for clip_len={clip_len}'
    starts = np.linspace(lo, hi, args.num_clips, dtype=int).tolist()
    print(f'{args.episode}: {T} frames, clip_len={clip_len}, n_history={n_history}, '
          f'starts={starts}')

    sim = CacheDiffSimWithFrictionBatch(cfg, 1, 1, wp_device, requires_grad=False)
    mu = friction.mu[None].repeat(1, 1)

    clips = []  # each: dict(start, gt[clip_len+1], pred[clip_len+1], err[clip_len+1])
    with torch.no_grad():
        for f0 in starts:
            x = xs[f0:f0 + 1].to(device)
            v = vs[f0:f0 + 1].to(device)
            his_idx = [max(0, f0 - k) for k in range(n_history, 0, -1)]
            x_his = torch.cat([xs[i] for i in his_idx], dim=-1)[None].to(device)
            v_his = torch.cat([vs[i] for i in his_idx], dim=-1)[None].to(device)

            pred_cloth = [xs[f0]]
            for step in range(clip_len):
                tgt = f0 + 1 + step
                g_x = gp[tgt][None].to(device)
                g_v = gp_v[tgt][None].to(device)
                g_x, g_v, g_mask = prepare_gripper_points(cfg, g_x[:, None], g_v[:, None])
                g_x, g_v, g_mask = g_x[:, 0], g_v[:, 0], g_mask[:, 0]

                x_in = torch.cat([x, g_x], dim=1)
                v_in = torch.cat([v, g_v], dim=1)
                zhist = torch.zeros((1, g_x.shape[1], n_history * 3), device=device)
                x_his_in = torch.cat([x_his, zhist], dim=1)
                v_his_in = torch.cat([v_his, zhist.clone()], dim=1)
                enabled = torch.cat([torch.ones(1, n_cloth, dtype=torch.bool, device=device),
                                     g_mask], dim=1)

                statics = StaticsBatch()
                statics.init(shape=(1, x_in.shape[1]), device=wp_device)
                statics.update_clip_bound(clip_bound)
                statics.update_enabled(enabled.cpu())
                colliders = CollidersBatch()
                colliders.init(shape=(1, 0), device=wp_device)

                pred = material(x_in, v_in, x_his_in, v_his_in, enabled)
                x_out, v_out = sim(statics, colliders, 0, x_in, v_in, mu, pred)
                x, v = x_out[:, :n_cloth], v_out[:, :n_cloth]
                pred_cloth.append(x[0].cpu())

                if n_history > 0:
                    x_his = torch.cat([x_his.reshape(1, n_cloth, -1, 3)[:, :, 1:],
                                       x[:, :, None]], dim=2).reshape(1, n_cloth, -1)
                    v_his = torch.cat([v_his.reshape(1, n_cloth, -1, 3)[:, :, 1:],
                                       v[:, :, None]], dim=2).reshape(1, n_cloth, -1)

            pred = torch.stack(pred_cloth, dim=0)              # (clip_len+1, n_cloth, 3)
            gt = xs[f0:f0 + clip_len + 1]                      # (clip_len+1, n_cloth, 3)
            err = (pred - gt).norm(dim=-1).mean(-1) * 1000.0   # mm, per frame
            clips.append(dict(start=f0,
                              gt=to_world(gt).numpy(),
                              pred=to_world(pred).numpy(),
                              grip=to_world(gp[f0:f0 + clip_len + 1].reshape(
                                  clip_len + 1, -1, 3)).numpy(),
                              n_grip_pts=gp.shape[2],
                              err=err.numpy()))
            print(f'  clip @frame {f0:4d}:  per-step cloth L2 (mm) = '
                  f'{[round(e, 1) for e in err.tolist()]}')

    # --- render: 2 rows per clip (GT, PRED) x (clip_len+1) columns ---
    allpts = np.concatenate([c['gt'].reshape(-1, 3) for c in clips], axis=0)
    lo_, hi_ = allpts.min(0), allpts.max(0)
    pad = 0.05 * (hi_ - lo_ + 1e-6)
    lims = list(zip(lo_ - pad, hi_ + pad))

    ncol = clip_len + 1
    nrow = 2 * len(clips)
    fig = plt.figure(figsize=(2.6 * ncol, 2.6 * nrow))
    for ci, c in enumerate(clips):
        P = c['n_grip_pts']
        for j in range(ncol):
            gl, gr = c['grip'][j][:P], c['grip'][j][P:]
            a_gt = fig.add_subplot(nrow, ncol, 2 * ci * ncol + j + 1, projection='3d')
            draw(a_gt, c['gt'][j], gl, gr, lims,
                 f'GT  clip@{c["start"]}  +{j}')
            a_pr = fig.add_subplot(nrow, ncol, (2 * ci + 1) * ncol + j + 1, projection='3d')
            draw(a_pr, c['pred'][j], gl, gr, lims,
                 f'PRED  +{j}  ({c["err"][j]:.1f} mm)')
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'train_clips.png', dpi=110)
    plt.close(fig)
    print(f'saved -> {out / "train_clips.png"}')


if __name__ == '__main__':
    main()
