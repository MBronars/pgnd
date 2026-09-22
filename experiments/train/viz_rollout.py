"""Visualize a cloth-folding rollout: predicted vs ground-truth.

Takes pred_1000frames.npy from rollout_cloth.py ([n_frames, 9021, 3], world
metres) and the converted OOD episode as ground truth, and renders:
  - <out>/rollout.mp4         side-by-side pred | GT animation
  - <out>/rollout_strip.png   static panels at a few key frames

Usage:
    python experiments/train/viz_rollout.py \
        --pred experiments/log/cloth_fold/pred_1000frames.npy \
        --gt   experiments/log/data/cloth_ood/episode_0000 \
        --out  experiments/log/cloth_fold/viz
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))
from pgnd.ffmpeg import make_video

def load_gt(gt_dir):
    """GT cloth + gripper points (list, num_grippers entries) for an episode."""
    cloth = np.load(gt_dir / 'traj.npz')['xyz']            # (T, N_cloth, 3)
    gp = np.load(gt_dir / 'gripper_points.npz')
    keys = [k for k in ('left', 'right') if k in gp]       # 1 (pushing) or 2 (cloth)
    grippers = [gp[k] for k in keys]                       # each (T, P, 3)
    return cloth, grippers


def split(frame, n_cloth, n_ee, num_grippers):
    """frame: (n_cloth + num_grippers*n_ee, 3) -> (cloth, [g0, g1, ...])"""
    parts = [frame[:n_cloth]]
    for g in range(num_grippers):
        parts.append(frame[n_cloth + g*n_ee : n_cloth + (g+1)*n_ee])
    return parts[0], parts[1:]


_GRIPPER_COLORS = ('#d62728', '#2ca02c', '#ff7f0e', '#9467bd')

def draw(ax, cloth, grippers, lims, title):
    ax.clear()
    ax.scatter(cloth[:, 0], cloth[:, 1], cloth[:, 2], s=1.0, c='#1f77b4', alpha=0.5)
    for g, pts in enumerate(grippers):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.5,
                   c=_GRIPPER_COLORS[g % len(_GRIPPER_COLORS)], alpha=0.6)
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ax.set_box_aspect((lims[0][1] - lims[0][0],
                       lims[1][1] - lims[1][0],
                       lims[2][1] - lims[2][0]))
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=25, azim=-60)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--gt', required=True, help='converted OOD episode dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stride', type=int, default=10, help='frame subsample for the video')
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--load_skip', type=int, default=1,
                    help='match training/rollout dataset_load_skip_frame: GT is '
                         'subsampled with [::load_skip] so each pred[t] aligns with '
                         'gt[t] (otherwise pred runs at 1/load_skip the GT rate).')
    args = ap.parse_args()

    pred = np.load(args.pred)                              # (T_pred, n_cloth + ng*n_ee, 3)
    gt_cloth, gt_grippers = load_gt(Path(args.gt))         # list of (T, P, 3)
    if args.load_skip > 1:
        gt_cloth = gt_cloth[::args.load_skip]
        gt_grippers = [g[::args.load_skip] for g in gt_grippers]
        print(f'GT subsampled by load_skip={args.load_skip}: '
              f'{gt_cloth.shape[0]} frames after subsample')
    num_grippers = len(gt_grippers)
    n_ee = gt_grippers[0].shape[1]
    # Derive n_cloth from pred shape (rollout may subsample cloth to n_particles
    # to match the training pipeline; pred = [cloth_pred(N_p) | g1(P) | ... | gK(P)]).
    n_cloth = pred.shape[1] - num_grippers * n_ee
    # If the rollout saved cloth indices alongside, use them to subsample GT for
    # apples-to-apples comparison; otherwise assume pred[:n_cloth] aligns with
    # gt_cloth[:n_cloth] (no subsample).
    pred_path = Path(args.pred)
    idx_path = pred_path.with_name(pred_path.stem + '_cloth_indices.npy')
    if idx_path.exists() and n_cloth != gt_cloth.shape[1]:
        cloth_indices = np.load(idx_path)
        assert len(cloth_indices) == n_cloth, \
            f'indices ({len(cloth_indices)}) != n_cloth from pred ({n_cloth})'
        gt_cloth = gt_cloth[:, cloth_indices]
        print(f'using cloth_indices ({len(cloth_indices)}) from {idx_path.name}')
    else:
        assert n_cloth == gt_cloth.shape[1], \
            f'pred n_cloth={n_cloth} but GT has {gt_cloth.shape[1]} and no indices file'
    T = min(pred.shape[0], gt_cloth.shape[0])
    print(f'pred frames {pred.shape[0]}, gt frames {gt_cloth.shape[0]} -> compare {T}'
          f'  (n_cloth={n_cloth}, n_ee={n_ee})')

    # shared fixed axis limits from GT (stable animation)
    allpts = np.concatenate([gt_cloth[:T].reshape(-1, 3)]
                            + [g[:T].reshape(-1, 3) for g in gt_grippers], axis=0)
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.05 * (hi - lo + 1e-6)
    lims = list(zip(lo - pad, hi + pad))

    # per-frame cloth L2 error (mm), for the title
    cloth_err_mm = np.linalg.norm(pred[:T, :n_cloth] - gt_cloth[:T], axis=-1).mean(-1) * 1000.0

    out = Path(args.out)
    frame_dir = out / 'frames'
    frame_dir.mkdir(parents=True, exist_ok=True)

    # --- animation ---
    idxs = list(range(0, T, args.stride))
    for k, f in enumerate(idxs):
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        pc, pg = split(pred[f], n_cloth, n_ee, num_grippers)
        draw(ax1, pc, pg, lims, f'PRED  frame {f}   cloth L2 {cloth_err_mm[f]:.1f} mm')
        draw(ax2, gt_cloth[f], [g[f] for g in gt_grippers], lims, f'GROUND TRUTH  frame {f}')
        fig.tight_layout()
        fig.savefig(frame_dir / f'{k:04d}.png', dpi=80)
        plt.close(fig)
        if (k + 1) % 20 == 0:
            print(f'  rendered {k + 1}/{len(idxs)} frames')
    make_video(frame_dir, out / 'rollout.mp4', '%04d.png', args.fps)
    print(f'video -> {out / "rollout.mp4"}')

    # --- static strip at key frames ---
    keys = [f for f in (0, 100, 250, 500, T - 1) if f < T]
    fig = plt.figure(figsize=(4 * len(keys), 8))
    for j, f in enumerate(keys):
        pc, pg = split(pred[f], n_cloth, n_ee, num_grippers)
        a1 = fig.add_subplot(2, len(keys), j + 1, projection='3d')
        draw(a1, pc, pg, lims, f'PRED f{f}  ({cloth_err_mm[f]:.1f} mm)')
        a2 = fig.add_subplot(2, len(keys), len(keys) + j + 1, projection='3d')
        draw(a2, gt_cloth[f], [g[f] for g in gt_grippers], lims, f'GT f{f}')
    fig.tight_layout()
    fig.savefig(out / 'rollout_strip.png', dpi=110)
    plt.close(fig)
    print(f'strip -> {out / "rollout_strip.png"}')
    print(f'cloth L2 (mm) @ frames 10/100/500/{T-1}: '
          f'{cloth_err_mm[min(10,T-1)]:.1f} / {cloth_err_mm[min(100,T-1)]:.1f} / '
          f'{cloth_err_mm[min(500,T-1)]:.1f} / {cloth_err_mm[T-1]:.1f}')


if __name__ == '__main__':
    main()
