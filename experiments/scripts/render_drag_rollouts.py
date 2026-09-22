#!/usr/bin/env python3
"""Render one GIF per mpm_drag rollout npz: prediction | ground truth, in world metres.

The drag sphere (PGND's gripper) is drawn at its centre, filled while it is closed.

    python experiments/scripts/render_drag_rollouts.py --rollouts <dir> [--fps 10] [--stride 2]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def draw(ax, pts, g_xyz, closed, lims, title):
    ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], s=1.2, c='#3b6ea5', depthshade=False)   # sim y is world up
    ax.scatter([g_xyz[0]], [g_xyz[2]], [-g_xyz[1]], s=90, marker='o',
               facecolors='#d1495b' if closed else 'none', edgecolors='#d1495b', linewidths=1.5)
    ax.set_xlim(lims[0]); ax.set_ylim(lims[2]); ax.set_zlim(-lims[1][1], -lims[1][0])
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=22, azim=-60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rollouts', required=True)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--stride', type=int, default=2, help='render every Nth frame')
    ap.add_argument('--dpi', type=int, default=70)
    a = ap.parse_args()
    d = Path(a.rollouts)
    for npz in sorted(d.glob('ep*.npz')):
        z = np.load(npz, allow_pickle=True)
        pred, gt, grip, err = z['pred'], z['gt'], z['gripper'], z['err_mm']
        scale = float(z['scale'])
        allp = np.concatenate([gt.reshape(-1, 3), pred.reshape(-1, 3)])
        lo, hi = allp.min(0), allp.max(0)
        pad = .05 * (hi - lo + 1e-6)
        lims = list(zip(lo - pad, hi + pad))
        frames = []
        idx = list(range(0, len(gt), a.stride))
        for j in idx:
            fig = plt.figure(figsize=(8, 4))
            closed = grip[j, -1] < .5
            draw(fig.add_subplot(121, projection='3d'), pred[j], grip[j, :3], closed, lims,
                 f'PGND  step {j + 1}/{len(gt)}   error {err[j]:.0f} mm')
            draw(fig.add_subplot(122, projection='3d'), gt[j], grip[j, :3], closed, lims, 'ground truth')
            fig.suptitle(f"{str(z['domain'])}   {'dragging' if closed else 'released'}", fontsize=9)
            fig.tight_layout()
            p = d / f'{npz.stem}_{j:04d}.png'
            fig.savefig(p, dpi=a.dpi)
            plt.close(fig)
            frames.append(p)
        gif = d / f'{npz.stem}.gif'
        imgs = [Image.open(p) for p in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 / a.fps), loop=0)
        for p in frames:
            p.unlink()
        print(f'{gif}  {len(idx)} frames, mean error {err.mean():.1f} mm, {gif.stat().st_size / 1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
