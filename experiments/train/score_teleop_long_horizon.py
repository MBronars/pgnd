"""Per-frame stats for teleop_long_horizon rollouts.

For each episode, computes per-frame L2 (cm) on OBJECT particles (rows 0..n_obj-1
of pred.npy) against GT (traj.npz['xyz'] in the converted episode dir).

Aggregates across episodes:
  - mean/median/p25/p75/p95 per frame
  - per-horizon scalars at t = 10, 25, 50, 100, 150, 199
  - TH10  = mean(per_frame_cm[1..10])     -- short-horizon
  - TH50  = mean(per_frame_cm[1..50])
  - All   = mean(per_frame_cm[1..F])
  - Final = per_frame_cm[F]                -- worst-case at horizon end

Outputs:
  <rollouts>/episode_NNNN/metrics.npz                       (per-traj per-frame array)
  <rollouts>/episode_NNNN/summary.json                      (per-traj scalars)
  <rollouts>/per_frame_l2_cm_all.npy                        (n_eps, F) stacked array
  <rollouts>/summary.json                                   (aggregate stats)
"""
import argparse
import json
from pathlib import Path

import numpy as np


CONTEXT_LEN = 2     # pred frames 0..CONTEXT_LEN-1 echo the GT context
SPEC_HORIZONS = (10, 25, 50, 100, 150, 199)


def per_frame_cm(pred_path: Path, gt_dir: Path):
    pred = np.load(pred_path)                                  # (T_pred, n_obj+n_grip, 3) m
    idx_path = pred_path.with_name(pred_path.stem + '_cloth_indices.npy')
    n_obj = np.load(idx_path).shape[0] if idx_path.exists() else None
    gt = np.load(gt_dir / 'traj.npz')['xyz']                  # (T_gt, n_obj_full, 3)
    if n_obj is None:
        n_obj = gt.shape[1]
    pred_obj = pred[:, :n_obj]
    T = min(pred_obj.shape[0], gt.shape[0])
    pred_obj = pred_obj[:T]
    gt_obj = gt[:T]
    cm = np.linalg.norm(pred_obj - gt_obj, axis=-1).mean(-1) * 100.0  # (T,)
    return cm[CONTEXT_LEN:]                                          # drop context echo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rollouts', required=True)
    ap.add_argument('--gt_root', required=True)
    args = ap.parse_args()

    rollouts = Path(args.rollouts)
    gt_root = Path(args.gt_root)
    eps = sorted(d.name for d in rollouts.iterdir() if d.is_dir() and d.name.startswith('episode_'))
    print(f'scoring {len(eps)} episodes from {rollouts}')

    per_ep = {}
    stack = []
    for ep in eps:
        pred_path = rollouts / ep / 'pred.npy'
        gt_dir = gt_root / ep
        if not pred_path.exists() or not (gt_dir / 'traj.npz').exists():
            print(f'  {ep}: SKIP')
            continue
        cm = per_frame_cm(pred_path, gt_dir)                            # (F,)
        F = cm.shape[0]
        horizons = {f't={h}': float(cm[h - 1]) for h in SPEC_HORIZONS if F >= h}
        scores = {
            'F': int(F),
            'horizons_cm': horizons,
            'TH10_cm':  float(cm[:min(10, F)].mean()),
            'TH50_cm':  float(cm[:min(50, F)].mean()),
            'All_cm':   float(cm.mean()),
            'Final_cm': float(cm[-1]),
            'per_frame_cm': cm.tolist(),
        }
        np.savez(rollouts / ep / 'metrics.npz',
                 per_frame_cm=cm.astype(np.float32))
        with open(rollouts / ep / 'summary.json', 'w') as f:
            json.dump(scores, f, indent=2)
        per_ep[ep] = scores
        stack.append(cm)
        h_str = ' '.join(f'{k}={v:6.2f}' for k, v in horizons.items())
        print(f'  {ep}: F={F}  {h_str}  TH10={scores["TH10_cm"]:5.2f}  TH50={scores["TH50_cm"]:5.2f}  All={scores["All_cm"]:5.2f}  Final={scores["Final_cm"]:5.2f}')

    if not stack:
        print('no scored episodes')
        return

    # Aggregate (trim to shortest F if any variation)
    Fmin = min(c.shape[0] for c in stack)
    M = np.stack([c[:Fmin] for c in stack], axis=0)               # (n_eps, F)
    mean_pf  = M.mean(0)
    med_pf   = np.median(M, 0)
    p25_pf   = np.percentile(M, 25, 0)
    p75_pf   = np.percentile(M, 75, 0)
    p95_pf   = np.percentile(M, 95, 0)
    np.save(rollouts / 'per_frame_l2_cm_all.npy', M.astype(np.float32))

    horizons = {f't={h}': float(mean_pf[h - 1]) for h in SPEC_HORIZONS if Fmin >= h}
    agg = {
        'n_episodes': len(stack),
        'F': int(Fmin),
        'horizons_cm_mean':   horizons,
        'horizons_cm_median': {f't={h}': float(med_pf[h - 1]) for h in SPEC_HORIZONS if Fmin >= h},
        'TH10_cm':  float(mean_pf[:min(10, Fmin)].mean()),
        'TH50_cm':  float(mean_pf[:min(50, Fmin)].mean()),
        'All_cm':   float(mean_pf.mean()),
        'Final_cm': float(mean_pf[-1]),
        'per_frame_cm_mean':   mean_pf.tolist(),
        'per_frame_cm_median': med_pf.tolist(),
        'per_frame_cm_p25':    p25_pf.tolist(),
        'per_frame_cm_p75':    p75_pf.tolist(),
        'per_frame_cm_p95':    p95_pf.tolist(),
    }
    print('\nAGGREGATE (mean across episodes):')
    print('  horizons (cm):  ' + '  '.join(f'{k}={v:6.2f}' for k, v in horizons.items()))
    print(f'  TH10={agg["TH10_cm"]:5.2f}  TH50={agg["TH50_cm"]:5.2f}  All={agg["All_cm"]:5.2f}  Final={agg["Final_cm"]:5.2f}')

    out = {'episodes': per_ep, 'aggregate': agg}
    with open(rollouts / 'summary.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'summary -> {rollouts / "summary.json"}')


if __name__ == '__main__':
    main()
