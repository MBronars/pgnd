"""Score pickup val rollouts per pickup_dataset_docs.md sec 10.

Metric: per-frame per-particle L2 distance (cm) on OBJECT particles
(`obj_ids == 0` in the source). The pred.npy from rollout_cloth.py has rows
0..n_particles-1 = the (subsampled or full) object particles in the same
order as the GT object particles -- so we just compare row-by-row.

For each frame t in [1, 38] (the model predicts frames CONTEXT_LEN..T-1):
  cm_per_frame[t] = mean over object particles of ||pred_obj[t] - gt_obj[t]||_2 * 100

Reported (from the spec):
  - per-horizon values at t = 5, 10, 20, 30, 38
  - TH10  = mean(cm_per_frame[1..10])
  - All   = mean(cm_per_frame[1..38])
  - Final = cm_per_frame[38]

Outputs:
  - <val_root>/episode_NNNNNN/metrics.npz
  - <val_root>/episode_NNNNNN/summary.json  (per-traj, schema matches aggregate_v9_vs_physctrl.py)
  - <val_root>/per_frame_l2_cm.npy          (per-traj, F=38, copy of metrics)
  - <val_root>/summary.json                 (aggregated across episodes)

Usage:
  python experiments/train/score_pickup_val.py \\
      --val_root experiments/log/pickup/val_rollouts \\
      --gt_root  experiments/log/data/pickup/sub_episodes_v
"""
import argparse
import json
from pathlib import Path

import numpy as np


SPEC_HORIZONS = (5, 10, 20, 30, 38)
CONTEXT_LEN = 2   # pred frames 0..1 echo context; 2..T-1 are predicted


def score_episode(pred_path: Path, gt_dir: Path):
    """Returns dict of scalar metrics + per-frame L2 (cm) array."""
    pred = np.load(pred_path)                                  # (T_pred, n_obj + n_grip, 3) world m
    idx_path = pred_path.with_name(pred_path.stem + '_cloth_indices.npy')
    obj_indices = np.load(idx_path) if idx_path.exists() else None
    if obj_indices is not None:
        n_obj = obj_indices.shape[0]
    else:
        n_obj = pred.shape[1]   # fallback (won't happen for pickup)

    gt = np.load(gt_dir / 'traj.npz')['xyz']                  # (T_gt, N_obj_full, 3)
    if obj_indices is not None and obj_indices.shape[0] != gt.shape[1]:
        gt_obj = gt[:, obj_indices]
    else:
        gt_obj = gt

    pred_obj = pred[:, :n_obj]                                 # (T_pred, n_obj, 3)
    T = min(pred_obj.shape[0], gt_obj.shape[0])
    pred_obj = pred_obj[:T]
    gt_obj = gt_obj[:T]

    # per-frame L2 cm; rollout includes context echo at rows 0..CONTEXT_LEN-1.
    # We score frames [CONTEXT_LEN, T) i.e. predicted frames; index them 1..(T-CONTEXT_LEN).
    per_frame_cm = np.linalg.norm(pred_obj - gt_obj, axis=-1).mean(-1) * 100.0  # (T,)
    pred_only = per_frame_cm[CONTEXT_LEN:]                                       # (F,)
    F = pred_only.shape[0]

    # Horizon metrics (1-indexed: t=k means k frames after context end).
    horizons = {}
    for h in SPEC_HORIZONS:
        if F >= h:
            horizons[f't={h}'] = float(pred_only[h - 1])
    th10 = float(pred_only[:min(10, F)].mean()) if F else float('nan')
    all_  = float(pred_only.mean()) if F else float('nan')
    final = float(pred_only[-1]) if F else float('nan')
    return {
        'n_obj': int(n_obj),
        'F': int(F),
        'horizons_cm': horizons,
        'TH10_cm': th10,
        'All_cm': all_,
        'Final_cm': final,
        'per_frame_cm': pred_only.tolist(),
    }, pred_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_root', required=True)
    ap.add_argument('--gt_root', required=True)
    args = ap.parse_args()
    val_root = Path(args.val_root)
    gt_root = Path(args.gt_root)
    episodes = sorted(d.name for d in val_root.iterdir() if d.is_dir() and d.name.startswith('episode_'))
    print(f'scoring {len(episodes)} episodes from {val_root}')

    all_scores = {}
    per_frame_stack = []
    for ep in episodes:
        pred_path = val_root / ep / 'pred.npy'
        gt_dir = gt_root / ep
        if not pred_path.exists() or not (gt_dir / 'traj.npz').exists():
            print(f'  {ep}: SKIP')
            continue
        scores, per_frame = score_episode(pred_path, gt_dir)
        per_ep_summary = val_root / ep / 'summary.json'
        with open(per_ep_summary, 'w') as f:
            json.dump(scores, f, indent=2)
        np.save(val_root / ep / 'per_frame_l2_cm.npy', per_frame.astype(np.float32))
        all_scores[ep] = scores
        per_frame_stack.append(per_frame)
        h_str = ' '.join(f'{k}={v:5.2f}' for k, v in scores['horizons_cm'].items())
        print(f'  {ep}: F={scores["F"]}  {h_str}  TH10={scores["TH10_cm"]:5.2f}  All={scores["All_cm"]:5.2f}  Final={scores["Final_cm"]:5.2f}')

    if per_frame_stack:
        # Mean across episodes; trim to shortest F (should be same for all -- T=40 means F=38).
        Fmin = min(pf.shape[0] for pf in per_frame_stack)
        stack = np.stack([pf[:Fmin] for pf in per_frame_stack], axis=0)
        mean_per_frame = stack.mean(0)
        agg_horizons = {f't={h}': float(mean_per_frame[h - 1]) for h in SPEC_HORIZONS if Fmin >= h}
        agg = {
            'n_episodes': len(per_frame_stack),
            'F': int(Fmin),
            'horizons_cm': agg_horizons,
            'TH10_cm': float(mean_per_frame[:min(10, Fmin)].mean()),
            'All_cm':  float(mean_per_frame.mean()),
            'Final_cm': float(mean_per_frame[-1]),
            'per_frame_cm_mean': mean_per_frame.tolist(),
        }
        print('\nMEAN across episodes:')
        print('  ' + ' '.join(f'{k}={v:5.2f}' for k, v in agg['horizons_cm'].items())
              + f"  TH10={agg['TH10_cm']:5.2f}  All={agg['All_cm']:5.2f}  Final={agg['Final_cm']:5.2f}")
        out = {'episodes': all_scores, 'mean': agg}
    else:
        out = {'episodes': {}, 'mean': None}

    with open(val_root / 'summary.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'summary -> {val_root / "summary.json"}')


if __name__ == '__main__':
    main()
