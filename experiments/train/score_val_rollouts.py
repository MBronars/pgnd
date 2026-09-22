"""Score per-episode rollouts (pred.npy) against GT for the 8 val episodes.

Reads each <val_rollouts>/episode_NNNN/pred.npy + corresponding GT in
cloth_diverse_v1/<episode_NNNN>/traj.npz, subsamples GT with the same
load_skip used at rollout time, and reports:
  - Cloth L2 to GT (mm) at frames {10, 50, 100, 500, 1000}  (last-clamped)
  - Mesh edge distortion (mm) at the same frames
      (K=8 KNN edges on the FIRST-FRAME predicted cloth topology;
       compare predicted edge length vs GT edge length at each query frame)

Outputs:
  - <val_rollouts>/episode_NNNN/metrics.npz
        keys: frames, cloth_l2_mm, mesh_edge_distortion_mm,
              cloth_points_pred (T,N,3), cloth_points_gt (T,N,3)
  - <val_rollouts>/summary.json
        per-episode metrics + mean across episodes
"""
import argparse
import json
from pathlib import Path

import numpy as np


SPEC_FRAMES = (10, 50, 100, 500, 1000)
K_KNN = 8


def knn_edges(points, k):
    """First-frame topology: for each point, indices of its k nearest neighbors.

    Returns (N, k) int array of neighbor indices (excluding self).
    """
    N = points.shape[0]
    # pairwise distance (N, N) -- N<=2000 here, ~16M floats, fine
    d2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return np.argpartition(d2, kth=k, axis=1)[:, :k]


def edge_lengths(points, edges):
    """edges: (N, k); points: (N, 3); returns (N, k) Euclidean lengths."""
    i = np.arange(points.shape[0])[:, None]               # (N, 1)
    return np.linalg.norm(points[i] - points[edges], axis=-1)


def score_episode(pred_path: Path, gt_dir: Path, load_skip: int, frames=SPEC_FRAMES, k=K_KNN):
    pred = np.load(pred_path)                              # (T_pred, n_cloth + ng*P, 3)
    idx_path = pred_path.with_name(pred_path.stem + '_cloth_indices.npy')
    cloth_indices = np.load(idx_path) if idx_path.exists() else None
    n_cloth = cloth_indices.shape[0] if cloth_indices is not None else None

    gt = np.load(gt_dir / 'traj.npz')['xyz']               # (T_gt, N_full, 3)
    if load_skip > 1:
        gt = gt[::load_skip]
    if cloth_indices is not None:
        gt_cloth = gt[:, cloth_indices]                    # (T_gt, n_cloth, 3)
    else:
        # Fall back: assume pred[:n_cloth] aligns with GT cloth
        # (n_cloth is derived from pred minus gripper points)
        # We don't know n_grippers here, so this path only works if pred has same N as GT.
        assert pred.shape[1] == gt.shape[1], \
            f'no cloth_indices file and pred N={pred.shape[1]} != gt N={gt.shape[1]}'
        n_cloth = pred.shape[1]
        gt_cloth = gt

    if n_cloth is None:
        n_cloth = cloth_indices.shape[0]
    pred_cloth = pred[:, :n_cloth]                          # (T_pred, n_cloth, 3)

    T = min(pred_cloth.shape[0], gt_cloth.shape[0])
    pred_cloth = pred_cloth[:T]
    gt_cloth = gt_cloth[:T]

    # query frames: clamp to last available
    q_frames = [min(f, T - 1) for f in frames]

    # Cloth L2 (mm) per query frame: mean over particles
    l2_mm = []
    for f in q_frames:
        l2_mm.append(float(np.linalg.norm(pred_cloth[f] - gt_cloth[f], axis=-1).mean() * 1000.0))

    # Mesh edge distortion: K=8 KNN on first-frame PRED topology,
    # compare predicted vs GT edge lengths at each query frame, mean over edges (mm).
    edges = knn_edges(pred_cloth[0], k=k)                  # (N, k)
    edge_dist_mm = []
    for f in q_frames:
        e_pred = edge_lengths(pred_cloth[f], edges)
        e_gt = edge_lengths(gt_cloth[f], edges)
        edge_dist_mm.append(float(np.abs(e_pred - e_gt).mean() * 1000.0))

    return {
        'T_compared': int(T),
        'frames_requested': list(frames),
        'frames_used': q_frames,
        'cloth_l2_mm': l2_mm,
        'mesh_edge_distortion_mm': edge_dist_mm,
        'n_cloth': int(n_cloth),
    }, pred_cloth, gt_cloth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_root', required=True,
                    help='dir containing episode_NNNN/pred.npy')
    ap.add_argument('--gt_root', required=True,
                    help='dir containing episode_NNNN/traj.npz (cloth_diverse_v1)')
    ap.add_argument('--load_skip', type=int, default=3,
                    help='match training dataset_load_skip_frame so GT lines up with pred')
    args = ap.parse_args()

    val_root = Path(args.val_root)
    gt_root = Path(args.gt_root)
    episodes = sorted([d.name for d in val_root.iterdir() if d.is_dir() and d.name.startswith('episode_')])
    print(f'scoring {len(episodes)} episodes from {val_root}')

    summary = {'episodes': {}, 'load_skip': args.load_skip, 'frames_requested': list(SPEC_FRAMES), 'k_knn': K_KNN}
    for ep in episodes:
        pred_path = val_root / ep / 'pred.npy'
        gt_dir = gt_root / ep
        if not pred_path.exists():
            print(f'  {ep}: SKIP (no pred.npy)')
            continue
        if not (gt_dir / 'traj.npz').exists():
            print(f'  {ep}: SKIP (no GT at {gt_dir})')
            continue
        scores, pred_cloth, gt_cloth = score_episode(pred_path, gt_dir, args.load_skip)
        # save per-episode metrics + per-frame point clouds for the reporting bundle
        out_npz = val_root / ep / 'metrics.npz'
        np.savez(out_npz,
                 frames_used=np.array(scores['frames_used'], dtype=np.int32),
                 cloth_l2_mm=np.array(scores['cloth_l2_mm'], dtype=np.float32),
                 mesh_edge_distortion_mm=np.array(scores['mesh_edge_distortion_mm'], dtype=np.float32),
                 cloth_points_pred=pred_cloth.astype(np.float32),
                 cloth_points_gt=gt_cloth.astype(np.float32))
        summary['episodes'][ep] = scores
        print(f'  {ep}: T={scores["T_compared"]:>4}  '
              f'L2(mm) @ {scores["frames_used"]}: '
              + ' / '.join(f'{x:6.2f}' for x in scores['cloth_l2_mm'])
              + '   edge(mm): '
              + ' / '.join(f'{x:5.2f}' for x in scores['mesh_edge_distortion_mm']))

    # means across episodes per frame index
    if summary['episodes']:
        l2_stack = np.array([scores['cloth_l2_mm'] for scores in summary['episodes'].values()])
        ed_stack = np.array([scores['mesh_edge_distortion_mm'] for scores in summary['episodes'].values()])
        summary['mean_across_episodes'] = {
            'frames_requested': list(SPEC_FRAMES),
            'cloth_l2_mm': l2_stack.mean(0).tolist(),
            'mesh_edge_distortion_mm': ed_stack.mean(0).tolist(),
        }
        print('\nMEAN across episodes (frames {10,50,100,500,1000} clamped to last):')
        print('  cloth L2 (mm):              ' + ' / '.join(f'{x:6.2f}' for x in summary['mean_across_episodes']['cloth_l2_mm']))
        print('  mesh edge distortion (mm):  ' + ' / '.join(f'{x:5.2f}' for x in summary['mean_across_episodes']['mesh_edge_distortion_mm']))

    out_json = val_root / 'summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'summary -> {out_json}')


if __name__ == '__main__':
    main()
