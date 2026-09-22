#!/usr/bin/env python3
"""Convert pushing_data_actions_04_18 WebDataset shards -> PGND episode dirs.

Source layout (one .tar = one WDS sample with B=4 demos stacked):
  <name>.particle_pos.pyd       (B, T, 256, 3)  float32  -- gripper + blob mixed
  <name>.obj_ids.pyd            (B, 256)        int32    -- 0=blob, 1=gripper
  <name>.valid_particles.pyd    (B, T, 256)     bool
  <name>.start_state.pyd        (B, 12)         float32
  (others ignored)

Output layout (one PGND episode per source demo):
  <out>/sub_episodes_v/episode_NNNN/
      traj.npz             {'xyz': (T, N_blob, 3) float32}      object/cloth particles
      gripper_points.npz   {'left': (T, N_grip, 3) float32}     single gripper
      eef_traj.txt         (T, 3)   gripper centroid per frame (placeholder for Path B)
      eef_rot.txt          (T, 9)   identity rotation per frame (placeholder)
      eef_gripper.txt      (T, 1)   zeros (placeholder)
      meta.txt             3 floats: [demo_id_in_split, 0, T-1]

Train tars come first (episode_0000..episode_{N_train-1}); test tars follow
(episode_{N_train}..episode_{N_train+N_test-1}). Pass training_start/end_episode
accordingly in the train script.

Usage:
    python experiments/scripts/convert_pushing.py \
        --src /data/group_data/katefgroup/VLA/3DPCWM/pushing_data_actions_04_18 \
        --out /data/group_data/katefgroup/VLA/3DPCWM/pushing_pgnd/sub_episodes_v \
        [--max-train-tars N] [--max-test-tars N] [--workers W]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import tarfile
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm


def _read_pyds(tar_path: Path):
    """Iterate (basename, fields_dict) per WDS sample in a tar."""
    samples = {}
    with tarfile.open(tar_path, 'r') as t:
        for m in t.getmembers():
            if not m.name.endswith('.pyd'):
                continue
            # name like "train_00000003.particle_pos.pyd"  -> base="train_00000003"
            base, field, _ = m.name.rsplit('.', 2)
            samples.setdefault(base, {})[field] = pickle.loads(t.extractfile(m).read())
    return samples


def _write_episode(out_dir: Path, blob_pos: np.ndarray, grip_pos: np.ndarray, src_idx: int):
    """Write one PGND-format episode dir from blob + gripper particle trajectories."""
    out_dir.mkdir(parents=True, exist_ok=True)
    T = blob_pos.shape[0]
    np.savez(out_dir / 'traj.npz', xyz=blob_pos.astype(np.float32))
    np.savez(out_dir / 'gripper_points.npz', left=grip_pos.astype(np.float32))
    # placeholder eef (Path B doesn't use these for the model, only for the
    # `grippers` 15-d array which is ignored in gripper_points=True mode)
    eef_xyz = grip_pos.mean(axis=1).astype(np.float32)               # (T, 3)
    np.savetxt(out_dir / 'eef_traj.txt', eef_xyz, fmt='%.6e')
    eef_rot = np.broadcast_to(np.eye(3, dtype=np.float32).reshape(1, 9), (T, 9))
    np.savetxt(out_dir / 'eef_rot.txt', eef_rot, fmt='%.6e')
    np.savetxt(out_dir / 'eef_gripper.txt', np.zeros((T, 1), dtype=np.float32), fmt='%.6e')
    np.savetxt(out_dir / 'meta.txt', np.array([float(src_idx), 0.0, float(T - 1)], dtype=np.float32))


def _convert_tar(args_tuple: Tuple[str, int, Path]) -> int:
    tar_path, ep_start, out_root = args_tuple
    tar_path = Path(tar_path)
    samples = _read_pyds(tar_path)
    n_written = 0
    ep_idx = ep_start
    for sample_name in sorted(samples.keys()):
        s = samples[sample_name]
        if 'particle_pos' not in s or 'obj_ids' not in s:
            continue
        ppos = s['particle_pos']   # (B, T, N, 3)
        oids = s['obj_ids']        # (B, N)
        B = ppos.shape[0]
        for b in range(B):
            blob_mask = (oids[b] == 0)
            grip_mask = (oids[b] == 1)
            blob_pos = ppos[b][:, blob_mask, :]
            grip_pos = ppos[b][:, grip_mask, :]
            out_dir = out_root / f'episode_{ep_idx:06d}'  # 6-digit padding; we may have >9999 episodes
            _write_episode(out_dir, blob_pos, grip_pos, ep_idx)
            ep_idx += 1
            n_written += 1
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='pushing_data_actions_04_18 root (with train/ test/)')
    ap.add_argument('--out', required=True, help='PGND-style sub_episodes_v output dir')
    ap.add_argument('--max-train-tars', type=int, default=-1)
    ap.add_argument('--max-test-tars', type=int, default=-1)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Count B (demos per shard) from the first tar to determine episode starts.
    train_tars = sorted((src / 'train').glob('*.tar'))
    test_tars = sorted((src / 'test').glob('*.tar'))
    if args.max_train_tars >= 0:
        train_tars = train_tars[:args.max_train_tars]
    if args.max_test_tars >= 0:
        test_tars = test_tars[:args.max_test_tars]

    # Peek at first train tar to learn B and N samples per tar.
    first = _read_pyds(train_tars[0])
    sample0 = first[sorted(first.keys())[0]]
    B = sample0['particle_pos'].shape[0]
    samples_per_tar = len(first)
    demos_per_tar = B * samples_per_tar
    print(f'train tars: {len(train_tars)}, test tars: {len(test_tars)}')
    print(f'each tar: {samples_per_tar} WDS samples x B={B} = {demos_per_tar} demos')
    print(f'estimate -> train demos {len(train_tars)*demos_per_tar}, test demos {len(test_tars)*demos_per_tar}')

    # Assign episode index ranges per tar so workers can write without coordination.
    tar_jobs: List[Tuple[str, int, Path]] = []
    ep = 0
    for tp in train_tars:
        tar_jobs.append((str(tp), ep, out))
        ep += demos_per_tar
    n_train_total = ep
    for tp in test_tars:
        tar_jobs.append((str(tp), ep, out))
        ep += demos_per_tar
    n_total = ep
    print(f'planned episode range: 0..{n_total - 1}  (train: 0..{n_train_total - 1}, test: {n_train_total}..{n_total - 1})')

    with Pool(args.workers) as p:
        written = list(tqdm(p.imap_unordered(_convert_tar, tar_jobs), total=len(tar_jobs)))
    print(f'wrote {sum(written)} episode dirs total')

    # Write a metadata file recording the train/test split
    meta = {
        'n_train_demos': n_train_total,
        'n_test_demos': n_total - n_train_total,
        'n_total': n_total,
        'B_per_sample': B,
        'samples_per_tar': samples_per_tar,
        'train_tars_used': len(train_tars),
        'test_tars_used': len(test_tars),
    }
    with open(out / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'wrote {out / "metadata.json"}')


if __name__ == '__main__':
    main()
