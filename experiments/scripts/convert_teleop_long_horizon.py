#!/usr/bin/env python3
"""Convert teleop_long_horizon_clean .pkl files -> PGND episode dirs.

Source format (one .pkl = one trajectory, T=201, N=1158=512 obj+646 grip):
  particle_pos  : (T, N, 3)  float32  -- world positions, z-up
  obj_ids       : (N,)        int32    -- 0=object, 1=gripper
  (other fields ignored: link_pos/link_quat/joint_*/goal_pose/reward/...)

Output (PGND episode-dir format, identical to convert_pickup.py):
  <out>/episode_NNNN/
      traj.npz             {'xyz': (T, 512, 3) f32}     object particles
      gripper_points.npz   {'left': (T, 646, 3) f32}    single gripper (Panda arm)
      eef_traj.txt         (T, 3)
      eef_rot.txt          (T, 9)
      eef_gripper.txt      (T, 1)
      meta.txt             [src_idx, 0, T-1]

Episode index = position in sorted-by-filename list of the .pkl files.

Usage:
    python experiments/scripts/convert_teleop_long_horizon.py \\
        --src /data/group_data/katefgroup/VLA/3DPCWM/teleop_long_horizon_clean \\
        --out experiments/log/data/teleop_long_horizon/sub_episodes_v
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np


def _write_episode(out_dir: Path, obj_pos: np.ndarray, grip_pos: np.ndarray, src_idx: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    T = obj_pos.shape[0]
    np.savez(out_dir / 'traj.npz', xyz=obj_pos.astype(np.float32))
    np.savez(out_dir / 'gripper_points.npz', left=grip_pos.astype(np.float32))
    eef_xyz = grip_pos.mean(axis=1).astype(np.float32)               # (T, 3) centroid
    np.savetxt(out_dir / 'eef_traj.txt', eef_xyz, fmt='%.6e')
    eef_rot = np.broadcast_to(np.eye(3, dtype=np.float32).reshape(1, 9), (T, 9))
    np.savetxt(out_dir / 'eef_rot.txt', eef_rot, fmt='%.6e')
    np.savetxt(out_dir / 'eef_gripper.txt', np.zeros((T, 1), dtype=np.float32), fmt='%.6e')
    np.savetxt(out_dir / 'meta.txt', np.array([float(src_idx), 0.0, float(T - 1)], dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pkls = sorted(glob.glob(str(src / '*.pkl')))
    print(f'found {len(pkls)} pkls in {src}')

    src_names = []
    for ep_idx, pkl_path in enumerate(pkls):
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        ppos = d['particle_pos']                  # (T, N, 3)
        oids = d['obj_ids']                       # (N,)
        obj_pos = ppos[:, oids == 0, :]           # (T, n_obj, 3)
        grip_pos = ppos[:, oids == 1, :]          # (T, n_grip, 3)
        ep_dir = out / f'episode_{ep_idx:04d}'
        _write_episode(ep_dir, obj_pos, grip_pos, ep_idx)
        src_names.append(Path(pkl_path).name)
        print(f'  episode_{ep_idx:04d} <- {Path(pkl_path).name}  '
              f'T={ppos.shape[0]}, obj={obj_pos.shape[1]}, grip={grip_pos.shape[1]}')

    meta = {
        'n_total': len(pkls),
        'T_per_episode': 201,
        'src_dir': str(src),
        'src_files': src_names,
    }
    with open(out / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'wrote {len(pkls)} episodes -> {out}')
    print(f'metadata -> {out / "metadata.json"}')


if __name__ == '__main__':
    main()
