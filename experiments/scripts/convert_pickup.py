#!/usr/bin/env python3
"""Convert the pickup_pose 4-success WebDataset shards -> PGND episode dirs.

Faithful to pickup_dataset_docs.md:
  - Pulls from all 4 task roots: place_success, convex_success, pick_success, upright_success
  - Train/val split is shard-level: last 10% of shards per root = val
  - Applies the 4-success combined blocklist (299 entries) by (shard_suffix, key, elem)
  - Each source traj => one PGND episode (T=40 frames, 512 obj + 671 grip particles)

Output layout:
  <out_root>/sub_episodes_v/episode_NNNNNN/
      traj.npz             {'xyz': (40, 512, 3) f32}        object particles
      gripper_points.npz   {'left': (40, 671, 3) f32}       gripper particles (single gripper)
      eef_traj.txt         (40, 3)  gripper centroid (placeholder for Path B)
      eef_rot.txt          (40, 9)  identity rotation per frame (placeholder)
      eef_gripper.txt      (40, 1)  zeros (placeholder)
      meta.txt             [src_idx, 0, T-1]
  <out_root>/sub_episodes_v/metadata.json
      {n_train, n_val, n_total, roots, blocked_count_per_root, ...}

Episode index layout (so PGND train uses contiguous train range, val after):
  train: [0, n_train)        -- from train shards across all 4 roots, in root-then-shard order
  val:   [n_train, n_total)  -- from val shards across all 4 roots, same order

Usage:
    python experiments/scripts/convert_pickup.py \\
        --out /data/group_data/katefgroup/VLA/3DPCWM/pickup_pgnd/sub_episodes_v \\
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import tarfile
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


ROOTS = [
    '/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_place_success',
    '/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_convex_success',
    '/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_pick_success',
    '/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_upright_success',
]
BLOCKLIST_PATH = '/data/user_data/mbronars/packages/pick_and_place/reports/phantom_blocklists/robert_pickup_data_4success_combined.json'
VAL_FRACTION = 0.10


def _shard_suffix(path: str) -> str:
    """Normalize a shard path to its '/<root_name>/gpu0/NNNNN.tar' suffix.

    Both group_data and robertre symlinked paths share the same '/robert_pickup_data_*/gpu0/NNNNN.tar'
    tail, so matching on that suffix lets us reuse the blocklist regardless of which prefix it stored.
    """
    parts = path.split(os.sep)
    # find the 'robert_pickup_data_*' segment
    for i, p in enumerate(parts):
        if p.startswith('robert_pickup_data_'):
            return os.sep.join(parts[i:])
    return path  # fall back


def _load_blocklist() -> Dict[Tuple[str, str, int], list]:
    """Returns dict keyed by (shard_suffix, key, elem) -> flags."""
    with open(BLOCKLIST_PATH) as f:
        bl = json.load(f)['blocklist']
    return {(_shard_suffix(e['shard']), e['key'], int(e['elem'])): e.get('flags', []) for e in bl}


def _collect_shards():
    """Return (train_tars, val_tars) as lists of (root_name, shard_path) preserving root order."""
    train, val = [], []
    for root in ROOTS:
        root_name = Path(root).name
        shards = sorted(Path(root, 'gpu0').glob('*.tar'))
        if not shards:
            raise RuntimeError(f'no shards under {root}/gpu0')
        n_val = max(1, round(len(shards) * VAL_FRACTION))
        train.extend((root_name, str(s)) for s in shards[:-n_val])
        val.extend((root_name, str(s)) for s in shards[-n_val:])
    return train, val


def _read_pyds(tar_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Read one tar -> {batch_key: {field: np.ndarray}}."""
    samples: Dict[str, Dict[str, np.ndarray]] = {}
    with tarfile.open(tar_path, 'r') as t:
        for m in t.getmembers():
            if not m.name.endswith('.pyd'):
                continue
            base, field, _ = m.name.rsplit('.', 2)  # 'batch_00000000.particle_pos.pyd' -> base 'batch_00000000', field 'particle_pos'
            samples.setdefault(base, {})[field] = pickle.loads(t.extractfile(m).read())
    return samples


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


def _process_tar(args_tuple):
    """Write all non-blocked trajs from one tar to staging/tar_{gidx:05d}/episode_{local:04d}."""
    gidx, tar_path, staging_root_str, blocked = args_tuple
    staging_root = Path(staging_root_str)
    suffix = _shard_suffix(tar_path)
    samples = _read_pyds(tar_path)
    local = 0
    n_blocked = 0
    src_idx = gidx * 10_000  # encode tar index into source idx so we can trace later
    tar_dir = staging_root / f'tar_{gidx:05d}'
    for key in sorted(samples.keys()):  # 'batch_00000000', etc.
        s = samples[key]
        if 'particle_pos' not in s or 'obj_ids' not in s:
            continue
        pp = s['particle_pos']   # (B, T, N, 3)
        oi = s['obj_ids']        # (B, N)
        B = pp.shape[0]
        for elem in range(B):
            if (suffix, key, elem) in blocked:
                n_blocked += 1
                continue
            obj_mask = (oi[elem] == 0)
            grip_mask = (oi[elem] == 1)
            obj_pos = pp[elem][:, obj_mask, :]
            grip_pos = pp[elem][:, grip_mask, :]
            ep_dir = tar_dir / f'episode_{local:04d}'
            _write_episode(ep_dir, obj_pos, grip_pos, src_idx + local)
            local += 1
    return gidx, local, n_blocked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='target sub_episodes_v dir')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-train-tars', type=int, default=-1)
    ap.add_argument('--max-val-tars', type=int, default=-1)
    args = ap.parse_args()

    out_root = Path(args.out)
    staging_root = out_root.parent / (out_root.name + '_staging')
    out_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)

    print(f'loading blocklist ({BLOCKLIST_PATH})...')
    blocked = _load_blocklist()
    print(f'  {len(blocked)} blocklist entries (matched by shard-suffix + key + elem)')

    train_tars, val_tars = _collect_shards()
    if args.max_train_tars >= 0:
        train_tars = train_tars[:args.max_train_tars]
    if args.max_val_tars >= 0:
        val_tars = val_tars[:args.max_val_tars]
    all_tars = train_tars + val_tars
    print(f'train tars: {len(train_tars)}, val tars: {len(val_tars)} (total {len(all_tars)})')

    jobs = [(gidx, tar_path, str(staging_root), blocked) for gidx, (_, tar_path) in enumerate(all_tars)]
    with Pool(args.workers) as p:
        results = list(tqdm(p.imap_unordered(_process_tar, jobs), total=len(jobs), desc='convert'))
    # sort results by gidx for deterministic ordering
    results.sort(key=lambda r: r[0])
    n_written_per_tar = {gidx: n for gidx, n, _ in results}
    n_blocked_total = sum(b for _, _, b in results)
    print(f'wrote {sum(n_written_per_tar.values())} episodes; blocklist dropped {n_blocked_total}')

    # Compact: rename staging tars to final contiguous episode_NNNNNN dirs.
    # train tars first (0..n_train-1), then val tars.
    final_idx = 0
    for gidx in range(len(all_tars)):
        tar_dir = staging_root / f'tar_{gidx:05d}'
        if not tar_dir.exists():
            continue
        eps = sorted(tar_dir.glob('episode_*'))
        for ep_src in eps:
            ep_dst = out_root / f'episode_{final_idx:06d}'
            shutil.move(str(ep_src), str(ep_dst))
            final_idx += 1
        # remove empty tar_dir
        try:
            tar_dir.rmdir()
        except OSError:
            pass
        if gidx + 1 == len(train_tars):
            n_train = final_idx
            print(f'  train end at episode_{n_train - 1:06d} (n_train={n_train})')
    n_total = final_idx
    n_val = n_total - n_train
    print(f'val: [{n_train:06d}, {n_total:06d}) (n_val={n_val})')

    # Clean staging root
    try:
        staging_root.rmdir()
    except OSError:
        print(f'staging root {staging_root} not empty -- check leftovers')

    meta = {
        'n_train': n_train,
        'n_val': n_val,
        'n_total': n_total,
        'roots': ROOTS,
        'val_fraction': VAL_FRACTION,
        'blocklist_path': BLOCKLIST_PATH,
        'blocklist_dropped': n_blocked_total,
        'train_start_episode': 0,
        'train_end_episode': n_train,         # exclusive
        'val_start_episode': n_train,
        'val_end_episode': n_total,           # exclusive
        'T_per_episode': 40,
        'n_obj_particles': 512,
        'n_grip_particles': 671,
    }
    with open(out_root / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'metadata -> {out_root / "metadata.json"}')


if __name__ == '__main__':
    main()
