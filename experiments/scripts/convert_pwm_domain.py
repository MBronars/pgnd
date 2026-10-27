#!/usr/bin/env python3
"""Convert ONE pwm-data-v1 domain (rl_push or full yikai) -> PGND episode dirs, for a Table 1 specialist.

The single-domain version of convert_three.py (docs/TABLE1_PUSHING_PICKPLACE.md in gpwm: Pushing = rl_push,
Pick and Place = the FULL yikai release). Every episode is read through gpwm's own loader, so the episodes,
exclusions, start frames and point construction are gpwm's, not a re-implementation:
  rl_push  data.rl_push.UnifiedRLPushDataset, exclusions rl_push_exclusions_20260918.json on both splits;
           frames [RL_PUSH_FIRST_START=3, len); only episodes the loader trains on (len >= 3 + 4 + 16);
           the deterministic per-object 512-point subset the loader uses for val/eval.
  yikai    data.indexed_episodes.IndexedEpisodeDataset on <release>/yikai/<split> (whole scenes, no table);
           frames [first start (10), frames).
Both: world metres, z up, table top at z = 0, the 671-point Franka gripper (hand 241 + left 212 + right 218)
stored as PGND's recorded gripper points, object rows in traj.npz. Works on pwm-data-v1 and on the compact
release (gpwm's loader reads both).

Axis and layout as convert_three.py: stored s = world * (1, -1, -1); PGND preprocess gives (x, z, -y), world up = +y.
  <out>/episode_NNNNNN/{traj.npz, gripper_points.npz, eef_traj.txt, eef_rot.txt, eef_gripper.txt, meta.txt}
  <out>/metadata.json   train / val episode ranges, counts, provenance, extents
  <out>/episodes.jsonl  one line per episode: idx, split, source id, frames, n_obj, extents
Train episodes first, then test.

Run in the gpwm env from the gpwm checkout's parent environment (imports gpwm read-only):
    python experiments/scripts/convert_pwm_domain.py --domain rl_push --gpwm <gpwm checkout> \\
        --release /ocean/projects/cis260367p/mbronars/projects/GPWM/data --out <dir>/sub_episodes_v --workers 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

CONTEXT, FUTURE = 4, 16                                      # gpwm windows: 4 given + 16 predicted
N_GRIP = 671
SPLITS = ('train', 'test')
FLIP = np.array([1.0, -1.0, -1.0], np.float32)               # world -> PGND source axes (see docstring)
RL_PUSH_EXCL = 'rl_push/manifests/rl_push_exclusions_20260918.json'

_DS = {}   # split -> gpwm dataset, built in the parent, inherited by forked workers


def _write_episode(ep_dir: Path, obj: np.ndarray, grip: np.ndarray, obj_ids=None):
    """obj (T, No, 3), grip (T, 671, 3) in world metres; obj_ids (No,) gpwm object id of each object row (yikai)."""
    assert grip.shape[1] == N_GRIP and obj.shape[0] == grip.shape[0], (obj.shape, grip.shape)
    assert np.isfinite(obj).all() and np.isfinite(grip).all()
    obj, grip = (obj * FLIP).astype(np.float32), (grip * FLIP).astype(np.float32)
    T = obj.shape[0]
    ep_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ep_dir / 'traj.npz', xyz=obj)
    np.savez(ep_dir / 'gripper_points.npz', left=grip)
    np.savetxt(ep_dir / 'eef_traj.txt', grip.mean(axis=1), fmt='%.6e')
    np.savetxt(ep_dir / 'eef_rot.txt', np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (T, 1)), fmt='%.6e')
    np.savetxt(ep_dir / 'eef_gripper.txt', np.zeros((T, 1), np.float32), fmt='%.6e')
    if obj_ids is not None:   # per-row object identity: lets a run drop whole objects (gripper-centred box)
        np.save(ep_dir / 'object_ids.npy', np.asarray(obj_ids, np.int32))


def _stats(obj, grip):
    """World-frame extents, for choosing the PGND grid / preprocess_scale."""
    lo = np.minimum(obj.min((0, 1)), grip.min((0, 1)))
    hi = np.maximum(obj.max((0, 1)), grip.max((0, 1)))
    return dict(frames=int(obj.shape[0]), n_obj=int(obj.shape[1]),
                obj_ext=(obj.max((0, 1)) - obj.min((0, 1))).round(4).tolist(),
                all_ext=(hi - lo).round(4).tolist(), all_lo=lo.round(4).tolist(), all_hi=hi.round(4).tolist())


def _rl_push_unit(split, obj_idx):
    from data.rl_push import window_points
    ds = _DS[split]
    obj = ds.objects[obj_idx]
    rows = ds._object_rows(obj)
    out = []
    for i, e in ds.episodes:
        if i != obj_idx:
            continue
        n = int(obj.lengths[e])
        if n < ds.first_start + CONTEXT + FUTURE:
            continue
        pts, _ = window_points(obj, e, ds.first_start, n - ds.first_start, ds.links, rows)
        out.append((f'{obj.file}:{e}', pts[:, N_GRIP:], pts[:, :N_GRIP], None))
    return out


def _yikai_unit(split, lo, hi):
    ds = _DS[split]
    out = []
    for rec in ds.records[lo:hi]:
        b = ds.window(rec, ds.first_start)
        p = b.positions[0].numpy()
        g = (b.object_ids[0] == 1).numpy()
        assert g.sum() == N_GRIP and g[:N_GRIP].all(), (rec['episode_id'], int(g.sum()))
        ids = b.object_ids[0].numpy()
        out.append((f'{rec["episode_id"]}:first{ds.first_start}', p[:, ~g], p[:, g], ids[~g]))
    return out


def _run(job):
    uid, domain, split, args, staging = job
    eps = {'rl_push': _rl_push_unit, 'yikai': _yikai_unit}[domain](split, *args)
    unit_dir = Path(staging) / f'{uid:06d}'
    recs = []
    for k, (src, obj, grip, ids) in enumerate(eps):
        _write_episode(unit_dir / f'e{k:05d}', obj, grip, ids)
        recs.append(dict(split=split, source=src, **_stats(obj, grip),
                         **({} if ids is None else dict(n_objects=int(len(np.unique(ids)))))))
    return uid, recs


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, choices=('rl_push', 'yikai'))
    ap.add_argument('--gpwm', required=True, help='gpwm checkout (branch mb/psc-port on PSC), imported read-only')
    ap.add_argument('--release', required=True, help='pwm-data-v1 (or the compact release) root')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit-units', type=int, default=0, help='smoke: first N units per split')
    args = ap.parse_args()
    sys.path.insert(0, args.gpwm)
    rel = Path(args.release)

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} exists and is not empty -- refusing to overwrite')
    staging = out.parent / (out.name + '_staging')
    if staging.exists():
        raise SystemExit(f'{staging} exists -- remove a previous partial run first')
    out.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True)

    lim = (lambda xs: xs[:args.limit_units]) if args.limit_units else (lambda xs: xs)
    jobs = []   # ordered: split-major, then unit -> final episode order
    for split in SPLITS:
        if args.domain == 'rl_push':
            from data.rl_push import UnifiedRLPushDataset
            _DS[split] = UnifiedRLPushDataset(str(rel / 'rl_push' / split), split='all', context_len=CONTEXT,
                                              future_len=FUTURE, training=False, raw=True,
                                              manifests=[str(rel / RL_PUSH_EXCL)])
            units = [(i,) for i in range(len(_DS[split].objects))]
        else:
            from data.indexed_episodes import IndexedEpisodeDataset
            _DS[split] = IndexedEpisodeDataset(str(rel / 'yikai' / split), domain='yikai', context_len=1,
                                               future_len=-1, training=False, shuffle=False, raw=True)
            n = len(_DS[split].records)
            units = [(lo, min(lo + 20, n)) for lo in range(0, n, 20)]
        for a in lim(units):
            jobs.append((len(jobs), args.domain, split, a, str(staging)))
    print(f'{len(jobs)} units', flush=True)

    results = {}
    with get_context('fork').Pool(args.workers) as pool:
        for k, (uid, recs) in enumerate(pool.imap_unordered(_run, jobs, chunksize=1)):
            results[uid] = recs
            if k % 50 == 0:
                print(f'  {k + 1}/{len(jobs)} units', flush=True)

    # compact staging -> contiguous episode_NNNNNN in job order
    episodes, ranges, idx = [], {}, 0
    for uid, _, split, _, _ in jobs:
        r = ranges.setdefault(split, [idx, idx])
        for k, rec in enumerate(results[uid]):
            os.rename(staging / f'{uid:06d}' / f'e{k:05d}', out / f'episode_{idx:06d}')
            np.savetxt(out / f'episode_{idx:06d}' / 'meta.txt', np.array([idx, 0.0, rec['frames'] - 1], np.float32))
            episodes.append(dict(idx=idx, **rec))
            idx += 1
        r[1] = idx
        (staging / f'{uid:06d}').rmdir() if (staging / f'{uid:06d}').exists() else None
    staging.rmdir()

    with open(out / 'episodes.jsonl', 'w') as f:
        for e in episodes:
            f.write(json.dumps(e) + '\n')
    ext = {s: np.array([e['all_ext'] for e in episodes if e['split'] == s]) for s in SPLITS}
    rev = subprocess.run(['git', '-C', args.gpwm, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    meta = dict(
        domain=args.domain, ranges=ranges,   # split -> [start, end) episode indices
        train_start_episode=ranges['train'][0], train_end_episode=ranges['train'][1],
        val_start_episode=ranges['test'][0], val_end_episode=ranges['test'][1],
        counts={k: v[1] - v[0] for k, v in ranges.items()},
        frames={s: int(sum(e['frames'] for e in episodes if e['split'] == s)) for s in SPLITS},
        max_extent_m={s: ext[s].max(0).round(4).tolist() for s in SPLITS if len(ext[s])},
        p99_extent_m={s: np.percentile(ext[s], 99, axis=0).round(4).tolist() for s in SPLITS if len(ext[s])},
        axis='stored = world * (1, -1, -1); PGND preprocess then gives (x, z, -y), world up = +y',
        dt=0.05, n_gripper_points=N_GRIP, limit_units=args.limit_units, gpwm_rev=rev, release=str(rel),
        index_sha256={s: _sha(rel / args.domain / s / 'INDEX.json') for s in SPLITS
                      if (rel / args.domain / s / 'INDEX.json').exists()},
        rules=(f'frames [3, len), len >= {3 + CONTEXT + FUTURE}, seeded 512-pt object subset, exclusions {RL_PUSH_EXCL}'
               if args.domain == 'rl_push' else 'frames [first start, frames), INDEX.json records, whole scene'),
    )
    with open(out / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: meta[k] for k in ('counts', 'frames', 'max_extent_m', 'p99_extent_m')}, indent=1))
    print(f'wrote {idx} episodes -> {out}')


if __name__ == '__main__':
    main()
