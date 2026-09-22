#!/usr/bin/env python3
"""Convert the three-domain PWM mix (pickup + rl_push + yikai_small) -> PGND episode dirs.

Reproduces what the pwm_final_paper unified loader feeds the PWM three-domain run
(/data/group_data/katefgroup/runs_pwm_mbronars/datasets/yikai_small/TRAINING.md):
  pickup   release pwm-data-v1, train exclusions pickup_exclusions_train_20260919.json, test none;
           recipe shift_m (-0.45, 0, -0.11); all 40 frames (no first-start rule).
  rl_push  release pwm-data-v1, exclusions rl_push_exclusions_20260918.json on both splits; frames
           [RL_PUSH_FIRST_START=3, len); only episodes the loader trains on (len >= 3 + 4 + 16);
           the deterministic per-object 512-point subset the loader uses for val/eval.
  yikai    yikai_small INDEX.json records; frames [ds._first(record), frames) (10 unless the index
           carries a per-record first_start); shift (-0.45, 0, -table_top_z_env).
All three: world metres, z up, table top at z = 0, the shared 671-point Franka gripper
(hand 241 + left 212 + right 218, articulated fingers) first, then object rows.

Axis: PGND's preprocess maps a source point s to (s_x, -s_z, s_y) and drops the table to the
lowest y. We store s = (x, -y, -z) of the world point (a 180 deg rotation about x), so PGND sees
(x, z, -y): world up = PGND +y.

Output layout (one PGND episode per source episode):
  <out>/episode_NNNNNN/
      traj.npz             {'xyz': (T, N_obj, 3) f32}    object points (512 or 1024)
      gripper_points.npz   {'left': (T, 671, 3) f32}     gripper points (one "gripper")
      eef_traj.txt         (T, 3)  gripper centroid (placeholder, Path B uses the points)
      eef_rot.txt          (T, 9)  identity (placeholder)
      eef_gripper.txt      (T, 1)  zeros (placeholder)
      meta.txt             [global_idx, 0, T-1]
  <out>/metadata.json      split/domain episode ranges + provenance
  <out>/episodes.jsonl     one line per episode: idx, domain, split, source id, frames, n_obj

Order: train pickup, train rl_push, train yikai, then test pickup, test rl_push, test yikai.

Run in the pwm211 env (imports pwm_final_paper read-only):
    python experiments/scripts/convert_three.py \\
        --out /data/group_data/katefgroup/runs_pwm_mbronars/datasets/pgnd_three/sub_episodes_v \\
        --workers 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
from multiprocessing import get_context
from pathlib import Path

import numpy as np

PWM = '/data/user_data/mbronars/packages/pwm_final_paper'
sys.path.insert(0, PWM)

REL = '/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1'
YS = '/data/group_data/katefgroup/runs_pwm_mbronars/datasets/yikai_small'
PICKUP_EXCL = f'{REL}/pickup/manifests/pickup_exclusions_train_20260919.json'
RL_PUSH_EXCL = f'{REL}/rl_push/manifests/rl_push_exclusions_20260918.json'
PICKUP_SHIFT = np.array([-0.45, 0.0, -0.11], np.float32)   # recipe shift_m (unified_dataset.py:924-925)
CONTEXT, FUTURE = 4, 16                                      # DOMAIN_FUTURE_LEN 16, rl_push training filter
N_GRIP = 671
DOMAINS = ('pickup', 'rl_push', 'yikai')
SPLITS = ('train', 'test')
FLIP = np.array([1.0, -1.0, -1.0], np.float32)               # world -> PGND source axes (see docstring)

_RL = {}   # split -> UnifiedRLPushDataset, built in the parent, inherited by forked workers
_YK = {}   # split -> IndexedEpisodeDataset


def _write_episode(ep_dir: Path, obj: np.ndarray, grip: np.ndarray):
    """obj (T, No, 3), grip (T, 671, 3) in world metres."""
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


def _stats(obj, grip):
    """World-frame extents, for choosing preprocess_scale."""
    return dict(frames=int(obj.shape[0]), n_obj=int(obj.shape[1]),
                obj_ext=(obj.max((0, 1)) - obj.min((0, 1))).round(4).tolist(),
                all_ext=(np.maximum(obj.max((0, 1)), grip.max((0, 1)))
                         - np.minimum(obj.min((0, 1)), grip.min((0, 1)))).round(4).tolist())


# ---------------------------------------------------------------- units (one per worker call)

def _pickup_unit(split, tar):
    from data.blocklists import canonical_shard, filter_blocklisted, load_blocklists
    root = f'{REL}/pickup/{split}'
    bl = load_blocklists([PICKUP_EXCL]) if split == 'train' else {}
    samples = {}
    with tarfile.open(tar) as tf:
        for m in tf:
            if not m.isfile():
                continue
            key, suf = m.name.split('.', 1)
            if suf in ('particle_pos.pyd', 'obj_ids.pyd'):
                samples.setdefault(key, {'__key__': key, '__url__': tar})[suf] = pickle.load(tf.extractfile(m))
    rel = os.path.relpath(canonical_shard(tar), canonical_shard(root))
    out = []
    for s in filter_blocklisted([samples[k] for k in sorted(samples)], bl, [root]):
        pos, oid = np.asarray(s['particle_pos.pyd']), np.asarray(s['obj_ids.pyd'])
        keep = _kept_rows(s, samples[s['__key__']])
        for r in range(pos.shape[0]):
            p = pos[r] + PICKUP_SHIFT
            g = oid[r] == 1
            out.append((f'{rel}:{s["__key__"]}:{keep[r]}', p[:, ~g], p[:, g]))
    return out


def _kept_rows(filtered, original):
    """Original row index of each row left in a filtered sample (for provenance)."""
    a, b = np.asarray(filtered['particle_pos.pyd']), np.asarray(original['particle_pos.pyd'])
    if a.shape[0] == b.shape[0]:
        return list(range(a.shape[0]))
    rows, j = [], 0
    for i in range(b.shape[0]):
        if j < a.shape[0] and np.array_equal(a[j], b[i]):
            rows.append(i)
            j += 1
    assert len(rows) == a.shape[0]
    return rows


def _rl_push_unit(split, obj_idx):
    from data.rl_push import window_points
    ds = _RL[split]
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
        out.append((f'{obj.file}:{e}', pts[:, N_GRIP:], pts[:, :N_GRIP]))
    return out


def _yikai_unit(split, lo, hi):
    ds = _YK[split]
    out = []
    for rec in ds.records[lo:hi]:
        b = ds.window(rec, ds._first(rec))
        p = b.positions[0].numpy()
        g = (b.object_ids[0] == 1).numpy()
        assert g.sum() == N_GRIP and g[:N_GRIP].all()
        out.append((f'{rec["episode_id"]}:first{ds._first(rec)}', p[:, ~g], p[:, g]))
    return out


def _run(job):
    uid, domain, split, args, staging = job
    fn = {'pickup': _pickup_unit, 'rl_push': _rl_push_unit, 'yikai': _yikai_unit}[domain]
    eps = fn(split, *args)
    unit_dir = Path(staging) / f'{uid:06d}'
    recs = []
    for k, (src, obj, grip) in enumerate(eps):
        _write_episode(unit_dir / f'e{k:05d}', obj, grip)
        recs.append(dict(domain=domain, split=split, source=src, **_stats(obj, grip)))
    return uid, recs


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit-units', type=int, default=0, help='smoke: first N units per domain/split')
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} exists and is not empty -- refusing to overwrite')
    staging = out.parent / (out.name + '_staging')
    if staging.exists():
        raise SystemExit(f'{staging} exists -- remove a previous partial run first')
    out.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True)

    from data.indexed_episodes import IndexedEpisodeDataset
    from data.rl_push import UnifiedRLPushDataset
    for split in SPLITS:
        _RL[split] = UnifiedRLPushDataset(f'{REL}/rl_push/{split}', split='all', context_len=CONTEXT,
                                          future_len=FUTURE, training=False, raw=True, manifests=[RL_PUSH_EXCL])
        _YK[split] = IndexedEpisodeDataset(f'{YS}/{split}', domain='yikai', context_len=1, future_len=-1,
                                           training=False, shuffle=False, raw=True)

    lim = (lambda xs: xs[:args.limit_units]) if args.limit_units else (lambda xs: xs)
    jobs = []   # ordered: split-major, then domain, then unit -> final episode order
    for split in SPLITS:
        tars = sorted(t for t in map(str, Path(f'{REL}/pickup/{split}').rglob('*.tar')) if '_quarantine' not in t)
        units = [('pickup', (t,)) for t in lim(tars)]
        units += [('rl_push', (i,)) for i in lim(list(range(len(_RL[split].objects))))]
        n = len(_YK[split].records)
        units += [('yikai', (lo, min(lo + 100, n))) for lo in lim(list(range(0, n, 100)))]
        for domain, a in units:
            jobs.append((len(jobs), domain, split, a, str(staging)))
    print(f'{len(jobs)} units', flush=True)

    results = {}
    with get_context('fork').Pool(args.workers) as pool:
        for k, (uid, recs) in enumerate(pool.imap_unordered(_run, jobs, chunksize=1)):
            results[uid] = recs
            if k % 50 == 0:
                print(f'  {k + 1}/{len(jobs)} units', flush=True)

    # compact staging -> contiguous episode_NNNNNN in job order
    episodes, ranges, idx = [], {}, 0
    for uid, domain, split, _, _ in jobs:
        r = ranges.setdefault(f'{split}/{domain}', [idx, idx])
        for k, rec in enumerate(results[uid]):
            os.rename(staging / f'{uid:06d}' / f'e{k:05d}', out / f'episode_{idx:06d}')
            np.savetxt(out / f'episode_{idx:06d}' / 'meta.txt',
                       np.array([idx, 0.0, rec['frames'] - 1], np.float32))
            episodes.append(dict(idx=idx, **rec))
            idx += 1
        r[1] = idx
        (staging / f'{uid:06d}').rmdir() if (staging / f'{uid:06d}').exists() else None
    staging.rmdir()

    with open(out / 'episodes.jsonl', 'w') as f:
        for e in episodes:
            f.write(json.dumps(e) + '\n')
    rev = subprocess.run(['git', '-C', PWM, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    meta = dict(
        ranges=ranges,   # 'split/domain' -> [start, end) episode indices
        train_start_episode=ranges['train/pickup'][0], train_end_episode=ranges['train/yikai'][1],
        val_start_episode=ranges['test/pickup'][0], val_end_episode=ranges['test/yikai'][1],
        counts={k: v[1] - v[0] for k, v in ranges.items()},
        frames={k: int(sum(e['frames'] for e in episodes if f"{e['split']}/{e['domain']}" == k)) for k in ranges},
        axis='stored = world * (1, -1, -1); PGND preprocess then gives (x, z, -y), world up = +y',
        dt=0.05, n_gripper_points=N_GRIP, limit_units=args.limit_units,
        pwm_final_paper_rev=rev,
        sources=dict(pickup=f'{REL}/pickup', rl_push=f'{REL}/rl_push', yikai=YS,
                     pickup_exclusions=PICKUP_EXCL, rl_push_exclusions=RL_PUSH_EXCL,
                     yikai_index_sha256={s: _sha(f'{YS}/{s}/INDEX.json') for s in SPLITS}),
        rules=dict(pickup='all frames, shift (-0.45, 0, -0.11)',
                   rl_push=f'frames [3, len), len >= {3 + CONTEXT + FUTURE}, seeded 512-pt object subset',
                   yikai='frames [ds._first(rec), frames), INDEX.json records'),
    )
    with open(out / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta['counts'], indent=1))
    print(json.dumps(meta['frames'], indent=1))
    print(f'wrote {idx} episodes -> {out}')


if __name__ == '__main__':
    main()
