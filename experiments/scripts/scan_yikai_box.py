#!/usr/bin/env python3
"""Size PGND's fixed box for full yikai: what a gripper-centred box of each candidate size drops.

Per 20-frame window (4 given + 16 predicted): the box is centred in x/y on the gripper centroid at the window's
first frame, its floor is the episode's table plane (lowest object point), and PGND keeps (clip_bound 1.5 + 0.5)
cells = 4 cm free at every wall. An object is dropped for the window when any of its points leaves the box in any of
the 20 frames. It "moves" when one of its points moves more than --move-mm within the window, and is "near" when its
centroid comes within --near-m of the gripper centroid (x/y) during the window.

Reads every episode through gpwm's loader (IndexedEpisodeDataset, as convert_pwm_domain.py), nothing is written but
the report. Run in the gpwm env:
    python experiments/scripts/scan_yikai_box.py --gpwm <gpwm checkout> --release <pwm-data-v1 or compact root> \\
        --out <report.json> --workers 32
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

W, N_GRIP, DX, MARGIN = 20, 671, 0.02, 0.04
CANDIDATES = {'50x50x50': (50, 50, 50), '60x60x50': (60, 60, 50), '70x70x50': (70, 70, 50), '80x80x50': (80, 80, 50)}
_DS = {}
ARGS = None


def _scan(job):
    split, i = job
    ds = _DS[split]
    rec = ds.records[i]
    b = ds.window(rec, ds.first_start)
    p = b.positions[0].numpy().astype(np.float64)              # (T, N, 3) world metres, z up
    ids = b.object_ids[0].numpy()
    g = ids == 1
    assert g.sum() == N_GRIP, (rec['episode_id'], int(g.sum()))
    grip, obj, oid = p[:, g], p[:, ~g], ids[~g]
    T = len(p)
    if T < W:
        return split, rec['episode_id'], None
    starts = np.arange(0, T - W + 1, ARGS.stride[split])
    zfloor = obj[..., 2].min()
    c = grip[starts].mean(1)                                    # (S, 3) gripper centroid at each window start
    gc = grip.mean(1)                                           # (T, 3)
    # gripper extent over each window
    g_lo = sliding_window_view(grip.min(1), W, axis=0).min(-1)[starts]   # (S, 3)
    g_hi = sliding_window_view(grip.max(1), W, axis=0).max(-1)[starts]
    per_obj = []
    for k in np.unique(oid):
        o = obj[:, oid == k]                                    # (T, n, 3)
        lo = sliding_window_view(o.min(1), W, axis=0).min(-1)[starts]     # (S, 3) object bbox over each window
        hi = sliding_window_view(o.max(1), W, axis=0).max(-1)[starts]
        disp = np.array([np.linalg.norm(o[s:s + W] - o[s], axis=-1).max() for s in starts])
        oc = o.mean(1)
        near = np.array([np.linalg.norm(oc[s:s + W, :2] - gc[s:s + W, :2], axis=-1).min() for s in starts])
        per_obj.append((int(k), lo, hi, disp, near))
    out = {}
    for name, (nx, ny, nz) in CANDIDATES.items():
        hx, hy, hz = nx * DX / 2 - MARGIN, ny * DX / 2 - MARGIN, nz * DX - 2 * MARGIN
        def inside(lo, hi):
            return ((hi[:, 0] - c[:, 0] <= hx) & (c[:, 0] - lo[:, 0] <= hx) & (hi[:, 1] - c[:, 1] <= hy)
                    & (c[:, 1] - lo[:, 1] <= hy) & (hi[:, 2] - zfloor <= hz) & (lo[:, 2] - zfloor >= -1e-6))
        grip_out = ~inside(g_lo, g_hi)
        drop_obj = 0
        drop_mv = np.zeros(len(starts), bool)
        drop_near = np.zeros(len(starts), bool)
        cases = []
        for k, lo, hi, disp, near in per_obj:
            dropped = ~inside(lo, hi)
            mv = dropped & (disp > ARGS.move_mm / 1000)
            nr = mv & (near < ARGS.near_m)
            drop_obj += int(dropped.sum())
            drop_mv |= mv
            drop_near |= nr
            if name == ARGS.report_box:
                for j in np.flatnonzero(nr):
                    cases.append(dict(start=int(starts[j]), object=k, near_m=round(float(near[j]), 3),
                                      moved_mm=round(float(disp[j] * 1000), 1)))
        out[name] = dict(windows=len(starts), objects=len(per_obj) * len(starts), drop_obj=drop_obj,
                         drop_mv=int(drop_mv.sum()), drop_near=int(drop_near.sum()), grip_out=int(grip_out.sum()),
                         cases=cases)
    return split, rec['episode_id'], out


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpwm', required=True)
    ap.add_argument('--release', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--move-mm', type=float, default=5.0)
    ap.add_argument('--near-m', type=float, default=0.25)
    ap.add_argument('--report-box', default='70x70x50')
    ap.add_argument('--train-stride', type=int, default=2)
    ap.add_argument('--test-stride', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ARGS = ap.parse_args()
    ARGS.stride = {'train': ARGS.train_stride, 'test': ARGS.test_stride}
    sys.path.insert(0, ARGS.gpwm)
    from data.indexed_episodes import IndexedEpisodeDataset
    jobs = []
    for split in ('train', 'test'):
        _DS[split] = IndexedEpisodeDataset(str(Path(ARGS.release) / 'yikai' / split), domain='yikai', context_len=1,
                                           future_len=-1, training=False, shuffle=False, raw=True)
        n = len(_DS[split].records)
        jobs += [(split, i) for i in range(min(n, ARGS.limit) if ARGS.limit else n)]
    print(f'{len(jobs)} episodes', flush=True)
    agg = {s: {k: dict(windows=0, objects=0, drop_obj=0, drop_mv=0, drop_near=0, grip_out=0) for k in CANDIDATES}
           for s in ('train', 'test')}
    cases, eps_near = [], {s: set() for s in ('train', 'test')}
    with get_context('fork').Pool(ARGS.workers) as pool:
        for n, (split, ep, out) in enumerate(pool.imap_unordered(_scan, jobs, chunksize=4)):
            if out is None:
                continue
            for k, v in out.items():
                for f in agg[split][k]:
                    agg[split][k][f] += v[f]
            for cse in out[ARGS.report_box]['cases']:
                cases.append(dict(split=split, episode=ep, **cse))
                eps_near[split].add(ep)
            if n % 1000 == 0:
                print(f'  {n}/{len(jobs)}', flush=True)
    report = dict(settings=dict(window=W, cell_m=DX, wall_margin_m=MARGIN, move_mm=ARGS.move_mm, near_m=ARGS.near_m,
                                stride=ARGS.stride, centre='gripper centroid x/y at the window start; floor = table'),
                  totals=agg, report_box=ARGS.report_box,
                  episodes_with_near_drops={s: sorted(v) for s, v in eps_near.items()}, near_drop_cases=cases)
    Path(ARGS.out).write_text(json.dumps(report, indent=1))
    print('box | split | windows | objects dropped | windows dropping a moving object | ... near the gripper | gripper out')
    for k in CANDIDATES:
        for s in ('train', 'test'):
            a = agg[s][k]
            w = max(a['windows'], 1)
            print(f'{k} | {s} | {a["windows"]} | {a["drop_obj"] / max(a["objects"], 1):.1%} | {a["drop_mv"] / w:.3%} | '
                  f'{a["drop_near"]} ({a["drop_near"] / w:.4%}) | {a["grip_out"] / w:.3%}')
    print(f'{ARGS.report_box}: {len(cases)} near-gripper moving drops in '
          f'{len(eps_near["train"])} train / {len(eps_near["test"])} test episodes -> {ARGS.out}')


if __name__ == '__main__':
    main()
