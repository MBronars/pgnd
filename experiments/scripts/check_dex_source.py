#!/usr/bin/env python3
"""Check pgnd.data.pwm_dex_source against pwm_final_paper's own loader (run in pwm211).

For --limit records per split, sampled across every shard: object and hand points must match
IndexedEpisodeDataset(domain="dex").window(record, 0) (raw=True, future_len=-1: whole episode).
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

PGND = Path(__file__).resolve().parents[2]
sys.path.insert(0, '/data/user_data/mbronars/packages/pwm_final_paper')
REL = '/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/dex_v2_filtered2'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--per-shard', type=int, default=2)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('pwm_dex_source', PGND / 'pgnd/data/pwm_dex_source.py')
    src = importlib.util.module_from_spec(spec)
    sys.modules['pwm_dex_source'] = src
    spec.loader.exec_module(src)
    from data.indexed_episodes import IndexedEpisodeDataset
    ds = IndexedEpisodeDataset(f'{REL}/{a.split}', domain='dex', context_len=4, future_len=-1,
                               training=False, shuffle=False, raw=True)
    picks, seen = [], {}
    for i, r in enumerate(ds.records):                       # first/last env of every shard
        seen.setdefault(r['file'], []).append(i)
    for f, ids in seen.items():
        picks += sorted({ids[0], ids[-1], ids[len(ids) // 2]})[:max(a.per_shard, 1)]
    wants = [(i, ds.records[i]) for i in picks]
    got = {}
    for f in {ds.records[i]['file'] for i in picks}:
        for i, ep in src._read_shard((f, [(i, r) for i, r in wants if r['file'] == f])):
            got[i] = ep
    worst = 0.0
    for i in picks:
        r = ds.records[i]
        b = ds.window(r, 0)
        P, act = b.positions[0].numpy(), b.action_mask[0].numpy().astype(bool)
        ep = got[i]
        assert P[:, ~act].shape == ep['obj'].shape and P[:, act].shape == ep['hand'].shape, (P.shape, ep['obj'].shape, ep['hand'].shape)
        worst = max(worst, float(np.abs(P[:, ~act] - ep['obj']).max()), float(np.abs(P[:, act] - ep['hand']).max()))
    print(f'{a.split}: {len(picks)} episodes from {len(seen)} shards match (max |diff| {worst:.3g} m); '
          f'hands {sorted({got[i]["bucket"] for i in picks})}')


if __name__ == '__main__':
    main()
