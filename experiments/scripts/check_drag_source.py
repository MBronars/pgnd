#!/usr/bin/env python3
"""Check pgnd.data.pwm_drag_source against pwm_final_paper's own loader (run in pwm211).

For every record of the chosen split (or --limit of them): positions, driven patch and
per-transition active flag must be identical to IndexedEpisodeDataset(domain="mpm_drag")
.window(record, 0) (raw=True: world metres, shifted).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PGND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PGND))
sys.path.insert(0, '/data/user_data/mbronars/packages/pwm_final_paper')
REL = '/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/mpm_drag'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    import importlib.util   # load the reader by path: the pgnd package imports warp (not in pwm211)
    spec = importlib.util.spec_from_file_location('pwm_drag_source', PGND / 'pgnd/data/pwm_drag_source.py')
    src = importlib.util.module_from_spec(spec)
    sys.modules['pwm_drag_source'] = src   # so worker processes can unpickle its functions
    spec.loader.exec_module(src)
    load_episodes, MPM_SHIFT = src.load_episodes, src.MPM_SHIFT
    from data.indexed_episodes import IndexedEpisodeDataset
    from data.unified_dataset import CANONICAL_TRANSLATION
    assert np.array_equal(np.asarray(CANONICAL_TRANSLATION['mpm'], np.float32), MPM_SHIFT)
    ds = IndexedEpisodeDataset(f'{REL}/{a.split}', domain='mpm_drag', context_len=1, future_len=-1,
                               training=False, raw=True)
    eps = load_episodes(f'{REL}/{a.split}', workers=8)
    n = len(eps) if not a.limit else a.limit
    worst = 0.0
    for rec, ep in list(zip(ds.records, eps))[:n]:
        b = ds.window(rec, 0)
        pos = b.positions[0].numpy()
        sel = b.action_mask[0].numpy().astype(bool)
        act = b.action_active[0].numpy()                     # [T-1, N] = active & selected
        assert rec['episode_id'] == ep['episode_id']
        assert pos.shape == ep['pos'].shape, (pos.shape, ep['pos'].shape)
        worst = max(worst, float(np.abs(pos - ep['pos']).max()))
        assert np.array_equal(sel, ep['selected'])
        assert np.array_equal(act, ep['active'][:, None] & ep['selected'][None]), rec['episode_id']
    print(f'{a.split}: {n} episodes identical (positions max |diff| {worst:.3g} m, patch, active flags)')


if __name__ == '__main__':
    main()
