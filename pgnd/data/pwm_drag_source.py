"""Read the PWM mpm_drag release (pwm-data-v1/mpm_drag/<split>) directly -- no converted copy.

Reproduces what pwm_final_paper's IndexedEpisodeDataset(domain="mpm_drag").window(record, 0)
returns for a whole episode (data/recording_adapters.py mpm_window + data/mpm_drag.py commands):
the INDEX.json records (the index is the episode selection; no exclusions exist), all frames,
positions + CANONICAL_TRANSLATION["mpm"] (-0.75, -0.75, -0.10) in world metres (z up, floor at
z = 0), the driven patch (is_grabbed) and the drag-active flag per transition. Checked against
the PWM loader by experiments/scripts/check_drag_source.py (pwm211 env).

Depends only on numpy + the standard library, so it runs in the pgnd env (no webdataset).
"""
import json
import pickle
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MPM_SHIFT = np.array([-0.75, -0.75, -0.10], np.float32)   # pwm_final_paper CANONICAL_TRANSLATION["mpm"]
DT = 0.0417
_FIELDS = ('particle_pos.pyd', 'valid_particles.pyd', 'is_grabbed.pyd', 'drag_active.pyd', 'provenance.json')


def load_index(root):
    path = Path(root) / 'INDEX.json'
    index = json.load(open(path))
    if index.get('domain') != 'mpm_drag':
        raise ValueError(f'{path}: domain {index.get("domain")!r}, expected mpm_drag')
    return index


def _read_file(args):
    """All requested rows of one TAR -> list of (record position, episode dict)."""
    path, wants = args   # wants: [(pos, record)]
    stat = Path(path).stat()
    keys = {rec['key'] for _, rec in wants}
    for _, rec in wants:
        if (stat.st_size, stat.st_mtime_ns) != (rec['size'], rec['mtime_ns']):
            raise ValueError(f'Source shard changed since indexing: {path}')
    samples = defaultdict(dict)
    with tarfile.open(path) as archive:
        for member in archive:
            key, _, suffix = member.name.partition('.')
            if key in keys and suffix in _FIELDS:
                with archive.extractfile(member) as f:
                    samples[key][suffix] = json.load(f) if suffix.endswith('.json') else pickle.load(f)
    out = []
    for pos_i, rec in wants:
        s, row = samples[rec['key']], rec['row']
        if set(s) != set(_FIELDS):
            raise ValueError(f'Incomplete sample {path}:{rec["key"]}')
        prov = s['provenance.json'][row]
        if prov['episode_id'] != rec['episode_id']:
            raise ValueError('provenance episode identity differs from the index')
        times = np.asarray(prov['state_time_s'])
        if len(times) != rec['frames'] or not np.allclose(np.diff(times), DT, rtol=0, atol=1e-12):
            raise ValueError(f'{rec["episode_id"]}: length or 0.0417 s clock differs from the index')
        if not np.asarray(s['valid_particles.pyd'][row]).all():
            raise ValueError(f'{rec["episode_id"]}: padded / invalid particles')
        pos = np.asarray(s['particle_pos.pyd'][row], np.float32) + MPM_SHIFT     # [T, N, 3]
        selected = np.asarray(s['is_grabbed.pyd'][row]).astype(bool)             # [N]
        active = np.asarray(s['drag_active.pyd'][row]).astype(bool)[:-1]         # [T-1] per transition
        out.append((pos_i, dict(episode_id=rec['episode_id'], n_obj=len(rec['asset_ids']),
                                bucket=f"obj{len(rec['asset_ids'])}",
                                pos=pos, selected=selected, active=active)))
    return out


def load_episodes(root, workers=8, limit=0):
    """Every episode of one split, in INDEX.json record order (limit > 0: that many, evenly spaced -- smokes)."""
    records = load_index(root)['records']
    if limit:
        records = [records[i] for i in np.linspace(0, len(records) - 1, min(limit, len(records))).round().astype(int)]
    by_file = defaultdict(list)
    for i, rec in enumerate(records):
        by_file[rec['file']].append((i, rec))
    episodes = [None] * len(records)
    jobs = list(by_file.items())
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = pool.map(_read_file, jobs, chunksize=4)
            for res in results:
                for i, ep in res:
                    episodes[i] = ep
    else:
        for job in jobs:
            for i, ep in _read_file(job):
                episodes[i] = ep
    return episodes


FLIP = np.array([1.0, -1.0, -1.0], np.float32)   # world -> PGND source axes: preprocess then puts world up on +y


def to_pgnd_arrays(ep):
    """One episode -> keyword arrays for PGND's preprocess_arrays() (xyz, eef_xyz, eef_rot, eef_gripper).

    xyz: every particle, driven patch included (PWM predicts it too).
    Gripper = PGND's stock sphere (sim.gripper_forcing), driven by the patch's RECORDED motion
    (user 2026-09-22): centre = centroid c of the driven patch particles, closed while the drag
    is active (drive + hold), open after release, translation only (eef_rot None = identity).
    Timing: the forcing step that predicts frame k from k-1 reads gripper entry k -- position
    eef[k] and forward-difference velocity eef[k+1] - eef[k] -- so eef[k] = c[k-1] gives it the
    patch centre at frame k-1 and the patch's actual step c[k] - c[k-1]; flag[k] = transition
    k-1 -> k active. Stored with world * (1, -1, -1) axes.
    """
    pos, sel, active = ep['pos'], ep['selected'], ep['active']
    c = pos[:, sel].mean(1)                                    # [T, 3] patch centroid, world
    eef = np.concatenate([c[:1], c[:-1]], 0)                   # eef[k] = c[k-1] (eef[0] = c[0], unused)
    closed = np.concatenate([[False], active])                 # flag[k] = transition k-1 -> k active
    eef_gripper = np.where(closed, 0.0, 1.0)[:, None].astype(np.float32)   # PGND: < 0.5 = closed
    return dict(xyz=(pos * FLIP).astype(np.float32), eef_xyz=(eef * FLIP).astype(np.float32), eef_rot=None,
                eef_gripper=eef_gripper, gripper_points=None)
