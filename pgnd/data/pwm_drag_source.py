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

from .release_files import same_file

MPM_SHIFT = np.array([-0.75, -0.75, -0.10], np.float32)   # pwm_final_paper CANONICAL_TRANSLATION["mpm"]
DT = 0.0417
_FIELDS = ('particle_pos.pyd', 'valid_particles.pyd', 'is_grabbed.pyd', 'drag_active.pyd', 'provenance.json',
           'grab_velocity.pyd', 'material_properties.pyd')


def load_index(root):
    path = Path(root) / 'INDEX.json'
    index = json.load(open(path))
    if index.get('domain') != 'mpm_drag':
        raise ValueError(f'{path}: domain {index.get("domain")!r}, expected mpm_drag')
    for rec in index['records']:   # pwm-data-v1-compact: paths relative to the INDEX.json folder
        if isinstance(rec.get('file'), str) and not Path(rec['file']).is_absolute():
            rec['file'] = str(Path(root) / rec['file'])
    return index


def _read_file(args):
    """All requested rows of one TAR -> list of (record position, episode dict)."""
    path, wants = args   # wants: [(pos, record)]
    stat = Path(path).stat()
    keys = {rec['key'] for _, rec in wants}
    for _, rec in wants:
        if not same_file(stat, rec['size'], rec['mtime_ns']):
            raise ValueError(f'Source shard changed since indexing: {path}')
    samples = defaultdict(dict)
    with tarfile.open(path) as archive:
        for member in archive:
            key, _, suffix = member.name.partition('.')
            if key in keys and suffix.endswith('.cpk'):        # pwm-data-v1-compact: lossless zstd positions
                from .gpwm_compact_codec import decode_member
                with archive.extractfile(member) as f:
                    pyd, arr = decode_member(suffix, f.read())
                if pyd in _FIELDS:
                    samples[key][pyd] = arr
            elif key in keys and suffix in _FIELDS:
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
        vel = np.asarray(s['grab_velocity.pyd'][row], np.float32)                 # [T-1, N, 3] commanded, patch only
        vsel = vel[:, selected]
        if vsel.size and np.abs(vsel - vsel[:, :1]).max() > 1e-7:
            raise ValueError(f'{rec["episode_id"]}: commanded velocity differs across the patch')
        cmd_vel = vsel[:, 0] if vsel.size else np.zeros((len(pos) - 1, 3), np.float32)   # [T-1, 3] m/s
        if np.abs(cmd_vel[~active]).max(initial=0) > 0:
            raise ValueError(f'{rec["episode_id"]}: commanded velocity outside the active drag')
        out.append((pos_i, dict(episode_id=rec['episode_id'], n_obj=len(rec['asset_ids']),
                                bucket=f"obj{len(rec['asset_ids'])}",
                                pos=pos, selected=selected, active=active, cmd_vel=cmd_vel,
                                mat=material_features(np.asarray(s['material_properties.pyd'][row])))))
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
N_MATERIAL_FEATURES = 3


def material_features(props):
    """Per-particle material input [N, 3] from material_properties [N, 5] (E, nu, rho, type, fifth value): the
    three columns that vary in mpm_drag (user, 2026-10-25) -- log10(E) - 5 (E in 1e4 / 1e5 / 1e6), the elastic (0) /
    plasticine (1) flag, and the fifth value / 1e4 (10,000 for plasticine, 0 otherwise). Poisson ratio (0.2) and
    density (1000) are constant and left out."""
    props = np.asarray(props, np.float64)
    return np.stack([np.log10(props[:, 0]) - 5.0, props[:, 3], props[:, 4] / 1e4], -1).astype(np.float32)


def reanchored_sphere_path(centroid, cmd_vel, active, dt=DT):
    """Untied drag sphere (user, 2026-10-25): at the first frame of every drag event the sphere sits on the patch
    centroid of that frame; during the drag it moves ONLY by the commanded velocity (never by the patch); between
    drags it stays put (open, so it forces nothing). centroid [T, 3], cmd_vel [T-1, 3], active [T-1] -> path [T, 3]."""
    T = len(centroid)
    u = np.empty((T, 3), np.float64)
    u[0] = centroid[0]
    for t in range(T - 1):
        if active[t] and (t == 0 or not active[t - 1]):
            u[t] = centroid[t]                          # re-anchor at the drag's first frame
        u[t + 1] = u[t] + (cmd_vel[t] * dt if active[t] else 0.0)
    return u.astype(np.float32)


def to_pgnd_arrays(ep, sphere='tied'):
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
    if sphere == 'reanchor':                                   # sim.drag_sphere=reanchor: untied, command-driven
        c = reanchored_sphere_path(c, ep['cmd_vel'], active)
    elif sphere != 'tied':
        raise ValueError(f'sim.drag_sphere must be tied or reanchor, got {sphere!r}')
    eef = np.concatenate([c[:1], c[:-1]], 0)                   # eef[k] = c[k-1] (eef[0] = c[0], unused)
    closed = np.concatenate([[False], active])                 # flag[k] = transition k-1 -> k active
    eef_gripper = np.where(closed, 0.0, 1.0)[:, None].astype(np.float32)   # PGND: < 0.5 = closed
    return dict(xyz=(pos * FLIP).astype(np.float32), eef_xyz=(eef * FLIP).astype(np.float32), eef_rot=None,
                eef_gripper=eef_gripper, gripper_points=None)
