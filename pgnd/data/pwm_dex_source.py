"""Read the PWM dex release (pwm-data-v1/dex_v2_filtered2/<split>) directly -- no converted copy.

Reproduces pwm_final_paper's IndexedEpisodeDataset(domain="dex").window(record, 0) for a whole
episode (data/recording_adapters.py dex_window): the INDEX.json records (exclusions are already
applied by the index), frames 0..frames-1 (native index 2*t), bodies [object, link_0..link_L-1]
posed from the recorded poses: world = R(q_wxyz) @ template + origin, templates from the
per-shard geometry file (object_<shape_id>, hand_<k>). No shift (CANONICAL_TRANSLATION dex = 0),
z up, metres. The same checks as the PWM loader: shard and geometry size/mtime, geometry
sha256, link-name order, frames == (episode_length + 1) // 2, stride 2, 0.025 s control clock,
finite unit quaternions. Checked against the PWM loader by experiments/scripts/check_dex_source.py.

Depends only on numpy + the standard library (runs in the pgnd env).
"""
import hashlib
import json
import multiprocessing
import struct
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

FLIP = np.array([1.0, -1.0, -1.0], np.float32)   # world -> PGND source axes: preprocess then puts world up on +y
FLOOR_MARGIN_IN_HAND = 0.10                      # m of clearance between the scene and the grid floor (in-hand tasks)


def npz_array(path, key):
    """One member of a ZIP_STORED npz as a read-only memmap (pwm_final_paper data/npz_arrays.py)."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(key + '.npy')
        if info.compress_type != zipfile.ZIP_STORED:
            with archive.open(info) as stream:
                return np.lib.format.read_array(stream, allow_pickle=False)
    with open(path, 'rb') as stream:
        stream.seek(info.header_offset)
        header = stream.read(30)
        if len(header) != 30 or header[:4] != b'PK\x03\x04':
            raise ValueError(f'Invalid ZIP local header: {path}:{key}')
        name_size, extra_size = struct.unpack('<HH', header[26:30])
        stream.seek(name_size + extra_size, 1)
        member_start = stream.tell()
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        if dtype.hasobject:
            raise ValueError(f'Object arrays are not supported: {path}:{key}')
        offset = stream.tell()
        size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if offset + size != member_start + info.file_size:
            raise ValueError(f'NPY size disagrees with ZIP member: {path}:{key}')
    if size == 0:
        return np.empty(shape, dtype=dtype, order='F' if fortran else 'C')
    return np.memmap(path, dtype=dtype, mode='r', shape=shape, order='F' if fortran else 'C', offset=offset)


def quat_wxyz_to_matrix(q):
    """[..., 4] wxyz -> [..., 3, 3] (the standard formula, float32 like the PWM loader)."""
    w, x, y, z = np.moveaxis(q.astype(np.float32), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def hand_of(teacher):
    return 'sharpa' if teacher.endswith('sharpa') else 'xhand1'


def in_hand(teacher):
    """In-hand / in-palm tasks have no ground; the screwdriver tasks stand on the floor at z = 0."""
    return not teacher.startswith('screwdriver')


@lru_cache(maxsize=None)
def _geometry(path, size, mtime_ns, sha256):
    stat = Path(path).stat()
    if (stat.st_size, stat.st_mtime_ns) != (size, mtime_ns):
        raise ValueError(f'Dex geometry changed: {path}')
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha256:
        raise ValueError(f'Dex geometry sha256 differs from the index: {path}')
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def _read_shard(args):
    """All requested envs of one shard -> list of (record position, episode dict)."""
    path, wants = args[:2]
    prepare = args[2] if len(args) > 2 else None
    stat = Path(path).stat()
    names = [str(n) for n in npz_array(path, 'link_names')]
    lengths = np.asarray(npz_array(path, 'episode_length'))
    shape_ids = np.asarray(npz_array(path, 'episode_shape_id'))
    if float(npz_array(path, 'control_dt')) != .025:
        raise ValueError(f'Dex source clock changed: {path}')
    link_pos, link_quat = npz_array(path, 'link_pos'), npz_array(path, 'link_quat')
    obj_pos, obj_quat = npz_array(path, 'obj_pos'), npz_array(path, 'obj_quat')
    out = []
    for i, rec in wants:
        if (stat.st_size, stat.st_mtime_ns) != (rec['size'], rec['mtime_ns']):
            raise ValueError(f'Source shard changed since indexing: {path}')
        geo = _geometry(rec['geometry_file'], rec['geometry_size'], rec['geometry_mtime_ns'], rec['geometry_sha256'])
        if names != [str(n) for n in geo['link_names']]:
            raise ValueError('Dex geometry and shard link order differ')
        env = rec['env']
        length = int(lengths[env])
        if rec['stride'] != 2 or rec['frames'] != (length + 1) // 2:
            raise ValueError('Dex requires indexed first-episode length and native stride 2')
        idx = np.arange(rec['frames']) * 2
        lp = np.asarray(link_pos[idx, env], np.float32)                       # [T, L, 3]
        lq = np.asarray(link_quat[idx, env], np.float32)                      # [T, L, 4]
        op = np.asarray(obj_pos[idx, env], np.float32)                        # [T, 3]
        oq = np.asarray(obj_quat[idx, env], np.float32)                       # [T, 4]
        quats = np.concatenate([oq[:, None], lq], 1)
        if not np.isfinite(quats).all() or (np.abs(np.linalg.norm(quats, axis=-1) - 1) > 1e-4).any():
            raise ValueError('Dex requires finite unit wxyz quaternions')
        rot = quat_wxyz_to_matrix(quats)                                      # [T, 1+L, 3, 3]
        shape = int(shape_ids[env])
        obj = np.einsum('tij,pj->tpi', rot[:, 0], geo[f'object_{shape}'].astype(np.float32)) + op[:, None]
        hand = np.concatenate([np.einsum('tij,pj->tpi', rot[:, k + 1], geo[f'hand_{k}'].astype(np.float32))
                               + lp[:, k, None] for k in range(len(names))], 1)
        ep = dict(episode_id=rec['episode_id'], teacher=rec['teacher'], bucket=hand_of(rec['teacher']),
                  obj=obj.astype(np.float32), hand=hand.astype(np.float32))
        if prepare is not None:
            # keep the poses (a few hundred kB) instead of the points (~10 MB): prepare() sees the
            # full points once and returns what the dataset needs; points are rebuilt per window
            extra = prepare(ep)
            ep = dict(episode_id=ep['episode_id'], teacher=ep['teacher'], bucket=ep['bucket'],
                      frames=int(rec['frames']), geometry=(rec['geometry_file'], rec['geometry_size'],
                      rec['geometry_mtime_ns'], rec['geometry_sha256']), shape=shape, n_links=len(names),
                      obj_pos=op, obj_quat=oq, link_pos=lp, link_quat=lq, **extra)
        out.append((i, ep))
    return out


def load_episodes(root, workers=8, limit=0, prepare=None, part=None):
    """Every episode of one split, in INDEX.json record order (limit > 0: that many, evenly spaced -- smokes).
    prepare: optional picklable callable(ep) -> dict, run in the worker; the episode then keeps only
    its poses + prepare()'s output (see LazyPoints), not the points."""
    index = json.load(open(Path(root) / 'INDEX.json'))
    if index.get('domain') != 'dex':
        raise ValueError(f'{root}: domain {index.get("domain")!r}, expected dex')
    records = index['records']
    for rec in records:   # pwm-data-v1-compact: paths relative to the INDEX.json folder (as gpwm's loader resolves them)
        for key in ('file', 'geometry_file'):
            if isinstance(rec.get(key), str) and not Path(rec[key]).is_absolute():
                rec[key] = str(Path(root) / rec[key])
    if limit:
        records = [records[i] for i in np.linspace(0, len(records) - 1, min(limit, len(records))).round().astype(int)]
    if part is not None:   # (rank, world): this process's share, records[rank::world] (PWM's rank sharding)
        records = records[part[0]::part[1]]
    by_file = defaultdict(list)
    for i, rec in enumerate(records):
        by_file[rec['file']].append((i, rec))
    # split big shards so workers share the load (each shard holds ~1,000 envs)
    jobs = [(f, wants[lo:lo + 200], prepare) for f, wants in by_file.items() for lo in range(0, len(wants), 200)]
    episodes = [None] * len(records)
    if workers > 1:
        # spawn, not fork: prepare() runs PGND's preprocessing (torch) and the parent has already
        # initialized torch/warp/CUDA -- forking there deadlocks the workers (0 CPU, seen 2026-09-22)
        ctx = multiprocessing.get_context('spawn' if prepare is not None else 'fork')
        with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
            for res in pool.map(_read_shard, jobs):
                for i, ep in res:
                    episodes[i] = ep
    else:
        for job in jobs:
            for i, ep in _read_shard(job):
                episodes[i] = ep
    return episodes


def to_pgnd_arrays(ep):
    """One episode -> keyword arrays for PGND's preprocess_arrays().

    Objects: the 512 object points. Gripper: the recorded, posed hand points as ONE PGND gripper
    point cloud (sim.gripper_points; the network sees them each step, only object points are
    predicted). eef_* are placeholders (hand centroid, identity rotation, zero flag), unused in
    gripper_points mode. Floor (user 2026-09-22): screwdriver tasks keep PGND's stock drop (the
    object's lowest point = the real ground at z = 0); in-hand tasks put the grid floor
    FLOOR_MARGIN_IN_HAND below the lowest point of object AND hand over the episode, so neither
    touches it. Stored with world * (1, -1, -1) axes.
    """
    obj, hand = ep['obj'] * FLIP, ep['hand'] * FLIP
    out = dict(xyz=obj.astype(np.float32), eef_xyz=hand.mean(1).astype(np.float32), eef_rot=None,
               eef_gripper=None, gripper_points=hand[:, None].astype(np.float32))
    if in_hand(ep['teacher']):
        out.update(floor_points=np.concatenate([obj, hand], 1), floor_margin=FLOOR_MARGIN_IN_HAND)
    return out


# ---------------------------------------------------------------- lazy points (poses -> window points)

_R_PGND = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])   # pgnd.data.dataset.preprocess axis map


@lru_cache(maxsize=64)
def _templates(path, size, mtime_ns, sha256, shape, n_links):
    geo = _geometry(path, size, mtime_ns, sha256)
    return (geo[f'object_{shape}'].astype(np.float32),
            [geo[f'hand_{k}'].astype(np.float32) for k in range(n_links)])


def world_points(ep, ts, which):
    """World points of frames ts: 'obj' [len(ts), 512, 3] or 'hand' [len(ts), Nh, 3] (float32, as _read_shard)."""
    obj_t, hand_t = _templates(*ep['geometry'], ep['shape'], ep['n_links'])
    if which == 'obj':
        return (np.einsum('tij,pj->tpi', quat_wxyz_to_matrix(ep['obj_quat'][ts]), obj_t)
                + ep['obj_pos'][ts][:, None]).astype(np.float32)
    rot = quat_wxyz_to_matrix(ep['link_quat'][ts])                              # [t, L, 3, 3]
    return np.concatenate([np.einsum('tij,pj->tpi', rot[:, k], hand_t[k]) + ep['link_pos'][ts][:, k, None]
                           for k in range(ep['n_links'])], 1).astype(np.float32)


class LazyPoints:
    """Stands in for an episode's preprocessed point tensor ([T, N, 3], or [T, 1, N, 3] for the
    gripper): indexing with an int or slice builds just those frames from the poses, with the same
    float operations as preprocess_arrays (FLIP, axis map, scale, the episode's global translation)."""

    def __init__(self, ep, which, scale, translation, gripper_axis=False):
        self.ep, self.which, self.gripper_axis = ep, which, gripper_axis
        self.scale, self.translation = scale, np.asarray(translation, np.float64)
        n = 512 if which == 'obj' else sum(len(t) for t in _templates(*ep['geometry'], ep['shape'], ep['n_links'])[1])
        self.shape = (ep['frames'], 1, n, 3) if gripper_axis else (ep['frames'], n, 3)

    def size(self, d=None):
        return self.shape if d is None else self.shape[d]

    def __len__(self):
        return self.shape[0]

    def frames(self, ts):
        import torch
        p = world_points(self.ep, np.asarray(ts), self.which) * FLIP
        p = np.einsum('nij,jk->nik', p, _R_PGND.T) * self.scale + self.translation
        p = torch.tensor(p).float()
        return p[:, None] if self.gripper_axis else p

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.frames(range(*idx.indices(self.shape[0])))
        return self.frames([int(idx)])[0]


def prepare_for_pgnd(ep, cfg, dt_frame, dx):
    """Worker-side: PGND's preprocess_arrays on the full episode once, keeping only what the dataset
    needs -- the grid placement (info), the gripper state tensor, and the per-frame mean object
    displacement for importance sampling. Object points must all survive the bounds check."""
    from pgnd.data.dataset import preprocess_arrays
    info, frames = preprocess_arrays(cfg, ep['episode_id'], **to_pgnd_arrays(ep), dt_frame=dt_frame, dx=dx)
    if info['num_particles'] != ep['obj'].shape[1]:
        raise ValueError(f"{ep['episode_id']}: {ep['obj'].shape[1] - info['num_particles']} object points outside the grid")
    x = frames['x']
    return dict(info=info, grippers=frames['grippers'],
                frame_motion=(x[1:] - x[:-1]).norm(dim=-1).mean(dim=-1))
