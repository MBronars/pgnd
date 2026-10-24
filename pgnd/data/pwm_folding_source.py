"""Read the PWM folding release (pwm-data-v1/folding_8mm/<split>) directly and serve PGND batches.

Windows and batches follow pwm_final_paper's IndexedEpisodeDataset(domain="folding") as our folding
model trains on it (data/indexed_episodes.py + data/recording_adapters.py folding_window):
  - the INDEX.json records (official train / test garments), 30 Hz, every frame (no stride);
  - the garment's retained cloth vertices (cloth-subset-8mm.npz, every one kept) and the 2 x 1,000
    recorded gripper points (rows n_cloth.. of particle_pos);
  - a batch = batch_size windows of ONE clip, starts drawn with replacement with
    p(start) ∝ (mean cloth-vertex step over the window's 16 future transitions + floor)^alpha
    (the recipe's data.window_motion for folding: alpha 1.0, floor 0.02 mm);
  - window = n_history + 1 given frames + num_steps_train predicted frames (4 + 16 = PWM C4/F16).
Each clip goes through PGND's own preprocess_arrays (axes, scale, grid placement over the whole clip);
velocities as PGND: cloth backward differences, gripper points forward differences, at the frame dt.
Gripper = PGND's gripper-point mode (sim.gripper_points=True): the recorded points of both hands every
frame, no sphere forcing (user 2026-10-22).

Validation: a FIXED set -- val_batches test clips evenly spaced in INDEX order, starts from
np.random.default_rng(sha256(episode_id)) as the PWM loader draws evaluation windows.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .dataset import preprocess_arrays
from .release_files import same_file

FLIP = np.array([1.0, -1.0, -1.0], np.float32)   # world (z up) -> PGND source axes; preprocess puts world up on +y
N_GRIP = 2
P_GRIP = 1000


RELATIVE_PATH_FIELDS = ('file', 'subset_file', 'template_file', 'geometry_file')   # as gpwm's indexed_episodes.py


def resolve_paths(records, root):
    """pwm-data-v1-compact writes record paths relative to the INDEX.json folder (the release can move between
    clusters); resolve them as gpwm's loader does. pwm-data-v1 paths are absolute and pass through."""
    for rec in records:
        for key in RELATIVE_PATH_FIELDS:
            if isinstance(rec.get(key), str) and not Path(rec[key]).is_absolute():
                rec[key] = str(Path(root) / rec[key])
    return records


def load_index(root):
    index = json.load(open(Path(root) / 'INDEX.json'))
    if index.get('domain') != 'folding':
        raise ValueError(f'{root}: domain {index.get("domain")!r}, expected folding')
    return resolve_paths(index['records'], root)


def read_clip(rec):
    """Raw clip in world metres: cloth [T, n_keep, 3] (retained vertices), grippers [T, 2, 1000, 3].

    Reads both releases: pwm-data-v1 (every native vertex in particle_pos + the garment's subset file) and
    pwm-data-v1-compact (the clip holds only the retained vertices, as `positions_cpk`: cloth lossless, the gripper
    links as rigid bodies within 0.00013 mm), decoded with gpwm's codec as gpwm's loader does."""
    st = Path(rec['file']).stat()
    if not same_file(st, rec['size'], rec['mtime_ns']):
        raise ValueError(f'Source changed since indexing: {rec["file"]}')
    with np.load(rec['file'], allow_pickle=False) as z:
        nc = int(z['n_cloth'])
        if not np.isclose(float(z['dt']), 1 / 30):
            raise ValueError(f'{rec["file"]}: dt {float(z["dt"])}')
        if 'positions_cpk' in z.files:
            from .gpwm_compact_codec import loads, unpack_positions
            p = unpack_positions(loads(z['positions_cpk'].tobytes())['positions'])
        else:
            p = z['particle_pos']
    if p.shape != (rec['frames'], nc + N_GRIP * P_GRIP, 3):
        raise ValueError(f'{rec["file"]}: particle_pos {p.shape}')
    if 'subset_file' not in rec:       # compact clip: n_cloth is already the retained subset
        if nc != rec['particles'] - N_GRIP * P_GRIP:
            raise ValueError(f'{rec["file"]}: {nc} cloth vertices vs index {rec["particles"]}')
        return p[:, :nc], p[:, nc:].reshape(len(p), N_GRIP, P_GRIP, 3)
    st = Path(rec['subset_file']).stat()
    if not same_file(st, rec['subset_size'], rec['subset_mtime_ns']):
        raise ValueError(f'Subset changed since indexing: {rec["subset_file"]}')
    with np.load(rec['subset_file'], allow_pickle=False) as s:
        keep = np.asarray(s['indices'], dtype=np.int64)        # sorted unique vertex ids (PWM _folding_subset)
    if keep.ndim != 1 or np.any(np.diff(keep) <= 0):
        raise ValueError(f'{rec["subset_file"]}: subset must be sorted unique ids')
    if len(keep) != rec['particles'] - N_GRIP * P_GRIP or keep[-1] >= nc:
        raise ValueError(f'{rec["subset_file"]}: {len(keep)} kept vertices vs index {rec["particles"]}')
    cloth = p[:, keep]
    grip = p[:, nc:].reshape(len(p), N_GRIP, P_GRIP, 3)
    return cloth, grip


def motion_weights(cloth, n_given, n_pred, alpha, floor_m):
    """PWM _start_weights: starts 0..T-n_given-n_pred, score = mean cloth step over the n_pred future transitions."""
    m = np.linalg.norm(np.diff(cloth, axis=0), axis=-1).mean(1)          # [T-1]
    total = np.concatenate([[0.0], np.cumsum(m)])
    starts = np.arange(0, len(cloth) - n_given - n_pred + 1)
    score = (total[starts + n_given + n_pred - 1] - total[starts + n_given - 1]) / n_pred
    return torch.tensor((score + floor_m) ** alpha, dtype=torch.float64)


class FoldingClip:
    """One clip after PGND preprocessing, ready to cut windows from."""

    def __init__(self, cfg, rec):
        self.cfg, self.rec = cfg, rec
        cloth, grip = read_clip(rec)
        self.raw_cloth = cloth
        dt = float(cfg.sim.dt)
        info, frames = preprocess_arrays(
            cfg, rec['episode_id'], (cloth * FLIP).astype(np.float32),
            eef_xyz=(grip.mean(2) * FLIP).reshape(len(grip), -1).astype(np.float32),   # unused in gripper-point mode
            eef_rot=None, eef_gripper=None, gripper_points=(grip * FLIP).astype(np.float32),
            dt_frame=dt, dx=cfg.sim.num_grids[-1])
        if info['num_particles'] != cloth.shape[1]:
            raise ValueError(f'{rec["episode_id"]}: {cloth.shape[1] - info["num_particles"]} cloth points left the box '
                             f'(preprocess_scale {cfg.sim.preprocess_scale})')
        self.x = frames['x']                                   # [T, N, 3] sim frame
        self.g = frames['gripper_points']                      # [T, 2, P, 3]
        self.v = torch.empty_like(self.x)                      # backward: v[t] = x[t] - x[t-1], v[0] = v[1]
        self.v[1:] = (self.x[1:] - self.x[:-1]) / dt
        self.v[0] = self.v[1]
        self.gv = torch.empty_like(self.g)                     # forward: v[t] = x[t+1] - x[t], v[-1] = v[-2]
        self.gv[:-1] = (self.g[1:] - self.g[:-1]) / dt
        self.gv[-1] = self.gv[-2]
        self.clip_bound = float(info['clip_bound'])

    def window(self, start):
        """Same tuple as RealTeleopBatchDataset.__getitem__ (skip_frame 1, every point)."""
        H, S = int(self.cfg.sim.n_history), int(self.cfg.sim.num_steps_train)
        frame = start + H
        x, v = self.x[frame], self.v[frame]
        x_his = torch.cat([self.x[frame + h] for h in range(-H, 0)], dim=-1) if H else x.new_zeros(len(x), 0)
        v_his = torch.cat([self.v[frame + h] for h in range(-H, 0)], dim=-1) if H else v.new_zeros(len(v), 0)
        enabled = torch.ones(len(x), dtype=torch.bool)
        init_state = (x, v, x_his, v_his, torch.tensor(self.clip_bound), enabled, torch.tensor([0, frame]))
        sl = slice(frame + 1, frame + S + 1)
        actions = torch.zeros(S, N_GRIP, 15)                   # sphere colliders unused (gripper-point mode)
        return init_state, actions, (self.x[sl], self.v[sl]), (self.g[sl], self.gv[sl])

    def batch(self, starts):
        items = [self.window(s) for s in starts]
        return torch.utils.data.default_collate(items)


class FoldingTrainBatches(torch.utils.data.IterableDataset):
    """Endless training batches: each worker walks its share of the records in a fresh random order per pass
    (PWM: records[rank*workers + worker :: world*workers], randperm per pass), one batch per clip."""

    def __init__(self, cfg, records):
        super().__init__()
        self.cfg, self.records = cfg, records
        wm = cfg.train.pwm_window_motion
        self.alpha, self.floor_m = float(wm.alpha), float(wm.floor_mm) / 1000.0

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
        mine = self.records[rank * nw + wid::world * nw]       # PWM: records[rank*workers + worker :: world*workers]
        if not mine:
            raise ValueError(f'loader worker {wid} of {nw} has no folding clips; use fewer train.num_workers')
        H, S, B = int(self.cfg.sim.n_history), int(self.cfg.sim.num_steps_train), int(self.cfg.train.batch_size)
        while True:
            for i in torch.randperm(len(mine)).tolist():
                clip = FoldingClip(self.cfg, mine[i])
                w = motion_weights(clip.raw_cloth, H + 1, S, self.alpha, self.floor_m) if self.alpha > 0 else None
                n = len(clip.x) - (H + 1) - S + 1
                starts = (torch.multinomial(w / w.sum(), B, replacement=True) if w is not None
                          else torch.randint(n, (B,))).tolist()
                yield clip.batch(starts)


def fixed_val_batches(cfg, records, n_batches):
    """n_batches clips evenly spaced in INDEX order, batch_size starts each from a per-episode seed."""
    H, S, B = int(cfg.sim.n_history), int(cfg.sim.num_steps_train), int(cfg.train.batch_size)
    pick = np.linspace(0, len(records) - 1, min(n_batches, len(records))).round().astype(int)
    out = []
    for i in pick:
        rec = records[i]
        clip = FoldingClip(cfg, rec)
        seed = int.from_bytes(hashlib.sha256(rec['episode_id'].encode()).digest()[:8], 'little')
        starts = np.random.default_rng(seed).integers(0, len(clip.x) - (H + 1) - S + 1, size=B).tolist()
        out.append(clip.batch(starts))
    return out
