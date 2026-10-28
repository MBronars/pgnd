"""Full yikai (Table 1 Pick and Place) for PGND: a FIXED box shifted per window onto the gripper.

Whole cluttered scenes span up to 1.8 m, so no fixed PGND box holds an episode; a box that grows with the data
was ruled out (user, 2026-10-27/28). Instead, per window of n_history + 1 given + num_steps_train predicted frames:
  - the box is cfg.sim.num_grids (yikai: 70 x 50 x 70 cells of 2 cm = 1.4 m wide, 1.0 m tall, PGND axes x, up, z);
  - it is shifted so the gripper centroid at the window's FIRST frame sits at the box centre in x / z (the two
    horizontal axes) and the episode's table plane (its lowest object point) sits on the box floor, as stock PGND's
    preprocess_with_table does;
  - an object any of whose points leaves the box (PGND's clip margin included) in any frame of the window is
    disabled for that window (enabled = False: no loss, the simulator does not move it, and with
    model.material.mask_disabled the network ignores it);
  - a window whose gripper leaves the box, or that would disable an object moving > 5 mm within 25 cm of the gripper
    (an object being handled), is not drawn for training or validation (scan_yikai_box.py: 0.0011% of training
    windows at 70 x 70 x 50, ~1% for the gripper).
Batches: train.batch_size windows of ONE episode (gpwm's yikai batches: 8 windows of one episode per GPU), uniform
starts over the converted frames (which begin at gpwm's first start, frame 10), every point, the 671-point recorded
gripper (Path B). Same tuples as RealTeleopBatchDataset / pwm_folding_source.FoldingClip.

Episodes come from experiments/scripts/convert_pwm_domain.py --domain yikai (compact rigid bodies, world metres).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .gpwm_compact_codec import loads, unpack_positions

MOVE_M, NEAR_M = 0.005, 0.25          # a disabled object moving > 5 mm within 25 cm of the gripper = being handled
MAX_TRIES = 64                        # random starts tried per window before an episode is given up for this pass


def load_records(root, split):
    """Episode files of one split, in converter order (metadata.json ranges)."""
    root = Path(root)
    meta = json.load(open(root / 'metadata.json'))
    lo, hi = meta['ranges'][split]
    eps = [json.loads(line) for line in open(root / 'episodes.jsonl')]
    return [dict(path=str(root / f'episode_{e["idx"]:06d}.cpk'), episode_id=e['source'], frames=e['frames'])
            for e in eps[lo:hi]]


class YikaiEpisode:
    """One converted episode in PGND's sim axes (x, up, z) at preprocess_scale, not yet placed in the box."""

    def __init__(self, cfg, rec):
        self.cfg, self.rec = cfg, rec
        d = loads(Path(rec['path']).read_bytes())
        p = unpack_positions(d['positions'])                       # [T, 671 + No, 3] world metres, z up
        p = np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)       # world -> PGND (x, z, -y): up = +y (as preprocess)
        p = torch.from_numpy(p * float(cfg.sim.preprocess_scale)).float()
        n_grip = int(d['n_grip'])
        self.g = p[:, :n_grip]                                     # [T, P, 3]
        self.x = p[:, n_grip:]                                     # [T, No, 3]
        self.ids = torch.from_numpy(np.asarray(d['object_ids'], np.int64))
        self.bodies = [self.ids == k for k in torch.unique(self.ids)]
        dt = float(cfg.sim.dt)
        self.v = torch.empty_like(self.x)                          # backward: v[t] = x[t] - x[t-1], v[0] = v[1]
        self.v[1:] = (self.x[1:] - self.x[:-1]) / dt
        self.v[0] = self.v[1]
        self.gv = torch.empty_like(self.g)                         # forward: v[t] = x[t+1] - x[t], v[-1] = v[-2]
        self.gv[:-1] = (self.g[1:] - self.g[:-1]) / dt
        self.gv[-1] = self.gv[-2]
        self.floor_y = float(self.x[..., 1].min())                 # the table plane: the episode's lowest object point
        H, S = int(cfg.sim.n_history), int(cfg.sim.num_steps_train)
        self.W = H + 1 + S                                         # frames a window reads
        self.n_starts = len(self.x) - self.W + 1
        dx = float(cfg.sim.num_grids[-1])
        self.box = torch.tensor([float(cfg.sim.num_grids[i]) * dx for i in range(3)])
        self.lo = dx * (float(cfg.model.clip_bound) + 0.5)          # PGND keeps this much free at every wall
        self.clip_bound = float(cfg.model.clip_bound)

    def place(self, start):
        """(shift [3], enabled [No] bool, valid) for the window starting at `start` (its first given frame)."""
        sl = slice(start, start + self.W)
        c = self.g[start].mean(0)
        shift = torch.stack([self.box[0] / 2 - c[0], torch.tensor(self.lo + 1e-5 - self.floor_y), self.box[2] / 2 - c[2]])
        hi = self.box - self.lo
        def inside(pts):                                           # [w, n, 3] -> all points of all frames inside
            q = pts + shift
            return bool(((q >= self.lo - 1e-6) & (q <= hi + 1e-6)).all())
        gw = self.g[sl]
        valid = inside(gw)
        enabled = torch.ones(self.x.shape[1], dtype=torch.bool)
        gc = gw.mean(1)                                            # [w, 3] gripper centroid per frame
        for m in self.bodies:
            o = self.x[sl][:, m]
            if inside(o):
                continue
            enabled[m] = False
            moved = float((o - o[:1]).norm(dim=-1).max())
            near = float((o.mean(1)[:, [0, 2]] - gc[:, [0, 2]]).norm(dim=-1).min())
            if moved > MOVE_M and near < NEAR_M:                   # an object being handled would be dropped
                valid = False
        return shift, enabled, valid

    def window(self, start, shift, enabled):
        """Same tuple as RealTeleopBatchDataset.__getitem__ (skip_frame 1, every point), placed by `shift`."""
        H, S = int(self.cfg.sim.n_history), int(self.cfg.sim.num_steps_train)
        frame = start + H
        x, v = self.x[frame] + shift, self.v[frame]
        x_his = torch.cat([self.x[frame + h] + shift for h in range(-H, 0)], dim=-1) if H else x.new_zeros(len(x), 0)
        v_his = torch.cat([self.v[frame + h] for h in range(-H, 0)], dim=-1) if H else v.new_zeros(len(v), 0)
        init_state = (x, v, x_his, v_his, torch.tensor(self.clip_bound), enabled.clone(), torch.tensor([0, frame]))
        sl = slice(frame + 1, frame + S + 1)
        actions = torch.zeros(S, 1, 15)                            # sphere colliders unused (gripper-point mode)
        return (init_state, actions, (self.x[sl] + shift, self.v[sl]),
                ((self.g[sl] + shift)[:, None], self.gv[sl][:, None]))

    def draw(self, n, rng):
        """n valid windows with uniform random starts (None when the episode cannot supply them)."""
        if self.n_starts <= 0:
            return None
        items, tries = [], 0
        while len(items) < n and tries < MAX_TRIES * n:
            tries += 1
            s = int(rng.integers(0, self.n_starts))
            shift, enabled, valid = self.place(s)
            if valid:
                items.append(self.window(s, shift, enabled))
        return torch.utils.data.default_collate(items) if len(items) == n else None


class YikaiTrainBatches(torch.utils.data.IterableDataset):
    """Endless training batches: each loader worker walks its share of the episodes in a fresh random order per pass
    (as pwm_folding_source.FoldingTrainBatches), one batch of train.batch_size windows per episode."""

    def __init__(self, cfg, records):
        super().__init__()
        self.cfg, self.records = cfg, records

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
        mine = self.records[rank * nw + wid::world * nw]
        if not mine:
            raise ValueError(f'loader worker {wid} of {nw} has no yikai episodes; use fewer train.num_workers')
        rng = np.random.default_rng([int(self.cfg.seed), rank, wid, int.from_bytes(os.urandom(4), 'little')])
        B = int(self.cfg.train.batch_size)
        while True:
            for i in rng.permutation(len(mine)):
                batch = YikaiEpisode(self.cfg, mine[i]).draw(B, rng)
                if batch is not None:
                    yield batch


def fixed_val_batches(cfg, records, n_batches):
    """n_batches test episodes evenly spaced in converter order, batch_size valid windows each from a per-episode seed."""
    B = int(cfg.train.batch_size)
    pick = np.linspace(0, len(records) - 1, min(n_batches, len(records))).round().astype(int)
    out = []
    for i in pick:
        rec = records[i]
        seed = int.from_bytes(hashlib.sha256(rec['episode_id'].encode()).digest()[:8], 'little')
        batch = YikaiEpisode(cfg, rec).draw(B, np.random.default_rng(seed))
        if batch is not None:
            out.append(batch)
    return out
