from pathlib import Path
from typing import Union, Optional
from omegaconf import DictConfig

import os
import torch
import time
import shutil
import json
import yaml
import random
import glob
import kornia
import numpy as np
import pickle as pkl
import open3d as o3d
from torch.utils.data import Dataset
from dgl.geometry import farthest_point_sampler
from sklearn.neighbors import NearestNeighbors



def preprocess_arrays(cfg, name, xyz, eef_xyz, eef_rot, eef_gripper, gripper_points, dt_frame, dx,
                      floor_points=None, floor_margin=0.0):
    """PGND's episode preprocessing on in-memory arrays (the body of the stock file-based
    preprocess(), which now only loads/saves around it).

    xyz (T, N, 3); eef_xyz (T, 3 * n_grippers); eef_rot (T, 9 * n_grippers) or None (identity);
    eef_gripper (T, n_grippers) or None (zeros); gripper_points (T, n_grippers, P, 3) or None.
    floor_points (T, M, 3) or None: with preprocess_with_table, the grid floor goes floor_margin
    (metres) below the lowest of xyz AND these points instead of at xyz's lowest point (stock).
    Returns (info dict for info.yaml, frames dict of tensors for frames.pt).
    """
    # Original PGND check assumed 30 Hz source data; we removed it because we
    # also train on 20 Hz pushing data. dt_frame is just a velocity-normalization
    # scalar -- as long as it's consistent train+eval, the model is self-consistent.

    # transform xyz: x -> x, z -> -y, y -> z
    R = np.array(
        [[1, 0, 0],
         [0, 0, -1],
         [0, 1, 0]]
    )

    eef_global_T = np.array([cfg.model.eef_t[0], cfg.model.eef_t[1], cfg.model.eef_t[2]])  # 1018_sloth: 0.185, 1018_rope_short: 0.01

    n_frames = eef_xyz.shape[0]
    eef_xyz = eef_xyz.reshape(eef_xyz.shape[0], -1, 3)  # (n_frames, n_grippers, 3)
    eef_xyz = eef_xyz + eef_global_T

    xyz = np.einsum('nij,jk->nik', xyz, R.T)
    eef_xyz = np.einsum('nij,jk->nik', eef_xyz, R.T)  # (n_frames, n_grippers, 3)

    if eef_rot is not None:  # (n_frames, -1)
        eef_rot = eef_rot.reshape(eef_rot.shape[0], -1, 3, 3)  # (n_frames, n_grippers, 3, 3)
        eef_rot = R @ eef_rot @ R.T
        eef_rot = torch.from_numpy(eef_rot).reshape(-1, 3, 3)  # (n_frames, n_grippers, 3, 3)
        eef_quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(eef_rot).numpy().reshape(n_frames, -1, 4)  # (n_frames, n_grippers, 4)
    else:
        eef_quat = np.zeros((xyz.shape[0], cfg.sim.num_grippers, 4), dtype=np.float32)  # (n_frames, n_grippers, 4)
        eef_quat[:, :, 0] = 1.0  # (n_frames, n_grippers, 4)

    if eef_gripper is not None:  # (n_frames, n_grippers)
        eef_gripper = eef_gripper.reshape(eef_gripper.shape[0], -1)
        assert cfg.sim.num_grippers == eef_gripper.shape[1]
    else:
        eef_gripper = np.zeros((n_frames, cfg.sim.num_grippers), dtype=np.float32)  # (n_frames, n_grippers)
    assert cfg.sim.num_grippers == eef_quat.shape[1]

    # per-frame gripper point clouds (PGND Path B: used directly, no template FK).
    # transformed with the SAME R / scale / global_translation as the cloth so
    # grippers and cloth stay in a consistent sim frame.
    has_gripper_points = gripper_points is not None
    if has_gripper_points:  # (n_frames, num_grippers, P, 3)
        assert gripper_points.shape[1] == cfg.sim.num_grippers, \
            f'gripper_points has {gripper_points.shape[1]} grippers, cfg expects {cfg.sim.num_grippers}'
        assert gripper_points.shape[0] == xyz.shape[0], 'gripper/cloth frame count mismatch'
        gripper_points = np.einsum('nkpj,jl->nkpl', gripper_points, R.T)

    scale = cfg.sim.preprocess_scale
    floor_y = None
    if floor_points is not None:   # same transform as xyz: R, then scale
        fp = np.einsum('nij,jk->nik', floor_points, R.T) * scale
        floor_y = min(float(fp[:, :, 1].min()), float((xyz[:, :, 1] * scale).min())) - floor_margin * scale
    xyz = xyz * scale
    eef_xyz = eef_xyz * scale
    if has_gripper_points:
        gripper_points = gripper_points * scale

    n_frames = xyz.shape[0]

    # construct colliders: position, velocity, height, radius, axis
    eef_vel = np.zeros_like(eef_xyz)  # (n_frames, n_grippers, 3)
    eef_vel[:-1] = (eef_xyz[1:] - eef_xyz[:-1]) / dt_frame
    eef_vel[-1] = eef_vel[-2]

    # construct rotational velocity
    eef_rot_this = kornia.geometry.conversions.quaternion_to_rotation_matrix(torch.from_numpy(eef_quat[:-1]).reshape(-1, 4))  # (n_frames-1 * n_grippers, 3, 3)
    eef_rot_next = kornia.geometry.conversions.quaternion_to_rotation_matrix(torch.from_numpy(eef_quat[1:]).reshape(-1, 4))  # (n_frames-1 * n_grippers, 3, 3)
    eef_rot_delta = eef_rot_this.bmm(eef_rot_next.inverse())
    eef_aa = kornia.geometry.conversions.rotation_matrix_to_axis_angle(eef_rot_delta)  # (n_frames-1 * n_grippers, 3)

    eef_quat_vel = np.zeros((n_frames, cfg.sim.num_grippers, 3), dtype=eef_quat.dtype)
    eef_quat_vel[:-1] = eef_aa.reshape(n_frames - 1, -1, 3) / dt_frame  # (n_frames-1, n_grippers, 3), radian per second
    eef_quat_vel[-1] = eef_quat_vel[-2]

    grippers = np.zeros((n_frames, cfg.sim.num_grippers, 15))
    grippers[:, :, :3] = eef_xyz
    grippers[:, :, 3:6] = eef_vel
    grippers[:, :, 6:10] = eef_quat
    grippers[:, :, 10:13] = eef_quat_vel
    grippers[:, :, 13] = cfg.model.gripper_radius
    grippers[:, :, 14] = eef_gripper
    
    # box = num_grids x dx per axis (stock PGND hard-coded a [0, 1] box: 50 x 0.02)
    box = np.array([cfg.sim.num_grids[i] * dx for i in range(3)], dtype=np.float64)
    lo = dx * (cfg.model.clip_bound + 0.5)

    if cfg.sim.preprocess_with_table:  # drop everything to the table plane (y=0)
        global_translation = np.array([
            box[0] / 2 - (xyz[:, :, 0].max() + xyz[:, :, 0].min()) / 2,
            lo + 1e-5 - (xyz[:, :, 1].min() if floor_y is None else floor_y),
            box[2] / 2 - (xyz[:, :, 2].max() + xyz[:, :, 2].min()) / 2,
        ], dtype=xyz.dtype)
    else:
        global_translation = np.array([
            box[0] / 2 - (xyz[:, :, 0].max() + xyz[:, :, 0].min()) / 2,
            box[1] / 2 - (xyz[:, :, 1].max() + xyz[:, :, 1].min()) / 2,
            box[2] / 2 - (xyz[:, :, 2].max() + xyz[:, :, 2].min()) / 2,
        ], dtype=xyz.dtype)
    xyz += global_translation
    grippers[:, :, :3] += global_translation
    if has_gripper_points:
        gripper_points += global_translation
    
    if not (xyz[:, :, 0].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
        and xyz[:, :, 0].max() <= box[0] - lo + 1e-6 \
        and xyz[:, :, 1].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
        and xyz[:, :, 1].max() <= box[1] - lo + 1e-6 \
        and xyz[:, :, 2].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
        and xyz[:, :, 2].max() <= box[2] - lo + 1e-6):
        print(name, 'out of bound')
        xyz_max = xyz.max(axis=0)
        xyz_max_mask = (xyz_max[:, 0] > box[0] - lo + 1e-6) | \
            (xyz_max[:, 1] > box[1] - lo + 1e-6) | \
            (xyz_max[:, 2] > box[2] - lo + 1e-6)
        xyz_min = xyz.min(axis=0)
        xyz_min_mask = (xyz_min[:, 0] < dx * (cfg.model.clip_bound + 0.5) - 1e-6) | \
            (xyz_min[:, 1] < dx * (cfg.model.clip_bound + 0.5) - 1e-6) | \
            (xyz_min[:, 2] < dx * (cfg.model.clip_bound + 0.5) - 1e-6)
        xyz_mask = xyz_max_mask | xyz_min_mask
        xyz = xyz[:, ~xyz_mask]
        
        assert (xyz[:, :, 0].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
            and xyz[:, :, 0].max() <= box[0] - lo + 1e-6 \
            and xyz[:, :, 1].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
            and xyz[:, :, 1].max() <= box[1] - lo + 1e-6 \
            and xyz[:, :, 2].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
            and xyz[:, :, 2].max() <= box[2] - lo + 1e-6)

    info = {
        'num_particles': xyz.shape[1],
        'clip_bound': cfg.model.clip_bound,
        # transform that maps raw metres -> sim frame, for inverting predictions:
        #   xyz_sim = (xyz_raw @ R.T) * scale + global_translation
        'preprocess_scale': float(scale),
        'global_translation': [float(t) for t in global_translation],
    }
    frames = {
        'x': torch.tensor(xyz).float(),                  # (n_frames, n_particles, 3)
        'grippers': torch.tensor(grippers).float(),      # (n_frames, n_grippers, 15)
    }
    if has_gripper_points:
        frames['gripper_points'] = torch.tensor(gripper_points).float()  # (n_frames, n_grippers, P, 3)
    return info, frames


def preprocess(cfg, dataset_root_episode, source_data_root_episode, dt_frame, dx):
    print('Preprocessing episode:', dataset_root_episode.name)
    src = source_data_root_episode
    xyz = np.load(src / 'traj.npz')['xyz']
    eef_xyz = np.loadtxt(src / 'eef_traj.txt')
    eef_rot = np.loadtxt(src / 'eef_rot.txt') if os.path.exists(src / 'eef_rot.txt') else None
    eef_gripper = np.loadtxt(src / 'eef_gripper.txt') if os.path.exists(src / 'eef_gripper.txt') else None
    gripper_points = None
    if os.path.exists(src / 'gripper_points.npz'):
        gp = np.load(src / 'gripper_points.npz')
        # PGND-original loads ['left', 'right']; we also accept a 1-gripper layout
        # ({'left': ...} only) for single-paddle datasets like pushing.
        keys = ['left', 'right'][:cfg.sim.num_grippers]
        assert all(k in gp for k in keys), \
            f'gripper_points.npz missing keys {keys}; has {list(gp.keys())}'
        gripper_points = np.stack([gp[k] for k in keys], axis=1)  # (n_frames, num_grippers, P, 3)
    info, frames = preprocess_arrays(cfg, dataset_root_episode.name, xyz, eef_xyz, eef_rot, eef_gripper,
                                     gripper_points, dt_frame, dx)
    with open(dataset_root_episode / 'info.yaml', 'w') as f:
        yaml.dump(info, f)
    # save the whole episode as a single file (one .pt per episode, not one per
    # frame -- ~400k tiny files would cripple NFS metadata throughput).
    torch.save(frames, dataset_root_episode / 'frames.pt')
    return True  # necessary for success check


class _FiniteDiff:
    """Per-episode finite-difference velocities, computed on access instead of stored
    (storing them doubled resident RAM, ~140 GB for the three-domain mix).

    backward: v[t] = (x[t] - x[t-1]) / dt, v[0] = v[1]   -- particles (PGND stock)
    forward:  v[t] = (x[t+1] - x[t]) / dt, v[-1] = v[-2] -- gripper points, matching stock
              PGND's gripper velocity (preprocess: eef_vel[:-1] = (eef[1:] - eef[:-1]) / dt)
    """

    def __init__(self, xs, dt, forward=False):
        self.xs, self.dt, self.forward = xs, dt, forward

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        xs = self.xs[i]
        if hasattr(xs, 'frames'):          # LazyPoints: velocities for the requested frames only
            return _LazyDiff(xs, self.dt, self.forward)
        vs = torch.empty_like(xs)
        if self.forward:
            vs[:-1] = (xs[1:] - xs[:-1]) / self.dt
            vs[-1] = vs[-2]
        else:
            vs[1:] = (xs[1:] - xs[:-1]) / self.dt
            vs[0] = vs[1]
        return vs


class _LazyDiff:
    """Finite-difference velocities of a LazyPoints episode, built for the requested frames only
    (same values as _FiniteDiff: backward v[t] = (x[t] - x[t-1]) / dt with v[0] = v[1]; forward
    v[t] = (x[t+1] - x[t]) / dt with v[-1] = v[-2])."""

    def __init__(self, xs, dt, forward):
        self.xs, self.dt, self.forward, self.T = xs, dt, forward, len(xs)

    def _pairs(self, ts):
        ts = np.asarray(list(ts))
        if self.forward:
            a = np.minimum(ts, self.T - 2)
            return a + 1, a
        a = np.maximum(ts, 1)
        return a, a - 1

    def _diff(self, ts):
        hi, lo = self._pairs(ts)
        return (self.xs.frames(hi) - self.xs.frames(lo)) / self.dt

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._diff(range(*idx.indices(self.T)))
        return self._diff([int(idx)])[0]


def _episode_domains(source_data_root, episode_dirs):
    """Domain name per episode dir from the converter's metadata.json 'ranges' (None if absent)."""
    meta_path = Path(source_data_root) / 'metadata.json'
    if not meta_path.exists():
        return None
    ranges = json.load(open(meta_path)).get('ranges')
    if not ranges:
        return None
    out = []
    for ep in episode_dirs:
        idx = int(ep.name.split('_')[-1])
        names = [k.split('/')[-1] for k, (a, b) in ranges.items() if a <= idx < b]
        assert len(names) == 1, f'{ep.name}: in {len(names)} domain ranges'
        out.append(names[0])
    return out


def fps(x, enabled, n, device, random_start=False):
    assert torch.diff(enabled * 1.0).sum() in [0.0, -1.0]
    start_idx = random.randint(0, enabled.sum() - 1) if random_start else 0
    fps_idx = farthest_point_sampler(x[enabled][None], n, start_idx=start_idx)[0]
    fps_idx = fps_idx.to(x.device)
    return fps_idx


class RealTeleopBatchDataset(Dataset):

    def __init__(self,
                cfg: DictConfig, 
                dataset_root: Union[str, Path],
                source_data_root: Union[str, Path], 
                device: torch.device, 
                num_steps: int, 
                train=False,
                eval_episode_name=None,
                dataset_non_overwrite=None,
                episode_range=None,
                source_split=None,
        ) -> None:
        super().__init__()
        self.device = device
        dataset_root = Path(dataset_root).resolve()
        dataset_root.mkdir(parents=True, exist_ok=True)
        self.root = dataset_root
        self.is_train = train

        self.interval = cfg.sim.interval
        self.dt = eval(cfg.sim.dt) if isinstance(cfg.sim.dt, str) else cfg.sim.dt
        self.num_steps = num_steps
        self.skip_frame = cfg.train.dataset_skip_frame
        self.n_particles = cfg.sim.n_particles
        # subsample gripper_points to this count per gripper at __getitem__ time
        # (matches PGND's RealGripperDataset n=500). None / unset / <=0 = no subsample.
        ngp = cfg.sim.n_gripper_particles if 'n_gripper_particles' in cfg.sim else None
        self.n_gripper_particles = int(ngp) if (ngp is not None and ngp) else None
        self.n_history = cfg.sim.n_history
        self.uniform = cfg.sim.uniform or not train  # always use uniform sampling during test
        print('Using uniform sampling:', self.uniform)
        self.load_skip_frame = cfg.train.dataset_load_skip_frame
        self.dx = cfg.sim.num_grids[-1]
        self.preprocess_scale = cfg.sim.preprocess_scale

        self.return_downsample_indices = (eval_episode_name is not None)

        with torch.no_grad():
            # train.source_format=pwm_mpm_drag | pwm_dex: read that PWM release split directly
            # (pwm_drag_source / pwm_dex_source) and preprocess in memory -- no converted copy,
            # no frames.pt cache. The split replaces the episode-index ranges (train -> 'train',
            # val -> 'test'); each episode's 'bucket' (object count / hand) is its batch bucket.
            direct = cfg.train.get('source_format', 'pgnd') in ('pwm_mpm_drag', 'pwm_dex')
            lazy_points = cfg.train.get('source_format', 'pgnd') == 'pwm_dex'
            frame_motion = []
            dt_frame = self.dt * self.interval / self.skip_frame / self.load_skip_frame
            if direct:
                prepare = None
                if cfg.train.source_format == 'pwm_mpm_drag':
                    from .pwm_drag_source import load_episodes, to_pgnd_arrays
                else:
                    import functools
                    from .pwm_dex_source import load_episodes, prepare_for_pgnd, to_pgnd_arrays
                    prepare = functools.partial(prepare_for_pgnd, cfg=cfg, dt_frame=dt_frame, dx=self.dx)
                split = source_split or ('train' if train else 'test')
                pwm_eps = load_episodes(Path(cfg.train.pwm_root) / split, workers=int(cfg.train.num_workers),
                                        limit=int(cfg.train.get('pwm_max_episodes') or 0),
                                        **({'prepare': prepare} if prepare is not None else {}))
                episodes = [e['episode_id'] for e in pwm_eps]
                direct_domains = [e['bucket'] for e in pwm_eps]
                print(f'{cfg.train.source_format} {split}: {len(episodes)} episodes from {cfg.train.pwm_root}')
            else:
                source_data_root = Path(source_data_root).resolve()
                episodes = list(sorted(Path(str(source_data_root)).glob('episode_*')))
                if len(episodes) == 0:
                    import ipdb; ipdb.set_trace()
                if eval_episode_name is not None:
                    assert not train
                    episodes = [e for e in episodes if e.name == eval_episode_name]
                if train:
                    episodes = episodes[cfg.train.training_start_episode:cfg.train.training_end_episode]
                if episode_range is not None:
                    # explicit episode slice (e.g. validation set) -- mutually
                    # exclusive with train slicing / single-episode eval
                    assert not train and eval_episode_name is None
                    episodes = episodes[episode_range[0]:episode_range[1]]
                # train.exclude_episodes: JSON {"episodes": [episode dir names]} dropped AFTER the
                # index-range slicing above, so training_*/eval_* ranges keep their meaning.
                if cfg.train.get('exclude_episodes'):
                    excluded = set(json.load(open(cfg.train.exclude_episodes))['episodes'])
                    n_before = len(episodes)
                    episodes = [e for e in episodes if e.name not in excluded]
                    print(f'exclude_episodes: dropped {n_before - len(episodes)} of {n_before} '
                          f'({len(excluded)} listed in {cfg.train.exclude_episodes})')

            episodes_clean = []
            episodes_xs = []
            episodes_enabled = []
            episodes_grippers = []
            episodes_gripper_points = []  # per-frame gripper point clouds (Path B)
            episodes_clip_bound = []
            len_sim_list = []
            traj_list = []
            meta_list = []
            use_grippers = False
            max_num_particles = 0
            max_num_frames = 0
            overwrite = None if eval_episode_name is None else True

            def load_cached(source_data_root_episode):
                """File mode: preprocess into <dataset_root>/<episode>/frames.pt once, then load.
                Returns (frames, info, cache dir) or None if preprocessing failed."""
                nonlocal overwrite
                episode_name = source_data_root_episode.name
                dataset_root_episode = dataset_root / episode_name
                # one cache file per episode ('frames.pt') -- NOT one per frame:
                # ~400k tiny files would cripple NFS for both preprocessing and
                # every subsequent dataset load.
                frames_path = dataset_root_episode / 'frames.pt'
                cached = frames_path.exists()
                if cached:
                    if dataset_non_overwrite:
                        overwrite = False
                    while overwrite is None:
                        feedback = input(f'{dataset_root_episode} already exists, overwrite? [y/n] ')
                        ret = feedback.casefold()
                        if ret == 'y':
                            overwrite = True
                            break
                        elif ret == 'n':
                            overwrite = False
                            break
                    if overwrite:
                        shutil.rmtree(dataset_root_episode, ignore_errors=True)
                        cached = False
                if not cached:
                    dataset_root_episode.mkdir(parents=True, exist_ok=True)
                    success = preprocess(cfg, dataset_root_episode, source_data_root_episode, dt_frame, self.dx)
                    if not success:
                        shutil.rmtree(dataset_root_episode, ignore_errors=True)
                        return None
                frames = torch.load(frames_path, map_location='cpu')
                with open(dataset_root_episode / 'info.yaml', 'r') as f:
                    info = yaml.safe_load(f)
                return frames, info, dataset_root_episode

            for episode_id in range(len(episodes)):
                if direct and lazy_points:
                    # dex: the episode kept its poses, not its points (prepare_for_pgnd); build a
                    # LazyPoints view per body set and let __getitem__ materialize only its window
                    from .pwm_dex_source import LazyPoints
                    ep = pwm_eps[episode_id]
                    info = ep['info']
                    frames = {'x': LazyPoints(ep, 'obj', self.preprocess_scale, info['global_translation']),
                              'grippers': ep['grippers'],
                              'gripper_points': LazyPoints(ep, 'hand', self.preprocess_scale,
                                                           info['global_translation'], gripper_axis=True)}
                    frame_motion.append(ep['frame_motion'])
                    dataset_root_episode = episodes[episode_id]
                elif direct:
                    info, frames = preprocess_arrays(cfg, episodes[episode_id], **to_pgnd_arrays(pwm_eps[episode_id]),
                                                     dt_frame=dt_frame, dx=self.dx)
                    pwm_eps[episode_id] = None   # release the raw copy
                    dataset_root_episode = episodes[episode_id]
                else:
                    loaded = load_cached(episodes[episode_id])
                    if loaded is None:
                        continue
                    frames, info, dataset_root_episode = loaded
                episodes_clean.append(episode_id)

                xs = frames['x'][::self.load_skip_frame]
                n_frames_ep = xs.size(0)
                traj_list.append(dataset_root_episode)

                num_particles = xs.size(1)
                max_num_particles = max(max_num_particles, num_particles)
                max_num_frames = max(n_frames_ep, max_num_frames)

                len_sim = max(1, n_frames_ep - (self.num_steps + self.n_history) * self.skip_frame)
                len_sim_list.append(len_sim)

                episodes_xs.append(xs)

                grippers = frames['grippers'][::self.load_skip_frame]
                episodes_grippers.append(grippers)

                # per-frame gripper point clouds (Path B): (n_frames, n_grippers, P, 3)
                if 'gripper_points' in frames:
                    gp = frames['gripper_points'][::self.load_skip_frame]
                    episodes_gripper_points.append(gp)

                clip_bound = info['clip_bound']
                episodes_clip_bound.append(clip_bound)

            # Keep episodes as per-episode lists (no padding / no stacking): the
            # full-trajectory cloth episodes vary widely in length and particle
            # count, and padding them to a common (max_frames, max_particles)
            # shape would blow up RAM. __getitem__ indexes one episode at a time,
            # so a list of variable-shaped tensors works unchanged.
            for xs in episodes_xs:
                episodes_enabled.append(torch.ones(xs.size(1), dtype=torch.bool))

            self.episode_xs = episodes_xs              # list of (n_frames_i, n_particles_i, 3)
            v_dt = self.dt * self.interval / self.skip_frame
            self.episode_vs = _FiniteDiff(episodes_xs, v_dt)
            self.episode_enabled = episodes_enabled    # list of (n_particles_i,) all-True
            self.episode_grippers = episodes_grippers
            self.episode_gripper_points = episodes_gripper_points  # list of (n_frames_i, n_grippers, P, 3)
            self.episode_gripper_vs = _FiniteDiff(episodes_gripper_points, v_dt, forward=True)
            self.has_gripper_points = len(episodes_gripper_points) > 0
            self.episode_clip_bound = torch.tensor(episodes_clip_bound, dtype=torch.float32)
            self.len_sim_list = np.array(len_sim_list)
            self.total_len_sim = sum(len_sim_list)
            self.cum_len_sim = np.cumsum(self.len_sim_list)

            # domain of every kept episode (three-domain converter metadata), and the
            # dataset indices (window starts) belonging to each domain, for
            # domain-pure batches. None for single-domain sources.
            domains = direct_domains if direct else _episode_domains(source_data_root, episodes)
            if domains is not None:
                self.episode_domain = [domains[i] for i in episodes_clean]
                starts = np.concatenate([[0], self.cum_len_sim[:-1]])
                self.domain_indices = {}
                for ep, d in enumerate(self.episode_domain):
                    self.domain_indices.setdefault(d, []).append(
                        np.arange(starts[ep], starts[ep] + self.len_sim_list[ep]))
                self.domain_indices = {d: np.concatenate(v) for d, v in self.domain_indices.items()}
            else:
                self.episode_domain, self.domain_indices = None, None
            self.episode_frame_motion = frame_motion if lazy_points else None
            self.max_num_particles = max_num_particles
            self.traj_list = traj_list
            self.meta_list = meta_list

            # motion-weighted importance-sampling weight per dataset index
            # (only meaningful for training; eval rolls out a single episode and
            # the weights-vs-windows count would mismatch when num_steps is large)
            if train:
                self.sample_weights = self._compute_sample_weights()

        print('Finished loading dataset')
        print('Total number of episodes:', len(episodes_clean))
        print('Total number of frames:', self.total_len_sim)
        assert len(self.episode_xs) == len(episodes_clean)

    def _compute_sample_weights(self):
        """One importance-sampling weight per dataset index (window start).

        score = mean per-particle cloth motion (mm, raw scale) over the rollout
        window, floored at 0.1 mm; weight = score ** 2  (DATASET_SPEC sec. 4).
        Up-weights windows with real cloth motion so training isn't dominated by
        static-cloth frames.
        """
        # window spans W consecutive per-frame displacements
        W = (self.num_steps + self.n_history) * self.skip_frame
        weights = []
        for episode_id, xs in enumerate(self.episode_xs):
            len_sim = int(self.len_sim_list[episode_id])
            # per-consecutive-frame mean particle displacement (preprocessed units); lazy
            # episodes carry it from load time (their points are not kept)
            fm = (self.episode_frame_motion[episode_id] if self.episode_frame_motion is not None
                  else (xs[1:] - xs[:-1]).norm(dim=-1).mean(dim=-1))  # (n_frames - 1,)
            csum = torch.cat([fm.new_zeros(1), fm.cumsum(0)])  # (n_frames,)
            # sliding window sum: score for window start r = mean(fm[r : r+W])
            avail = min(len_sim, csum.shape[0] - W)
            wsum = csum[W:W + avail] - csum[:avail]            # (avail,)
            # back to raw metres -> mm (preprocessed xs are raw * preprocess_scale)
            scores = wsum / W / self.preprocess_scale * 1000.0
            w = torch.clamp(scores, min=0.1) ** 2.0
            if avail < len_sim:  # short episode safety pad (shouldn't trigger here)
                w = torch.cat([w, w.new_full((len_sim - avail,), w.mean())])
            weights.append(w)
        weights = torch.cat(weights)
        assert weights.shape[0] == self.total_len_sim, \
            f'{weights.shape[0]} weights vs {self.total_len_sim} windows'
        return weights

    def __len__(self) -> int:
        return self.total_len_sim


    def __getitem__(self, index):
        cum_len_sim = self.cum_len_sim
        episode = int(np.searchsorted(cum_len_sim, index, side='right'))
        frame = index - cum_len_sim[episode - 1] if episode > 0 else index

        # history frames
        frame = frame + self.n_history * self.skip_frame

        ep_vs = self.episode_vs[episode]  # computed once per sample (see _FiniteDiff)
        x = self.episode_xs[episode][frame]
        v = ep_vs[frame]  # velocity from step to step+1
        enabled = self.episode_enabled[episode]

        # downsample
        if not self.n_particles or self.n_particles <= 0:
            downsample_indices = torch.arange(int(enabled.sum()))   # keep every point
        elif self.uniform:
            downsample_indices = fps(x, enabled, self.n_particles, self.device, random_start=True)
        else:
            downsample_indices = torch.randperm(enabled.sum())[:self.n_particles]
        x = x[downsample_indices]
        v = v[downsample_indices]

        x_his = torch.zeros((x.size(0), 0), dtype=x.dtype, device=x.device)
        v_his = torch.zeros((x.size(0), 0), dtype=v.dtype, device=v.device)
        for his_id in range(-self.n_history, 0):
            his_frame = frame + his_id * self.skip_frame
            assert his_frame >= 0
            x_his_frame = self.episode_xs[episode][his_frame]
            v_his_frame = ep_vs[his_frame]
            x_his_frame = x_his_frame[downsample_indices]
            v_his_frame = v_his_frame[downsample_indices]
            x_his = torch.cat([x_his, x_his_frame], dim=-1)
            v_his = torch.cat([v_his, v_his_frame], dim=-1)

        enabled = torch.ones(x.size(0), dtype=torch.bool)
        clip_bound = self.episode_clip_bound[episode]
        episode_vec = torch.tensor([episode, frame])
        init_state = (x, v, x_his, v_his, clip_bound, enabled, episode_vec)

        end_frame = frame + self.num_steps * self.skip_frame + 1

        actions = self.episode_grippers[episode][frame+self.skip_frame:end_frame:self.skip_frame]

        gt_xs = self.episode_xs[episode][frame+self.skip_frame:end_frame:self.skip_frame]
        gt_vs = ep_vs[frame+self.skip_frame:end_frame:self.skip_frame]
        
        gt_xs = gt_xs[:, downsample_indices]
        gt_vs = gt_vs[:, downsample_indices]
        
        gt_states = (gt_xs, gt_vs)

        # per-frame gripper point clouds + velocities for the rollout window,
        # aligned with `actions` / `gt_states` (Path B: used directly, no FK).
        if self.has_gripper_points:
            gripper_x = self.episode_gripper_points[episode][frame+self.skip_frame:end_frame:self.skip_frame]
            gripper_v = self.episode_gripper_vs[episode][frame+self.skip_frame:end_frame:self.skip_frame]
            # match PGND's RealGripperDataset: subsample to n_gripper_particles per
            # gripper (FPS at eval, randperm at train). Same indices across all
            # frames of the window so point identity is consistent.
            P = gripper_x.shape[-2]
            if self.n_gripper_particles is not None and P > self.n_gripper_particles:
                n_g = self.n_gripper_particles
                num_grip = gripper_x.shape[1]
                if self.uniform:
                    # FPS per gripper (pick from frame 0)
                    enabled_g = torch.ones(P, dtype=torch.bool)
                    sub_x, sub_v = [], []
                    for g in range(num_grip):
                        idx_g = fps(gripper_x[0, g], enabled_g, n_g, self.device, random_start=True).cpu()
                        sub_x.append(gripper_x[:, g, idx_g])
                        sub_v.append(gripper_v[:, g, idx_g])
                    gripper_x = torch.stack(sub_x, dim=1)
                    gripper_v = torch.stack(sub_v, dim=1)
                else:
                    # randperm at train; same indices for both grippers (symmetric template)
                    idx_g = torch.randperm(P)[:n_g]
                    gripper_x = gripper_x[:, :, idx_g]
                    gripper_v = gripper_v[:, :, idx_g]
        else:
            gripper_x = torch.zeros(0)
            gripper_v = torch.zeros(0)
        gripper_states = (gripper_x, gripper_v)  # each (num_steps, n_grippers, P, 3)

        if self.return_downsample_indices:
            # eval mode: match PGND-original 4-tuple (no gripper_states; eval.py
            # uses RealGripperDataset separately for Path A, and rollout_cloth.py
            # reads ds.episode_* directly without calling __getitem__).
            return init_state, actions, gt_states, downsample_indices
        else:
            return init_state, actions, gt_states, gripper_states
