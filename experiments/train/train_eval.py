from pathlib import Path
import random
import time
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm, trange
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from PIL import Image
import warp as wp
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import kornia
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from pgnd.sim import Friction, CacheDiffSimWithFrictionBatch, StaticsBatch, CollidersBatch
from pgnd.material import PGNDModel
from pgnd.data import RealTeleopBatchDataset, RealGripperDataset
from pgnd.utils import Logger, get_root, mkdir

from train.pv_train import do_train_pv
from train.pv_dataset import do_dataset_pv
from train.metric_eval import do_metric

root: Path = get_root(__file__)

def dataloader_wrapper(dataloader, name):
    cnt = 0
    while True:
        cnt += 1
        for data in dataloader:
            yield data

class DomainBatchSampler(torch.utils.data.Sampler):
    """Endless domain-pure batches: each batch picks a domain (bucket) at random, then
    batch_size window starts from it -- by importance weight within the bucket when weights
    are given, else uniformly (with replacement, like WeightedRandomSampler).
    domain_probs None = uniform over buckets (the PWM three-domain round robin); else a
    {bucket: weight} dict, e.g. episode counts (PWM batch_across_episodes visits every
    episode once per pass, so a bucket's share of batches follows its episode count)."""

    def __init__(self, domain_indices, batch_size, weights=None, seed=0, domain_probs=None):
        self.domains = sorted(domain_indices)
        p = [1.0 if domain_probs is None else float(domain_probs[d]) for d in self.domains]
        self.domain_p = torch.tensor(p, dtype=torch.double) / sum(p)
        self.indices = {d: torch.as_tensor(domain_indices[d], dtype=torch.long) for d in self.domains}
        self.weights = None if weights is None else {
            d: torch.as_tensor(weights, dtype=torch.double)[self.indices[d]] for d in self.domains}
        self.batch_size = batch_size
        self.gen = torch.Generator().manual_seed(seed)

    def __iter__(self):
        while True:
            d = self.domains[int(torch.multinomial(self.domain_p, 1, generator=self.gen))]
            idx = self.indices[d]
            if self.weights is None:
                pick = torch.randint(len(idx), (self.batch_size,), generator=self.gen)
            else:
                pick = torch.multinomial(self.weights[d], self.batch_size, replacement=True, generator=self.gen)
            yield idx[pick].tolist()


def transform_gripper_points(cfg, gripper_points, gripper):
    """PGND-original template-FK: takes the EE-local gripper point template
    (gripper_points: (bsz, P, 3)) and the per-frame gripper poses (gripper:
    (bsz, num_steps, num_grippers, 15)) and returns the world-frame gripper
    points + velocities + in-bounds mask. Restored from PGND HEAD so eval.py
    (which uses gripper_points via RealGripperDataset) still imports. Our own
    Path-B cloth pipeline uses prepare_gripper_points() below instead."""
    dx = cfg.sim.num_grids[-1]
    gripper_xyz = gripper[:, :, :, :3]
    gripper_v = gripper[:, :, :, 3:6]
    gripper_quat = gripper[:, :, :, 6:10]
    num_steps = gripper_xyz.shape[1]
    num_grippers = gripper_xyz.shape[2]
    gripper_mat = kornia.geometry.conversions.quaternion_to_rotation_matrix(gripper_quat)
    gripper_points = gripper_points[:, None, None].repeat(1, num_steps, num_grippers, 1, 1)
    gripper_x = gripper_points @ gripper_mat + gripper_xyz[:, :, :, None]
    bsz = gripper_x.shape[0]
    num_points = gripper_x.shape[3]

    gripper_quat_vel = gripper[:, :, :, 10:13]
    gripper_angular_vel = torch.linalg.norm(gripper_quat_vel, dim=-1, keepdims=True)
    gripper_quat_axis = gripper_quat_vel / (gripper_angular_vel + 1e-10)

    gripper_v_expand = gripper_v[:, :, :, None].repeat(1, 1, 1, num_points, 1)
    gripper_points_from_axis = gripper_x - gripper_xyz[:, :, :, None]
    grid_from_gripper_axis = gripper_points_from_axis - \
        (gripper_quat_axis[:, :, :, None] * gripper_points_from_axis).sum(dim=-1, keepdims=True) * gripper_quat_axis[:, :, :, None]
    gripper_v_expand = torch.cross(gripper_quat_vel[:, :, :, None], grid_from_gripper_axis, dim=-1) + gripper_v_expand
    gripper_v = gripper_v_expand.reshape(bsz, num_steps, num_grippers * num_points, 3)
    gripper_x = gripper_x.reshape(bsz, num_steps, num_grippers * num_points, 3)

    gripper_x_mask = (gripper_x[:, :, :, 0] > dx * (cfg.model.clip_bound + 0.5)) \
                   & (gripper_x[:, :, :, 0] < 1 - (dx * (cfg.model.clip_bound + 0.5))) \
                   & (gripper_x[:, :, :, 1] > dx * (cfg.model.clip_bound + 0.5)) \
                   & (gripper_x[:, :, :, 1] < 1 - (dx * (cfg.model.clip_bound + 0.5))) \
                   & (gripper_x[:, :, :, 2] > dx * (cfg.model.clip_bound + 0.5)) \
                   & (gripper_x[:, :, :, 2] < 1 - (dx * (cfg.model.clip_bound + 0.5)))
    return gripper_x, gripper_v, gripper_x_mask


def prepare_gripper_points(cfg, gripper_x, gripper_v):
    """Path B: gripper points come straight from the dataset as per-frame world-
    frame point clouds (already in the sim frame after preprocess). This just
    flattens the per-gripper axis and computes the in-bounds mask -- replacing
    the old template-FK `transform_gripper_points`.

    gripper_x, gripper_v: (bsz, num_steps, num_grippers, P, 3)
    returns: gripper_x, gripper_v  (bsz, num_steps, num_grippers*P, 3)
             gripper_mask          (bsz, num_steps, num_grippers*P)  in-bounds
    """
    dx = cfg.sim.num_grids[-1]
    bsz, num_steps, num_grippers, num_points, _ = gripper_x.shape
    gripper_x = gripper_x.reshape(bsz, num_steps, num_grippers * num_points, 3)
    gripper_v = gripper_v.reshape(bsz, num_steps, num_grippers * num_points, 3)

    lo = dx * (cfg.model.clip_bound + 0.5)
    box = torch.tensor([cfg.sim.num_grids[i] * dx for i in range(3)], device=gripper_x.device, dtype=gripper_x.dtype)
    gripper_mask = ((gripper_x > lo) & (gripper_x < box - lo)).all(dim=-1)
    return gripper_x, gripper_v, gripper_mask


class Trainer:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        print(OmegaConf.to_yaml(cfg, resolve=True))

        wp.init()
        wp.ScopedTimer.enabled = False
        wp.set_module_options({'fast_math': False})
        wp.config.verify_autograd_array_access = True

        gpus = [int(gpu) for gpu in cfg.gpus]
        wp_devices = [wp.get_device(f'cuda:{gpu}') for gpu in gpus]
        torch_devices = [torch.device(f'cuda:{gpu}') for gpu in gpus]
        device_count = len(torch_devices)
    
        assert device_count == 1
        self.wp_device = wp_devices[0]
        self.torch_device = torch_devices[0]

        seed = cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        torch.autograd.set_detect_anomaly(True)

        torch.backends.cudnn.benchmark = True

        # path
        log_root: Path = root / 'log'
        exp_root: Path = log_root / cfg.train.name
        mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume)
        OmegaConf.save(cfg, exp_root / 'hydra.yaml', resolve=True)

        ckpt_root: Path = exp_root / 'ckpt'
        ckpt_root.mkdir(parents=True, exist_ok=True)
        
        self.log_root = log_root
        self.ckpt_root = ckpt_root

        self.use_pv = cfg.train.use_pv
        self.dataset_non_overwrite = cfg.train.dataset_non_overwrite
        if not self.use_pv:
            print('not using pv rendering...')
        
        assert self.cfg.train.source_dataset_name is not None
        self.use_gs = cfg.train.use_gs

        # logging
        self.verbose = False
        if not cfg.debug:
            logger = Logger(cfg, project='pgnd-train')
            self.logger = logger

    def load_train_dataset(self):
        cfg = self.cfg
        if cfg.train.dataset_name is None:
            cfg.train.dataset_name = Path(cfg.train.name).parent / 'dataset'

        source_dataset_root = self.log_root / str(cfg.train.source_dataset_name)
        if cfg.train.source_format == 'pgnd':   # pwm_mpm_drag / pwm_dex read cfg.train.pwm_root instead
            assert os.path.exists(source_dataset_root)

        dataset = RealTeleopBatchDataset(
            cfg, 
            dataset_root=self.log_root / cfg.train.dataset_name / 'state',
            source_data_root=source_dataset_root, 
            device=self.torch_device,
            num_steps=cfg.sim.num_steps_train,
            train=True,
            dataset_non_overwrite=self.dataset_non_overwrite,
        )
        self.dataset = dataset
        # Path B: per-frame gripper points are loaded by RealTeleopBatchDataset
        # itself (no separate RealGripperDataset / template).

    def init_train(self):
        cfg = self.cfg

        # motion-weighted importance sampling: up-weight windows with real cloth
        # motion so training isn't dominated by static-cloth frames (spec sec. 4).
        if cfg.train.domain_batches:
            assert self.dataset.domain_indices is not None, 'domain_batches needs converter metadata ranges'
            weights = self.dataset.sample_weights if cfg.train.importance_sampling else None
            probs = None
            if cfg.train.domain_weighting == 'episodes':
                probs = {d: self.dataset.episode_domain.count(d) for d in self.dataset.domain_indices}
            elif str(cfg.train.domain_weighting).startswith('only:'):   # debug: one bucket only (memory checks)
                only = str(cfg.train.domain_weighting)[5:]
                probs = {d: float(d == only) for d in self.dataset.domain_indices}
                assert sum(probs.values()) == 1, (only, list(probs))
            else:
                assert cfg.train.domain_weighting == 'uniform', cfg.train.domain_weighting
            batch_sampler = DomainBatchSampler(self.dataset.domain_indices, cfg.train.batch_size,
                                               weights=weights, seed=cfg.seed, domain_probs=probs)
            train_loader = DataLoader(
                self.dataset, batch_sampler=batch_sampler,
                num_workers=cfg.train.num_workers, pin_memory=True)
            print(f'Domain-pure batches ({cfg.train.domain_weighting}):',
                  dict(zip(batch_sampler.domains, batch_sampler.domain_p.numpy().round(3))),
                  {d: len(v) for d, v in batch_sampler.indices.items()}, 'windows;',
                  'importance sampling within domain:', bool(cfg.train.importance_sampling))
        elif cfg.train.importance_sampling:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=self.dataset.sample_weights,
                num_samples=len(self.dataset),
                replacement=True,
            )
            train_loader = DataLoader(
                self.dataset, batch_size=cfg.train.batch_size, sampler=sampler,
                num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)
            print('Importance sampling: ON')
        else:
            train_loader = DataLoader(
                self.dataset, batch_size=cfg.train.batch_size, shuffle=True,
                num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)
            print('Importance sampling: OFF (uniform)')
        dataloader = dataloader_wrapper(train_loader, 'dataset')
        self.dataloader = dataloader

        # material model
        material_requires_grad = cfg.model.material.requires_grad
        material: nn.Module = PGNDModel(cfg)
        material.to(self.torch_device)
        material.requires_grad_(material_requires_grad)
        material.train(True)

        # friction
        friction: nn.Module = Friction(np.array([cfg.model.friction.value]))
        friction.to(self.torch_device)
        friction.requires_grad_(False)
        friction.train(False)

        # auto-resume: if resuming without an explicit iteration, pick up the
        # latest checkpoint on disk (lets a re-submitted sbatch job continue).
        # checkpoints: latest.pt (every save interval), best.pt (lowest mean val),
        # final.pt (end of training) -- no per-step snapshots.
        latest = self.ckpt_root / 'latest.pt'
        self.best_val = float('inf')
        if cfg.resume and cfg.train.resume_iteration == 0 and latest.exists():
            ckpt = torch.load(latest, map_location=self.torch_device)
            cfg.train.resume_iteration = int(ckpt['iteration'])
            material.load_state_dict(ckpt['material'])
            self.best_val = float(ckpt.get('best_val', float('inf')))
            print(f'auto-resume: continuing from iteration {cfg.train.resume_iteration}')
        elif cfg.model.ckpt:
            ckpt = torch.load(self.log_root / cfg.model.ckpt, map_location=self.torch_device)
            material.load_state_dict(ckpt['material'])

        if material_requires_grad:
            material_optimizer = torch.optim.Adam(material.parameters(), lr=cfg.train.material_lr, weight_decay=cfg.train.material_wd)
            material_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=material_optimizer, T_max=cfg.train.num_iterations)
            if cfg.train.resume_iteration > 0:
                material_lr_scheduler.last_epoch = cfg.train.resume_iteration - 1
                material_lr_scheduler.step()

        criterion = nn.MSELoss(reduction='mean')
        criterion.to(self.torch_device)

        total_step_count = 0
        if cfg.resume and cfg.train.resume_iteration > 0:
            total_step_count = cfg.train.resume_iteration * cfg.sim.num_steps_train * int(cfg.train.grad_accum)
        losses_log = defaultdict(int)
        loss_factor_v = cfg.train.loss_factor_v
        loss_factor_x = cfg.train.loss_factor_x
    
        self.loss_factor_v = loss_factor_v
        self.loss_factor_x = loss_factor_x
        self.material_requires_grad = material_requires_grad
        self.material = material
        self.material_optimizer = material_optimizer
        self.material_lr_scheduler = material_lr_scheduler
        self.criterion = criterion
        self.total_step_count = total_step_count
        self.losses_log = losses_log
        self.friction = friction
    
    def train(self, start_iteration, end_iteration, save=True):
        cfg = self.cfg
        self.material.train(True)
        for iteration in trange(start_iteration, end_iteration, dynamic_ncols=True):
            if self.material_requires_grad:
                self.material_optimizer.zero_grad()

            # train.grad_accum micro-batches of train.batch_size per optimizer step (gradients
            # summed with loss / grad_accum, i.e. the mean over grad_accum x batch_size windows);
            # grad_accum 1 is stock PGND.
            accum = int(cfg.train.grad_accum)
            for micro in range(accum):
                losses = defaultdict(int)

                init_state, actions, gt_states, gripper_states = next(self.dataloader)
                x, v, x_his, v_his, clip_bound, enabled, episode_vec = init_state
                x = x.to(self.torch_device)
                v = v.to(self.torch_device)
                x_his = x_his.to(self.torch_device)
                v_his = v_his.to(self.torch_device)

                actions = actions.to(self.torch_device)

                if cfg.sim.gripper_points:
                    gripper_x, gripper_v = gripper_states
                    gripper_x = gripper_x.to(self.torch_device)
                    gripper_v = gripper_v.to(self.torch_device)
                    gripper_x, gripper_v, gripper_mask = prepare_gripper_points(cfg, gripper_x, gripper_v)  # (bsz, num_steps, num_grippers*P, 3)

                gt_x, gt_v = gt_states
                gt_x = gt_x.to(self.torch_device)
                gt_v = gt_v.to(self.torch_device)

                # gt_x: (bsz, num_steps_total)
                batch_size = gt_x.shape[0]
                num_steps_total = gt_x.shape[1]
                num_particles = gt_x.shape[2]

                if cfg.sim.gripper_points:
                    num_gripper_particles = gripper_x.shape[2]
                    num_particles_orig = num_particles
                    num_particles = num_particles + num_gripper_particles

                sim = CacheDiffSimWithFrictionBatch(cfg, num_steps_total, batch_size, self.wp_device, requires_grad=True)

                statics = StaticsBatch()
                statics.init(shape=(batch_size, num_particles), device=self.wp_device)
                statics.update_clip_bound(clip_bound)
                statics.update_enabled(enabled)
                colliders = CollidersBatch()

                if cfg.sim.gripper_points:
                    assert not cfg.sim.gripper_forcing
                    num_grippers = 0
                else:
                    num_grippers = cfg.sim.num_grippers

                colliders.init(shape=(batch_size, num_grippers), device=self.wp_device)
                if num_grippers > 0:
                    assert len(actions.shape) > 2
                    colliders.initialize_grippers(actions[:, 0])

                enabled = enabled.to(self.torch_device)  # (bsz, num_particles)
                enabled_mask = enabled.unsqueeze(-1).repeat(1, 1, 3)  # (bsz, num_particles, 3)

                for step in range(num_steps_total):
                    if num_grippers > 0:
                        colliders.update_grippers(actions[:, step])

                    x_in = x.clone()
                    if step == 0:
                        x_in_gt = x.clone()
                        v_in_gt = v.clone()
                    else:
                        x_in_gt = x_in_gt + v_in_gt * cfg.sim.dt * cfg.sim.interval

                    if cfg.sim.gripper_points:
                        x = torch.cat([x, gripper_x[:, step]], dim=1)  # gripper_x: (bsz, num_steps, num_particles, 3)
                        v = torch.cat([v, gripper_v[:, step]], dim=1)
                        x_his = torch.cat([x_his, torch.zeros((gripper_x.shape[0], gripper_x.shape[2], cfg.sim.n_history * 3), device=x_his.device, dtype=x_his.dtype)], dim=1)
                        v_his = torch.cat([v_his, torch.zeros((gripper_x.shape[0], gripper_x.shape[2], cfg.sim.n_history * 3), device=v_his.device, dtype=v_his.dtype)], dim=1)
                        if enabled.shape[1] < num_particles:
                            enabled = torch.cat([enabled, gripper_mask[:, step]], dim=1)
                        statics.update_enabled(enabled.cpu())

                    pred = self.material(x, v, x_his, v_his, enabled)
                    x, v = sim(statics, colliders, step, x, v, self.friction.mu.clone()[None].repeat(batch_size, 1), pred)

                    if cfg.sim.gripper_forcing:
                        assert not cfg.sim.gripper_points
                        gripper_xyz = actions[:, step, :, :3]  # (bsz, num_grippers, 3)
                        gripper_v = actions[:, step, :, 3:6]  # (bsz, num_grippers, 3)
                        x_from_gripper = x_in[:, None] - gripper_xyz[:, :, None]  # (bsz, num_grippers, num_particles, 3)
                        x_gripper_distance = torch.norm(x_from_gripper, dim=-1)  # (bsz, num_grippers, num_particles)
                        x_gripper_distance_mask = x_gripper_distance < cfg.model.gripper_radius
                        x_gripper_distance_mask = x_gripper_distance_mask.unsqueeze(-1).repeat(1, 1, 1, 3)  # (bsz, num_grippers, num_particles, 3)
                        gripper_v_expand = gripper_v[:, :, None].repeat(1, 1, num_particles, 1)  # (bsz, num_grippers, num_particles, 3)

                        gripper_closed = actions[:, step, :, -1] < 0.5  # (bsz, num_grippers)  # 1: open, 0: close
                        x_gripper_distance_mask = torch.logical_and(x_gripper_distance_mask, gripper_closed[:, :, None, None].repeat(1, 1, num_particles, 3))

                        gripper_quat_vel = actions[:, step, :, 10:13]  # (bsz, num_grippers, 3)
                        gripper_angular_vel = torch.linalg.norm(gripper_quat_vel, dim=-1, keepdims=True)  # (bsz, num_grippers, 1)
                        gripper_quat_axis = gripper_quat_vel / (gripper_angular_vel + 1e-10)  # (bsz, num_grippers, 3)

                        grid_from_gripper_axis = x_from_gripper - \
                            (gripper_quat_axis[:, :, None] * x_from_gripper).sum(dim=-1, keepdims=True) * gripper_quat_axis[:, :, None]  # (bsz, num_grippers, num_particles, 3)
                        gripper_v_expand = torch.cross(gripper_quat_vel[:, :, None], grid_from_gripper_axis, dim=-1) + gripper_v_expand

                        for i in range(gripper_xyz.shape[1]):
                            x_gripper_distance_mask_single = x_gripper_distance_mask[:, i]
                            x[x_gripper_distance_mask_single] = x_in[x_gripper_distance_mask_single] + cfg.sim.dt * gripper_v_expand[:, i][x_gripper_distance_mask_single]
                            v[x_gripper_distance_mask_single] = gripper_v_expand[:, i][x_gripper_distance_mask_single]

                    if cfg.sim.n_history > 0:
                        if cfg.sim.gripper_points:
                            x_his_particles = torch.cat([x_his[:, :num_particles_orig].reshape(batch_size, num_particles_orig, -1, 3)[:, :, 1:], x[:, :num_particles_orig, None].detach()], dim=2)
                            v_his_particles = torch.cat([v_his[:, :num_particles_orig].reshape(batch_size, num_particles_orig, -1, 3)[:, :, 1:], v[:, :num_particles_orig, None].detach()], dim=2)
                            x_his = x_his_particles.reshape(batch_size, num_particles_orig, -1)
                            v_his = v_his_particles.reshape(batch_size, num_particles_orig, -1)
                        else:
                            x_his = torch.cat([x_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], x[:, :, None].detach()], dim=2)
                            v_his = torch.cat([v_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], v[:, :, None].detach()], dim=2)
                            x_his = x_his.reshape(batch_size, num_particles, -1)
                            v_his = v_his.reshape(batch_size, num_particles, -1)
                    
                    if cfg.sim.gripper_points:
                        x = x[:, :num_particles_orig]
                        v = v[:, :num_particles_orig]
                        enabled = enabled[:, :num_particles_orig]

                    if self.verbose:
                        print('x', x.min().item(), x.max().item())
                        print('v', v.min().item(), v.max().item())

                    if self.loss_factor_x > 0:
                        loss_x = self.criterion(x[enabled_mask > 0], gt_x[:, step][enabled_mask > 0]) * self.loss_factor_x
                        losses['loss_x'] += loss_x
                        self.losses_log['loss_x'] += loss_x.item()
                
                    if self.loss_factor_v > 0:
                        loss_v = self.criterion(v[enabled_mask > 0], gt_v[:, step][enabled_mask > 0]) * self.loss_factor_v
                        losses['loss_v'] += loss_v
                        self.losses_log['loss_v'] += loss_v.item()

                    with torch.no_grad():
                        if self.loss_factor_x > 0:
                            loss_x_trivial = self.criterion((x_in_gt + v_in_gt * cfg.sim.dt * cfg.sim.interval)[enabled_mask > 0], gt_x[:, step][enabled_mask > 0]) * self.loss_factor_x
                            self.losses_log['loss_x_trivial'] += loss_x_trivial.item()

                        if self.loss_factor_v > 0:
                            loss_v_trivial = self.criterion(v_in_gt[enabled_mask > 0], gt_v[:, step][enabled_mask > 0]) * self.loss_factor_v
                            self.losses_log['loss_v_trivial'] += loss_v_trivial.item()

                        loss_x_sanity = self.criterion(x_in[enabled_mask > 0], (x - v * cfg.sim.dt * cfg.sim.interval)[enabled_mask > 0]) * self.loss_factor_x
                        self.losses_log['loss_x_sanity'] += loss_x_sanity.item()  # if > 0 then clipping issue

                        if step > 0:
                            loss_x_gt_sanity = self.criterion((gt_x[:, step - 1] + gt_v[:, step] * cfg.sim.dt * cfg.sim.interval)[enabled_mask > 0], gt_x[:, step][enabled_mask > 0]) * self.loss_factor_x
                            self.losses_log['loss_x_gt_sanity'] += loss_x_gt_sanity.item()
                        else:
                            loss_x_gt_sanity = self.criterion((x_in + gt_v[:, step] * cfg.sim.dt * cfg.sim.interval)[enabled_mask > 0], gt_x[:, step][enabled_mask > 0]) * self.loss_factor_x
                            self.losses_log['loss_x_gt_sanity'] += loss_x_gt_sanity.item()

                    if save and not cfg.debug:
                        self.logger.add_scalar('main/iteration', iteration, step=self.total_step_count)
                        for loss_k, loss_v in losses.items():
                            self.logger.add_scalar(f'main/{loss_k}', loss_v.item(), step=self.total_step_count)
                    self.total_step_count += 1

                loss = sum(losses.values())
                try:
                    (loss / accum).backward()
                except Exception as e:
                    print(f'loss.backward() failed: {e.with_traceback()}')
                    continue

            if self.material_requires_grad:
                material_grad_norm = clip_grad_norm_(
                    self.material.parameters(),
                    max_norm=cfg.train.material_grad_max_norm,
                    error_if_nonfinite=True)
                self.material_optimizer.step()

            if (iteration + 1) % cfg.train.iteration_log_interval == 0:
                msgs = [
                    cfg.train.name,
                    time.strftime('%H:%M:%S'),
                    'iteration {:{width}d}/{}'.format(iteration + 1, cfg.train.num_iterations, width=len(str(cfg.train.num_iterations))),
                ]

                msgs.extend([
                    'pred.norm {:.4f}'.format(pred.norm().item()),
                ])

                if self.material_requires_grad:
                    material_lr = self.material_optimizer.param_groups[0]['lr']
                    msgs.extend([
                        'e-lr {:.2e}'.format(material_lr),
                        'e-|grad| {:.4f}'.format(material_grad_norm),
                    ])

                for loss_k, loss_v in self.losses_log.items():
                    n_log = cfg.train.iteration_log_interval * int(cfg.train.grad_accum)
                    msgs.append('{} {:.8f}'.format(loss_k, loss_v / n_log))
                    if save and not cfg.debug:
                        self.logger.add_scalar('stat/mean_{}'.format(loss_k), loss_v / n_log, step=self.total_step_count)
                
                msg = ','.join(msgs)
                print('[{}]'.format(msg))
                self.losses_log = defaultdict(int)
            
            if save and not cfg.debug:
                self.logger.add_scalar('stat/pred_norm', pred.norm().item(), step=self.total_step_count)

            if self.material_requires_grad:
                material_lr = self.material_optimizer.param_groups[0]['lr']
                if save and not cfg.debug:
                    self.logger.add_scalar('stat/material_lr', material_lr, step=self.total_step_count)
                    self.logger.add_scalar('stat/material_grad_norm', material_grad_norm, step=self.total_step_count)

            if save and (iteration + 1) % cfg.train.iteration_save_interval == 0:
                self.save_ckpt('latest.pt', iteration + 1)
            # numbered snapshots alongside latest/best/final, matching the GNS runs' layout
            # (user 2026-09-22): <run>/train/ckpt/model-<step>.pt every train.snapshot_every steps
            if save and cfg.train.snapshot_every and (iteration + 1) % cfg.train.snapshot_every == 0:
                self.save_ckpt(f'model-{iteration + 1}.pt', iteration + 1)

            if self.material_requires_grad:
                self.material_lr_scheduler.step()

    # number of (fixed) val batches scored per validate() call -- keeps the
    # 500-iteration validation cheap while covering all 8 val episodes.
    VAL_MAX_BATCHES = 40

    def _build_val_loader(self):
        """Build the validation dataloader(s) once: short-horizon (num_steps_train)
        windows over the held-out val episodes. With converter domain metadata,
        one domain-pure loader per domain (a fixed random subset of its windows)."""
        cfg = self.cfg
        val_dataset = RealTeleopBatchDataset(
            cfg,
            dataset_root=self.log_root / cfg.train.dataset_name / 'state',
            source_data_root=self.log_root / str(cfg.train.source_dataset_name),
            device=self.torch_device,
            num_steps=cfg.sim.num_steps_train,
            train=False,
            episode_range=(cfg.train.eval_start_episode, cfg.train.eval_end_episode),
            dataset_non_overwrite=True,
        )
        if val_dataset.domain_indices is not None:
            self.val_loaders = {}
            for d, idx in sorted(val_dataset.domain_indices.items()):
                gen = torch.Generator().manual_seed(0)  # fixed subset every call
                sub = torch.utils.data.Subset(val_dataset, idx.tolist())
                self.val_loaders[d] = DataLoader(
                    sub, batch_size=cfg.train.batch_size, shuffle=True, generator=gen,
                    num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)
            print('val windows per domain:', {d: len(v) for d, v in val_dataset.domain_indices.items()})
        else:
            gen = torch.Generator().manual_seed(0)  # fixed subset every call
            self.val_loaders = {'all': DataLoader(
                val_dataset, batch_size=cfg.train.batch_size, shuffle=True, generator=gen,
                num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)}

    @torch.no_grad()
    def validate(self):
        """Position MSE (x loss_factor_x) per val loader: {domain or 'all': loss}."""
        if not hasattr(self, 'val_loaders'):
            self._build_val_loader()
        self.material.eval()
        out = {d: self._val_loss(loader) for d, loader in self.val_loaders.items()}
        self.material.train(True)
        return out

    def _val_loss(self, loader):
        """Mean position MSE over up to VAL_MAX_BATCHES batches, same short-horizon
        forward as training. Supports both gripper_points and gripper_forcing modes."""
        cfg = self.cfg
        forcing = bool(cfg.sim.gripper_forcing)
        dt = eval(cfg.sim.dt) if isinstance(cfg.sim.dt, str) else cfg.sim.dt

        total_loss, n_steps = 0.0, 0
        for batch_idx, (init_state, actions, gt_states, gripper_states) in enumerate(loader):
            if batch_idx >= self.VAL_MAX_BATCHES:
                break
            x, v, x_his, v_his, clip_bound, enabled, episode_vec = init_state
            x = x.to(self.torch_device); v = v.to(self.torch_device)
            x_his = x_his.to(self.torch_device); v_his = v_his.to(self.torch_device)
            actions = actions.to(self.torch_device)

            gt_x, gt_v = gt_states
            gt_x = gt_x.to(self.torch_device)

            batch_size, num_steps_total, num_particles = gt_x.shape[:3]
            num_particles_orig = num_particles

            if forcing:
                num_grippers = cfg.sim.num_grippers
            else:
                num_grippers = 0
                gripper_x, gripper_v = gripper_states
                gripper_x = gripper_x.to(self.torch_device)
                gripper_v = gripper_v.to(self.torch_device)
                gripper_x, gripper_v, gripper_mask = prepare_gripper_points(cfg, gripper_x, gripper_v)
                num_particles = num_particles + gripper_x.shape[2]

            sim = CacheDiffSimWithFrictionBatch(cfg, num_steps_total, batch_size, self.wp_device, requires_grad=False)
            statics = StaticsBatch()
            statics.init(shape=(batch_size, num_particles), device=self.wp_device)
            statics.update_clip_bound(clip_bound)
            statics.update_enabled(enabled)
            colliders = CollidersBatch()
            colliders.init(shape=(batch_size, num_grippers), device=self.wp_device)
            if num_grippers > 0:
                colliders.initialize_grippers(actions[:, 0])

            enabled = enabled.to(self.torch_device)
            enabled_mask = enabled.unsqueeze(-1).repeat(1, 1, 3)

            for step in range(num_steps_total):
                if num_grippers > 0:
                    colliders.update_grippers(actions[:, step])
                x_in = x.clone()

                if forcing:
                    pred = self.material(x, v, x_his, v_his, enabled)
                    x, v = sim(statics, colliders, step, x, v,
                               self.friction.mu[None].repeat(batch_size, 1), pred)
                    # kinematic gripper forcing -- mirrors the training loop
                    gripper_xyz = actions[:, step, :, :3]
                    gripper_v = actions[:, step, :, 3:6]
                    x_from_gripper = x_in[:, None] - gripper_xyz[:, :, None]
                    g_mask = (torch.norm(x_from_gripper, dim=-1) < cfg.model.gripper_radius)
                    g_mask = g_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
                    gripper_closed = actions[:, step, :, -1] < 0.5
                    g_mask = torch.logical_and(g_mask, gripper_closed[:, :, None, None].repeat(1, 1, num_particles, 3))
                    gripper_v_expand = gripper_v[:, :, None].repeat(1, 1, num_particles, 1)
                    gripper_quat_vel = actions[:, step, :, 10:13]
                    gripper_quat_axis = gripper_quat_vel / (torch.linalg.norm(gripper_quat_vel, dim=-1, keepdims=True) + 1e-10)
                    grid_from_gripper_axis = x_from_gripper - \
                        (gripper_quat_axis[:, :, None] * x_from_gripper).sum(dim=-1, keepdims=True) * gripper_quat_axis[:, :, None]
                    gripper_v_expand = torch.cross(gripper_quat_vel[:, :, None], grid_from_gripper_axis, dim=-1) + gripper_v_expand
                    for i in range(gripper_xyz.shape[1]):
                        mi = g_mask[:, i]
                        x[mi] = x_in[mi] + dt * gripper_v_expand[:, i][mi]
                        v[mi] = gripper_v_expand[:, i][mi]
                else:
                    x = torch.cat([x, gripper_x[:, step]], dim=1)
                    v = torch.cat([v, gripper_v[:, step]], dim=1)
                    zhist = torch.zeros((batch_size, gripper_x.shape[2], cfg.sim.n_history * 3),
                                        device=x_his.device, dtype=x_his.dtype)
                    x_his = torch.cat([x_his, zhist], dim=1)
                    v_his = torch.cat([v_his, zhist.clone()], dim=1)
                    if enabled.shape[1] < num_particles:
                        enabled = torch.cat([enabled, gripper_mask[:, step]], dim=1)
                    statics.update_enabled(enabled.cpu())

                    pred = self.material(x, v, x_his, v_his, enabled)
                    x, v = sim(statics, colliders, step, x, v,
                               self.friction.mu[None].repeat(batch_size, 1), pred)

                if cfg.sim.n_history > 0:
                    x_his = torch.cat([x_his[:, :num_particles_orig].reshape(batch_size, num_particles_orig, -1, 3)[:, :, 1:],
                                       x[:, :num_particles_orig, None]], dim=2).reshape(batch_size, num_particles_orig, -1)
                    v_his = torch.cat([v_his[:, :num_particles_orig].reshape(batch_size, num_particles_orig, -1, 3)[:, :, 1:],
                                       v[:, :num_particles_orig, None]], dim=2).reshape(batch_size, num_particles_orig, -1)
                x = x[:, :num_particles_orig]
                v = v[:, :num_particles_orig]
                enabled = enabled[:, :num_particles_orig]

                # same scaling as the training loss_x so the curves are comparable
                loss_x = nn.functional.mse_loss(x[enabled_mask > 0], gt_x[:, step][enabled_mask > 0]) * cfg.train.loss_factor_x
                total_loss += loss_x.item()
                n_steps += 1

        return total_loss / max(n_steps, 1)

    def eval_episode(self, iteration: int, episode: int, save: bool = True):
        cfg = self.cfg

        log_root: Path = root / 'log'
        eval_name = f'{cfg.train.name}/eval/{cfg.train.dataset_name.split("/")[-1]}/{iteration:06d}'
        exp_root: Path = log_root / eval_name
        if save:
            state_root: Path = exp_root / 'state'
            mkdir(state_root, overwrite=cfg.overwrite, resume=cfg.resume)
            episode_state_root = state_root / f'episode_{episode:04d}'
            mkdir(episode_state_root, overwrite=cfg.overwrite, resume=cfg.resume)
            OmegaConf.save(cfg, exp_root / 'hydra.yaml', resolve=True)

        if cfg.train.dataset_name is None:
            cfg.train.dataset_name = Path(cfg.train.name).parent / 'dataset'
        assert cfg.train.source_dataset_name is not None

        source_dataset_root = self.log_root / str(cfg.train.source_dataset_name)
        assert os.path.exists(source_dataset_root)

        eval_dataset = RealTeleopBatchDataset(
            cfg, 
            dataset_root=self.log_root / cfg.train.dataset_name / 'state',
            source_data_root=source_dataset_root,
            device=self.torch_device,
            num_steps=self.cfg.sim.num_steps,
            eval_episode_name=f'episode_{episode:04d}',
        )
        eval_dataloader = dataloader_wrapper(
            DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True),
            'dataset'
        )
        init_state, actions, gt_states, gripper_states, downsample_indices = next(eval_dataloader)

        x, v, x_his, v_his, clip_bound, enabled, episode_vec = init_state
        x = x.to(self.torch_device)
        v = v.to(self.torch_device)
        x_his = x_his.to(self.torch_device)
        v_his = v_his.to(self.torch_device)
    
        actions = actions.to(self.torch_device)

        if cfg.sim.gripper_points:
            gripper_x, gripper_v = gripper_states
            gripper_x = gripper_x.to(self.torch_device)
            gripper_v = gripper_v.to(self.torch_device)
            gripper_x, gripper_v, gripper_mask = prepare_gripper_points(cfg, gripper_x, gripper_v)

        gt_x, gt_v = gt_states
        gt_x = gt_x.to(self.torch_device)
        gt_v = gt_v.to(self.torch_device)
    
        # gt_states: (bsz, num_steps_total)
        batch_size = gt_x.shape[0]
        num_steps_total = gt_x.shape[1]
        num_particles = gt_x.shape[2]
        assert batch_size == 1

        if cfg.sim.gripper_points:
            num_gripper_particles = gripper_x.shape[2]
            num_particles_orig = num_particles
            num_particles = num_particles + num_gripper_particles

        sim = CacheDiffSimWithFrictionBatch(cfg, num_steps_total, batch_size, self.wp_device, requires_grad=True)

        statics = StaticsBatch()
        statics.init(shape=(batch_size, num_particles), device=self.wp_device)
        statics.update_clip_bound(clip_bound)
        statics.update_enabled(enabled)
        colliders = CollidersBatch()
        
        self.material.eval()
        self.friction.eval()

        if cfg.sim.gripper_points:
            assert not cfg.sim.gripper_forcing
            num_grippers = 0
        else:
            num_grippers = cfg.sim.num_grippers

        colliders.init(shape=(batch_size, num_grippers), device=self.wp_device)
        if num_grippers > 0:
            assert len(actions.shape) > 2
            colliders.initialize_grippers(actions[:, 0])

        enabled = enabled.to(self.torch_device)
        enabled_mask = enabled.unsqueeze(-1).repeat(1, 1, 3)  # (bsz, num_particles, 3)

        colliders_save = colliders.export()
        colliders_save = {key: torch.from_numpy(colliders_save[key])[0].to(x.device).to(x.dtype) for key in colliders_save}
        ckpt = dict(x=x[0], v=v[0], **colliders_save)

        if save:
            torch.save(ckpt, episode_state_root / f'{0:04d}.pt')

        losses = {}
        with torch.no_grad():
            for step in trange(num_steps_total):
                if num_grippers > 0:
                    colliders.update_grippers(actions[:, step])
                if cfg.sim.gripper_forcing:
                    x_in = x.clone()
                else:
                    x_in = None

                if cfg.sim.gripper_points:
                    x = torch.cat([x, gripper_x[:, step]], dim=1)  # gripper_points: (bsz, num_steps, num_particles, 3)
                    v = torch.cat([v, gripper_v[:, step]], dim=1)
                    x_his = torch.cat([x_his, torch.zeros((gripper_x.shape[0], gripper_x.shape[2], cfg.sim.n_history * 3), device=x_his.device, dtype=x_his.dtype)], dim=1)
                    v_his = torch.cat([v_his, torch.zeros((gripper_x.shape[0], gripper_x.shape[2], cfg.sim.n_history * 3), device=v_his.device, dtype=v_his.dtype)], dim=1)
                    if enabled.shape[1] < num_particles:
                        enabled = torch.cat([enabled, gripper_mask[:, step]], dim=1)
                    statics.update_enabled(enabled.cpu())

                pred = self.material(x, v, x_his, v_his, enabled)

                if pred.isnan().any():
                    print('pred isnan', pred.min().item(), pred.max().item())
                    break
                if pred.isinf().any():
                    print('pred isinf', pred.min().item(), pred.max().item())
                    break

                x, v = sim(statics, colliders, step, x, v, self.friction.mu[None].repeat(batch_size, 1), pred)

                if cfg.sim.gripper_forcing:
                    assert not cfg.sim.gripper_points
                    gripper_xyz = actions[:, step, :, :3]  # (bsz, num_grippers, 3)
                    gripper_v = actions[:, step, :, 3:6]  # (bsz, num_grippers, 3)
                    x_from_gripper = x_in[:, None] - gripper_xyz[:, :, None]  # (bsz, num_grippers, num_particles, 3)
                    x_gripper_distance = torch.norm(x_from_gripper, dim=-1)  # (bsz, num_grippers, num_particles)
                    x_gripper_distance_mask = x_gripper_distance < cfg.model.gripper_radius
                    x_gripper_distance_mask = x_gripper_distance_mask.unsqueeze(-1).repeat(1, 1, 1, 3)  # (bsz, num_grippers, num_particles, 3)
                    gripper_v_expand = gripper_v[:, :, None].repeat(1, 1, num_particles, 1)  # (bsz, num_grippers, num_particles, 3)

                    gripper_closed = actions[:, step, :, -1] < 0.5  # (bsz, num_grippers)  # 1: open, 0: close
                    x_gripper_distance_mask = torch.logical_and(x_gripper_distance_mask, gripper_closed[:, :, None, None].repeat(1, 1, num_particles, 3))

                    gripper_quat_vel = actions[:, step, :, 10:13]  # (bsz, num_grippers, 3)
                    gripper_angular_vel = torch.linalg.norm(gripper_quat_vel, dim=-1, keepdims=True)  # (bsz, num_grippers, 1)
                    gripper_quat_axis = gripper_quat_vel / (gripper_angular_vel + 1e-10)  # (bsz, num_grippers, 3)

                    grid_from_gripper_axis = x_from_gripper - \
                        (gripper_quat_axis[:, :, None] * x_from_gripper).sum(dim=-1, keepdims=True) * gripper_quat_axis[:, :, None]  # (bsz, num_grippers, num_particles, 3)
                    gripper_v_expand = torch.cross(gripper_quat_vel[:, :, None], grid_from_gripper_axis, dim=-1) + gripper_v_expand

                    for i in range(gripper_xyz.shape[1]):
                        x_gripper_distance_mask_single = x_gripper_distance_mask[:, i]
                        x[x_gripper_distance_mask_single] = x_in[x_gripper_distance_mask_single] + cfg.sim.dt * gripper_v_expand[:, i][x_gripper_distance_mask_single]
                        v[x_gripper_distance_mask_single] = gripper_v_expand[:, i][x_gripper_distance_mask_single]
                
                if cfg.sim.n_history > 0:
                    if cfg.sim.gripper_points:
                        x_his_particles = torch.cat([x_his[:, :num_particles_orig].reshape(batch_size, num_particles_orig, -1, 3)[:, :, 1:], x[:, :num_particles_orig, None].detach()], dim=2)
                        v_his_particles = torch.cat([v_his[:, :num_particles_orig].reshape(batch_size, num_particles_orig, -1, 3)[:, :, 1:], v[:, :num_particles_orig, None].detach()], dim=2)
                        x_his = x_his_particles.reshape(batch_size, num_particles_orig, -1)
                        v_his = v_his_particles.reshape(batch_size, num_particles_orig, -1)
                    else:
                        x_his = torch.cat([x_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], x[:, :, None].detach()], dim=2)
                        v_his = torch.cat([v_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], v[:, :, None].detach()], dim=2)
                        x_his = x_his.reshape(batch_size, num_particles, -1)
                        v_his = v_his.reshape(batch_size, num_particles, -1)

                if cfg.sim.gripper_points:
                    extra_save = {
                        'gripper_x': gripper_x[0, step],
                        'gripper_v': gripper_v[0, step],
                        'grippers': actions[0, step],
                    }
                    x = x[:, :num_particles_orig]
                    v = v[:, :num_particles_orig]
                    enabled = enabled[:, :num_particles_orig]
                else:
                    extra_save = {}

                colliders_save = colliders.export()
                colliders_save = {key: torch.from_numpy(colliders_save[key])[0].to(x.device).to(x.dtype) for key in colliders_save}
                
                loss_x = nn.functional.mse_loss(x[enabled_mask > 0], gt_x[:, step][enabled_mask > 0])
                loss_v = nn.functional.mse_loss(v[enabled_mask > 0], gt_v[:, step][enabled_mask > 0])
                losses[step] = dict(loss_x=loss_x.item(), loss_v=loss_v.item())

                ckpt = dict(x=x[0], v=v[0], **colliders_save, **extra_save)

                if save and step % cfg.sim.skip_frame == 0:
                    torch.save(ckpt, episode_state_root / f'{int(step / cfg.sim.skip_frame):04d}.pt')

        metrics = None
        if save:
            for loss_k in losses[0].keys():
                plt.figure(figsize=(10, 5))
                loss_list = [losses[step][loss_k] for step in losses]
                plt.plot(loss_list)
                plt.title(loss_k)
                plt.grid()
                plt.savefig(state_root / f'episode_{episode:04d}_{loss_k}.png', dpi=300)

            # particle visualization
            if self.use_pv:
                do_train_pv(
                    cfg,
                    log_root,
                    iteration,
                    [f'episode_{episode:04d}'],
                    eval_dirname=f'eval',
                    dataset_name=cfg.train.dataset_name.split("/")[-1],
                    eval_postfix='',
                )

            # gaussian splatting visualization
            if self.use_gs:
                from .gs import do_gs
                do_gs(
                    cfg,
                    log_root,
                    iteration,
                    [f'episode_{episode:04d}'],
                    eval_dirname=f'eval',
                    dataset_name=cfg.train.dataset_name.split("/")[-1],
                    eval_postfix='',
                    camera_id=1,
                    with_mask=True,
                    with_bg=True,
                )

            # particle visualization of ground truth
            if self.use_pv:
                _ = do_dataset_pv(
                    cfg,
                    log_root / str(cfg.train.dataset_name),
                    [f'episode_{episode:04d}'],
                    save_dir=log_root / f'{cfg.train.name}/eval/{cfg.train.dataset_name.split("/")[-1]}/{iteration:06d}/pv',
                    downsample_indices=downsample_indices,
                )
            
            metrics = do_metric(
                cfg,
                log_root,
                iteration,
                [f'episode_{episode:04d}'],
                downsample_indices,
                eval_dirname=f'eval',
                dataset_name=cfg.train.dataset_name.split("/")[-1],
                eval_postfix='',
                camera_id=1,
                use_gs=self.use_gs,
            )

        return metrics


    def eval(self, eval_iteration: int, save: bool = True):
        cfg = self.cfg

        metrics_list = []
        start_episode = cfg.train.eval_start_episode
        end_episode = cfg.train.eval_end_episode if save else cfg.train.eval_start_episode + 2
        for episode in range(start_episode, end_episode):
            metrics = self.eval_episode(eval_iteration, episode, save=save)
            metrics_list.append(metrics)

        if not save:
            return

        metrics_list = np.array(metrics_list)[:, 0]  # (n_episodes, n_frames, 10 or 3)
        if self.use_gs:
            metric_names = ['mde', 'chamfer', 'emd', 'jscore', 'fscore', 'jfscore', 'perception', 'psnr', 'ssim']
        else:
            metric_names = ['mde', 'chamfer', 'emd']
    
        median_metric = np.median(metrics_list, axis=0)
        step_75_metric = np.percentile(metrics_list, 75, axis=0)
        step_25_metric = np.percentile(metrics_list, 25, axis=0)
        mean_metric = np.mean(metrics_list, axis=0)
        std_metric = np.std(metrics_list, axis=0)

        for i, metric_name in enumerate(metric_names):
            # plot error
            x = np.arange(1, len(median_metric) + 1)
            plt.figure(figsize=(8, 5))
            plt.plot(x, median_metric[:, i])
            plt.xlabel(f"prediction steps, dt={cfg.sim.dt}")
            plt.ylabel(metric_name)
            plt.grid()

            ax = plt.gca()
            x = np.arange(1, len(median_metric) + 1)
            ax.fill_between(x, step_25_metric[:, i], step_75_metric[:, i], alpha=0.2)

            save_dir = root / 'log' / cfg.train.name / 'eval' / cfg.train.dataset_name.split("/")[-1] / f'{eval_iteration:06d}' / 'metric'
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{i:02d}-{metric_name}.png'), dpi=300)
            plt.close()
        
        # send to wandb
        if not cfg.debug:
            for i, metric_name in enumerate(metric_names):
                self.logger.add_scalar(f'metric/{metric_name}-mean', mean_metric[:, i].mean(), step=self.total_step_count)
                self.logger.add_scalar(f'metric/{metric_name}-std', std_metric[:, i].mean(), step=self.total_step_count)
                img = np.array(Image.open(os.path.join(save_dir, f'{i:02d}-{metric_name}.png')).convert('RGB'))
                self.logger.add_image(f'metric_curve/{metric_name}', img, step=self.total_step_count)

    def save_ckpt(self, name, iteration):
        tmp = self.ckpt_root / (name + '.tmp')
        torch.save({'material': self.material.state_dict(), 'iteration': iteration,
                    'best_val': self.best_val}, tmp)
        os.replace(tmp, self.ckpt_root / name)

    def test_cuda_mem(self):
        self.init_train()
        self.train(0, 10, save=False)
        self.eval(10, save=False)

@hydra.main(version_base='1.2', config_path=str(root / 'cfg'), config_name='default')
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.load_train_dataset()
    trainer.init_train()
    for iteration in range(cfg.train.resume_iteration, cfg.train.num_iterations, cfg.train.iteration_eval_interval):
        start_iteration = iteration
        end_iteration = min(iteration + cfg.train.iteration_eval_interval, cfg.train.num_iterations)
        trainer.train(start_iteration, end_iteration)

        # lightweight validation on held-out episodes (per domain when available)
        val = trainer.validate()
        val_loss = float(np.mean(list(val.values())))
        if val_loss < trainer.best_val:
            trainer.best_val = val_loss
            trainer.save_ckpt('best.pt', end_iteration)
        trainer.save_ckpt('latest.pt', end_iteration)
        per = '  '.join(f'{d} {v:.6f}' for d, v in val.items())
        print(f'[val] iteration {end_iteration}: val_loss_x mean {val_loss:.6f}  best {trainer.best_val:.6f}  |  {per}')
        if not cfg.debug:
            trainer.logger.add_scalar('val/loss_x', val_loss, step=trainer.total_step_count)
            trainer.logger.add_scalar('val/loss_x_best', trainer.best_val, step=trainer.total_step_count)
            for d, v in val.items():
                trainer.logger.add_scalar(f'val/loss_x_{d}', v, step=trainer.total_step_count)
    trainer.save_ckpt('final.pt', cfg.train.num_iterations)


if __name__ == '__main__':
    main()
