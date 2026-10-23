#!/usr/bin/env python3
"""Where does the dex loader's memory go? Load N episodes and account for it.

Reports, for each N: parent RSS, the sum of worker RSS, /dev/shm in use, and the actual bytes of
the arrays the parent keeps per episode -- so the growth per episode is measurable rather than
guessed. Job 10538530 was OOM-killed at 64 GB while loading all 16,999.

    python experiments/scripts/diag_dex_memory.py --counts 500 2000 4000 --workers 8
"""
import argparse
import functools
import gc
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = '/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/dex_v2_filtered2/train'


def rss_mb(pid=None):
    pid = pid or os.getpid()
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        return 0.0
    return 0.0


def children_rss_mb():
    out = subprocess.run(['ps', '-o', 'rss=', '--ppid', str(os.getpid())], capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split()) / 1024


def shm_mb():
    out = subprocess.run(['df', '-m', '/dev/shm'], capture_output=True, text=True).stdout.splitlines()
    return float(out[-1].split()[2]) if len(out) > 1 else 0.0


def episode_bytes(ep):
    total = 0
    for k, v in ep.items():
        if isinstance(v, np.ndarray):
            total += v.nbytes
        elif torch.is_tensor(v):
            total += v.element_size() * v.numel()
        elif isinstance(v, dict):
            total += sum(x.nbytes for x in v.values() if isinstance(x, np.ndarray))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--counts', type=int, nargs='+', default=[500, 2000, 4000])
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    from pgnd.data.pwm_dex_source import load_episodes, prepare_for_pgnd
    with __import__('hydra').initialize_config_dir(
            config_dir=str(Path(__file__).resolve().parents[1] / 'cfg'), version_base='1.2'):
        cfg = __import__('hydra').compose(config_name='default', overrides=[
            'sim.dt=0.05', 'sim.gripper_points=True', 'sim.gripper_forcing=False', 'model.eef_t=[0.0,0.0,0.0]'])
    prep = functools.partial(prepare_for_pgnd, cfg=cfg, dt_frame=0.05, dx=0.02)
    base, base_shm = rss_mb(), shm_mb()
    print(f'baseline: parent {base:.0f} MB, /dev/shm {base_shm:.0f} MB', flush=True)
    for n in a.counts:
        eps = load_episodes(ROOT, workers=a.workers, limit=n, prepare=prep)
        kept = sum(episode_bytes(e) for e in eps) / 1024 ** 2
        print(f'{n:6d} episodes: parent RSS {rss_mb():8.0f} MB (+{rss_mb() - base:7.0f})  '
              f'workers {children_rss_mb():7.0f} MB  /dev/shm {shm_mb() - base_shm:6.0f} MB  '
              f'arrays kept {kept:7.0f} MB = {kept / n * 1024:.0f} kB/episode', flush=True)
        del eps
        gc.collect()
        print(f'        after release: parent RSS {rss_mb():.0f} MB, /dev/shm {shm_mb() - base_shm:.0f} MB', flush=True)


if __name__ == '__main__':
    main()
