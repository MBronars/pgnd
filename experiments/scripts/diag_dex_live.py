#!/usr/bin/env python3
"""Sample memory WHILE the dex loader runs: cgroup total, parent, and each worker.

Job 10538530 was OOM-killed inside load_episodes at 64 GB, while the arrays it keeps come to
3.3 GB (diag_dex_memory.py). This watches the workers during the load instead of after it.
"""
import argparse, functools, os, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = '/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/dex_v2_filtered2/train'


def cgroup_mb():
    for p in ('/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory/memory.usage_in_bytes'):
        try:
            return int(open(p).read()) / 1024 ** 2
        except OSError:
            continue
    return -1.0


def procs():
    out = subprocess.run(['ps', '-eo', 'pid,ppid,rss,args'], capture_output=True, text=True).stdout.splitlines()
    me = os.getpid()
    rows = []
    for line in out[1:]:
        pid, ppid, rss, args = line.split(None, 3)
        if int(pid) == me or int(ppid) == me:
            rows.append((int(pid) == me, int(rss) / 1024))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    from pgnd.data.pwm_dex_source import load_episodes, prepare_for_pgnd
    import hydra
    with hydra.initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'cfg'), version_base='1.2'):
        cfg = hydra.compose(config_name='default', overrides=['sim.dt=0.05', 'sim.gripper_points=True',
                                                              'sim.gripper_forcing=False', 'model.eef_t=[0.0,0.0,0.0]'])
    prep = functools.partial(prepare_for_pgnd, cfg=cfg, dt_frame=0.05, dx=0.02)
    done = threading.Event()
    peak = {'cgroup': 0.0, 'parent': 0.0, 'worker_sum': 0.0, 'worker_max': 0.0}

    def watch():
        t0 = time.time()
        while not done.wait(5):
            rows = procs()
            parent = sum(r for me, r in rows if me)
            workers = [r for me, r in rows if not me]
            cg = cgroup_mb()
            peak['cgroup'] = max(peak['cgroup'], cg)
            peak['parent'] = max(peak['parent'], parent)
            peak['worker_sum'] = max(peak['worker_sum'], sum(workers))
            peak['worker_max'] = max(peak['worker_max'], max(workers, default=0))
            print(f'{time.time() - t0:6.0f}s  cgroup {cg:8.0f} MB  parent {parent:7.0f}  '
                  f'{len(workers)} workers sum {sum(workers):8.0f} max {max(workers, default=0):7.0f} MB', flush=True)

    threading.Thread(target=watch, daemon=True).start()
    t0 = time.time()
    eps = load_episodes(ROOT, workers=a.workers, limit=a.limit, prepare=prep)
    done.set()
    time.sleep(1)
    print(f'loaded {len(eps)} episodes in {(time.time() - t0) / 60:.1f} min', flush=True)
    print('PEAK: ' + '  '.join(f'{k} {v:.0f} MB' for k, v in peak.items()), flush=True)


if __name__ == '__main__':
    main()
