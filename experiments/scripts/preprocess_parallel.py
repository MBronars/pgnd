#!/usr/bin/env python3
"""Run PGND's own preprocess() over every source episode in parallel.

RealTeleopBatchDataset preprocesses uncached episodes serially inside the training job
(~111k episodes for the three-domain mix). This fills the same cache ahead of time --
<log>/<train.dataset_name>/state/episode_*/{frames.pt, info.yaml} -- with the exact
function, cfg and dt_frame the dataset would use, so training only loads.

Takes the same hydra overrides as train_eval.py (only sim.* / model.* / train.dataset_name /
train.source_dataset_name / train.dataset_*skip_frame matter):
    python experiments/scripts/preprocess_parallel.py --workers 16 -- \\
        train.name=three/train train.source_dataset_name=data/three/sub_episodes_v \\
        train.dataset_name=three/dataset sim.dt=0.05 sim.preprocess_scale=1.0 ...
"""
import argparse
import sys
from multiprocessing import get_context
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.append(str(EXP.parent))
sys.path.append(str(EXP))

from hydra import compose, initialize_config_dir  # noqa: E402

_CFG = None


def _one(args):
    src, dst, dt_frame, dx = args
    from pgnd.data.dataset import preprocess
    dst = Path(dst)
    if (dst / 'frames.pt').exists():
        return 'cached'
    dst.mkdir(parents=True, exist_ok=True)
    try:
        preprocess(_CFG, dst, Path(src), dt_frame, dx)
    except Exception as e:  # the dataset drops an episode whose preprocess fails; report it
        return f'FAILED {Path(src).name}: {type(e).__name__}: {e}'
    return 'ok'


def main():
    global _CFG
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('overrides', nargs='*')
    a = ap.parse_args()
    with initialize_config_dir(config_dir=str(EXP / 'cfg'), version_base='1.2'):
        _CFG = compose(config_name='default', overrides=a.overrides)
    cfg = _CFG
    log_root = EXP / 'log'
    src_root = (log_root / str(cfg.train.source_dataset_name)).resolve()
    dst_root = log_root / cfg.train.dataset_name / 'state'
    dt = eval(cfg.sim.dt) if isinstance(cfg.sim.dt, str) else cfg.sim.dt
    # identical to RealTeleopBatchDataset: dt * interval / skip_frame / load_skip_frame
    dt_frame = dt * cfg.sim.interval / cfg.train.dataset_skip_frame / cfg.train.dataset_load_skip_frame
    dx = cfg.sim.num_grids[-1]
    eps = sorted(src_root.glob('episode_*'))
    print(f'{len(eps)} episodes: {src_root} -> {dst_root}  dt_frame={dt_frame} dx={dx}', flush=True)
    jobs = [(str(e), str(dst_root / e.name), dt_frame, dx) for e in eps]
    counts, failed = {}, []
    with get_context('fork').Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap(_one, jobs, chunksize=16)):
            key = r.split(' ')[0]
            counts[key] = counts.get(key, 0) + 1
            if key == 'FAILED':
                failed.append(r)
            if k % 5000 == 0:
                print(f'  {k + 1}/{len(jobs)} {counts}', flush=True)
    print(counts)
    for f in failed:
        print(f)


if __name__ == '__main__':
    main()
