# PGND for Table 1 Pushing (rl_push) and Pick and Place (full yikai): launching on babel

Set up on PSC 2026-10-27/28 (branch `gpwm-baselines`); both runs are launched on babel. What the columns are and why
every model is a single-domain specialist: gpwm `docs/TABLE1_PUSHING_PICKPLACE.md` (branch `mb/psc-port`).

| run | data | box | batch | job file | status |
|---|---|---|---|---|---|
| `pgnd_rl_push_std` | rl_push, 50,815 train + 200 test (gpwm's counts) | 55^3 of 2 cm, whole episode (as drag: the widest object extent is 1.00 m) | 2 GPUs x 16 = 32/step, 200k | `experiments/scripts/train_rl_push_std.sbatch` | conversion checked; one PSC start died on a cache race (fixed, below); not trained |
| `pgnd_yikai_box` | full yikai, 11,015 train + 200 test | FIXED 70 x 70 x 50 of 2 cm (x, y, up), shifted per window onto the gripper | 2 GPUs x 8 (one episode) = 16/step, 200k | `experiments/scripts/train_yikai_box.sbatch` | CPU checks only; needs a GPU smoke |

Every setting against the released recipe and gpwm's specialist is in the header of `train_rl_push_std.sh` /
`train_yikai_box.sh`.

## Steps on babel

```bash
cd /data/user_data/mbronars/packages/pgnd && git fetch && git checkout gpwm-baselines && git pull
R=/data/group_data/katefgroup/runs_gns_mbronars
mkdir -p $R/pgnd_data $R/pgnd_rl_push_std $R/pgnd_yikai_box          # slurm cannot create the --output folders
# 1. data (CPU): confirm GPWM_PY / GPWM_REPO in the file first
sbatch --export=ALL,DOMAIN=rl_push experiments/scripts/convert_pwm_domain.sbatch
sbatch --export=ALL,DOMAIN=yikai   experiments/scripts/convert_pwm_domain.sbatch
# 2. yikai GPU smoke (memory and speed are unmeasured), then the runs
mkdir -p $R/pgnd_yikai_box_smoke
sbatch --export=ALL,NAME=yikai_box_smoke,ITERS=200,EVAL_EVERY=100 --time=01:00:00 \
       --output=$R/pgnd_yikai_box_smoke/slurm_%j.out --error=$R/pgnd_yikai_box_smoke/slurm_%j.err \
       experiments/scripts/train_yikai_box.sbatch
sbatch experiments/scripts/train_rl_push_std.sbatch
sbatch experiments/scripts/train_yikai_box.sbatch
```

Check within minutes of a start: `nccl ok: rank r of 2 passed the first barrier`, then `iteration 20/200000` with
`loss_x_sanity 0`; the log's `peak` field is GPU memory. Resubmitting the same command resumes from
`train/ckpt/latest.pt`. Instead of converting rl_push again, the PSC copy can be pulled (30 GB):
`/ocean/projects/cis260367p/mbronars/projects/GPWM/trainings/pgnd/datasets/rl_push/sub_episodes_v` via
`data.bridges2.psc.edu` (PSC transfers are started from babel).

## The yikai box (user decisions 2026-10-27/28)

Whole cluttered scenes span up to 1.8 m, so no fixed box holds an episode, and a box sized to always hold every
point would be ~13x the released grid (ruled out). Instead, per window (3 history + current + 16 predicted frames):

- the box is a constant, `sim.num_grids=[70,50,70,0.02]` (1.4 m wide, 1.0 m tall; 2.0x the released 50^3 cells);
- it is shifted so the gripper centroid at the window's first frame is centred horizontally, and the episode's
  table plane (lowest object point) sits on the box floor (z is not centred);
- an object leaving the box during the window is disabled for that window: no loss, the simulator does not move
  it, and `model.material.mask_disabled=True` keeps it out of the network's centring, active-grid selection and
  20 cm aggregation (CPU check: output identical to the enabled points alone, 3e-8; with the flag off, i.e.
  released behaviour, it would differ by 0.19);
- windows whose gripper leaves the box, or that would disable an object being handled (> 5 mm motion within 25 cm
  of the gripper), are not drawn.

Full-release scan (`experiments/scripts/scan_yikai_box.py`, 2.6M train / 95k test windows) at 70 x 70 x 50:
21% of objects disabled (distractors far from the gripper), windows that would drop a handled object 0.0011%
(train) / 0.0126% (test: 12 windows of 2 of the 200 test episodes), gripper leaves the box in ~1.1% of windows.

## Code added for these runs

- `experiments/scripts/convert_pwm_domain.py`: one domain through gpwm's loader; rl_push as PGND episode dirs,
  yikai as compact rigid bodies (`episode_NNNNNN.cpk`; every body rigid, worst rebuild 0.00012 mm).
- `pgnd/data/pwm_yikai_source.py`: the per-window box, one episode per batch, fixed validation batches.
- `experiments/train/train_eval.py`: `train.source_format=pwm_yikai_box`.
- `pgnd/material/pgnd.py`: `model.material.mask_disabled` (default false = released PGND, unchanged for every other run).
- `pgnd/data/dataset.py`: the per-episode cache (`info.yaml`, `frames.pt`) is written to a temporary name and renamed.
  With 2 GPUs both ranks preprocess the same episode; a rank read a half-written file
  (`failed finding central directory`, PSC 2026-10-28). Affects any 2-GPU run on converted (`pgnd`-format) episodes.

## Not done yet

- Table numbers: PGND has no exporter writing gpwm's prediction layout for `scripts/unified/score_predictions.py`
  (for any domain). For yikai, objects disabled in a rollout window must be held at their positions (zero motion)
  in the dump, since the scorer requires every row; say so in the paper.
- The yikai GPU smoke above (memory, speed).
