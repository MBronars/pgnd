#!/bin/bash
#SBATCH --job-name=pgnd_convert
#SBATCH --output=/data/group_data/katefgroup/runs_gns_mbronars/pgnd_data/convert_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=general
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# babel: one pwm-data-v1 domain -> PGND training episodes (experiments/scripts/convert_pwm_domain.py), through gpwm's
# own loader. CPU only. Writes $DATA/<DOMAIN>/{sub_episodes_v (rl_push) | episodes (yikai, compact)}.
#   sbatch --export=ALL,DOMAIN=rl_push experiments/scripts/convert_pwm_domain.sbatch
#   sbatch --export=ALL,DOMAIN=yikai   experiments/scripts/convert_pwm_domain.sbatch
# CONFIRM GPWM_PY (a python with gpwm's dependencies: the pwm211 env convert_three.py used) and GPWM_REPO (a gpwm
# checkout whose data/rl_push.py and data/indexed_episodes.py read pwm-data-v1) before the first submit.
# On PSC (2026-10-27/28): rl_push 50,815 train + 200 test episodes (gpwm's counts), 30 GB, 10 min on 32 cores;
# yikai 40-episode smoke: every body rigid (worst rebuild 0.00012 mm), 0.43 MB per episode (~5 GB for the release).
set -euo pipefail
DOMAIN=${DOMAIN:?set DOMAIN=rl_push or yikai}
GPWM_REPO=${GPWM_REPO:-/data/user_data/mbronars/packages/pwm_final_paper}
GPWM_PY=${GPWM_PY:-/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python}
RELEASE=${RELEASE:-/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1}
DATA=${DATA:-/data/group_data/katefgroup/runs_gns_mbronars/pgnd_data}
OUT=$DATA/$DOMAIN/$([ "$DOMAIN" = yikai ] && echo episodes || echo sub_episodes_v)
cd /data/user_data/mbronars/packages/pgnd
echo "job $SLURM_JOB_ID on $(hostname) $(date): $DOMAIN -> $OUT"
"$GPWM_PY" -W ignore experiments/scripts/convert_pwm_domain.py --domain "$DOMAIN" --gpwm "$GPWM_REPO" --release "$RELEASE" \
  --out "$OUT" --workers "$SLURM_CPUS_PER_TASK"
du -sh "$DATA/$DOMAIN"
