#!/bin/bash
#SBATCH --job-name=pgnd_convert
#SBATCH --account=cis260367p
#SBATCH --partition=RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=62G
#SBATCH --time=12:00:00
#SBATCH --output=/ocean/projects/cis260367p/mbronars/projects/GPWM/trainings/pgnd/slurm/%x_%j.out
# One pwm-data-v1 domain -> PGND episode dirs (experiments/scripts/convert_pwm_domain.py) on the PSC compact release,
# through gpwm's own loader (gpwm env, branch mb/psc-port). Writes trainings/pgnd/datasets/<DOMAIN>/sub_episodes_v.
#   sbatch --export=ALL,DOMAIN=rl_push psc/convert_pwm_domain.psc.sbatch
set -euo pipefail
DOMAIN=${DOMAIN:?set DOMAIN=rl_push or yikai}
G=/ocean/projects/cis260367p/mbronars/projects/GPWM
source "$(conda info --base)/etc/profile.d/conda.sh"; set +u; conda activate gpwm; set -u
cd $G/baselines/pgnd
python -W ignore experiments/scripts/convert_pwm_domain.py --domain $DOMAIN --gpwm $G/gpwm_worktrees/psc-port \
  --release $G/data --out $G/trainings/pgnd/datasets/$DOMAIN/sub_episodes_v --workers $SLURM_CPUS_PER_TASK
du -sh $G/trainings/pgnd/datasets/$DOMAIN
