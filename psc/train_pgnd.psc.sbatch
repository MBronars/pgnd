#!/bin/bash
#SBATCH --job-name=pgnd_gpwm
#SBATCH --account=cis260367p
#SBATCH --partition=GPU-shared
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l40s-48:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G   # dex peaked ~175 GB host memory on babel; GPU-shared: 63 GB/GPU default, 128000M/GPU max
#SBATCH --time=48:00:00
#SBATCH --output=/ocean/projects/cis260367p/mbronars/projects/GPWM/trainings/pgnd/slurm/%x_%j.out
#SBATCH --error=/ocean/projects/cis260367p/mbronars/projects/GPWM/trainings/pgnd/slurm/%x_%j.err
# PGND baseline for gpwm on PSC Bridges-2. RUN=folding (experiments/scripts/train_folding.sh) or RUN=dex
# (experiments/scripts/train_dex_std.sh): every training setting comes from those scripts, unchanged; this file only
# sets the PSC paths and scheduler lines. CONFIRM the partition / gres / memory lines on Bridges-2 first (PSC_SETUP.md;
# dex held ~175 GB host memory on one process on babel, split across the 2 processes here).
# Resubmit the same command to resume from latest.pt (weights + optimizer).
#   sbatch --job-name=pgnd_folding --export=ALL,RUN=folding psc/train_pgnd.psc.sbatch
set -uo pipefail
RUN=${RUN:?set RUN=folding, RUN=dex or RUN=drag}
PGND_REPO=${PGND_REPO:-/ocean/projects/cis260367p/mbronars/projects/GPWM/baselines/pgnd}
COMPACT=${COMPACT:-/ocean/projects/cis260367p/mbronars/projects/GPWM/data}         # pwm-data-v1-compact
RUNS=${RUNS:-/ocean/projects/cis260367p/mbronars/projects/GPWM/trainings/pgnd}
CONDA_ENV=${CONDA_ENV:-pgnd}
export NPROC=${NPROC:-2}
case $RUN in
  folding) NAME=${NAME:-folding}; SCRIPT=experiments/scripts/train_folding.sh;  export PWM_ROOT=$COMPACT/folding_8mm ;;
  dex)     NAME=${NAME:-dex_std}; SCRIPT=experiments/scripts/train_dex_std.sh; export PWM_ROOT=$COMPACT/dex_v2_filtered2 ;;
  drag)    NAME=${NAME:-drag_std}; SCRIPT=experiments/scripts/train_drag_std.sh; export PWM_ROOT=$COMPACT/mpm_drag ;;
  *) echo "unknown RUN=$RUN" >&2; exit 2 ;;
esac
export NAME
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"
cd "$PGND_REPO"
# PGND writes runs under experiments/log/<NAME>/; keep the run itself in the trainings folder
mkdir -p "$RUNS/$NAME" "$RUNS/slurm" experiments/log
[ -e "experiments/log/$NAME" ] || ln -s "$RUNS/$NAME" "experiments/log/$NAME"
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}     # babel: first-barrier hangs on some nodes; 0.84M params, no cost
echo "$(date) job ${SLURM_JOB_ID:-local} $(hostname) RUN=$RUN NAME=$NAME data=$PWM_ROOT run=$RUNS/$NAME"; nvidia-smi -L
WORKERS=${WORKERS:-3} bash "$SCRIPT" "$@"
