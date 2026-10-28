#!/bin/bash
#SBATCH --job-name=pgnd_rl_push_std
#SBATCH --output=/data/group_data/katefgroup/runs_gns_mbronars/pgnd_rl_push_std/slurm_%j.out
#SBATCH --error=/data/group_data/katefgroup/runs_gns_mbronars/pgnd_rl_push_std/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-o9-20,babel-o9-24,babel-p9-32,babel-n5-32,babel-q5-28,babel-n5-28

# PGND on rl_push (Table 1 Pushing) under the standard baseline rules, 2 GPUs x 16 = 32 windows/step, 200,000 steps
# (= gpwm's rl_push specialist); every setting in train_rl_push_std.sh. Needs the converted episodes first
# (convert_pwm_domain.sbatch DOMAIN=rl_push). Host memory: each of the 2 processes holds all 51,015 episodes (~30 GB).
# The episode cache (PGND's frames.pt per episode, ~30 GB) goes next to the data on group_data, not in the repo.
# 2026-10-28 on PSC: a 2-GPU start died reading a half-written frames.pt (both ranks preprocess the same episode);
# fixed in pgnd/data/dataset.py (temporary file + rename). Not yet trained anywhere.
# Resubmit to resume from latest.pt (weights + optimizer). This job does not self-chain.
set -uo pipefail
export PATH=/data/user_data/mbronars/.miniconda3/envs/pgnd/bin:$PATH
export NCCL_P2P_DISABLE=1          # see train_dex_std.sbatch: first-collective hangs on some babel nodes
NAME=${NAME:-rl_push_std}; SRC=${SRC:-rl_push}
DATA=${DATA:-/data/group_data/katefgroup/runs_gns_mbronars/pgnd_data}
RUN=/data/group_data/katefgroup/runs_gns_mbronars/pgnd_$NAME
cd /data/user_data/mbronars/packages/pgnd
mkdir -p "$RUN" experiments/log/data
[ -e "experiments/log/$NAME" ] || ln -s "$RUN" "experiments/log/$NAME"                 # the run folder
[ -e "experiments/log/data/$SRC" ] || ln -s "$DATA/$SRC" "experiments/log/data/$SRC"     # converted episodes
[ -e "experiments/log/$SRC" ] || ln -s "$DATA/$SRC" "experiments/log/$SRC"               # PGND's episode cache
echo "job $SLURM_JOB_ID on $(hostname) $(date)"; nvidia-smi -L
NAME=$NAME SRC=$SRC NPROC=2 WORKERS=3 bash experiments/scripts/train_rl_push_std.sh
