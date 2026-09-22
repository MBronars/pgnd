#!/bin/bash
#SBATCH --job-name=pgnd_drag_memsmoke
#SBATCH --output=/data/user_data/mbronars/slurm/output/pgnd_drag_memsmoke_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/pgnd_drag_memsmoke_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=general
#SBATCH --exclude=babel-o9-20,babel-o9-24,babel-p9-32,babel-n5-32

# GPU memory smoke for PGND mpm_drag on one L40S: 4-object batches only (the worst case,
# 4,096 points), batch 8 x grad_accum 2 (train_drag.sh defaults), 16 steps, 3 history, 60^3 grid,
# eager. 10 optimizer steps + one val.
# Logs the training process's GPU memory (nvidia-smi) every 2 s; peak printed at the end.
export PATH=/data/user_data/mbronars/.miniconda3/envs/pgnd/bin:$PATH
cd /data/user_data/mbronars/packages/pgnd
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MEM=/data/group_data/katefgroup/runs_pwm_mbronars/pgnd_drag/memsmoke_${SLURM_JOB_ID}_gpumem.csv
( while true; do nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits >> $MEM; sleep 2; done ) &
POLL=$!
CUDA_VISIBLE_DEVICES=0 NAME=drag_memsmoke_${SLURM_JOB_ID} ITERS=10 EVAL_EVERY=10 WORKERS=8 \
    bash experiments/scripts/train_drag.sh train.domain_weighting=only:obj4
echo "train exit: $?"
kill $POLL
echo "peak GPU MiB (any process on this GPU): $(awk -F', ' '{print $2}' $MEM | sort -n | tail -1)"
