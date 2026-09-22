#!/bin/bash
#SBATCH --job-name=pgnd_drag
#SBATCH --output=/data/user_data/mbronars/slurm/output/pgnd_drag_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/pgnd_drag_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-o9-20,babel-o9-24,babel-p9-32,babel-n5-32

# PGND on the PWM mpm_drag release (see train_drag.sh): read directly from the release, no copy.
# general, one L40S (user freed 2 general GPUs, 2026-09-22); local 4-object memory check at
# batch 8 x accum 2 peaked at 27,774 MiB (A100 40 GB).
# Resubmitting resumes from experiments/log/drag/train/ckpt/latest.pt.
export PATH=/data/user_data/mbronars/.miniconda3/envs/pgnd/bin:$PATH
cd /data/user_data/mbronars/packages/pgnd
CUDA_VISIBLE_DEVICES=0 bash experiments/scripts/train_drag.sh
