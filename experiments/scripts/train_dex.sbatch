#!/bin/bash
#SBATCH --job-name=pgnd_dex
#SBATCH --output=/data/user_data/mbronars/slurm/output/pgnd_dex_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/pgnd_dex_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-o9-20,babel-o9-24,babel-p9-32,babel-n5-32

# PGND on the PWM dex release (see train_dex.sh): read directly from the release, no copy.
# general, one L40S (user 2026-09-22). Local check (A100 40 GB): Sharpa-only
# batches at batch 16, 16 steps, 3 history peaked at 13,828 MiB. RAM: all 16,999 train episodes
# in memory ~ 5.5M frames x ~2,170 points x 12 B ~ 143 GB (estimate), hence 200G.
# Resubmitting resumes from experiments/log/dex/train/ckpt/latest.pt.
export PATH=/data/user_data/mbronars/.miniconda3/envs/pgnd/bin:$PATH
cd /data/user_data/mbronars/packages/pgnd
CUDA_VISIBLE_DEVICES=0 bash experiments/scripts/train_dex.sh
