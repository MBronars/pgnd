#!/bin/bash
#SBATCH --job-name=pgnd_three
#SBATCH --output=/data/user_data/mbronars/slurm/output/pgnd_three_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/pgnd_three_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-o9-20,babel-o9-24,babel-p9-32,babel-n5-32

# PGND on pickup + rl_push + yikai_small (see train_three.sh). Data from prep job 10526242 (done).
# general, one L40S (user freed 2 general GPUs, 2026-09-22); memory check at batch 32, 16 steps,
# 3 history peaked at 36,552 MiB (A100 40 GB).
# Resubmitting resumes from experiments/log/three/train/ckpt/latest.pt.
export PATH=/data/user_data/mbronars/.miniconda3/envs/pgnd/bin:$PATH
cd /data/user_data/mbronars/packages/pgnd
CUDA_VISIBLE_DEVICES=0 bash experiments/scripts/train_three.sh
