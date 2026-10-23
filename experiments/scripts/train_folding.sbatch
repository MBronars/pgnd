#!/bin/bash
#SBATCH --job-name=pgnd_folding
#SBATCH --output=/data/group_data/katefgroup/runs_gns_mbronars/pgnd_folding/slurm_%j.out
#SBATCH --error=/data/group_data/katefgroup/runs_gns_mbronars/pgnd_folding/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-o9-20,babel-o9-24,babel-p9-32,babel-n5-32,babel-q5-28,babel-n5-28

# PGND on the PWM folding release, 2 GPUs x 4 windows of one clip = 8 windows/step, 200,000 steps -- the same
# steps and windows as our folding model (see train_folding.sh for every setting vs the released cloth recipe).
# Checks before launch (2026-10-23, RTX A6000): largest garment (10,000 cloth + 2,000 gripper points) at batch 4
# peaked 24.1 GiB with anomaly detection on; median garment 0.79 s per micro-batch of 4 with it off; 2-process
# smoke (gloo, one GPU): weights identical on both ranks every step; resume restores the optimizer state.
# Speed on 2 L40S not measured; if a 48 h leg ends first, resubmit this script: it resumes from latest.pt.
# This job does not self-chain.
export PATH=/data/user_data/mbronars/.miniconda3/envs/pgnd/bin:$PATH
# 2026-10-23: the first 2-GPU attempt (jobs 10902665 / 10902694, both on babel-n5-28) hung on the FIRST collective
# (WorkNCCL SeqNum=1 ALLREDUCE, both ranks) for the full 60 min -- the bad-node signature of babel-q5-28/n5-32.
# n5-28 is now excluded, and peer-to-peer GPU transfers are off (gradients of 0.84M parameters go through host
# memory; same math). The first collective prints "nccl ok" so a hang is visible within a minute.
export NCCL_P2P_DISABLE=1
cd /data/user_data/mbronars/packages/pgnd
echo "job $SLURM_JOB_ID on $(hostname) $(date)"; nvidia-smi -L
NPROC=2 WORKERS=3 bash experiments/scripts/train_folding.sh
