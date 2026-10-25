#!/bin/bash
#SBATCH --job-name=pgnd_drag_std
#SBATCH --output=/data/group_data/katefgroup/runs_gns_mbronars/pgnd_drag_std/slurm_%j.out
#SBATCH --error=/data/group_data/katefgroup/runs_gns_mbronars/pgnd_drag_std/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-o9-20,babel-o9-24,babel-p9-32,babel-n5-32,babel-q5-28,babel-n5-28

# PGND on mpm_drag under the standard baseline rules, 2 GPUs x 8 = 16 windows/step, 200,000 steps (= gpwm's mpm_drag
# model); command-driven re-anchored sphere + 3 material channels (user, 2026-10-25). Every setting vs the released
# recipe and vs the Sept run (pgnd_mpm_drag) is in train_drag_std.sh.
# Checks (2026-10-25): re-anchored sphere = patch centroid at each drag start, command-only steps during drags (tests/
# test_drag_reanchor_materials.py); window probabilities equal gpwm's to 1.1e-7; compact release reads identically;
# 1-GPU smoke on 40 episodes: peak 16.4 GiB at batch 8, validation runs. Speed on 2 L40S not measured.
# Resubmit to resume from latest.pt (weights + optimizer). This job does not self-chain.
export PATH=/data/user_data/mbronars/.miniconda3/envs/pgnd/bin:$PATH
# 2026-10-23: the first 2-GPU attempt (jobs 10902665 / 10902694, both on babel-n5-28) hung on the FIRST collective
# (WorkNCCL SeqNum=1 ALLREDUCE, both ranks) for the full 60 min -- the bad-node signature of babel-q5-28/n5-32.
# n5-28 is now excluded, and peer-to-peer GPU transfers are off (gradients of 0.84M parameters go through host
# memory; same math). The first collective prints "nccl ok" so a hang is visible within a minute.
export NCCL_P2P_DISABLE=1
cd /data/user_data/mbronars/packages/pgnd
echo "job $SLURM_JOB_ID on $(hostname) $(date)"; nvidia-smi -L
NPROC=2 WORKERS=3 bash experiments/scripts/train_drag_std.sh
