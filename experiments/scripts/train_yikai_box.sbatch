#!/bin/bash
#SBATCH --job-name=pgnd_yikai_box
#SBATCH --output=/data/group_data/katefgroup/runs_gns_mbronars/pgnd_yikai_box/slurm_%j.out
#SBATCH --error=/data/group_data/katefgroup/runs_gns_mbronars/pgnd_yikai_box/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-o9-20,babel-o9-24,babel-p9-32,babel-n5-32,babel-q5-28,babel-n5-28

# PGND on FULL yikai (Table 1 Pick and Place) in a fixed 70 x 70 x 50 box (2 cm cells) shifted per window onto the
# gripper; 2 GPUs x 8 = 16 windows/step, 200,000 steps (= gpwm's yikai specialist); every setting in
# train_yikai_box.sh. Needs the converted episodes first (convert_pwm_domain.sbatch DOMAIN=yikai). Episodes are read
# on demand by the loader workers (one episode per batch), so host memory stays small.
# Checked on PSC (2026-10-28, CPU): batches, box placement, masking. NOT yet run on a GPU: GPU memory (up to 3,584
# object + 671 gripper points at batch 8 on a 70 x 50 x 70 grid) and speed are unmeasured; run a short smoke first
# (ITERS=200 EVAL_EVERY=100 NAME=yikai_box_smoke) and check peak GiB in the log.
# Resubmit to resume from latest.pt (weights + optimizer). This job does not self-chain.
set -uo pipefail
export PATH=/data/user_data/mbronars/.miniconda3/envs/pgnd/bin:$PATH
export NCCL_P2P_DISABLE=1          # see train_dex_std.sbatch: first-collective hangs on some babel nodes
NAME=${NAME:-yikai_box}
DATA=${DATA:-/data/group_data/katefgroup/runs_gns_mbronars/pgnd_data}
RUN=/data/group_data/katefgroup/runs_gns_mbronars/pgnd_$NAME
cd /data/user_data/mbronars/packages/pgnd
mkdir -p "$RUN" experiments/log
[ -e "experiments/log/$NAME" ] || ln -s "$RUN" "experiments/log/$NAME"
echo "job $SLURM_JOB_ID on $(hostname) $(date)"; nvidia-smi -L
NAME=$NAME PWM_ROOT=$DATA/yikai/episodes NPROC=2 WORKERS=3 bash experiments/scripts/train_yikai_box.sh
