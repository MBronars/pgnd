#!/bin/bash
#SBATCH --job-name=pgnd_three_prep
#SBATCH --output=/data/user_data/mbronars/slurm/output/pgnd_three_prep_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/pgnd_three_prep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --exclude=babel-o9-20,babel-o9-24,babel-p9-32,babel-n5-32

# Data prep for PGND on the three-domain mix (no GPU work; the L40S only satisfies the
# 8-CPUs-per-GPU rule). 1) convert_three.py: PWM loader episodes -> PGND episode dirs
# (pwm211 env, reads pwm_final_paper read-only). 2) preprocess_parallel.py: PGND's own
# preprocess() -> frames.pt cache, so the training job only loads.
set -euo pipefail
cd /data/user_data/mbronars/packages/pgnd
G=/data/group_data/katefgroup/runs_pwm_mbronars/pgnd_three
echo "pwm_final_paper rev: $(git -C /data/user_data/mbronars/packages/pwm_final_paper rev-parse HEAD)"

PYTHONDONTWRITEBYTECODE=1 /data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python \
    experiments/scripts/convert_three.py --out $G/source/sub_episodes_v --workers 8

/data/user_data/mbronars/.miniconda3/envs/pgnd/bin/python experiments/scripts/preprocess_parallel.py --workers 8 -- \
    train.name=three/train train.source_dataset_name=data/three/sub_episodes_v train.dataset_name=three/dataset \
    sim.dt=0.05 sim.num_grippers=1 sim.gripper_points=True sim.preprocess_scale=1.0 sim.preprocess_with_table=True \
    'model.eef_t=[0.0,0.0,0.0]' 2>&1 | grep -v '^Preprocessing episode'
