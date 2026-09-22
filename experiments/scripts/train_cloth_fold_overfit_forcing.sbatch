#!/bin/bash
#SBATCH --job-name=cloth_ovf_fc
#SBATCH --output=/data/user_data/mbronars/slurm/output/cloth_ovf_fc_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/cloth_ovf_fc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=14:00:00
#SBATCH --partition=general

# DEBUG overfit run, gripper_forcing variant -- single trajectory (episode_0133),
# gripper modelled as a kinematic forcing sphere (PGND default, no point cloud).

source /home/mbronars/miniconda3/etc/profile.d/conda.sh
conda activate pgnd

cd /data/user_data/mbronars/packages/pgnd

bash experiments/scripts/train_cloth_fold_overfit_forcing.sh
