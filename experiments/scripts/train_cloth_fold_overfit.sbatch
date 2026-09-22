#!/bin/bash
#SBATCH --job-name=cloth_overfit
#SBATCH --output=/data/user_data/mbronars/slurm/output/cloth_overfit_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/cloth_overfit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=10:00:00
#SBATCH --partition=general

# DEBUG overfit run -- trains on a single trajectory (episode_0133).
# Only one episode is held in RAM (64G is ample); L40S requested so the
# batch_size=20 memory profile from the sweep holds.

source /home/mbronars/miniconda3/etc/profile.d/conda.sh
conda activate pgnd

cd /data/user_data/mbronars/packages/pgnd

bash experiments/scripts/train_cloth_fold_overfit.sh
