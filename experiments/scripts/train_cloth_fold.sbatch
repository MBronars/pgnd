#!/bin/bash
#SBATCH --job-name=cloth_train
#SBATCH --output=/data/user_data/mbronars/slurm/output/cloth_train_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/cloth_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --partition=general

# Trains the PGND cloth-folding baseline. The first run also preprocesses all
# 751 episodes into the .pt cache (~1-2 h, cached on group_data); resume=True in
# the train script means a re-submitted job continues from the latest checkpoint.
# ~200G RAM: RealTeleopBatchDataset holds the (unpadded) episode tensors in RAM.

source /home/mbronars/miniconda3/etc/profile.d/conda.sh
conda activate pgnd

cd /data/user_data/mbronars/packages/pgnd

bash experiments/scripts/train_cloth_fold.sh
