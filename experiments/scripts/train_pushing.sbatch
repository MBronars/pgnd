#!/bin/bash
#SBATCH --job-name=push_train
#SBATCH --output=/data/user_data/mbronars/slurm/output/push_train_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/push_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=24:00:00
#SBATCH --partition=general

# Pushing-domain PGND training. Smaller particle/episode counts than cloth so
# RAM is light (64G is more than enough -- the converted source data is ~1-2GB).

source /home/mbronars/miniconda3/etc/profile.d/conda.sh
conda activate pgnd

cd /data/user_data/mbronars/packages/pgnd

bash experiments/scripts/train_pushing.sh
