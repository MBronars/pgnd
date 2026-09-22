#!/bin/bash
#SBATCH --job-name=pickup_train
#SBATCH --output=/data/user_data/mbronars/slurm/output/pickup_train_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/pickup_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=48:00:00
#SBATCH --partition=general

# PGND on pickup_pose 4-success combined. 48k train + 5k val episodes,
# T=40 each, 512 obj + 671 grip particles. The converted source data lives
# on group storage (~30GB) so we want some headroom in RAM for the importance-
# sampling motion-score pre-pass; 96G is plenty.

source /home/mbronars/miniconda3/etc/profile.d/conda.sh
conda activate pgnd

cd /data/user_data/mbronars/packages/pgnd

bash experiments/scripts/train_pickup.sh
