#!/bin/bash
#SBATCH --job-name=cloth_convert
#SBATCH --output=/data/user_data/mbronars/slurm/output/cloth_convert_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/cloth_convert_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
# CPU-only job: the converter is pure numpy/scipy, no GPU needed.
#SBATCH --partition=cpu

source /home/mbronars/miniconda3/etc/profile.d/conda.sh
conda activate pgnd

cd /data/user_data/mbronars/packages/pgnd

# Converts SIM1 cloth-folding .npz trajectories -> PGND source-dataset format.
# Reads (read-only) from .../robert_sim1/sim1_cloth_*; writes only to
# .../3DPCWM/pgnd_cloth/. --resume skips episodes already fully converted, so
# this is safe to re-submit if the job is interrupted.
# --workers is pinned to the allocated CPU count; episodes are independent so
# the conversion is embarrassingly parallel.

# OOD eval trajectory (single file, fast)
python experiments/scripts/convert_cloth.py --resume --ood

# Full diverse_v1 set: 743 train + 8 val episodes
python experiments/scripts/convert_cloth.py --resume --workers ${SLURM_CPUS_PER_TASK}
