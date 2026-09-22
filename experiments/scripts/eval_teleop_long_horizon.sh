#!/bin/bash
# Roll out the PGND pickup model on every converted teleop_long_horizon episode.
# Outputs:
#   experiments/log/teleop_long_horizon/rollouts/episode_NNNN/
#       pred.npy                     [201, 512+646, 3] world metres
#       pred_cloth_indices.npy
set -e
source /home/mbronars/miniconda3/etc/profile.d/conda.sh
conda activate pgnd
cd /data/user_data/mbronars/packages/pgnd

CKPT=${CKPT:-experiments/log/pickup/train/ckpt/100000.pt}
CFG=$(dirname $(dirname $CKPT))/hydra.yaml
OUT_BASE=experiments/log/teleop_long_horizon/rollouts
SRC=experiments/log/data/teleop_long_horizon/sub_episodes_v
GPU=${GPU:-0}
mkdir -p "$OUT_BASE"

n=$(ls $SRC | grep -c '^episode_')
echo "ckpt: $CKPT (gpu=$GPU)"
echo "$n episodes to roll out"

for d in $(ls $SRC | grep '^episode_' | sort); do
    OUT=$OUT_BASE/$d
    if [ -f "$OUT/pred.npy" ]; then
        echo "=== $d : pred exists, skipping ==="
        continue
    fi
    echo "=== $d ==="
    CUDA_VISIBLE_DEVICES=$GPU python experiments/train/rollout_cloth.py \
        --ckpt "$CKPT" --cfg "$CFG" \
        --ood_source $SRC \
        --episode "$d" \
        --cache_root experiments/log/teleop_long_horizon/dataset/state \
        --out "$OUT/pred.npy" --gpu 0 --subsample_method fps 2>/dev/null
done
echo "=== ALL DONE ==="
