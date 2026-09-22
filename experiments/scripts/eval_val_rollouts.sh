#!/bin/bash
# Run cloth_fold_v2 rollout + viz on each of the 8 val episodes (743..750).
# Outputs:
#   experiments/log/cloth_fold_v2/val_rollouts/episode_0NNN/
#       pred.npy + pred_cloth_indices.npy + rollout.mp4 + rollout_strip.png + frames/
set -e
source /home/mbronars/miniconda3/etc/profile.d/conda.sh
conda activate pgnd
cd /data/user_data/mbronars/packages/pgnd

CKPT=experiments/log/cloth_fold_v2/train/ckpt/089000.pt
CFG=experiments/log/cloth_fold_v2/train/hydra.yaml
OUT_BASE=experiments/log/cloth_fold_v2/val_rollouts
mkdir -p "$OUT_BASE"

for i in 0743 0744 0745 0746 0747 0748 0749 0750; do
    EP=episode_$i
    OUT=$OUT_BASE/$EP
    echo "=== $EP ==="
    CUDA_VISIBLE_DEVICES=1 python experiments/train/rollout_cloth.py \
        --ckpt "$CKPT" --cfg "$CFG" \
        --ood_source experiments/log/data/cloth_diverse_v1 \
        --episode "$EP" \
        --cache_root experiments/log/cloth_fold_v2/dataset/state \
        --out "$OUT/pred.npy" --gpu 0 --subsample_method fps
    python experiments/train/viz_rollout.py \
        --pred "$OUT/pred.npy" \
        --gt experiments/log/data/cloth_diverse_v1/"$EP" \
        --out "$OUT" --stride 1 --fps 10 --load_skip 3
done
echo "=== ALL ROLLOUTS DONE ==="
