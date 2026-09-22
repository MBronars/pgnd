#!/bin/bash
# Run PGND rollout on the first N val episodes of the pickup dataset.
# N=64 matches what v9 / PhysCtrl reports headline numbers on.
# Outputs:
#   experiments/log/pickup/val_rollouts/episode_NNNNNN/
#       pred.npy                     [40, 512+671, 3] world metres (obj first, then gripper)
#       pred_cloth_indices.npy       (which obj indices the pred 512 rows correspond to)
#
# Run AFTER training finishes, with a chosen checkpoint:
#   CKPT=experiments/log/pickup/train/ckpt/099000.pt N=64 ./eval_pickup_val.sh
set -e
source /home/mbronars/miniconda3/etc/profile.d/conda.sh
conda activate pgnd
cd /data/user_data/mbronars/packages/pgnd

CKPT=${CKPT:-experiments/log/pickup/train/ckpt/100000.pt}
CFG=$(dirname $(dirname $CKPT))/hydra.yaml
OUT_BASE=experiments/log/pickup/val_rollouts
N=${N:-64}
GPU=${GPU:-0}
DO_VIZ=${DO_VIZ:-1}
META=/data/group_data/katefgroup/VLA/3DPCWM/pickup_pgnd/sub_episodes_v/metadata.json
N_TRAIN=$(python -c "import json; print(json.load(open('$META'))['train_end_episode'])")
mkdir -p "$OUT_BASE"

echo "ckpt: $CKPT (gpu=$GPU)"
echo "first val ep idx: $N_TRAIN ; evaluating $N episodes ; viz=$DO_VIZ"

for ((j=0; j<N; j++)); do
    i=$((N_TRAIN + j))
    EP=$(printf "episode_%06d" $i)
    OUT=$OUT_BASE/$EP
    if [ -f "$OUT/pred.npy" ]; then
        echo "=== $EP : pred exists, skipping rollout ==="
    else
        echo "=== $EP : rollout ==="
        CUDA_VISIBLE_DEVICES=$GPU python experiments/train/rollout_cloth.py \
            --ckpt "$CKPT" --cfg "$CFG" \
            --ood_source experiments/log/data/pickup/sub_episodes_v \
            --episode "$EP" \
            --cache_root experiments/log/pickup/dataset/state \
            --out "$OUT/pred.npy" --gpu 0 --subsample_method fps
    fi
    if [ "$DO_VIZ" = "1" ] && [ ! -f "$OUT/rollout.mp4" ]; then
        echo "    viz"
        python experiments/train/viz_rollout.py \
            --pred "$OUT/pred.npy" \
            --gt experiments/log/data/pickup/sub_episodes_v/"$EP" \
            --out "$OUT" --stride 1 --fps 10 --load_skip 1
    fi
done
echo "=== ALL PICKUP VAL ROLLOUTS DONE ==="
