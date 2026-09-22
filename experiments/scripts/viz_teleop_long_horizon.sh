#!/bin/bash
# Render side-by-side PRED vs GT animations for every teleop_long_horizon rollout.
# Output per episode:
#   experiments/log/teleop_long_horizon/rollouts/episode_NNNN/
#       rollout.mp4         side-by-side animation (stride=2, ~10s @ 10fps)
#       rollout_strip.png   static panels
#       frames/             per-frame PNGs
set -e
source /home/mbronars/miniconda3/etc/profile.d/conda.sh
conda activate pgnd
cd /data/user_data/mbronars/packages/pgnd

ROLLOUTS=experiments/log/teleop_long_horizon/rollouts
GT=experiments/log/data/teleop_long_horizon/sub_episodes_v
STRIDE=${STRIDE:-2}
FPS=${FPS:-10}

for d in $(ls $ROLLOUTS | grep '^episode_' | sort); do
    if [ ! -f "$ROLLOUTS/$d/pred.npy" ]; then continue; fi
    if [ -f "$ROLLOUTS/$d/rollout.mp4" ]; then
        echo "=== $d : mp4 exists, skipping ==="
        continue
    fi
    echo "=== $d : viz (stride=$STRIDE, fps=$FPS) ==="
    python experiments/train/viz_rollout.py \
        --pred "$ROLLOUTS/$d/pred.npy" \
        --gt "$GT/$d" \
        --out "$ROLLOUTS/$d" \
        --stride $STRIDE --fps $FPS --load_skip 1 2>/dev/null \
        | grep -E 'frames|video|strip|cloth L2' || true
done
echo "=== ALL VIZ DONE ==="
