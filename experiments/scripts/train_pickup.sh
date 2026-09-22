#!/bin/bash
# Train PGND on the pickup_pose 4-success combined dataset (rigid object + Panda arm gripper).
#
# Source: /data/group_data/katefgroup/VLA/3DPCWM/pickup_pgnd/sub_episodes_v
#   (symlink-mounted at experiments/log/data/pickup/sub_episodes_v).
# Converted from 4 task variants (place/convex/pick/upright _success) via convert_pickup.py,
# with the v9/PhysCtrl shard-level train/val split (last 10% of shards per root = val) and
# the combined 4-success blocklist (299 entries) applied.
#
# Each episode = one source trajectory: T=40 frames, 512 object + 671 gripper particles.
#
# Recipe (same flavor as pushing, scaled for larger particle counts + position range):
#   gripper_points=True (Path B: pre-recorded gripper particles, num_grippers=1)
#   num_steps_train=5, n_history=2 (PGND defaults; spec uses context_len=2)
#   dt=0.1, dataset_load_skip_frame=1 (T=40 is already short; no further sub)
#   n_particles=512, n_gripper_particles=671 (full counts from the data)
#   preprocess_scale=1.0  (workspace y extent ~0.86m must fit in the [0.04, 0.96] grid
#                           box; scale=2.0 was too aggressive and ~5% of episodes had
#                           every particle filtered out -> preprocess assertion crashed)
#   batch_size=24 (1183 particles vs cloth's 3000 at bs=20)
#   importance_sampling=True (object barely moves in approach phase, weight motion-rich windows)
#
# Episode split (from converter metadata.json; update if you re-convert):
#   train: [0, n_train),  val: [n_train, n_total)
#   n_train ~48k, n_val ~5k (depending on actual blocklist drops).
#   We pass a tiny eval window (n_train..n_train+64) for in-training sanity eval;
#   the headline metrics come from the offline rollout script (score_pickup_val.py).

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pull split bounds from the converter metadata so we don't have to hard-code them.
META=/data/group_data/katefgroup/VLA/3DPCWM/pickup_pgnd/sub_episodes_v/metadata.json
N_TRAIN=$(python -c "import json; print(json.load(open('$META'))['train_end_episode'])")
N_TOTAL=$(python -c "import json; print(json.load(open('$META'))['val_end_episode'])")
EVAL_END=$(( N_TRAIN + 64 ))
if [ $EVAL_END -gt $N_TOTAL ]; then EVAL_END=$N_TOTAL; fi
echo "split: train=[0,$N_TRAIN), in-train-eval=[$N_TRAIN,$EVAL_END), full-val=[$N_TRAIN,$N_TOTAL)"

CUDA_VISIBLE_DEVICES=0 python experiments/train/train_eval.py \
'gpus=[0]' \
overwrite=True \
resume=False \
debug=True \
train.name=pickup/train \
train.source_dataset_name=data/pickup/sub_episodes_v \
train.dataset_name=pickup/dataset \
'sim.dt=0.1' \
sim.interval=1 \
sim.num_grippers=1 \
sim.gripper_points=True \
sim.gripper_forcing=False \
sim.num_steps_train=5 \
sim.n_history=2 \
sim.n_particles=512 \
sim.n_gripper_particles=671 \
sim.preprocess_scale=1.0 \
sim.preprocess_with_table=True \
'model.eef_t=[0.0,0.0,0.0]' \
train.dataset_load_skip_frame=1 \
train.dataset_skip_frame=1 \
train.num_workers=8 \
train.batch_size=24 \
train.training_start_episode=0 \
train.training_end_episode=$N_TRAIN \
train.eval_start_episode=$N_TRAIN \
train.eval_end_episode=$EVAL_END \
train.num_iterations=100000 \
train.iteration_eval_interval=2000 \
train.iteration_log_interval=20 \
train.iteration_save_interval=2000 \
train.importance_sampling=True
