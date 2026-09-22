#!/bin/bash
# Train PGND on the pushing dataset (deformable blob + single rigid paddle).
#
# Source: /data/group_data/katefgroup/VLA/3DPCWM/pushing_pgnd/sub_episodes_v
#   (symlink-mounted at experiments/log/data/pushing/sub_episodes_v).
# Converted from pushing_data_actions_04_18 (WDS) via convert_pushing.py.
# Each episode = one demo, 149 frames, 128 blob particles + 128 gripper particles.
#
# Hybrid recipe (same as cloth_fold v2):
#   gripper_points=True (Path B: pre-recorded gripper particles, n_grippers=1)
#   material.radius=0.2 (default, local-neighborhood aggregation)
#   num_steps_train=5, n_history=2 (PGND defaults)
#   dt=0.1 + dataset_load_skip_frame=1 (149 frames is already short; no further sub)
#   n_particles=128 (we have exactly 128 blob particles per demo)
#   n_gripper_particles=128 (we have exactly 128 paddle particles per demo)
#   preprocess_scale=3.0 (~0.2m raw -> ~0.6 sim units, fits comfortably)
#   batch_size=32 (small particle count = small mask; bigger batch should fit)
#
# Episode split: training_start..end_episode and eval_start..end_episode are
# computed from the converter's metadata.json -- update if you re-convert.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 python experiments/train/train_eval.py \
'gpus=[0]' \
overwrite=True \
resume=False \
debug=True \
train.name=pushing/train \
train.source_dataset_name=data/pushing/sub_episodes_v \
train.dataset_name=pushing/dataset \
'sim.dt=0.1' \
sim.interval=1 \
sim.num_grippers=1 \
sim.gripper_points=True \
sim.gripper_forcing=False \
sim.num_steps_train=5 \
sim.n_history=2 \
sim.n_particles=128 \
sim.n_gripper_particles=128 \
sim.preprocess_scale=2.0 \
sim.preprocess_with_table=True \
'model.eef_t=[0.0,0.0,0.0]' \
train.dataset_load_skip_frame=2 \
train.dataset_skip_frame=1 \
train.num_workers=8 \
train.batch_size=32 \
train.training_start_episode=0 \
train.training_end_episode=500 \
train.eval_start_episode=500 \
train.eval_end_episode=510 \
train.num_iterations=50000 \
train.iteration_eval_interval=1000 \
train.iteration_log_interval=10 \
train.iteration_save_interval=1000 \
train.importance_sampling=True
