#!/bin/bash
# DEBUG: overfit PGND to a SINGLE cloth-folding trajectory, with the gripper
# modelled as a KINEMATIC FORCING sphere (gripper_forcing=True) instead of a
# point cloud. This is PGND's default gripper model (train_cloth.sh) -- the
# gripper is not particles; cloth particles within gripper_radius of a closed
# gripper are kinematically dragged to follow it. 1000 cloth particles only
# (no 2000 gripper points).
#
# Same single episode (episode_0133) and baseline temporal recipe (dt=0.1,
# load_skip=3) as train_cloth_fold_overfit.sh, so this is a clean A/B vs the
# gripper_points overfit -- only the gripper model changes. Runs 2x longer
# (30k iters) to give the overfit room to converge.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 python experiments/train/train_eval.py \
'gpus=[0]' \
overwrite=True \
resume=False \
debug=True \
train.name=cloth_fold_overfit_fc/train \
train.source_dataset_name=data/cloth_diverse_v1 \
train.dataset_name=cloth_fold_overfit_fc/dataset \
'sim.dt=0.1' \
sim.interval=1 \
sim.num_grippers=2 \
sim.gripper_points=False \
sim.gripper_forcing=True \
sim.num_steps_train=5 \
sim.n_history=2 \
sim.preprocess_scale=0.7 \
sim.preprocess_with_table=True \
'model.eef_t=[0.0,0.0,0.0]' \
train.dataset_load_skip_frame=3 \
train.dataset_skip_frame=1 \
train.num_workers=8 \
train.batch_size=20 \
train.training_start_episode=133 \
train.training_end_episode=134 \
train.eval_start_episode=133 \
train.eval_end_episode=134 \
train.num_iterations=30000 \
train.iteration_eval_interval=500 \
train.iteration_log_interval=10 \
train.iteration_save_interval=1000 \
train.importance_sampling=True
