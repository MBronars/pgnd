#!/bin/bash
# DEBUG: overfit PGND to a SINGLE cloth-folding trajectory.
#
# Episode: episode_0133 -- the shortest standard-resolution folding trajectory
# (670 frames, 7021 cloth pts). Trains AND evaluates on this one episode, so the
# val-loss IS the overfit loss. Goal: confirm the model can represent one full
# folding trajectory at all (if it cannot overfit one trajectory, no amount of
# data/iterations will help).
#
# Baseline temporal recipe (matches PGND train_cloth.sh): dt=0.1 +
# dataset_load_skip_frame=3 -> a num_steps_train=5 window spans 0.5s of real
# motion, the same as the PGND baseline (vs 0.17s under dt=1/30, load_skip=1).
# Everything else mirrors train_cloth_fold.sh.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 python experiments/train/train_eval.py \
'gpus=[0]' \
overwrite=True \
resume=False \
debug=True \
train.name=cloth_fold_overfit/train \
train.source_dataset_name=data/cloth_diverse_v1 \
train.dataset_name=cloth_fold_overfit/dataset \
'sim.dt=0.1' \
sim.interval=1 \
sim.num_grippers=2 \
sim.gripper_points=True \
sim.gripper_forcing=False \
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
train.num_iterations=15000 \
train.iteration_eval_interval=500 \
train.iteration_log_interval=10 \
train.iteration_save_interval=1000 \
train.importance_sampling=True
