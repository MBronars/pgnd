#!/bin/bash
# Train PGND on the PWM dex release (pwm-data-v1/dex_v2_filtered2).
#
# Source: the PWM release itself, read directly (train.source_format=pwm_dex,
#   pgnd/data/pwm_dex_source.py; matches the PWM loader to 7.8e-8 m, check_dex_source.py) and
#   preprocessed in memory -- no converted copy. Every indexed episode (16,999 train / 174 test;
#   exclusions are built into the index), all frames from 0 (20 Hz).
# Objects: the 512 points of the one rigid object. Gripper: the recorded, posed hand points
#   (XHand1 1,646 / Sharpa 1,682, fingers articulated) as PGND gripper points (Path B), forward-
#   difference velocities (as the three-domain run).
# Floor (user 2026-09-22): screwdriver tasks keep PGND's stock floor at the object's lowest point
#   (the real ground, z = 0); in-hand tasks get the floor 0.10 m below the lowest point of object
#   and hand over the episode.
# Batches: one hand per batch (all points kept), a hand's share of batches follows its episode count.
# Val: the test split (held-out objects), one loader per hand.
#
# Recipe, matched to the PWM specialist unet_dex_v2_filtered2_100m.yaml where PGND allows:
#   dt=0.05 (native 40 Hz at stride 2, as PWM), num_steps_train=16, n_history=3 (PWM C=4, F=16)
#   16 windows per optimizer step (PWM 8/GPU x 2 GPUs), iterations = ceil(3.2M / windows per
#   step) = 200,000; importance_sampling=True (as the other PGND runs)
#
# Env overrides: NAME, ITERS, EVAL_EVERY, BATCH, ACCUM, WORKERS, IMPORTANCE.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NAME=${NAME:-dex}
PWM_ROOT=/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/dex_v2_filtered2
BATCH=${BATCH:-16}
ACCUM=${ACCUM:-1}
PWM_WINDOWS=3200000
ITERS=${ITERS:-$(( (PWM_WINDOWS + BATCH * ACCUM - 1) / (BATCH * ACCUM) ))}
EVAL_EVERY=${EVAL_EVERY:-2000}
WORKERS=${WORKERS:-8}
IMPORTANCE=${IMPORTANCE:-True}

echo "train=$PWM_ROOT/train val=$PWM_ROOT/test  batch=$BATCH x accum=$ACCUM iters=$ITERS windows=$(( BATCH * ACCUM * ITERS ))"

python experiments/train/train_eval.py \
'gpus=[0]' \
overwrite=False \
resume=True \
debug=True \
train.name=$NAME/train \
train.source_format=pwm_dex \
train.pwm_root=$PWM_ROOT \
train.source_dataset_name=pwm_dex \
train.dataset_name=$NAME/dataset \
sim.dt=0.05 \
sim.interval=1 \
sim.num_grippers=1 \
sim.gripper_points=True \
sim.gripper_forcing=False \
sim.num_steps_train=16 \
sim.n_history=3 \
sim.n_particles=0 \
sim.preprocess_scale=1.0 \
sim.preprocess_with_table=True \
'model.eef_t=[0.0,0.0,0.0]' \
train.dataset_load_skip_frame=1 \
train.dataset_skip_frame=1 \
train.num_workers=$WORKERS \
train.batch_size=$BATCH \
train.grad_accum=$ACCUM \
train.num_iterations=$ITERS \
train.iteration_eval_interval=$EVAL_EVERY \
train.iteration_log_interval=20 \
train.iteration_save_interval=$EVAL_EVERY \
train.importance_sampling=$IMPORTANCE \
train.domain_batches=True \
train.domain_weighting=episodes \
"$@"
