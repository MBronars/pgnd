#!/bin/bash
# Train PGND on the PWM mpm_drag release (pwm-data-v1/mpm_drag).
#
# Source: the PWM release itself, read directly (train.source_format=pwm_mpm_drag,
#   pgnd/data/pwm_drag_source.py; identical to the PWM loader, check_drag_source.py) and
#   preprocessed in memory -- no converted copy, no frames.pt cache. Every indexed episode,
#   all frames, all particles (1-4 objects x 1,024, driven patch included).
# Gripper: PGND's stock sphere forcing (gripper_forcing=True), driven by the patch's RECORDED
#   motion -- centre = centroid of the 103 driven particles, velocity = its achieved step,
#   closed while the drag is active (drive + hold), open after release; radius
#   model.gripper_radius (stock 0.04 = 4 cm; the patch reaches <= 4.1 cm from its centroid).
# Batches: one object count per batch, all points kept (n_particles=0); a bucket's share of
#   batches follows its episode count, as PWM's batch_across_episodes.
# Val: the test folder (drag_unseen), one loader per object count. test_seen is not used.
#
# Recipe, matched to the PWM specialist unet_mpm_drag_100m.yaml where PGND allows:
#   dt=0.0417 (native, no subsampling)
#   num_steps_train=16, n_history=3: given 4 frames, predicts 16 (PWM C=4, F=16)
#   batch_size=8 x grad_accum=2 = PWM's 16 windows per optimizer step (batch 16 in one pass ran
#     out of memory on an L40S with 4-object scenes, job 10527165); iterations =
#     ceil(3.2M / (batch x accum)) (PWM: 200,000 steps x 16) -> 200,000
#   num_grids=[60,60,60,0.02]: 1.2 m box with 2 cm cells (largest scene spans 1.03 m)
#   preprocess_scale=1.0, importance_sampling=True (PWM draws mpm_drag windows by motion too)
#
# Env overrides: NAME, ITERS, EVAL_EVERY, BATCH, ACCUM, WORKERS, IMPORTANCE.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NAME=${NAME:-drag}
PWM_ROOT=/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/mpm_drag
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-2}
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
train.source_format=pwm_mpm_drag \
train.pwm_root=$PWM_ROOT \
train.source_dataset_name=pwm_mpm_drag \
train.dataset_name=$NAME/dataset \
sim.dt=0.0417 \
sim.interval=1 \
'sim.num_grids=[60,60,60,0.02]' \
sim.num_grippers=1 \
sim.gripper_points=False \
sim.gripper_forcing=True \
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
