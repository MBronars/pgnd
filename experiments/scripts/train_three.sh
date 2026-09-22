#!/bin/bash
# Train PGND on the three-domain PWM mix: pickup + rl_push + yikai_small
# (/data/group_data/katefgroup/runs_pwm_mbronars/datasets/yikai_small/TRAINING.md).
#
# Source: experiments/log/data/three/sub_episodes_v (converter: convert_three.py), one PGND
#   episode per loader episode, same episode sets / start frames / shifts as the PWM loader.
# Batches: domain-pure, domain drawn uniformly per batch (train.domain_batches=True).
# Val: every test episode of all three domains, one loader per domain (val/loss_x_<domain>).
#
# Recipe (PGND defaults unless noted):
#   dt=0.05 (native rate of all three domains, no subsampling: dataset_load_skip_frame=1)
#   num_steps_train=16, n_history=3: given 4 frames (current + 3 history), predicts 16 --
#     matched to PWM's 4 context / 16 future (PGND defaults are 5 / 2)
#   gripper_points=True (Path B: recorded 671-pt Franka gripper per frame, num_grippers=1)
#   n_particles=512 (every pickup/rl_push episode has 512 object points; 2-object yikai
#     episodes have 1024 and are FPS-subsampled to 512), n_gripper_particles=671 (all of them)
#   preprocess_scale=1.0 (largest per-episode extent in the smoke: 0.78 m objects, fits 0.92 m)
#   batch_size=16 x grad_accum=2 = PWM's 32 windows per optimizer step (batch 32 in one pass ran
#     out of GPU memory on an L40S at step ~20, job 10528470: 41.8 of 44.4 GiB)
#   num_iterations = ceil(6.4M / batch): batch x iterations matched to the PWM three-domain run
#     (200,000 steps x 32 windows); batch 32 -> 200,000, the same steps as PWM
#
# Env overrides: NAME (run name, default three), SRC (source name, default three),
#   ITERS, EVAL_EVERY, BATCH, ACCUM, WORKERS, IMPORTANCE (True/False).

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NAME=${NAME:-three}
SRC=${SRC:-three}
BATCH=${BATCH:-16}
ACCUM=${ACCUM:-2}
# batch x accum x iterations = the PWM three-domain run's 200,000 steps x 32 windows = 6.4M
PWM_WINDOWS=6400000
ITERS=${ITERS:-$(( (PWM_WINDOWS + BATCH * ACCUM - 1) / (BATCH * ACCUM) ))}
EVAL_EVERY=${EVAL_EVERY:-2000}
WORKERS=${WORKERS:-8}
IMPORTANCE=${IMPORTANCE:-True}
# 61 train episodes (57 yikai_small, 4 rl_push) whose objects span > 0.92 m do not fit the
# PGND box at scale 1.0 (14 lost every point in preprocessing, 47 some): excluded (user, 2026-09-22)
EXCLUDE=${EXCLUDE:-/data/group_data/katefgroup/runs_pwm_mbronars/pgnd_three/exclude_extent_over_0.92m.json}

META=experiments/log/data/$SRC/sub_episodes_v/metadata.json
read N_TRAIN_START N_TRAIN_END N_VAL_START N_VAL_END < <(python -c "
import json; m = json.load(open('$META'))
print(m['train_start_episode'], m['train_end_episode'], m['val_start_episode'], m['val_end_episode'])")
echo "train=[$N_TRAIN_START,$N_TRAIN_END) val=[$N_VAL_START,$N_VAL_END)  batch=$BATCH x accum=$ACCUM iters=$ITERS windows=$(( BATCH * ACCUM * ITERS ))"

python experiments/train/train_eval.py \
'gpus=[0]' \
overwrite=False \
resume=True \
debug=True \
train.name=$NAME/train \
train.source_dataset_name=data/$SRC/sub_episodes_v \
train.dataset_name=$SRC/dataset \
sim.dt=0.05 \
sim.interval=1 \
sim.num_grippers=1 \
sim.gripper_points=True \
sim.gripper_forcing=False \
sim.num_steps_train=16 \
sim.n_history=3 \
sim.n_particles=512 \
sim.n_gripper_particles=671 \
sim.preprocess_scale=1.0 \
sim.preprocess_with_table=True \
'model.eef_t=[0.0,0.0,0.0]' \
train.dataset_load_skip_frame=1 \
train.dataset_skip_frame=1 \
train.num_workers=$WORKERS \
train.batch_size=$BATCH \
train.grad_accum=$ACCUM \
train.training_start_episode=$N_TRAIN_START \
train.training_end_episode=$N_TRAIN_END \
train.eval_start_episode=$N_VAL_START \
train.eval_end_episode=$N_VAL_END \
train.num_iterations=$ITERS \
train.iteration_eval_interval=$EVAL_EVERY \
train.iteration_log_interval=20 \
train.iteration_save_interval=$EVAL_EVERY \
train.importance_sampling=$IMPORTANCE \
train.domain_batches=True \
train.exclude_episodes=$EXCLUDE \
"$@"
