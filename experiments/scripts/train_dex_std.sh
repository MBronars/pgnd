#!/bin/bash
# PGND on the PWM dex release (pwm-data-v1/dex_v2_filtered2) under the standard baseline rules (user 2026-10-22/23):
# PGND's released recipe unless our data or the comparison forces a change; same frames, points, steps and windows as
# our dex model (runs_pwm_robertre/pwm_v1_dex_v2_filtered2_100m_unet_bidir: 2 GPUs x batch 8 = 16 windows/step,
# 200,000 steps, 4 given + 16 predicted, uniform window starts).
#
#   source       PWM release read directly (pwm_dex_source.py; matches the PWM loader to 7.8e-8 m): 16,999 train /
#                174 test episodes (held-out objects), frames from 0 at 20 Hz (native 40 Hz, stride 2 as PWM)
#   frames       dt 0.05, every frame (released: 30 Hz data, every 3rd frame, dt 0.1)      -- same frames as ours
#   window       n_history 3 + current given, 16 predicted (released 2 / 5)                 -- ours: 4 given + 16
#   points       all 512 object points (n_particles 0; released 1,000 sampled)              -- same points as ours
#   gripper      gripper-point mode, the recorded posed hand points every frame (XHand1 1,646 / Sharpa 1,682,
#                fingers articulated); no spheres                                           -- user 2026-10-22
#   radius       model.material.radius 0.2 at preprocess_scale 1.0 = 20 cm (released default) -- user 2026-10-22
#   floor        screwdriver tasks: stock floor at the object's lowest point (the real ground); in-hand tasks
#                (no ground): floor 0.10 m below the lowest object/hand point                -- user 2026-09-22
#   windows      uniform starts (released uniform; our dex model has no motion weighting); the September run's
#                motion^2 importance sampling is OFF
#   batch        8 windows per GPU x 2 GPUs (gradients averaged) = 16/step; a batch holds one hand (the two hands'
#                point counts differ), drawn by episode share (PWM visits episodes uniformly)
#   iterations   200,000 (released 100,000)                                                 -- same steps as ours
#   anomaly      train.detect_anomaly=False (stock on: debug NaN checks, same math)
#   unchanged    50^3 grid of 0.02, network, MSE on positions x100 summed over steps, Adam 1e-4 cosine, grad
#                clip 0.1, friction 0 fixed, eval every 1,000
#
# Env overrides: NAME, NPROC, ITERS, BATCH, ACCUM, WORKERS (per GPU), EVAL_EVERY, LOG_EVERY; hydra overrides as args.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NAME=${NAME:-dex_std}
PWM_ROOT=/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/dex_v2_filtered2
NPROC=${NPROC:-2}
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-$(( 2 / NPROC ))}
ITERS=${ITERS:-200000}
EVAL_EVERY=${EVAL_EVERY:-1000}
LOG_EVERY=${LOG_EVERY:-20}
WORKERS=${WORKERS:-3}

echo "dex: train=$PWM_ROOT/train val=$PWM_ROOT/test  $NPROC GPU x batch=$BATCH x accum=$ACCUM iters=$ITERS windows=$(( NPROC * BATCH * ACCUM * ITERS ))"

if [ "$NPROC" -gt 1 ]; then
    LAUNCH="torchrun --standalone --nproc_per_node=$NPROC"
else
    LAUNCH=python
fi
$LAUNCH experiments/train/train_eval.py \
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
model.material.radius=0.2 \
'model.eef_t=[0.0,0.0,0.0]' \
train.dataset_load_skip_frame=1 \
train.dataset_skip_frame=1 \
train.num_workers=$WORKERS \
train.batch_size=$BATCH \
train.grad_accum=$ACCUM \
train.num_iterations=$ITERS \
train.iteration_eval_interval=$EVAL_EVERY \
train.iteration_log_interval=$LOG_EVERY \
train.iteration_save_interval=$EVAL_EVERY \
train.importance_sampling=False \
train.domain_batches=True \
train.domain_weighting=episodes \
train.detect_anomaly=False \
"$@"
