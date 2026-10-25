#!/bin/bash
# PGND on the PWM mpm_drag release under the standard baseline rules (user, 2026-10-22..25): PGND's released recipe unless
# our data or the comparison forces a change; same frames, points, windows and steps as gpwm's mpm_drag model
# (runs_pwm_robertre/pwm_v1_mpm_drag_100m_unet_bidir: 1 GPU x batch 16 = 16 windows/step, 200k steps planned (stopped at
# 184.7k; paper uses step_180000), 4 given + 16 predicted, motion-weighted starts alpha 1.0 floor 0.25 mm, batches of
# windows from different episodes with equal point counts).
#
#   source       PWM release read directly (pwm_drag_source.py): 1-4 objects x 1,024 particles, all frames (24 Hz)
#   frames       dt 0.0417, every frame (released: 10 Hz)                               -- same frames as gpwm
#   window       n_history 3 + current given, 16 predicted (released 2 / 5)              -- gpwm: 4 given + 16
#   points       every particle (n_particles 0; released 1,000 sampled)                  -- same points as gpwm
#   drag         PGND's sphere forcing (released), driven by the COMMAND (sim.drag_sphere=reanchor, user 2026-10-25):
#                at each drag's first frame the sphere sits on the patch centroid, then moves only by the commanded
#                velocity (grab_velocity, the action gpwm gets); open between drags. The Sept run tied it to the patch.
#                Stock gripper radius 0.04 (4 cm); the patch reaches 27.5 mm (median) / 45.8 mm (max) from its centroid.
#   materials    3 per-particle input channels (model.material.material_features=3, user 2026-10-25): log10 E - 5,
#                plasticine flag, fifth value / 1e4 -- gpwm gets per-object material too; released PGND has none
#   radius       model.material.radius 0.2 at preprocess_scale 1.0 = 20 cm (released default)   -- user 2026-10-22
#   grid         60^3 of 0.02 = 1.2 m box (released 50^3 = 1 m): the largest scene spans 1.03 m
#   windows      gpwm's weighting (train.window_motion alpha 1.0 floor 0.25 mm) instead of the Sept run's
#                motion^2 importance sampling; batches = one object count (equal point counts), by episode share
#   batch        8 windows per GPU x 2 GPUs (gradients averaged) = 16/step (released 32)    -- gpwm 16
#   iterations   200,000 (released 100,000)                                                 -- gpwm's planned steps
#   anomaly      train.detect_anomaly=False (debug checks only, same math)
#   unchanged    network, MSE on positions x100 summed over steps, Adam 1e-4 cosine, clip 0.1, friction 0 fixed
#
# Env overrides: NAME, PWM_ROOT, NPROC, ITERS, BATCH, ACCUM, WORKERS (per GPU), EVAL_EVERY, LOG_EVERY; hydra overrides as args.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NAME=${NAME:-drag_std}
PWM_ROOT=${PWM_ROOT:-/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/mpm_drag}
NPROC=${NPROC:-2}
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-$(( 2 / NPROC ))}
ITERS=${ITERS:-200000}
EVAL_EVERY=${EVAL_EVERY:-1000}
LOG_EVERY=${LOG_EVERY:-20}
WORKERS=${WORKERS:-3}

echo "drag: train=$PWM_ROOT/train val=$PWM_ROOT/test  $NPROC GPU x batch=$BATCH x accum=$ACCUM iters=$ITERS windows=$(( NPROC * BATCH * ACCUM * ITERS ))"

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
sim.drag_sphere=reanchor \
sim.num_steps_train=16 \
sim.n_history=3 \
sim.n_particles=0 \
sim.preprocess_scale=1.0 \
sim.preprocess_with_table=True \
model.material.radius=0.2 \
model.material.material_features=3 \
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
train.importance_sampling=True \
'train.window_motion={alpha:1.0,floor_mm:0.25}' \
train.domain_batches=True \
train.domain_weighting=episodes \
train.detect_anomaly=False \
"$@"
