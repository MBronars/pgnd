#!/bin/bash
# Train PGND on the PWM folding release (pwm-data-v1/folding_8mm), matched to our folding model
# (runs_pwm_robertre/pwm_v1_folding_8mm_100m_unet_bidir: 2 GPUs x batch 4 = 8 windows/step, 200k steps).
#
# Released PGND cloth recipe (experiments/scripts/train_cloth.sh at upstream ae050d1) unless noted:
#   source        PWM release read on demand (train.source_format=pwm_folding, pgnd/data/pwm_folding_source.py):
#                 official train / test garments, retained cloth vertices, 2 x 1,000 recorded gripper points
#   frames        30 Hz, every frame (released: 30 Hz data, every 3rd frame, dt 0.1)    -- same frames as ours
#   window        n_history 3 + current frame given, 16 predicted (released 2 / 5)       -- ours: 4 given + 16
#   points        every retained vertex, n_particles 0 (released: 1,000 sampled)         -- same points as ours
#   gripper       gripper-point mode, recorded points every frame, no spheres (released cloth: 2 spheres;
#                 released box recipe: gripper points from a rigid posed template)        -- user 2026-10-22
#   scale         preprocess_scale 0.57 (released cloth 0.8): the widest garment in train+test (1.596 m, a hooded
#                 jacket; scan of all 4,746 clips) must fit the 0.92-unit box: 0.92 / 1.596 = 0.576
#   radius        model.material.radius 0.114 = 20 cm real (0.20 m x scale; released rope / sloth / paper bag
#                 0.2 at scale 1.0 = 20 cm)                                               -- user 2026-10-22
#   windows       motion-weighted starts, alpha 1.0 floor 0.02 mm (released uniform)    -- ours trains with it
#   batch         4 windows of one clip per GPU x 2 GPUs (NPROC=2; gradients averaged) = 8 windows/step
#                 (released 32)  -- ours: 2 GPUs x 4 windows of one clip. NPROC=1 uses grad_accum 2 instead.
#   anomaly       train.detect_anomaly=False: stock PGND runs autograd anomaly detection (debug NaN checks on
#                 every backward op, same math); off = 3.4x faster per micro-batch (2.66 -> 0.79 s, 8.2k points)
#   iterations    200,000 (released 100,000)                                              -- same steps as ours
#   unchanged     50^3 grid of 0.02, network, MSE on positions x100 summed over steps, Adam 1e-4 cosine,
#                 grad clip 0.1, friction 0 fixed, eval every 1,000
#
# Env overrides: NAME, NPROC, ITERS, BATCH, ACCUM, WORKERS (per GPU), EVAL_EVERY, LOG_EVERY; extra hydra overrides as arguments.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NAME=${NAME:-folding}
PWM_ROOT=${PWM_ROOT:-/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/folding_8mm}
NPROC=${NPROC:-2}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-$(( 2 / NPROC ))}
ITERS=${ITERS:-200000}
EVAL_EVERY=${EVAL_EVERY:-1000}
LOG_EVERY=${LOG_EVERY:-20}
WORKERS=${WORKERS:-3}

echo "folding: train=$PWM_ROOT/train val=$PWM_ROOT/test  $NPROC GPU x batch=$BATCH x accum=$ACCUM iters=$ITERS windows=$(( NPROC * BATCH * ACCUM * ITERS ))"

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
train.source_format=pwm_folding \
train.pwm_root=$PWM_ROOT \
train.source_dataset_name=pwm_folding \
train.dataset_name=$NAME/dataset \
sim.dt=0.03333333333333333 \
sim.interval=1 \
'sim.num_grids=[50,50,50,0.02]' \
sim.num_grippers=2 \
sim.gripper_points=True \
sim.gripper_forcing=False \
sim.num_steps_train=16 \
sim.n_history=3 \
sim.n_particles=0 \
sim.preprocess_scale=0.57 \
sim.preprocess_with_table=True \
model.material.radius=0.114 \
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
train.domain_batches=False \
train.detect_anomaly=False \
"$@"
