#!/bin/bash
# PGND on FULL yikai: the Table 1 Pick and Place specialist (gpwm docs/TABLE1_PUSHING_PICKPLACE.md; the paper reports
# no PGND number in this column, the run exists for completeness), under the standard baseline rules, matched to
# gpwm's yikai specialist (TRAIN.md yikai_100m: 8 per GPU x 2 GPUs = 16 windows/step, 200k steps, 4 given + 16
# predicted, a GPU's batch = windows of ONE episode, uniform starts from frame 10).
#
#   source       experiments/scripts/convert_pwm_domain.py --domain yikai: every episode through gpwm's own loader, stored
#                as compact rigid bodies (pgnd/data/pwm_yikai_source.py reads them; ~5 GB for the release)
#   box          FIXED 70 x 50 x 70 cells of 2 cm (PGND axes x, up, z: 1.4 m wide, 1.0 m tall; 2.0x the released 50^3
#                cells), shifted per window: gripper centroid at the window's first frame centred in x / z, the
#                episode's table plane on the box floor (user, 2026-10-27/28). Never grows with the data. Objects
#                leaving the box during a window are disabled for it; windows whose gripper leaves the box, or that
#                would disable an object being handled (> 5 mm motion within 25 cm of the gripper), are not drawn.
#                Full-release scan (scan_yikai_box.py, 2.6M train / 95k test windows): 21% of objects disabled,
#                0.0011% of train windows would drop a handled object, ~1.1% have the gripper leave the box
#   masking      model.material.mask_disabled=True: disabled points take no part in the network's centring, active-grid
#                selection or 20 cm aggregation (checked on CPU: identical to the enabled points alone, 3e-8);
#                no loss on them and the simulator does not move them (stock enabled mask)
#   frames       dt 0.05, every frame (20 Hz)                                        -- same frames as gpwm
#   window       n_history 3 + current given, 16 predicted (released 2 / 5)          -- gpwm: 4 given + 16
#   points       every object point of the scene (2,048-3,584) + the recorded 671-point Franka gripper (Path B)
#   radius       model.material.radius 0.2 at preprocess_scale 1.0 = 20 cm (released default)
#   batch        8 windows of one episode per GPU x 2 GPUs (gradients averaged) = 16/step (released 32) -- gpwm 16
#   iterations   200,000 (released 100,000)                                           -- gpwm's steps
#   anomaly      train.detect_anomaly=False (debug checks only, same math)
#   unchanged    network, MSE on positions x100 summed over steps, Adam 1e-4 cosine, clip 0.1, friction 0 fixed
#
# Env overrides: NAME, PWM_ROOT (the converted episodes: metadata.json + episode_NNNNNN.cpk), NPROC, ITERS, BATCH, ACCUM,
# WORKERS (per GPU), EVAL_EVERY, LOG_EVERY, GRID; hydra overrides as args.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NAME=${NAME:-yikai_box}
PWM_ROOT=${PWM_ROOT:?set PWM_ROOT to the converted yikai episodes (convert_pwm_domain.py --domain yikai --out ...)}
NPROC=${NPROC:-2}
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-$(( 2 / NPROC ))}
ITERS=${ITERS:-200000}
EVAL_EVERY=${EVAL_EVERY:-1000}
LOG_EVERY=${LOG_EVERY:-20}
WORKERS=${WORKERS:-3}
GRID=${GRID:-[70,50,70,0.02]}

echo "yikai box: episodes=$PWM_ROOT  $NPROC GPU x batch=$BATCH x accum=$ACCUM iters=$ITERS windows=$(( NPROC * BATCH * ACCUM * ITERS )) grid=$GRID"

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
train.source_format=pwm_yikai_box \
train.pwm_root=$PWM_ROOT \
train.source_dataset_name=pwm_yikai_box \
train.dataset_name=$NAME/dataset \
sim.dt=0.05 \
sim.interval=1 \
"sim.num_grids=$GRID" \
sim.num_grippers=1 \
sim.gripper_points=True \
sim.gripper_forcing=False \
sim.num_steps_train=16 \
sim.n_history=3 \
sim.n_particles=0 \
sim.n_gripper_particles=671 \
sim.preprocess_scale=1.0 \
sim.preprocess_with_table=True \
model.material.radius=0.2 \
model.material.mask_disabled=True \
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
train.domain_batches=False \
train.detect_anomaly=False \
"$@"
