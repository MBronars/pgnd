#!/bin/bash
# PGND on rl_push alone: the Table 1 Pushing specialist (gpwm docs/TABLE1_PUSHING_PICKPLACE.md), under the standard
# baseline rules (PGND's released recipe unless our data or the comparison forces a change). Matched to gpwm's rl_push
# specialist (TRAIN.md rl_push_100m: 16 per GPU x 2 GPUs = 32 windows/step, 200k steps, 4 given + 16 predicted,
# random episode per window, uniform starts from frame 3).
#
#   source       experiments/scripts/convert_pwm_domain.py --domain rl_push: every episode read through gpwm's own loader
#                (exclusions rl_push_exclusions_20260918.json on both splits, frames [3, len), len >= 23, the
#                loader's seeded 512-point object subset), stored as PGND episode dirs (data/$SRC/sub_episodes_v)
#   frames       dt 0.05, every frame (20 Hz)                                        -- same frames as gpwm
#   window       n_history 3 + current given, 16 predicted (released 2 / 5)          -- gpwm: 4 given + 16
#   points       512 object points (all of them) + the recorded 671-point Franka gripper every frame (Path B)
#   grid         55^3 of 0.02 = 1.1 m box (released 50^3 = 1 m), preprocess_scale 1.0: the widest object extent of
#                any converted episode is 1.00 m (train; test 0.85 m; 4 train episodes exceed the released 0.92 m usable),
#                so nothing trained on leaves the grid (as drag). Gripper points outside the grid are masked by PGND
#   windows      uniform starts (importance_sampling False), as gpwm's rl_push loader
#   batch        16 windows per GPU x 2 GPUs (gradients averaged) = 32/step (released 32)  -- gpwm 32
#   iterations   200,000 (released 100,000)                                           -- gpwm's steps
#   anomaly      train.detect_anomaly=False (debug checks only, same math)
#   unchanged    network, MSE on positions x100 summed over steps, Adam 1e-4 cosine, clip 0.1, friction 0 fixed
#
# Env overrides: NAME, SRC, NPROC, ITERS, BATCH, ACCUM, WORKERS (per GPU), EVAL_EVERY, LOG_EVERY, GRID;
# hydra overrides as args.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NAME=${NAME:-rl_push_std}
SRC=${SRC:-rl_push}
NPROC=${NPROC:-2}
BATCH=${BATCH:-16}
ACCUM=${ACCUM:-$(( 2 / NPROC ))}
ITERS=${ITERS:-200000}
EVAL_EVERY=${EVAL_EVERY:-1000}
LOG_EVERY=${LOG_EVERY:-20}
WORKERS=${WORKERS:-3}
GRID=${GRID:-[55,55,55,0.02]}

META=experiments/log/data/$SRC/sub_episodes_v/metadata.json
read N_TRAIN_START N_TRAIN_END N_VAL_START N_VAL_END < <(python -c "
import json; m = json.load(open('$META'))
print(m['train_start_episode'], m['train_end_episode'], m['val_start_episode'], m['val_end_episode'])")
echo "rl_push: train=[$N_TRAIN_START,$N_TRAIN_END) val=[$N_VAL_START,$N_VAL_END)  $NPROC GPU x batch=$BATCH x accum=$ACCUM iters=$ITERS windows=$(( NPROC * BATCH * ACCUM * ITERS )) grid=$GRID"

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
train.source_dataset_name=data/$SRC/sub_episodes_v \
train.dataset_name=$SRC/dataset \
sim.dt=0.05 \
sim.interval=1 \
"sim.num_grids=$GRID" \
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
train.iteration_log_interval=$LOG_EVERY \
train.iteration_save_interval=$EVAL_EVERY \
train.importance_sampling=False \
train.detect_anomaly=False \
"$@"
