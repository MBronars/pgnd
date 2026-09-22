#!/bin/bash
# Train PGND on the SIM1 cloth-folding dataset (PointWorld baseline).
# Source data: /data/group_data/katefgroup/VLA/3DPCWM/pgnd_cloth/cloth_diverse_v1
#   (symlinked to experiments/log/data/cloth_diverse_v1)
# Episodes: train 0000-0742 (743), val 0743-0750 (8).
#
# Recipe -- "PGND-cloth temporal + box-style gripper" hybrid (per-task choices):
#   gripper handling (like PGND box):
#     gripper_points=True, gripper_forcing=False (Path B for us: pre-recorded
#     SIM1 gripper points). n_gripper_particles=500 subsamples per gripper to
#     match PGND's RealGripperDataset (which FPS-samples a 47k splat -> 500).
#     Halves gripper-side memory vs the old 1000/gripper.
#   material/model knobs (like PGND cloth):
#     material.radius=0.2 (local neighborhood aggregation, default), input_scale=2.0.
#     eef_t=[0,0,0] (Path B has gripper points already in world frame; eef_t is
#     only used in preprocess for the eef trajectory; not used by the model in
#     gripper_points mode).
#   temporal (matches PGND cloth):
#     dt=0.1 + dataset_load_skip_frame=3 -- a 5-step training window spans 0.5s
#     of motion (3 source frames per loaded step at 30 fps).
#   preprocess_scale=0.7 -- KEPT from previous (changing to PGND cloth's 0.8
#     would require re-preprocessing all 743 episodes; the cached frames.pt are
#     scale-baked). Re-run with =0.8 by changing the cache dir if desired.
#   num_steps_train=5: PGND default for every task. The 16-step setting we tried
#     earlier OOM'd; 5 + batch_size=20 is verified safe on a 44GB L40S.
#   batch_size=20: verified config; can try 32 once we confirm the new
#     n_gripper_particles=500 setting frees up ~30% of mask memory.
#   expandable_segments:True collapses PyTorch/Warp dual-allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 python experiments/train/train_eval.py \
'gpus=[0]' \
overwrite=True \
resume=False \
debug=True \
train.name=cloth_fold_v2/train \
train.source_dataset_name=data/cloth_diverse_v1 \
train.dataset_name=cloth_fold/dataset \
'sim.dt=0.1' \
sim.interval=1 \
sim.num_grippers=2 \
sim.gripper_points=True \
sim.gripper_forcing=False \
sim.num_steps_train=5 \
sim.n_history=2 \
sim.n_gripper_particles=500 \
sim.preprocess_scale=0.7 \
sim.preprocess_with_table=True \
'model.eef_t=[0.0,0.0,0.0]' \
train.dataset_load_skip_frame=3 \
train.dataset_skip_frame=1 \
train.num_workers=8 \
train.batch_size=20 \
train.training_start_episode=0 \
train.training_end_episode=743 \
train.eval_start_episode=743 \
train.eval_end_episode=751 \
train.num_iterations=100000 \
train.iteration_eval_interval=1000 \
train.iteration_log_interval=10 \
train.iteration_save_interval=1000 \
train.importance_sampling=True
