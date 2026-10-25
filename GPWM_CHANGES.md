# PGND as a gpwm baseline: every change from upstream

This fork (`MBronars/pgnd`, branch `gpwm-baselines`) trains PGND (Particle-Grid Neural Dynamics, `kywind/pgnd`) as a
baseline for **gpwm**, our particle world model (`RobertRen1122/particle_world_model`; file and run names that say
`pwm` mean the same model). The branch starts at upstream `ae050d1`; upstream has three newer commits (`a229c06`
checkpoints publish, `5de69f7` debug, `06ef5db` gripper saving / postprocess fixes) that are not merged.

## The rule (user, 2026-10-22)

PGND's released recipe unless our data or a fair comparison forces a change, and every change says why. Per dataset:
the same frames, the same points, the same steps and windows as the gpwm model it is compared with; gpwm's window
sampling where gpwm uses one; PGND's gripper-point mode on every dataset.

## Settings: released vs ours

| setting | released PGND | gpwm baseline | why |
|---|---|---|---|
| frames | 30 Hz data, every 3rd frame, dt 0.1 | every frame at the data's native rate (folding 30 Hz, dex 20 Hz) | same frames as gpwm |
| window | history 2 + current given, 5 predicted | history 3 + current given (4 frames), 16 predicted, rolled out one frame at a time, loss on all 16 | gpwm: 4 given + 16 |
| points | 1,000 sampled per window | every point (`sim.n_particles=0`); folding: every retained cloth vertex (2,600-10,000 per garment) | same points as gpwm |
| gripper | spheres that force closed-gripper material (cloth / rope), or a rigid point template posed by the end-effector (box) | gripper-point mode on every dataset with the RECORDED actuator points every frame (folding: 2 x 1,000 hand points; dex: the posed hand, 1,646 / 1,682 points) | one gripper treatment everywhere; picking and dex need articulating fingers; same gripper points as gpwm |
| neighbourhood radius | 0.2 in box units (rope / stuffed toy / paper bag at scale 1.0 = 20 cm; cloth at 0.8 = 25 cm) | 20 cm real everywhere: `model.material.radius = 0.20 x preprocess_scale` (folding 0.114, dex 0.2) | same physical reach on every dataset |
| scene scale | cloth 0.8 | folding 0.57 (widest garment 1.596 m, scan of all 4,746 clips; 0.92 / 1.596); dex 1.0 | the 0.92-unit box must hold the scene |
| floor | grid floor at the lowest point | folding / screwdriver: stock; in-hand dex tasks: 10 cm below the lowest object / hand point | in-hand tasks have no ground (user 2026-09-22) |
| window sampling | uniform | folding: gpwm's motion weighting (alpha 1.0, floor 0.02 mm); dex: uniform | gpwm's own sampling per dataset |
| batch / steps | 32 x 100k | folding 2 GPUs x 4 windows of one clip = 8 x 200k; dex 2 x 8 = 16 x 200k | same steps and windows as gpwm |
| grid, network, loss, optimizer | 50^3 cells of 0.02, PointNet + MLP, position MSE x100 summed over steps, Adam 1e-4 cosine, clip 0.1 | unchanged | |

## Code changes

| where | change | why |
|---|---|---|
| `pgnd/data/pwm_folding_source.py` (new) | folding reader: the gpwm release read on demand, a batch = 4 windows of one clip with gpwm's motion-weighted starts, fixed held-out set (40 test clips, per-episode seeded starts) | 4,510 clips x 902 frames do not fit in memory; gpwm's batches |
| `pgnd/data/pwm_dex_source.py`, `pwm_drag_source.py` (new) | dex / mpm_drag readers (match gpwm's loader to 7.8e-8 m); dex keeps poses, points made per window | |
| `pgnd/data/gpwm_compact_codec.py` (new, verbatim from gpwm) | reads pwm-data-v1-compact (the release on PSC) | |
| `pgnd/data/dataset.py` | in-memory preprocessing (`preprocess_arrays`), per-episode lists, per-frame gripper points (Path B), domain buckets, velocities computed on access; axis rotation in the points' dtype (exact; was a float64 promotion costing ~2 s per clip) | |
| `experiments/train/train_eval.py` | multi-GPU (torchrun: one process and warp device per GPU, gradients averaged after backward, rank 0 validates and saves); optimizer state saved and restored on resume (it used to restart Adam); `train.detect_anomaly` (stock on; off = 3.4x faster, same math); validation on a fixed subset (it scored a different subset every call); per-rank seeds; domain batches; `train.grad_accum`; latest / best / final checkpoints; peak memory in the log | |
| `experiments/cfg/train/default.yaml` | the new keys above (defaults keep stock behaviour) | |
| `experiments/scripts/convert_three.py`, `preprocess_parallel.py` | the Sept. three-domain converter | |
| `experiments/scripts/train_*.sh / .sbatch` | babel launchers; `train_folding.sh` and `train_dex_std.sh` list every setting against the released recipe | |
| `psc/train_pgnd.psc.sbatch` | PSC launcher (`PSC_SETUP.md`) | |

## mpm_drag (2026-10-25, `experiments/scripts/train_drag_std.sh`)

| setting | Sept run (`train_drag.sh`) | standard run | why |
|---|---|---|---|
| sphere | centred on the recorded patch centroid every frame, moved by the patch's achieved step | `sim.drag_sphere=reanchor`: placed on the patch centroid at each drag's first frame, then moved ONLY by the commanded velocity (`grab_velocity`); open between drags | the command is the action gpwm gets and the only one available when planning; a tied sphere follows predicted points (user) |
| materials | none (released PGND) | `model.material.material_features=3`: log10 E - 5, plasticine flag, fifth value / 1e4 per particle, appended to the PointNet input | material varies between episodes (E 1e4 / 1e5 / 1e6, elastic / plasticine) and gpwm is given it (user) |
| window sampling | motion^2 importance sampling, 0.1 mm floor | `train.window_motion` alpha 1.0, floor 0.25 mm: gpwm's mpm_drag weighting (probabilities equal to 1.1e-7) | gpwm's own window policy |
| batch / steps | 1 GPU x 8 x accum 2, 200k | 2 GPUs x 8, 200k (= gpwm 16 x 200k planned) | |

Measured on the release: during a drag the patch follows the command to 0.12 mm per step (median; max 0.64), at most
8.6 mm accumulated; between drags the patch can move on its own (up to 131 mm in 7 frames), which is why the sphere
is re-anchored at every drag start. The patch reaches 27.5 mm (median) / 45.8 mm (max) from its centroid; the sphere
keeps PGND's stock 4 cm radius. Both new options default to the old behaviour (`tied`, 0 channels).

## Known issue to keep in mind when reading validation

PGND's PointNet uses BatchNorm and validation uses its running statistics, which mostly reflect the last training
batches (each batch is one clip / one hand). Single validations can therefore jump (folding read 8.1 mm at one step
between 1.75-1.9 mm ones). Training is unaffected. In the Sept. three-domain run this was the main cause of its wild
validation curves (batches switched between very different domains).

## Checks

- 2026-10-23, compact vs pwm-data-v1: same episode sets; folding cloth bit-identical, gripper points within 0.000119 mm,
  preprocessed windows within 2.7e-6 box units (6 clips incl. the largest garment); dex points identical (6 held-out
  episodes). Training on the compact release runs (20 steps, loss falling, sanity losses 0).
- 2-process mode (gloo, one GPU): weights identical on both ranks every step; resume restores iteration and optimizer.
- Memory: largest garment 10,000 + 2,000 points at batch 4 x 16 steps: 24.1 GiB; dex batch 8: 6 GiB.
- Speed on 2 L40S: folding ~0.6-1.2 s/step, dex ~0.35 s/step.
