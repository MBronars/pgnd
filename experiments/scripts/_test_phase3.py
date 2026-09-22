"""Phase 3 integration test: preprocess + per-episode-list loader + gripper points.
Runs RealTeleopBatchDataset on a few short ik episodes and exercises __getitem__,
the DataLoader collate, and prepare_gripper_points. Throwaway script."""
import sys, shutil
from pathlib import Path
root = Path('/data/user_data/mbronars/packages/pgnd')
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'experiments'))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pgnd.data import RealTeleopBatchDataset
from train.train_eval import prepare_gripper_points

cfg = OmegaConf.create({
    'model': {'eef_t': [0.0, 0.0, 0.0], 'gripper_radius': 0.04, 'clip_bound': 1.5},
    'sim': {'num_grippers': 2, 'preprocess_scale': 0.7, 'preprocess_with_table': True,
            'interval': 1, 'dt': 0.0333333333, 'num_grids': [50, 50, 50, 0.02],
            'n_particles': 1000, 'n_history': 2, 'uniform': False},
    'train': {'dataset_skip_frame': 1, 'dataset_load_skip_frame': 1,
              'training_start_episode': 400, 'training_end_episode': 403},
})
NUM_STEPS = 16
test_root = root / 'experiments/log/_phase3_test'
if test_root.exists():
    shutil.rmtree(test_root)

ds = RealTeleopBatchDataset(
    cfg,
    dataset_root=test_root / 'state',
    source_data_root=root / 'experiments/log/data/cloth_diverse_v1',
    device=torch.device('cpu'),
    num_steps=NUM_STEPS,
    train=True,
    dataset_non_overwrite=True,
)
print(f'\n[init] episodes={len(ds.episode_xs)} has_gripper_points={ds.has_gripper_points} '
      f'total_len_sim={ds.total_len_sim}')
assert ds.has_gripper_points
assert isinstance(ds.episode_xs, list) and isinstance(ds.episode_gripper_points, list)

# --- Phase 4: importance-sampling weights ---
w = ds.sample_weights
print(f'[weights] shape={tuple(w.shape)} (expect ({ds.total_len_sim},))  '
      f'min={w.min():.4f} median={w.median():.4f} max={w.max():.2f}')
assert w.shape == (ds.total_len_sim,)
assert (w > 0).all() and w.max() > w.min(), 'weights should vary'
frac_floor = (w <= 0.1 ** 2 + 1e-9).float().mean()
print(f'[weights] fraction at motion floor (static windows) = {frac_floor:.3f}')
sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(ds), replacement=True)
drawn = torch.tensor([i for i, _ in zip(sampler, range(2000))])
mean_w_drawn = w[drawn].mean()
print(f'[weights] mean weight of 2000 sampled vs uniform mean: '
      f'{mean_w_drawn:.3f} vs {w.mean():.3f}  (sampled should be higher)')
assert mean_w_drawn > w.mean()

# --- check the per-episode frames.pt cache ---
fr = torch.load(test_root / 'state/episode_0400/frames.pt', map_location='cpu')
print(f'[frames.pt] keys={list(fr.keys())}  x={tuple(fr["x"].shape)}  '
      f'gripper_points={tuple(fr["gripper_points"].shape)}')
gp = fr['gripper_points']  # (n_frames, n_grippers, P, 3)
print(f'[frames.pt] gripper_points range [{gp.min():.3f}, {gp.max():.3f}]  '
      f'cloth x range [{fr["x"].min():.3f}, {fr["x"].max():.3f}]')
assert fr['x'].ndim == 3 and fr['gripper_points'].shape[1] == 2

# --- __getitem__ ---
init_state, actions, gt_states, gripper_states = ds[0]
x, v, x_his, v_his, clip_bound, enabled, episode_vec = init_state
gx, gv = gripper_states
print(f'\n[getitem] x={tuple(x.shape)} v={tuple(v.shape)} x_his={tuple(x_his.shape)}')
print(f'[getitem] actions={tuple(actions.shape)} gt_x={tuple(gt_states[0].shape)}')
print(f'[getitem] gripper_x={tuple(gx.shape)} gripper_v={tuple(gv.shape)}')
assert gx.shape == (NUM_STEPS, 2, 1000, 3), gx.shape
assert gx.shape == gv.shape

# --- DataLoader collate + prepare_gripper_points ---
dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, drop_last=True)
init_state, actions, gt_states, gripper_states = next(iter(dl))
gx, gv = gripper_states
print(f'\n[batch] gripper_x={tuple(gx.shape)} (expect (4,{NUM_STEPS},2,1000,3))')
gx2, gv2, gmask = prepare_gripper_points(cfg, gx, gv)
print(f'[prepare] gripper_x={tuple(gx2.shape)} gripper_v={tuple(gv2.shape)} mask={tuple(gmask.shape)}')
print(f'[prepare] in-bounds fraction = {gmask.float().mean():.3f}')
assert gx2.shape == (4, NUM_STEPS, 2000, 3)
assert gmask.shape == (4, NUM_STEPS, 2000)

shutil.rmtree(test_root)
print('\nPHASE 3 TEST PASSED')
