"""Convert SIM1 cloth-folding .npz trajectories into PGND source-dataset format.

Each output episode dir contains the files PGND's RealTeleopBatchDataset.preprocess()
consumes:
    traj.npz           key 'xyz' -> (n_frames, N_cloth, 3) cloth vertex positions
    eef_traj.txt       (n_frames, num_grippers*3)  EE translations  [L_xyz, R_xyz]
    eef_rot.txt        (n_frames, num_grippers*9)  EE rotation matrices (row-major)
    eef_gripper.txt    (n_frames, num_grippers)    gripper openness  [L_open, R_open]
    gripper_points.npz keys 'left'/'right' -> (n_frames, 1000, 3) per-frame gripper
                       point clouds in world frame (used directly, no template FK)

Source: /data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1
See pointworld_cloth_folding_instructions/DATASET_SPEC_FOR_POINTWORLD.md.

Usage:
    python experiments/scripts/convert_cloth.py            # full diverse_v1 set
    python experiments/scripts/convert_cloth.py --limit 4  # smoke test, 4 trajs
    python experiments/scripts/convert_cloth.py --ood      # OOD eval trajectory
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# --- dataset layout -----------------------------------------------------------
SRC_ROOT = '/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1'
OOD_FILE = ('/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/'
            'sim1_cloth_variant_0002/pointcloud_dataset/000086.npz')
CATEGORIES = ['bimanual', 'single_arm', 'ik_lift_random', 'ik_push']

# 8 validation trajectories (relative to SRC_ROOT) -- DO NOT TRAIN ON (spec sec. 2)
VAL = [
    'bimanual/existing_000000.npz',
    'bimanual/existing_000004.npz',
    'single_arm/000000_000000_left.npz',
    'single_arm/000001_000000_right.npz',
    'ik_lift_random/000000.npz',
    'ik_lift_random/000001.npz',
    'ik_push/000000.npz',
    'ik_push/000001.npz',
]

# NOTE: bimanual/existing_000037.npz is a diverse_v1 symlink that resolves to the
# OOD eval trajectory (variant_0002/000086.npz). Per user decision it is KEPT in
# the training set (training set = 743 trajs, matching the spec). This means the
# OOD long-horizon rollout is NOT fully held out -- the model will have seen it.
EXCLUDE = []

# --- preprocessing params (spec sec. 4) ---------------------------------------
SKIP_FRAMES = 30   # drop cloth-settling frames at trajectory start
STRIDE = 2         # native 60 Hz -> 30 Hz effective


def npz_array_shape(npz_path, key):
    """Read just the (header) shape of one array inside an .npz, without
    decompressing/loading the array data."""
    with zipfile.ZipFile(npz_path) as z:
        with z.open(key + '.npy') as f:
            version = np.lib.format.read_magic(f)
            shape, _fortran, _dtype = np.lib.format._read_array_header(f, version)
    return shape


def quat_to_rotmat(pose):
    """pose: (..., 7) = (x,y,z, qx,qy,qz,qw) -> rotation matrices (..., 3, 3)."""
    quat = pose[..., 3:7]  # scipy from_quat expects [x, y, z, w]
    return Rotation.from_quat(quat.reshape(-1, 4)).as_matrix().reshape(*pose.shape[:-1], 3, 3)


def convert_one(src_path, out_dir, resume=False):
    """Convert a single .npz trajectory into a PGND episode dir. Returns metadata.

    If resume=True and the episode is already fully converted, reads shapes from
    the existing files and skips the (slow) reconversion.
    """
    done = all((out_dir / f).exists() for f in
               ('traj.npz', 'gripper_points.npz', 'eef_traj.txt',
                'eef_rot.txt', 'eef_gripper.txt'))
    if resume and done:
        shape = npz_array_shape(out_dir / 'traj.npz', 'xyz')  # header-only, fast
        return {'source': str(src_path), 'n_cloth': int(shape[1]),
                'n_frames': int(shape[0]), 'n_native': None, 'resumed': True}

    d = np.load(src_path, allow_pickle=True)
    n_native = d['cloth_points'].shape[0]

    # frame selection: skip settling, then stride-2 subsample
    idx = np.arange(SKIP_FRAMES, n_native, STRIDE)
    if len(idx) < 4:
        raise ValueError(f'{src_path}: only {len(idx)} usable frames after skip/stride')

    xyz = d['cloth_points'][idx]                          # (T, N, 3)
    left_pose = d['left_ee_pose'][idx]                    # (T, 7)
    right_pose = d['right_ee_pose'][idx]                  # (T, 7)
    openness = d['openness'][idx]                         # (T, 2)  1=open, 0=closed
    left_pts = d['left_ee_points'][idx]                   # (T, 1000, 3) world frame
    right_pts = d['right_ee_points'][idx]                 # (T, 1000, 3) world frame

    T = xyz.shape[0]

    # eef translations: [L_xyz, R_xyz] -> (T, 6)
    eef_traj = np.concatenate([left_pose[:, :3], right_pose[:, :3]], axis=1)

    # eef rotations: quat -> 3x3 row-major flattened -> (T, 18)
    left_R = quat_to_rotmat(left_pose).reshape(T, 9)
    right_R = quat_to_rotmat(right_pose).reshape(T, 9)
    eef_rot = np.concatenate([left_R, right_R], axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'traj.npz', xyz=xyz.astype(np.float32))
    np.savez(out_dir / 'gripper_points.npz',
             left=left_pts.astype(np.float32), right=right_pts.astype(np.float32))
    np.savetxt(out_dir / 'eef_traj.txt', eef_traj)
    np.savetxt(out_dir / 'eef_rot.txt', eef_rot)
    np.savetxt(out_dir / 'eef_gripper.txt', openness)

    return {
        'source': str(src_path),
        'n_cloth': int(xyz.shape[1]),
        'n_frames': int(T),
        'n_native': int(n_native),
    }


def _convert_task(task):
    """Pool worker: convert one episode. task = (ep_id, rel, split, ds_root, resume)."""
    ep_id, rel, split, ds_root, resume = task
    ds_root = Path(ds_root)
    src = OOD_FILE if split == 'ood' else os.path.join(SRC_ROOT, rel)
    ep_dir = ds_root / f'episode_{ep_id:04d}'
    meta = convert_one(src, ep_dir, resume=resume)
    meta.update(split=split, episode_id=ep_id,
                category=(rel.split('/')[0] if rel else 'variant_0002'),
                rel_path=rel)
    return meta


def build_file_list():
    """Return (train_files, val_files) as paths relative to SRC_ROOT, with the
    contaminated trajectory excluded."""
    all_rel = []
    for cat in CATEGORIES:
        for f in sorted(glob.glob(os.path.join(SRC_ROOT, cat, '*.npz'))):
            all_rel.append(os.path.relpath(f, SRC_ROOT))

    assert len(all_rel) == 751, f'expected 751 total, got {len(all_rel)}'
    for v in VAL:
        assert v in all_rel, f'val traj missing: {v}'
    for e in EXCLUDE:
        assert e in all_rel, f'exclude traj missing: {e}'

    val_files = list(VAL)  # keep spec order
    train_files = [f for f in all_rel if f not in set(VAL) | set(EXCLUDE)]
    assert len(train_files) == 743, f'expected 743 train, got {len(train_files)}'
    assert not (set(train_files) & set(val_files)), 'train/val overlap'
    return train_files, val_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root',
                    default='/data/group_data/katefgroup/VLA/3DPCWM/pgnd_cloth',
                    help='root dir for converted PGND source datasets')
    ap.add_argument('--limit', type=int, default=None,
                    help='convert only the first N train trajs (smoke test)')
    ap.add_argument('--ood', action='store_true',
                    help='convert the OOD eval trajectory instead of diverse_v1')
    ap.add_argument('--resume', action='store_true',
                    help='skip episodes that are already fully converted')
    ap.add_argument('--workers', type=int, default=8,
                    help='number of parallel conversion processes')
    args = ap.parse_args()

    out_root = Path(args.out_root)

    if args.ood:
        ds_root = out_root / 'cloth_ood'
        ep_dir = ds_root / 'episode_0000'
        meta = convert_one(OOD_FILE, ep_dir, resume=args.resume)
        meta.update(split='ood', category='variant_0002', episode_id=0)
        with open(ds_root / 'manifest.json', 'w') as f:
            json.dump([meta], f, indent=2)
        print(f'OOD traj -> {ep_dir}  ({meta["n_frames"]} frames, N={meta["n_cloth"]})')
        return

    train_files, val_files = build_file_list()
    if args.limit is not None:
        train_files = train_files[:args.limit]
        val_files = val_files[:max(1, args.limit // 4)]

    ds_root = out_root / 'cloth_diverse_v1'
    # episodes: train first (0..n_train-1), then val -> matches PGND index-range split
    ordered = ([(f, 'train') for f in train_files]
               + [(f, 'val') for f in val_files])

    tasks = [(ep_id, rel, split, str(ds_root), args.resume)
             for ep_id, (rel, split) in enumerate(ordered)]

    # embarrassingly parallel: each episode is an independent file -> file conversion.
    # imap preserves input order, so the manifest stays aligned with episode_id.
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            manifest = list(tqdm(pool.imap(_convert_task, tasks, chunksize=1),
                                 total=len(tasks), desc='convert'))
    else:
        manifest = [_convert_task(t) for t in tqdm(tasks, desc='convert')]

    with open(ds_root / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    n_train = sum(m['split'] == 'train' for m in manifest)
    n_val = sum(m['split'] == 'val' for m in manifest)
    print(f'\nconverted {len(manifest)} episodes -> {ds_root}')
    print(f'  train: episode_0000..episode_{n_train - 1:04d}  ({n_train})')
    print(f'  val:   episode_{n_train:04d}..episode_{n_train + n_val - 1:04d}  ({n_val})')
    print(f'PGND config: train.training_start_episode=0 training_end_episode={n_train} '
          f'eval_start_episode={n_train} eval_end_episode={n_train + n_val}')


if __name__ == '__main__':
    main()
