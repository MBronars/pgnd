# Training the PGND baseline on PSC Bridges-2

Companion of gpwm's `docs/PSC_SETUP.md` (branch `mb/psc-port` of `RobertRen1122/particle_world_model`). What
changed from upstream PGND and why: `GPWM_CHANGES.md`. PGND reads the data itself (no gpwm checkout needed).

| item | PSC location |
|---|---|
| data: pwm-data-v1-compact | `/ocean/projects/cis260367p/mbronars/projects/GPWM/data` (already copied for the gpwm run) |
| this repo, branch `gpwm-baselines` | `$PGND_REPO`, e.g. `/ocean/projects/cis260367p/mbronars/projects/GPWM/code/pgnd` |
| run folders | `/ocean/projects/cis260367p/mbronars/projects/GPWM/trainings/pgnd/<name>/train/` (family `pgnd`) |

## 1. Code

```bash
git clone -b gpwm-baselines git@github.com:MBronars/pgnd.git "$PGND_REPO"
cd "$PGND_REPO" && git remote add upstream https://github.com/kywind/pgnd.git
```

## 2. Environment (`pgnd`; exact versions in `env/pgnd-2026-10-23.freeze.txt`)

Python 3.10, torch 2.5.1 + CUDA 12.4, warp-lang 1.1.0 (compiles its kernels for the GPU it runs on), dgl 2.4.0 from
DGL's wheel index. The bundled `diff-gaussian-rasterization` is only for rendering and is not needed to train.

```bash
conda create -n pgnd python=3.10 -y && conda activate pgnd
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html   # DGL's torch-2.4 build: the one the babel env runs with torch 2.5.1
grep -v -E '^(torch|torchvision|dgl|diff_gaussian_rasterization)==' env/pgnd-2026-10-23.freeze.txt > /tmp/pgnd_rest.txt
pip install -r /tmp/pgnd_rest.txt          # includes warp-lang 1.1.0 and zstandard (compact folding clips)
python -c "import torch, warp, dgl, zstandard; warp.init(); print(torch.cuda.get_arch_list())"
```

## 3. Data

Nothing to convert: `pgnd/data/pwm_folding_source.py` and `pwm_dex_source.py` read both pwm-data-v1 and the compact
release (relative index paths; the codec is vendored). The launcher sets `PWM_ROOT` (the dataset folder, not a code
checkout) to `$COMPACT/folding_8mm` or `$COMPACT/dex_v2_filtered2`.

## 4. Launch

`psc/train_pgnd.psc.sbatch` runs `RUN=folding`, `RUN=dex` or `RUN=drag` with every training setting from
`experiments/scripts/train_folding.sh` / `train_dex_std.sh` / `train_drag_std.sh` (2 GPUs; 8 / 16 / 16 windows per step; 200,000 steps) and
links `experiments/log/<name>` to the trainings folder. **Confirm the scheduler lines first**: account `cis260367p`,
partition `GPU-shared` and `--gres=gpu:h100-80:2` are assumptions, and dex needs a lot of host memory (175 GB peak on
one process on babel; each of the 2 processes loads half the episodes). Resubmit the same command to resume.

```bash
sbatch --job-name=pgnd_folding --export=ALL,RUN=folding psc/train_pgnd.psc.sbatch
sbatch --job-name=pgnd_dex     --export=ALL,RUN=dex     psc/train_pgnd.psc.sbatch
sbatch --job-name=pgnd_drag    --export=ALL,RUN=drag    psc/train_pgnd.psc.sbatch   # mpm_drag: re-anchored command sphere + materials
```

A run prints `nccl ok: rank r of 2 passed the first barrier` within a minute; if it does not, the node's GPU-to-GPU
transport is hanging (seen on babel nodes; `NCCL_P2P_DISABLE=1` is already set). To continue a babel run, copy its
`train/ckpt/latest.pt` to `trainings/pgnd/<name>/train/ckpt/latest.pt`.
