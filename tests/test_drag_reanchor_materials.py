"""mpm_drag: the re-anchored (untied) sphere and per-particle material channels (2026-10-25)."""
import subprocess, sys, importlib.util
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pgnd.data import pwm_drag_source as D          # noqa: E402

RELEASE = '/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/mpm_drag/train'


def test_reanchored_path_synthetic():
    T = 12
    c = np.zeros((T, 3), np.float32); c[6:] = [0.0, 0.0, -0.1]          # patch drops 10 cm between drags
    active = np.zeros(T - 1, bool); active[1:4] = True; active[7:10] = True
    cmd = np.zeros((T - 1, 3), np.float32); cmd[active] = [0.2, 0.0, 0.0]
    u = D.reanchored_sphere_path(c, cmd, active, dt=0.05)
    assert np.allclose(u[1], c[1]) and np.allclose(u[7], c[7])        # on the patch at each drag's first frame
    for t in np.flatnonzero(active):
        assert np.allclose(u[t + 1] - u[t], cmd[t] * 0.05)            # command-only steps during a drag
    assert np.allclose(u[4], u[6])                                   # still between drags
    assert not np.allclose(u[7], u[6])                               # control: the second anchor really moved it


def _episodes(n=6):
    return D.load_episodes(RELEASE, workers=1, limit=n)


def test_real_episodes_reanchor_follows_command_and_tied_is_unchanged():
    eps = _episodes()
    src = subprocess.run(['git', '-C', str(ROOT), 'show', '970dc0c:pgnd/data/pwm_drag_source.py'],
                         check=True, capture_output=True, text=True).stdout
    path = ROOT / '.pytest_cache' / 'pwm_drag_source_970dc0c.py'; path.parent.mkdir(exist_ok=True); path.write_text(src)
    spec = importlib.util.spec_from_file_location('drag_old', path); old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    for ep in eps:
        a_old = old.to_pgnd_arrays(ep); a_tied = D.to_pgnd_arrays(ep, sphere='tied')
        for k in ('xyz', 'eef_xyz', 'eef_gripper'):
            assert np.array_equal(a_old[k], a_tied[k]), k               # default behaviour unchanged
        a = D.to_pgnd_arrays(ep, sphere='reanchor')
        c = ep['pos'][:, ep['selected']].mean(1); act = ep['active']
        u = a['eef_xyz'] * D.FLIP                                      # back to world axes; eef[k] = sphere[k-1]
        sphere = np.concatenate([u[1:], (u[-1] + ep['cmd_vel'][-1] * 0)[None]], 0)
        starts = np.flatnonzero(act & ~np.r_[False, act[:-1]])
        for st in starts:
            assert np.allclose(sphere[st], c[st], atol=1e-6)
        for t in np.flatnonzero(act):
            assert np.allclose(sphere[t + 1] - sphere[t], ep['cmd_vel'][t] * D.DT, atol=1e-6)
        assert np.array_equal(a['eef_gripper'], a_tied['eef_gripper'])  # same open / closed flag
        m = ep['mat']; assert m.shape == (len(ep['selected']), 3) and np.isfinite(m).all()


def _cfg(n_mat):
    return OmegaConf.create({'model': {'material': {'radius': 0.2, 'requires_grad': True, 'output_scale': 1.0,
        'input_scale': 2.0, 'absolute_y': False, 'pe_num_func_res': 0, 'material_features': n_mat}},
        'sim': {'n_history': 3, 'num_grids': [50, 50, 50, 0.02]}})


def test_material_channels_off_is_released_and_on_is_used():
    from pgnd.material import PGNDModel
    torch.manual_seed(0); B, N = 2, 64
    x = torch.rand(B, N, 3) * 0.5 + 0.25; v = torch.randn(B, N, 3) * 0.01
    xh = (x[:, :, None] + torch.randn(B, N, 3, 3) * 0.01).reshape(B, N, 9); vh = torch.randn(B, N, 9) * 0.01
    en = torch.ones(B, N, dtype=torch.bool)
    m0 = PGNDModel(_cfg(0)).eval(); assert m0.encoder.stn.conv1.in_channels == 24
    m3 = PGNDModel(_cfg(3)).eval(); assert m3.encoder.stn.conv1.in_channels == 27
    mat = torch.zeros(B, N, 3); mat2 = mat.clone(); mat2[..., 0] = 1.0
    with torch.no_grad():
        a = m3(x, v, xh, vh, en, mat=mat); b = m3(x, v, xh, vh, en, mat=mat2)
    assert not torch.allclose(a, b)                                    # the material input changes the prediction
    out = m3(x, v, xh, vh, en, mat=mat2); out.abs().sum().backward()
    assert m3.encoder.stn.conv1.weight.grad[:, 24:].abs().sum() > 0    # and its channels get gradient
