# Vendored verbatim from gpwm (github.com/RobertRen1122/particle_world_model, branch mb/psc-port, commit 8630bd5,
# data/compact_codec.py) so PGND reads pwm-data-v1-compact without importing the gpwm repo. Do not edit here;
# re-copy from gpwm if the codec changes. Needs the zstandard package.
"""Compact storage for pwm-data-v1 particle arrays (pwm-data-v1-compact, 2026-10-22).

Two encodings, chosen per rigid body when a file is converted (scripts/data/convert_compact.py):
  * rigid: the body's frame-0 points (float32, exactly as recorded) and one pose per later frame (float64
    rotation matrix + translation, a Kabsch fit of the frame-0 points to that frame). Decoding returns the
    recorded frame 0 exactly and R_t @ p0 + t_t, rounded to float32, for every other frame.
  * lossless: byte-shuffled zstd of the float32 array (bit-exact). Deformable rows (cloth, MPM) always use it, and
    a body whose rigid rebuild misses the recording by more than RIGID_TOL_M falls back to it.
Decoders return exactly what the loaders read from the original files: C-contiguous float32 numpy arrays of the
original shape and point order. Nothing here touches torch or the training batch.
"""
from __future__ import annotations

import io
import pickle

import numpy as np

CODEC = "pwm-cpk-1"
RIGID_TOL_M = 1e-5            # 0.01 mm: worst rigid rebuild allowed before a body is stored losslessly


def _zstd():
    import zstandard
    return zstandard


def pack_array(a: np.ndarray, level: int = 9) -> dict:
    """Lossless: byte-shuffled zstd of a numeric array (bit-exact on decode)."""
    a = np.ascontiguousarray(a)
    raw = a.view(np.uint8).reshape(-1, a.dtype.itemsize).T.copy().tobytes()
    return {"codec": "shuffle-zstd", "dtype": a.dtype.str, "shape": tuple(a.shape),
            "data": _zstd().ZstdCompressor(level=level).compress(raw)}


def unpack_array(p: dict) -> np.ndarray:
    if p.get("codec") != "shuffle-zstd":
        raise ValueError(f"unknown array codec {p.get('codec')!r}")
    dtype, shape = np.dtype(p["dtype"]), tuple(p["shape"])
    raw = np.frombuffer(_zstd().ZstdDecompressor().decompress(p["data"]), dtype=np.uint8)
    n = int(np.prod(shape, dtype=np.int64))
    if raw.size != n * dtype.itemsize:
        raise ValueError("shuffle-zstd payload size disagrees with its shape")
    return np.ascontiguousarray(raw.reshape(dtype.itemsize, n).T).view(dtype).reshape(shape)


def kabsch(P: np.ndarray, Q: np.ndarray):
    """R [T, 3, 3], t [T, 3] (float64) minimizing |R P + t - Q[t]|; P [n, 3], Q [T, n, 3]."""
    P = P.astype(np.float64)
    Q = Q.astype(np.float64)
    pc, qc = P.mean(0), Q.mean(1)
    H = np.einsum("ni,tnj->tij", P - pc, Q - qc[:, None])
    U, _, Vt = np.linalg.svd(H)
    V = np.swapaxes(Vt, -1, -2)
    d = np.sign(np.linalg.det(V @ np.swapaxes(U, -1, -2)))
    D = np.tile(np.eye(3), (len(Q), 1, 1))
    D[:, 2, 2] = d
    R = V @ D @ np.swapaxes(U, -1, -2)
    return R, qc - np.einsum("tij,j->ti", R, pc)


def _rebuild(p0: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """[T, n, 3] float32: frame 0 is the recorded template itself, later frames R @ p0 + t."""
    out = (np.einsum("tij,nj->tni", R, p0.astype(np.float64)) + t[:, None]).astype(np.float32)
    out[0] = p0
    return out


def pack_positions(points: np.ndarray, body: np.ndarray | None, rigid_bodies=None) -> dict:
    """One episode's positions [T, N, 3] float32. `body` [N] assigns every point to a body (None = all
    deformable); `rigid_bodies` lists the body ids to try as rigid (default: every id in `body`).
    Returns a picklable dict; `report` holds the worst rigid rebuild error (m) and the bodies stored losslessly."""
    points = np.ascontiguousarray(points, dtype=np.float32)
    T, N, _ = points.shape
    if body is None:
        return {"shape": (T, N, 3), "deformable": pack_array(points), "bodies": [],
                "report": {"rigid_max_err_m": 0.0, "lossless_bodies": []}}
    body = np.asarray(body, dtype=np.int64)
    rigid = set(np.unique(body).tolist()) if rigid_bodies is None else set(int(k) for k in rigid_bodies)
    bodies, worst, fallback = [], 0.0, []
    deform = ~np.isin(body, sorted(rigid))
    for k in sorted(rigid):
        sel = np.flatnonzero(body == k)
        if sel.size == 0:
            continue
        pts = points[:, sel]
        if sel.size >= 3 and np.isfinite(pts).all():
            R, t = kabsch(pts[0], pts)
            err = float(np.abs(_rebuild(pts[0], R, t) - pts).max())
        else:
            err = float("inf")
        if err <= RIGID_TOL_M:
            bodies.append({"id": int(k), "rows": sel.astype(np.int32), "template": pts[0].copy(), "R": R, "t": t})
            worst = max(worst, err)
        else:
            deform[sel] = True
            fallback.append(int(k))
    out = {"shape": (T, N, 3), "bodies": bodies, "deform_rows": np.flatnonzero(deform).astype(np.int32),
           "report": {"rigid_max_err_m": worst, "lossless_bodies": fallback}}
    if deform.any():
        out["deformable"] = pack_array(points[:, deform])
    return out


def unpack_positions(p: dict) -> np.ndarray:
    T, N, _ = p["shape"]
    out = np.empty((T, N, 3), dtype=np.float32)
    filled = np.zeros(N, dtype=bool)
    for b in p["bodies"]:
        out[:, b["rows"]] = _rebuild(b["template"], b["R"], b["t"])
        filled[b["rows"]] = True
    if "deformable" in p:
        rows = p["deform_rows"] if "deform_rows" in p else np.arange(N)
        out[:, rows] = unpack_array(p["deformable"])
        filled[rows] = True
    if not filled.all():
        raise ValueError("compact positions leave points undefined")
    return out


def dumps(obj: dict) -> bytes:
    """A .cpk member: pickle of {"codec": CODEC, ...} (trusted local files, like the release's .pyd members)."""
    buf = io.BytesIO()
    pickle.dump(dict(obj, codec=CODEC), buf, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.getvalue()


def loads(data: bytes) -> dict:
    obj = pickle.loads(data)
    if not isinstance(obj, dict) or obj.get("codec") != CODEC:
        raise ValueError("not a pwm-cpk-1 member")
    return obj


def pack_batch_positions(positions: np.ndarray, bodies) -> dict:
    """A pre-batched member [B, T, N, 3]: one pack_positions per row; `bodies` is a list of B `body` arrays (or
    None entries for deformable rows)."""
    positions = np.asarray(positions)
    return {"rows": [pack_positions(positions[b], bodies[b]) for b in range(positions.shape[0])],
            "dtype": positions.dtype.str}


def unpack_batch_positions(p: dict) -> np.ndarray:
    return np.ascontiguousarray(np.stack([unpack_positions(r) for r in p["rows"]]).astype(np.dtype(p["dtype"])))


def decode_member(name: str, data: bytes):
    """(original member suffix, array) for a compact tar member: 'particle_pos.cpk' -> 'particle_pos.pyd'."""
    if name.endswith(".cpk"):
        return name[:-4] + ".pyd", unpack_batch_positions(loads(data)["positions"])
    raise ValueError(f"not a compact member: {name}")


def decode_wds_sample(sample: dict) -> dict:
    """WebDataset map stage: replace every '<field>.cpk' member (raw bytes) by the decoded '<field>.pyd'."""
    for key in [k for k in sample if k.endswith(".cpk")]:
        value = sample.pop(key)
        pyd, array = decode_member(key, value if isinstance(value, (bytes, bytearray)) else bytes(value))
        sample[pyd] = array
    return sample
