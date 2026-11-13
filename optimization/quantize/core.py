"""Route B pure functions: uint8/uint16 quantization of PLY attributes.

Every attribute is quantized to an integer grid with stored per-
attribute min/max (rotations use the fixed range [-1, 1] and are
renormalized on encode and decode). Storage is a compressed .npz
with a JSON metadata blob; dequantize.py writes a normal float32
PLY back with identical header comments and property order.

The archive is for storage/transfer only - Brush reads the
dequantized float32 PLY (same size as the input PLY).
"""
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from optimization.pruning.smart_prune import (
    renormalize_quaternions, require_properties)

FORMAT = "3dgs-quant-npz-v1"
META_KEY = "__meta__"
QUAT_LO, QUAT_HI = -1.0, 1.0
ALLOWED_BITS = (8, 16)
DEFAULT_OPTS = {"pos": 16, "rot": 8, "scale": 16,
                "opacity": 16, "dc": 16, "rest": 8}

OPT_KEYS = ("pos", "rot", "scale", "opacity", "dc", "rest")


def validate_opts(opts):
    for k in OPT_KEYS:
        if opts.get(k) not in ALLOWED_BITS:
            raise ValueError(f"bits for '{k}' must be one of {ALLOWED_BITS}, "
                             f"got {opts.get(k)}")
    return {k: int(opts[k]) for k in OPT_KEYS}


def bits_dtype(bits):
    if bits == 8:
        return np.uint8
    if bits == 16:
        return np.uint16
    raise ValueError(f"bits must be 8 or 16, got {bits}")


def bits_for_property(name, opts):
    if name in ("x", "y", "z"):
        return opts["pos"]
    if name == "opacity":
        return opts["opacity"]
    if name.startswith("rot_"):
        return opts["rot"]
    if name.startswith("scale_"):
        return opts["scale"]
    if name.startswith("f_dc_"):
        return opts["dc"]
    if name.startswith("f_rest_"):
        return opts["rest"]
    raise ValueError(f"unsupported property for quantization: {name}")


def quantize_values(values, bits, lo=None, hi=None):
    """Quantize to [0, 2^bits-1] over [lo, hi]. Returns (q, lo, hi)."""
    v = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite values cannot be quantized")
    if lo is None:
        lo = float(v.min())
    if hi is None:
        hi = float(v.max())
    if lo > hi:
        raise ValueError(f"bad range [{lo}, {hi}]")
    levels = (1 << bits) - 1
    if hi == lo:
        return np.zeros(v.shape, dtype=bits_dtype(bits)), lo, hi
    step = (hi - lo) / levels
    q = np.clip(np.round((v - lo) / step), 0, levels).astype(bits_dtype(bits))
    return q, float(lo), float(hi)


def dequantize_values(q, lo, hi, bits):
    v = np.asarray(q, dtype=np.float64)
    if hi == lo:
        return np.full(v.shape, float(lo))
    step = (hi - lo) / ((1 << bits) - 1)
    return v * step + float(lo)


def quantize_rotations(rot, bits):
    """Normalize quaternions, then quantize components in [-1, 1]."""
    qn = renormalize_quaternions(np.asarray(rot, dtype=np.float64))
    q, lo, hi = quantize_values(qn, bits, lo=QUAT_LO, hi=QUAT_HI)
    return q, lo, hi, qn


def dequantize_rotations(q, lo, hi, bits):
    vals = dequantize_values(q, lo, hi, bits)
    return renormalize_quaternions(vals)


def encode_attributes(v, opts):
    """Quantize every property of a structured vertex array.

    Returns (arrays, plan, original_f64): arrays[name] = uint8/uint16,
    plan[name] = {bits, min, max}, original[name] = float64 copy (the
    quantization target: for rot_* components that is the
    renormalized quaternion group).
    """
    opts = validate_opts(opts)
    require_properties(v)
    arrays, plan, original = {}, {}, {}
    rot_names = sorted(n for n in v.dtype.names if n.startswith("rot_"))
    rot_target = None
    if rot_names:
        mat = np.stack([v[n] for n in rot_names], axis=1).astype(np.float64)
        rot_target = renormalize_quaternions(mat)
    for name in v.dtype.names:
        bits = bits_for_property(name, opts)
        if rot_names and name in rot_names:
            col = rot_target[:, rot_names.index(name)]
            q, lo, hi = quantize_values(col, bits, lo=QUAT_LO, hi=QUAT_HI)
            target = col
        else:
            q, lo, hi = quantize_values(v[name], bits)
            target = np.asarray(v[name], dtype=np.float64)
        arrays[name] = q
        plan[name] = {"bits": bits, "min": lo, "max": hi}
        original[name] = target
    return arrays, plan, original


def decode_attributes(arrays, plan, names, count):
    """Rebuild a float32 structured array from quantized arrays."""
    out = np.zeros(count, dtype=np.dtype([(n, "<f4") for n in names]))
    rot_names = sorted(n for n in names if n.startswith("rot_"))
    for n in names:
        if n not in arrays:
            raise ValueError(f"archive is missing property array: {n}")
        if n not in plan:
            raise ValueError(f"archive is missing plan entry: {n}")
        p = plan[n]
        q = arrays[n]
        if len(q) != count:
            raise ValueError(f"property {n}: expected {count} values, "
                             f"got {len(q)}")
        if not (rot_names and n in rot_names):
            out[n] = dequantize_values(q, p["min"], p["max"],
                                       p["bits"]).astype(np.float32)
    if rot_names:
        cols = [dequantize_values(arrays[n], plan[n]["min"], plan[n]["max"],
                                  plan[n]["bits"]) for n in rot_names]
        mat = renormalize_quaternions(np.stack(cols, axis=1))
        for i, n in enumerate(rot_names):
            out[n] = mat[:, i].astype(np.float32)
    return out


def validate_roundtrip(original, decoded, plan):
    """Per-attribute error metrics of decoded vs original (float64 math)."""
    per_attribute = {}
    within_step = True
    for n in original:
        o = original[n]
        d = decoded[n].astype(np.float64)
        diff = np.abs(o - d)
        p = plan[n]
        step = ((p["max"] - p["min"]) / ((1 << p["bits"]) - 1)
                if p["max"] > p["min"] else 0.0)
        # the PLY stores float32, so the comparison itself can be off
        # by half an ulp of the decoded magnitudes
        half_ulp = float(np.spacing(np.abs(decoded[n]).max())) / 2.0
        if step <= 0:
            bound = half_ulp
        elif n.startswith("rot_"):
            # renormalizing the quantized 4-vector can amplify a
            # per-component quantization error delta = step/2 up to
            # 3*delta / (1 - 2*delta); see validate_roundtrip docstring
            delta = step / 2.0
            bound = 3.0 * delta / (1.0 - 2.0 * delta) + half_ulp
        else:
            bound = step / 2.0 * (1.0 + 1e-5) + 1e-12 + half_ulp
        ok = bool(diff.max() <= bound) if step > 0 else bool(diff.max() <= bound)
        within_step = within_step and ok
        per_attribute[n] = {"bits": p["bits"],
                            "min": p["min"], "max": p["max"],
                            "step": step,
                            "bound": bound,
                            "max_abs_err": float(diff.max()),
                            "mean_abs_err": float(diff.mean()),
                            "within_step_bound": ok}

    pos_err = np.concatenate([np.abs(original[k] - decoded[k].astype(np.float64))
                              for k in ("x", "y", "z")])
    extent = np.array([float(original[k].max() - original[k].min())
                       for k in ("x", "y", "z")])
    diag = float(np.linalg.norm(extent))
    axis_max = [float(np.abs(original[k] - decoded[k].astype(np.float64)).max())
                for k in ("x", "y", "z")]
    position = {
        "extent_xyz": [float(x) for x in extent],
        "diagonal": diag,
        "max_abs_err": float(pos_err.max()),
        "mean_abs_err": float(pos_err.mean()),
        "max_abs_err_per_axis": axis_max,
        "max_rel_to_axis_extent": [float(a / e) if e > 0 else 0.0
                                   for a, e in zip(axis_max, extent)],
        "max_rel_to_diagonal": float(pos_err.max() / diag) if diag > 0 else 0.0,
    }

    rot_names = [n for n in plan if n.startswith("rot_")]
    quat = None
    if rot_names:
        raw = np.stack([original[n] for n in sorted(rot_names)], axis=1)
        dec = np.stack([decoded[n].astype(np.float64)
                        for n in sorted(rot_names)], axis=1)
        pre = np.linalg.norm(dec, axis=1)
        post = np.linalg.norm(renormalize_quaternions(dec), axis=1)
        comp_err = np.abs(raw - dec)
        quat = {
            "pre_renorm_norm_min": float(pre.min()),
            "pre_renorm_norm_max": float(pre.max()),
            "pre_renorm_norm_mean": float(pre.mean()),
            "post_renorm_norm_min": float(post.min()),
            "post_renorm_norm_max": float(post.max()),
            "max_component_err": float(comp_err.max()),
            "mean_component_err": float(comp_err.mean()),
        }

    finite = bool(all(np.isfinite(decoded[n]).all() for n in decoded.dtype.names))
    return {"per_attribute": per_attribute,
            "within_step_bound": within_step,
            "position": position,
            "quaternion": quat,
            "finite": finite}


def encode_ply(src_path, npz_path, opts=None):
    """Quantize a PLY into a compressed .npz archive. Returns stats."""
    opts = validate_opts(opts or DEFAULT_OPTS)
    ply = PlyData.read(str(src_path), mmap=None)
    v = ply["vertex"].data
    names = list(v.dtype.names)
    arrays, plan, original = encode_attributes(v, opts)

    meta = {"format": FORMAT,
            "source": Path(src_path).name,
            "count": int(len(v)),
            "properties": names,
            "plan": plan,
            "comments": list(ply.comments)}
    payload = {META_KEY: np.frombuffer(json.dumps(meta).encode("utf-8"),
                                       dtype=np.uint8)}
    payload.update(arrays)
    np.savez_compressed(str(npz_path), **payload)

    decoded = decode_attributes(arrays, plan, names, len(v))
    report = validate_roundtrip(original, decoded, plan)
    return {"count": int(len(v)), "properties": names,
            "plan": plan, "report": report,
            "comments": list(ply.comments)}


def decode_ply(npz_path, dst_path):
    """Decode an archive back to a float32 PLY. Returns stats."""
    with np.load(str(npz_path), allow_pickle=False) as z:
        if META_KEY not in z.files:
            raise ValueError("not a 3dgs quantization archive "
                             f"(missing {META_KEY})")
        meta = json.loads(bytes(z[META_KEY]).decode("utf-8"))
        if meta.get("format") != FORMAT:
            raise ValueError(f"unsupported archive format: "
                             f"{meta.get('format')!r}")
        names = meta["properties"]
        plan = meta["plan"]
        count = meta["count"]
        arrays = {n: z[n] for n in names if n in z.files}
        if len(arrays) != len(names):
            missing = [n for n in names if n not in arrays]
            raise ValueError(f"archive missing arrays: {missing[:5]}")

    out = decode_attributes(arrays, plan, names, count)
    comments = list(meta.get("comments", []))
    ply = PlyData([PlyElement.describe(out, "vertex")],
                  text=False, byte_order="<", comments=comments)
    ply.write(str(dst_path))
    return {"count": count, "properties": names, "comments": comments,
            "source": meta.get("source", "?")}
