"""Normalisation core: centre, scale, optional upright rotation.

Transform (forward), applied to a Brush PLY:

    p_out           = s * R * (p_in - c)          centre c, rotation R, scale s
    log(scale)_out  = log(scale)_in + ln(s)       uniform ellipsoid scale
    rot_out         = q_R (x) rot_in              unit quaternions, rot_0 = w
    SH              rotated with sh_rotate so colour_out(d) = colour_in(R^T d)
    opacity, f_dc   unchanged (f_dc is direction independent)

The full 4x4 matrix (p_out = A p_in + b with A = sR, b = -sR c) and its
inverse are stored in a mandatory sidecar JSON so the desktop app can
map between original and normalized coordinates and undo the transform.

Evidence order for every automatic choice (design rule: report method
and confidence, never hard-code a scene number):

    centre:  manual --center > COLMAP ray intersection > PLY subject
             density (analyze_scene), falling back to the robust centre
    up:      manual --up-axis > average COLMAP camera up > dominant
             plane normal with the majority-mass sign (analyze_scene)
    scale:   manual --scale / --no-scale > subject radius p90 -> target
             radius, falling back to half the robust extent

Input is never modified; outputs are never overwritten.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from optimization import analyze_scene
from optimization.normalize import colmap, sh_rotate

POS_KEYS = ("x", "y", "z")
SCALE_KEYS = ("scale_0", "scale_1", "scale_2")
ROT_KEYS = ("rot_0", "rot_1", "rot_2", "rot_3")
DC_KEYS = ("f_dc_0", "f_dc_1", "f_dc_2")
REQUIRED = POS_KEYS + DC_KEYS + ("opacity",) + SCALE_KEYS + ROT_KEYS

TOOL = "optimization.normalize"
TOOL_VERSION = "1"

LOW_SUBJECT_CONFIDENCE = analyze_scene.LOW_SUBJECT_CONFIDENCE   # 0.35
MIN_PLANE_CONFIDENCE = analyze_scene.MIN_PLANE_CONFIDENCE       # 0.30
MIN_COLMAP_UP = 0.30      # mean resultant length: below = refuse
WARN_COLMAP_UP = 0.50     # mean resultant length: below = warn
UP_WARNING_CONFIDENCE = 0.35
TARGET_UP = np.array([0.0, 1.0, 0.0])   # Brush "Vertical axis: y"

# Round trip (apply then undo through the JSON sidecar) must stay near
# float32 precision, relative to each attribute group's data scale.
# A wrong inverse would show up at O(1) relative error.
ROUNDTRIP_REL_TOLERANCE = 1e-5

CONVENTION = (
    "p_out = s * R * (p_in - c); log(scale)_out = log(scale)_in + ln(s); "
    "rot_out = q_R (x) rot_in (unit quaternions, rot_0 = w); "
    "SH rotated so colour_out(d) = colour_in(R^T d); opacity and f_dc "
    "unchanged; matrix maps p_in -> p_out, matrix_inv maps back"
)


class StageError(Exception):
    """Stage failure (verification, writing) - exit code 3."""


# ----------------------------------------------------------------- reading

def read_vertex(path):
    """Validate and read a Brush PLY. Returns (PlyData, vertex array)."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"PLY not found: {path}")
    if path.suffix.lower() != ".ply":
        raise ValueError(f"not a .ply file: {path}")
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = v.dtype.names or ()
    missing = [k for k in REQUIRED if k not in names]
    if missing:
        raise ValueError(f"PLY is missing core properties: {missing}")
    if len(v) == 0:
        raise ValueError("PLY has no Gaussians")
    xyz = np.stack([v[k] for k in POS_KEYS], axis=1)
    if not np.isfinite(xyz).all():
        raise ValueError("PLY contains non-finite positions")
    return ply, v


def sh_degree_of(v):
    """SH degree from the f_rest property count (0/9/24/45)."""
    names = v.dtype.names or ()
    count = sum(1 for n in names if n.startswith("f_rest_"))
    for degree, expected in ((0, 0), (1, 9), (2, 24), (3, 45)):
        if count == expected:
            return degree
    raise ValueError(
        f"unexpected f_rest property count {count} (expected 0/9/24/45)")


# --------------------------------------------------------- matrix helpers

def rodrigues(axis, angle):
    k = np.asarray(axis, dtype=np.float64)
    kx, ky, kz = k
    kk = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return (np.eye(3) + math.sin(angle) * kk
            + (1.0 - math.cos(angle)) * (kk @ kk))


def rotation_between(a, b):
    """Rotation matrix taking unit vector a to unit vector b."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cosang = float(np.clip(a @ b, -1.0, 1.0))
    if cosang >= 1.0 - 1e-12:
        return np.eye(3)
    if cosang <= -1.0 + 1e-12:
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 \
            else np.array([0.0, 1.0, 0.0])
        axis = helper - (helper @ a) * a
        return rodrigues(axis / np.linalg.norm(axis), math.pi)
    axis = np.cross(a, b)
    return rodrigues(axis / np.linalg.norm(axis), math.acos(cosang))


def mat_to_quat(r):
    """3x3 rotation -> unit quaternion (w, x, y, z)."""
    r = np.asarray(r, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x = 0.25 * s, (r[2, 1] - r[1, 2]) / s
        y, z = (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w, x = (r[2, 1] - r[1, 2]) / s, 0.25 * s
        y, z = (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w, x = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s
        y, z = 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w, x = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s
        y, z = (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    """Hamilton product, last axis is (w, x, y, z); broadcasts."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def is_identity(r, atol=1e-12):
    return bool(np.allclose(np.asarray(r, dtype=np.float64),
                            np.eye(3), atol=atol))


# ------------------------------------------------------------ plan build

def _ply_subject(rep, warnings):
    """Subject centre from analyze_scene, robust-centre fallback."""
    sub = rep["subject"]
    robust = np.array(rep["centre"]["xyz"], dtype=np.float64)
    if sub.get("radius_p90") is None:
        warnings.append("no subject candidate (too few compact bright "
                        "Gaussians); used the robust centre")
        return robust, float(sub["confidence"]), \
            "robust centre (no subject candidate)", sub
    if sub["low_confidence"]:
        warnings.append(
            f"subject confidence {sub['confidence']:.3f} < "
            f"{LOW_SUBJECT_CONFIDENCE}; fell back to the robust centre")
        return (robust, float(sub["confidence"]),
                "robust centre (subject confidence too low)", sub)
    return (np.array(sub["centre"], dtype=np.float64),
            float(sub["confidence"]),
            "subject density centre (k-NN of compact bright Gaussians, "
            "dominant plane excluded)", sub)


def _frame_check(point, rep, warnings, label):
    """Warn when an external centre sits outside the PLY robust box."""
    if rep is None:
        return
    lo = np.array(rep["extent"]["percentile_box_min"], dtype=np.float64)
    hi = np.array(rep["extent"]["percentile_box_max"], dtype=np.float64)
    outside = np.linalg.norm(np.maximum(lo - point, 0.0)
                             + np.maximum(point - hi, 0.0))
    extent = float(rep["extent"]["robust_extent"])
    if outside > extent:
        warnings.append(
            f"{label} lies {outside:.4g} outside the PLY robust box "
            f"(robust extent {extent:.4g}); the PLY and the external "
            f"source may be in different coordinate frames")


def compute_plan(path, *, v=None, center=None, colmap_dir=None,
                 upright=False, up_axis=None, target_radius=1.0,
                 scale=None, no_scale=False):
    """Build the normalisation plan for one PLY (never writes files).

    Returns a dict with numpy arrays (center, R, up) plus methods,
    confidence, warnings and SH verification. Raises ValueError for
    user-fixable problems (exit 2) and StageError when the SH rotation
    verification fails (exit 3).
    """
    warnings = []
    methods = {}
    confidence = {}
    unit = ""

    need_scale = (not no_scale) and scale is None
    need_plane_up = ((upright or up_axis is not None)
                     and up_axis is None and colmap_dir is None)
    need_analyze = center is None or need_scale or need_plane_up
    rep = analyze_scene.analyze(path) if need_analyze else None

    sparse = None
    if colmap_dir is not None:
        sparse = colmap.read_sparse(colmap_dir)

    # ---- centre -------------------------------------------------------
    if center is not None:
        c = np.asarray(center, dtype=np.float64)
        methods["center"] = "manual --center"
        confidence["center"] = 1.0
        sub = None
    elif sparse is not None:
        centers, dirs, _ = colmap.poses(sparse["images"])
        try:
            c, c_conf, info = colmap.intersect_rays(centers, dirs)
            methods["center"] = (
                f"COLMAP least-squares intersection of {info['cameras']} "
                f"viewing axes (mean perpendicular offset "
                f"{info['mean_perpendicular_offset']:.4g})")
            confidence["center"] = c_conf
            _frame_check(c, rep, warnings, "COLMAP centre")
            sub = None
        except ValueError as err:
            warnings.append(f"COLMAP centre unavailable ({err}); "
                            f"fell back to the PLY centre")
            c, c_conf, c_method, sub = _ply_subject(rep, warnings)
            methods["center"] = c_method
            confidence["center"] = c_conf
    else:
        c, c_conf, c_method, sub = _ply_subject(rep, warnings)
        methods["center"] = c_method
        confidence["center"] = c_conf

    # scale still wants the subject even when the centre came from
    # COLMAP or --center
    if need_scale and sub is None and rep is not None:
        sub = rep["subject"]

    # ---- up / rotation ------------------------------------------------
    r_mat = np.eye(3)
    up = None
    up_angle = None
    if up_axis is not None:
        up = np.asarray(up_axis, dtype=np.float64)
        norm = float(np.linalg.norm(up))
        if norm <= 0:
            raise ValueError("--up-axis must be a non-zero vector")
        up = up / norm
        methods["upright"] = "manual --up-axis"
        confidence["upright"] = 1.0
    elif upright:
        if sparse is not None:
            _, _, ups = colmap.poses(sparse["images"])
            up, resultant = colmap.average_up(ups)
            if resultant < MIN_COLMAP_UP:
                raise ValueError(
                    "average camera up is unreliable "
                    f"(resultant {resultant:.3f} < {MIN_COLMAP_UP:g}); "
                    "pass --up-axis X Y Z")
            if resultant < WARN_COLMAP_UP:
                warnings.append(
                    f"average camera up is only moderately consistent "
                    f"(resultant {resultant:.3f} < {WARN_COLMAP_UP:g}); "
                    f"cameras may be rolled or spread")
            methods["upright"] = (
                f"average COLMAP camera up ({len(ups)} cameras, resultant "
                f"{resultant:.3f})")
            confidence["upright"] = float(resultant)
        else:
            plane = rep["plane"]
            upinfo = rep["up"]
            if (upinfo["vector"] is None
                    or plane.get("confidence", 0.0) < MIN_PLANE_CONFIDENCE):
                got = float(plane.get("confidence", 0.0))
                raise ValueError(
                    f"no reliable floor for --upright (plane confidence "
                    f"{got:.3f} < {MIN_PLANE_CONFIDENCE:g}); pass "
                    f"--up-axis X Y Z to set orientation manually")
            up = np.array(upinfo["vector"], dtype=np.float64)
            confidence["upright"] = float(upinfo["confidence"])
            share = upinfo["mass_share_on_up_side"]
            methods["upright"] = (
                f"dominant plane normal, sign from majority scene mass "
                f"({share * 100:.1f}% on the up side)")
            if confidence["upright"] < UP_WARNING_CONFIDENCE:
                warnings.append(
                    f"up confidence {confidence['upright']:.3f} < "
                    f"{UP_WARNING_CONFIDENCE}: the sign rests on a weak "
                    f"mass split, so the scene may end up flipped 180 "
                    f"deg; verify in Brush or pass --up-axis")
    if up is not None:
        r_mat = rotation_between(up, TARGET_UP)
        if is_identity(r_mat):
            r_mat = np.eye(3)
            up_angle = 0.0
        else:
            up_angle = float(np.degrees(np.arccos(np.clip(
                float(up @ TARGET_UP), -1.0, 1.0))))
        methods.setdefault("upright", "up vector")

    # ---- scale --------------------------------------------------------
    if no_scale:
        s = 1.0
        methods["scale"] = "disabled (--no-scale)"
        confidence["scale"] = None
        unit = "scale disabled (input units kept)"
    elif scale is not None:
        s = float(scale)
        if not math.isfinite(s) or s <= 0:
            raise ValueError("--scale must be a finite number > 0")
        methods["scale"] = f"manual --scale {s:g}"
        confidence["scale"] = 1.0
        unit = f"manual scale factor {s:g}"
    else:
        if rep is None:
            raise ValueError("internal: scene analysis required for scale")
        if sub is not None and sub.get("radius_p90") is not None:
            radius = float(sub["radius_p90"])
            radius_kind = "subject radius p90"
        else:
            radius = 0.5 * float(rep["extent"]["robust_extent"])
            radius_kind = "half the robust extent"
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("scene scale reference is not positive")
        s = float(target_radius) / radius
        methods["scale"] = (f"uniform scale so {radius_kind} "
                            f"({radius:.6g}) -> {target_radius:g}")
        confidence["scale"] = (float(sub["confidence"]) if sub is not None
                               else 0.0)
        unit = (f"{radius_kind} = {target_radius:g} in the output "
                f"(input {radius:.6g} x scale {s:.6g})")
        if sub is not None and sub["low_confidence"]:
            warnings.append(
                f"scale reference has low subject confidence "
                f"({sub['confidence']:.3f})")

    # ---- SH rotation verification -------------------------------------
    verification = {
        "sh_rotation_checked": False,
        "sh_rotation_max_error": 0.0,
        "sh_tolerance": float(sh_rotate.SH_TOLERANCE),
        "sh_basis": ("Brush crates/brush-render/src/kernels/sh.rs "
                     "(Sloan JCGT 2013)"),
    }
    if not is_identity(r_mat):
        if v is None:
            _, v = read_vertex(path)
        degree = sh_degree_of(v)
        verification["sh_degree"] = degree
        max_err = 0.0
        for block in _channel_blocks(v, degree):
            err, _ = sh_rotate.verify_rotation(
                r_mat, degree, coeffs=block, n_sets=64)
            max_err = max(max_err, err)
        if max_err > sh_rotate.SH_TOLERANCE:
            raise StageError(
                f"SH rotation verification failed: max colour error "
                f"{max_err:.3e} > {sh_rotate.SH_TOLERANCE:g}; refusing "
                f"to write a rotated scene")
        verification["sh_rotation_checked"] = True
        verification["sh_rotation_max_error"] = float(max_err)

    return {
        "source": str(path),
        "center": c,
        "R": r_mat,
        "s": s,
        "up": up,
        "up_angle_to_brush_y_deg": up_angle,
        "methods": methods,
        "confidence": confidence,
        "warnings": warnings,
        "unit": unit,
        "verification": verification,
    }


# ------------------------------------------------------------- transform

def _channel_blocks(v, degree):
    """3 arrays (n, (degree+1)**2): [f_dc, f_rest...] per channel."""
    nb = (degree + 1) ** 2
    cp = nb - 1
    blocks = []
    for ch in range(3):
        cols = [np.asarray(v[DC_KEYS[ch]], dtype=np.float64)]
        if cp:
            cols += [np.asarray(v[f"f_rest_{ch * cp + i}"],
                                dtype=np.float64) for i in range(cp)]
        blocks.append(np.stack(cols, axis=1))
    return blocks


def _transform(v, A, b, dlog, qR, r_sh):
    """Shared forward/inverse attribute transform (see module docstring).

    A, b: position affine  p_out = A p_in + b
    dlog: added to log(scale)
    qR:   (w,x,y,z) left-factor for rot, or None to skip
    r_sh: rotation matrix used for the SH coefficient matrix, or None
    """
    out = v.copy()
    xyz = np.stack([v[k] for k in POS_KEYS], axis=1).astype(np.float64)
    moved = xyz @ np.asarray(A, dtype=np.float64).T \
        + np.asarray(b, dtype=np.float64)
    for i, key in enumerate(POS_KEYS):
        out[key] = moved[:, i]

    if dlog != 0.0:
        for key in SCALE_KEYS:
            out[key] = v[key] + dlog

    if qR is not None:
        q = np.stack([v[k] for k in ROT_KEYS], axis=1).astype(np.float64)
        rotated = quat_mul(np.broadcast_to(np.asarray(qR, dtype=np.float64),
                                           q.shape), q)
        rotated /= np.linalg.norm(rotated, axis=1, keepdims=True)
        for i, key in enumerate(ROT_KEYS):
            out[key] = rotated[:, i]

    if r_sh is not None and not is_identity(r_sh):
        degree = sh_degree_of(v)
        if degree > 0:
            m = sh_rotate.rotation_matrix(r_sh, degree)
            m_rest = m[1:, 1:]
            cp = (degree + 1) ** 2 - 1
            blocks = _channel_blocks(v, degree)
            for ch in range(3):
                rest_new = blocks[ch][:, 1:] @ m_rest.T
                for i in range(cp):
                    out[f"f_rest_{ch * cp + i}"] = rest_new[:, i]
    return out


def apply_plan(v, plan):
    """Forward transform: original vertex array -> normalised copy."""
    s = float(plan["s"])
    r_mat = np.asarray(plan["R"], dtype=np.float64)
    c = np.asarray(plan["center"], dtype=np.float64)
    A = s * r_mat
    b = -A @ c
    dlog = math.log(s) if s != 1.0 else 0.0
    qR = None if is_identity(r_mat) else mat_to_quat(r_mat)
    r_sh = None if is_identity(r_mat) else r_mat
    return _transform(v, A, b, dlog, qR, r_sh)


def undo_vertex(v, sidecar):
    """Inverse transform from a sidecar dict (as read back from JSON)."""
    matrix = np.asarray(sidecar["matrix_inv"], dtype=np.float64)
    A = matrix[:3, :3]
    b = matrix[:3, 3]
    s = float(sidecar["s"])
    r_mat = np.asarray(sidecar["R"], dtype=np.float64)
    dlog = -math.log(s) if s != 1.0 else 0.0
    qR = None
    r_sh = None
    if not is_identity(r_mat):
        q_forward = mat_to_quat(r_mat)
        qR = np.array([q_forward[0], -q_forward[1], -q_forward[2],
                       -q_forward[3]])
        r_sh = r_mat.T
    return _transform(v, A, b, dlog, qR, r_sh)


def write_vertex(path, v, comments):
    """Write a Brush-style binary PLY with identical comments/order."""
    ply = PlyData([PlyElement.describe(v, "vertex")],
                  text=False, byte_order="<", comments=list(comments))
    ply.write(str(path))


# ---------------------------------------------------------------- sidecar

def build_sidecar(plan, source, output, roundtrip=None):
    """Mandatory sidecar JSON dict (all plain Python types)."""
    s = float(plan["s"])
    r_mat = np.asarray(plan["R"], dtype=np.float64)
    c = np.asarray(plan["center"], dtype=np.float64)
    A = s * r_mat
    b = -A @ c
    A_inv = r_mat.T / s
    b_inv = c
    matrix = np.eye(4)
    matrix[:3, :3] = A
    matrix[:3, 3] = b
    matrix_inv = np.eye(4)
    matrix_inv[:3, :3] = A_inv
    matrix_inv[:3, 3] = b_inv
    verification = dict(plan["verification"])
    if roundtrip is not None:
        verification["roundtrip_max_error"] = roundtrip
    return {
        "tool": TOOL,
        "tool_version": TOOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "source": str(source),
        "output": str(output),
        "convention": CONVENTION,
        "unit": plan["unit"],
        "matrix": matrix.tolist(),
        "matrix_inv": matrix_inv.tolist(),
        "R": r_mat.tolist(),
        "R_inv": r_mat.T.tolist(),
        "t": b.tolist(),
        "s": s,
        "center": c.tolist(),
        "up": (None if plan["up"] is None
               else [float(x) for x in plan["up"]]),
        "up_angle_to_brush_y_deg": plan["up_angle_to_brush_y_deg"],
        "methods": dict(plan["methods"]),
        "confidence": {k: (None if val is None else float(val))
                       for k, val in plan["confidence"].items()},
        "warnings": list(plan["warnings"]),
        "verification": verification,
    }


def sidecar_to_transform(sidecar):
    """Validate a sidecar dict; returns the arrays undo_vertex needs."""
    for key in ("matrix_inv", "s", "R"):
        if key not in sidecar:
            raise ValueError(f"sidecar is missing '{key}'")
    s = float(sidecar["s"])
    if not math.isfinite(s) or s <= 0:
        raise ValueError("sidecar scale s must be a finite number > 0")
    matrix = np.asarray(sidecar["matrix_inv"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("sidecar matrix_inv must be 4x4")
    r_mat = np.asarray(sidecar["R"], dtype=np.float64)
    if r_mat.shape != (3, 3):
        raise ValueError("sidecar R must be 3x3")
    return sidecar


def roundtrip_errors(v, plan, sidecar):
    """Max |apply then undo - original| per attribute group.

    Goes through the JSON sidecar values so serialization precision is
    part of the test.
    """
    applied = apply_plan(v, plan)
    restored = undo_vertex(applied, sidecar)
    errors = {}

    def group(keys):
        worst = 0.0
        for key in keys:
            delta = np.abs(np.asarray(restored[key], dtype=np.float64)
                           - np.asarray(v[key], dtype=np.float64))
            worst = max(worst, float(delta.max()) if len(delta) else 0.0)
        return worst

    errors["position"] = group(POS_KEYS)
    errors["scale"] = group(SCALE_KEYS)
    errors["rot"] = group(ROT_KEYS)
    errors["opacity"] = group(("opacity",))
    errors["f_dc"] = group(DC_KEYS)
    rest = tuple(k for k in (v.dtype.names or ())
                 if k.startswith("f_rest_"))
    errors["f_rest"] = group(rest) if rest else 0.0
    return errors


def roundtrip_worst_relative(v, rt):
    """Worst round-trip error relative to each group's data scale.

    Positions are scaled by the scene size (bbox diagonal of the
    ranges), everything else by max(1, max |value|) of its own fields,
    so the check is independent of scene units.
    """
    xyz = np.stack([np.asarray(v[k], dtype=np.float64)
                    for k in POS_KEYS], axis=1)
    pos_scale = max(1.0, float(np.linalg.norm(np.ptp(xyz, axis=0))))
    worst = float(rt["position"]) / pos_scale

    def value_scale(keys):
        vals = np.concatenate([
            np.asarray(v[k], dtype=np.float64).ravel() for k in keys])
        if not vals.size:
            return 1.0
        return max(1.0, float(np.max(np.abs(vals))))

    for name, keys in (("scale", SCALE_KEYS), ("rot", ROT_KEYS),
                       ("opacity", ("opacity",)), ("f_dc", DC_KEYS)):
        worst = max(worst, float(rt[name]) / value_scale(keys))
    rest = tuple(k for k in (v.dtype.names or ())
                 if k.startswith("f_rest_"))
    if rest:
        worst = max(worst, float(rt["f_rest"]) / value_scale(rest))
    return worst


# ---------------------------------------------------------------- report

def format_report(plan, sidecar=None, rt=None, sidecar_path=None):
    lines = []
    add = lines.append
    add(f"Normalise: {plan['source']}")
    c = np.asarray(plan["center"], dtype=np.float64)
    add(f"  centre    ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})  "
        f"confidence {plan['confidence'].get('center', float('nan')):.3f}")
    add(f"            method: {plan['methods'].get('center')}")
    add(f"  scale     s = {plan['s']:.6g}  ({plan['unit']})")
    add(f"            method: {plan['methods'].get('scale')}")
    if plan["up"] is None:
        add("  rotate    none (identity; SH untouched)")
    else:
        up = np.asarray(plan["up"], dtype=np.float64)
        angle = plan["up_angle_to_brush_y_deg"]
        add(f"  rotate    up ({up[0]:.4f}, {up[1]:.4f}, {up[2]:.4f}) "
            f"-> Brush +y, angle {angle:.2f} deg, "
            f"confidence {plan['confidence'].get('upright', float('nan')):.3f}")
        add(f"            method: {plan['methods'].get('upright')}")
    ver = plan["verification"]
    if ver.get("sh_rotation_checked"):
        add(f"  SH verify max colour error "
            f"{ver['sh_rotation_max_error']:.3e} over sample directions "
            f"(degree {ver.get('sh_degree')}, tolerance "
            f"{ver['sh_tolerance']:g}, Brush basis)")
    else:
        add("  SH verify not needed (identity rotation)")
    if rt is not None:
        add("  round-trip (apply then undo via sidecar), max |error|:")
        add("            position {position:.3e}  scale {scale:.3e}  "
            "rot {rot:.3e}  opacity {opacity:.3e}  "
            "f_dc {f_dc:.3e}  f_rest {f_rest:.3e}".format(**rt))
    if sidecar_path is not None:
        add(f"  sidecar   {sidecar_path} written "
            f"(matrix + matrix_inv + methods + confidence)")
    for warning in plan["warnings"]:
        add(f"  WARNING   {warning}")
    return lines
