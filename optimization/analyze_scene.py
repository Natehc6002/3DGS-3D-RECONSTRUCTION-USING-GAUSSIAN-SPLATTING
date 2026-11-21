"""Read-only scene diagnostics for any Gaussian splat PLY (Brush export).

Everything here is scene-relative: no threshold is a fixed number of
scene units, because COLMAP scale/origin/orientation are arbitrary per
project. Thresholds are percentiles of the scene itself, ratios of the
robust scene extent, or ratios of per-Gaussian sizes.

Reports:
    - robust centre (opacity- and size-weighted coordinate-wise median)
    - robust extent (p1..p99 percentile box diagonal) vs raw bbox
    - dominant plane (RANSAC on compact high-opacity Gaussians)
    - estimated up direction with confidence
    - subject candidate centre/radius with confidence (low when the
      scene has no clear subject - it never crashes on that)
    - needle count (anisotropy ratio AND size outliers)
    - Gaussians below the dominant plane by a margin (candidate mirror
      content)

The script never writes files (redirect stdout yourself if you want a
file) and never modifies the input.

Usage:
    python -m optimization.analyze_scene --ply scene.ply [--json]

Exit codes: 0 success, 2 bad arguments/paths.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

POS_KEYS = ("x", "y", "z")
SCALE_KEYS = ("scale_0", "scale_1", "scale_2")
REQUIRED = POS_KEYS + ("opacity",) + SCALE_KEYS

TOOL = "optimization.analyze_scene"
TOOL_VERSION = "1"

# All of these are RATIOS or percentiles (scene-relative), never
# absolute scene units.
PERCENTILE_LO, PERCENTILE_HI = 1.0, 99.0   # robust extent box
PLANE_TOL_RATIO = 0.01                      # inlier distance / extent
PLANE_ANISO_PCT = 90.0                      # compact candidate cutoff
NEEDLE_ANISO_RATIO = 8.0                    # max/min axis floor ratio
NEEDLE_ANISO_PCT = 99.0                     # anisotropy outlier pct
NEEDLE_SIZE_PCT = 99.0                      # size outlier pct
SUBJECT_INNER_RATIO = 0.15                  # inner radius / extent
SUBJECT_INNER_TARGET = 0.5                  # share giving confidence 1
SUBJECT_MAX_RADIUS_RATIO = 0.35             # radius/extent = score 0
BELOW_MARGIN_RATIO = 0.02                   # mirror margin / extent
RANSAC_ITERATIONS = 300                     # algorithm constant
RANSAC_SEED = 0
MIN_PLANE_CONFIDENCE = 0.30                 # below this: no mirror count
MIN_PLANE_CANDIDATES = 20
LOW_SUBJECT_CONFIDENCE = 0.35


def sigmoid(x):
    """Overflow-safe logistic function (opacity logit -> (0, 1))."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def clip01(x):
    return float(min(1.0, max(0.0, x)))


def weighted_median(values, weights):
    """Weighted median of 1-D samples (robust to outliers)."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    total = float(cw[-1])
    if not np.isfinite(total) or total <= 0:
        return float(np.median(values))
    idx = int(np.searchsorted(cw, 0.5 * total, side="left"))
    return float(v[min(idx, len(v) - 1)])


def weighted_percentile(values, weights, pct):
    """Weighted percentile (linear-free, nearest-rank on cumsum)."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    total = float(cw[-1])
    if not np.isfinite(total) or total <= 0:
        return float(np.percentile(values, pct))
    target = clip_pct(pct) / 100.0 * total
    idx = int(np.searchsorted(cw, target, side="left"))
    return float(v[min(idx, len(v) - 1)])


def clip_pct(pct):
    return float(min(100.0, max(0.0, pct)))


def load_scene(path):
    """Read a PLY and return scene arrays. Raises ValueError on purpose."""
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

    xyz = np.stack([v[k] for k in POS_KEYS], axis=1).astype(np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("PLY contains non-finite positions")
    axes = np.exp(np.stack([v[k] for k in SCALE_KEYS], axis=1)
                  .astype(np.float64))
    op = sigmoid(v["opacity"])
    volume = (4.0 / 3.0) * np.pi * np.prod(axes, axis=1)
    weights = op * volume
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        raise ValueError("PLY has no positive weight (opacity x volume)")
    return {"path": str(path), "xyz": xyz, "axes": axes, "opacity": op,
            "weights": weights, "count": len(xyz)}


def bounding_box(xyz):
    return xyz.min(axis=0), xyz.max(axis=0)


def percentile_box(xyz, lo=PERCENTILE_LO, hi=PERCENTILE_HI):
    return (np.percentile(xyz, lo, axis=0),
            np.percentile(xyz, hi, axis=0))


def box_diagonal(lo, hi):
    return float(np.linalg.norm(np.asarray(hi, dtype=np.float64)
                                - np.asarray(lo, dtype=np.float64)))


def robust_centre(xyz, weights):
    """Opacity- and size-weighted coordinate-wise median."""
    return np.array([weighted_median(xyz[:, i], weights) for i in range(3)])


def robust_extent(xyz):
    """Diagonal of the p1..p99 percentile box (scene-relative scale)."""
    lo, hi = percentile_box(xyz)
    return box_diagonal(lo, hi)


def anisotropy(axes):
    """max/min semi-axis ratio per Gaussian (shape, scale-invariant)."""
    axes = np.asarray(axes, dtype=np.float64)
    hi = axes.max(axis=1)
    lo = np.maximum(axes.min(axis=1), np.finfo(np.float64).tiny)
    return hi / lo


def fit_plane(xyz, axes, opacity, extent,
              iterations=RANSAC_ITERATIONS, seed=RANSAC_SEED):
    """Dominant plane by RANSAC over compact high-opacity Gaussians.

    Candidates: opacity >= scene median AND anisotropy <= scene p90
    (both scene-relative). Inlier tolerance: PLANE_TOL_RATIO * extent.
    The best model is refined by PCA on its inliers. Confidence is
    inlier_fraction * normal_stability, both in [0, 1].
    """
    n = len(xyz)
    aniso = anisotropy(axes)
    op_med = float(np.median(opacity))
    aniso_cut = float(np.percentile(aniso, PLANE_ANISO_PCT))
    cand = (opacity >= op_med) & (aniso <= aniso_cut)
    cand_idx = np.nonzero(cand)[0]

    tol = PLANE_TOL_RATIO * extent
    empty = {"normal": None, "point": None, "inlier_fraction": 0.0,
             "confidence": 0.0, "tolerance": float(tol),
             "candidate_count": int(len(cand_idx)), "inlier_count": 0,
             "normal_stability": 0.0,
             "method": "ransac on opacity>=median and anisotropy<=p90"}
    if len(cand_idx) < MIN_PLANE_CANDIDATES:
        empty["method"] = (f"insufficient candidates "
                           f"({len(cand_idx)} < {MIN_PLANE_CANDIDATES})")
        return empty

    pts = xyz[cand_idx]
    rng = np.random.default_rng(seed)
    best_count = -1
    best_normal = None
    best_point = None
    good_normals = []

    for _ in range(iterations):
        pick = rng.choice(len(pts), size=3, replace=False)
        p0, p1, p2 = pts[pick]
        nrm = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nrm)
        if norm <= 1e-12:
            continue
        nrm = nrm / norm
        d = -float(nrm @ p0)
        dist = np.abs(pts @ nrm + d)
        count = int((dist <= tol).sum())
        if count > best_count:
            if best_normal is not None:
                good_normals.append(best_normal)
            best_count = count
            best_normal = nrm
            best_point = p0
        elif best_normal is not None and count >= 0.9 * best_count:
            good_normals.append(nrm)

    if best_normal is None or best_count < 3:
        empty["method"] = "ransac found no valid plane model"
        return empty

    # Refine on the inliers of the best model (PCA smallest eigenvector).
    d = -float(best_normal @ best_point)
    inliers = np.abs(pts @ best_normal + d) <= tol
    if int(inliers.sum()) >= 3:
        ip = pts[inliers]
        centroid = ip.mean(axis=0)
        cov = np.cov((ip - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        refined = eigvecs[:, int(np.argmin(eigvals))]
        refined = refined / np.linalg.norm(refined)
        if float(refined @ best_normal) < 0:
            refined = -refined
        d = -float(refined @ centroid)
        inliers = np.abs(pts @ refined + d) <= tol
        best_normal, best_point = refined, centroid

    inlier_fraction = int(inliers.sum()) / float(len(pts))
    if good_normals:
        dots = np.abs([float(g @ best_normal) for g in good_normals])
        stability = float(np.mean(dots))
    else:
        stability = 1.0
    confidence = clip01(inlier_fraction * stability)
    return {"normal": [float(x) for x in best_normal],
            "point": [float(x) for x in best_point],
            "inlier_fraction": float(inlier_fraction),
            "confidence": confidence,
            "tolerance": float(tol),
            "candidate_count": int(len(pts)),
            "inlier_count": int(inliers.sum()),
            "normal_stability": float(stability),
            "method": ("ransac (opacity>=median, anisotropy<=p90) "
                       "refined by PCA on inliers")}


def signed_distances(xyz, plane):
    if plane.get("normal") is None:
        return None
    n = np.asarray(plane["normal"], dtype=np.float64)
    p = np.asarray(plane["point"], dtype=np.float64)
    return (xyz - p) @ n


def estimate_up(xyz, weights, plane):
    """Up direction from the dominant plane; sign from the mass side.

    COLMAP y points down, the Brush header says "Vertical axis: y"; we
    do NOT assume either here - the sign is chosen so that most of the
    scene weight sits on the positive (up) side, which is what a floor
    looks like. Confidence = plane confidence * side balance.
    """
    dist = signed_distances(xyz, plane)
    if dist is None or plane.get("confidence", 0.0) <= 0:
        return {"vector": None, "confidence": 0.0,
                "angle_to_brush_y_deg": None,
                "mass_share_on_up_side": None,
                "method": "no reliable dominant plane"}
    total = float(weights.sum())
    share_pos = float(weights[dist > 0].sum() / total)
    n = np.asarray(plane["normal"], dtype=np.float64)
    if share_pos < 0.5:
        n = -n
        share_up = 1.0 - share_pos
    else:
        share_up = share_pos
    side_balance = 2.0 * (share_up - 0.5)
    confidence = clip01(plane["confidence"] * side_balance)
    brush_y = np.array([0.0, 1.0, 0.0])
    cosang = clip_abs(float(n @ brush_y))
    angle = float(np.degrees(np.arccos(cosang)))
    return {"vector": [float(x) for x in n],
            "confidence": confidence,
            "angle_to_brush_y_deg": angle,
            "mass_share_on_up_side": float(share_up),
            "method": ("dominant plane normal, sign = majority of scene "
                       "weight above the plane")}


def clip_abs(x):
    return float(min(1.0, max(-1.0, x)))


def detect_subject(xyz, axes, opacity, extent, plane=None):
    """Subject candidate: densest compact high-opacity region.

    Density (k-NN) of candidates, not raw weight: walls and floors hold
    most of the opacity x volume weight, but a subject is where many
    compact bright Gaussians sit close together. Candidates exclude the
    dominant plane, keep opacity >= scene median and anisotropy <= p90.
    Confidence = how much denser the densest quartile is than the rest
    (a uniform scene has contrast ~1 and gets confidence ~0, never a
    crash).
    """
    n = len(xyz)
    keep = np.ones(n, dtype=bool)
    if plane is not None and plane.get("normal") is not None:
        dist_plane = signed_distances(xyz, plane)
        keep &= np.abs(dist_plane) > plane["tolerance"]

    aniso = anisotropy(axes)
    op_med = float(np.median(opacity))
    aniso_cut = float(np.percentile(aniso, PLANE_ANISO_PCT))
    cand = keep & (opacity >= op_med) & (aniso <= aniso_cut)
    idx = np.nonzero(cand)[0]

    global_centre = robust_centre(xyz, opacity * np.prod(axes, axis=1))
    fallback = {"centre": [float(x) for x in global_centre],
                "radius_p90": None, "cluster_count": int(len(idx)),
                "inner_radius": float(SUBJECT_INNER_RATIO * extent),
                "density_contrast": None, "confidence": 0.0,
                "offset_from_global": 0.0, "offset_ratio": 0.0,
                "low_confidence": True,
                "method": "too few candidates for a density estimate; "
                          "fell back to the robust centre"}
    if len(idx) < MIN_PLANE_CANDIDATES:
        return fallback

    pts = xyz[idx]
    op_c = opacity[idx]
    k = min(16, len(pts) - 1)
    dk = cKDTree(pts).query(pts, k=k + 1)[0][:, -1]

    dense = dk <= np.percentile(dk, 25.0)
    if int(dense.sum()) < 3 or int((~dense).sum()) < 3:
        return fallback

    centre = np.array([weighted_median(pts[dense, i], op_c[dense])
                       for i in range(3)])
    d_c = np.linalg.norm(pts[dense] - centre, axis=1)
    radius = float(np.percentile(d_c, 90.0))

    med_dense = float(np.median(dk[dense]))
    med_rest = float(np.median(dk[~dense]))
    contrast = med_rest / max(med_dense, np.finfo(np.float64).tiny)
    contrast_score = clip01((contrast - 1.0) / 2.0)

    offset = float(np.linalg.norm(centre - global_centre))
    radius_ratio = radius / extent if extent > 0 else 1.0
    compactness_score = clip01(1.0 - radius_ratio / SUBJECT_MAX_RADIUS_RATIO)
    confidence = max(contrast_score, compactness_score)
    return {"centre": [float(x) for x in centre],
            "radius_p90": radius,
            "radius_ratio": float(radius_ratio),
            "cluster_count": int(dense.sum()),
            "inner_radius": float(SUBJECT_INNER_RATIO * extent),
            "density_contrast": float(contrast),
            "contrast_score": float(contrast_score),
            "compactness_score": float(compactness_score),
            "confidence": confidence,
            "offset_from_global": offset,
            "offset_ratio": float(offset / extent) if extent > 0 else 0.0,
            "low_confidence": bool(confidence < LOW_SUBJECT_CONFIDENCE),
            "method": (f"k-NN density of compact bright candidates "
                       f"(opacity>=median, anisotropy<=p90, outside the "
                       f"plane); densest quartile centre; confidence = "
                       f"max(contrast score, compactness score) = "
                       f"max({contrast_score:.3f}, {compactness_score:.3f})")}


def count_needles(axes):
    """Needles = anisotropy outliers AND size outliers (both scene p99)."""
    aniso = anisotropy(axes)
    size = axes.max(axis=1)
    aniso_cut = max(NEEDLE_ANISO_RATIO,
                    float(np.percentile(aniso, NEEDLE_ANISO_PCT)))
    size_cut = float(np.percentile(size, NEEDLE_SIZE_PCT))
    aniso_out = aniso >= aniso_cut
    size_out = size >= size_cut
    needles = aniso_out & size_out
    return {"needles": int(needles.sum()),
            "aniso_outliers": int(aniso_out.sum()),
            "size_outliers": int(size_out.sum()),
            "anisotropy_threshold": float(aniso_cut),
            "size_threshold": float(size_cut),
            "median_anisotropy": float(np.median(aniso)),
            "p99_anisotropy": float(np.percentile(aniso, 99.0)),
            "median_max_axis": float(np.median(size)),
            "p99_max_axis": float(np.percentile(size, 99.0)),
            "needle_mask": needles,
            "method": (f"anisotropy >= max({NEEDLE_ANISO_RATIO}, p99) AND "
                       f"max axis >= p99 (both scene-relative)")}


def count_below_plane(xyz, weights, plane, extent):
    """Gaussians below the dominant plane by a scene-relative margin.

    Candidate mirror content. Reported only when the plane confidence is
    at least MIN_PLANE_CONFIDENCE; otherwise it refuses with a reason.
    """
    margin = BELOW_MARGIN_RATIO * extent
    dist = signed_distances(xyz, plane)
    if dist is None or plane.get("confidence", 0.0) < MIN_PLANE_CONFIDENCE:
        return {"count": None, "count_share": None, "weight_share": None,
                "margin": float(margin), "reliable": False,
                "reason": (f"plane confidence "
                           f"{plane.get('confidence', 0.0):.3f} < "
                           f"{MIN_PLANE_CONFIDENCE}; no reliable floor")}
    below = dist < -margin
    return {"count": int(below.sum()),
            "count_share": float(below.mean()),
            "weight_share": float(weights[below].sum() / weights.sum()),
            "margin": float(margin), "reliable": True,
            "reason": "plane confidence above threshold"}


def analyze(path):
    """Full read-only diagnostic. Returns a JSON-serializable dict."""
    scene = load_scene(path)
    xyz, axes = scene["xyz"], scene["axes"]
    weights, opacity = scene["weights"], scene["opacity"]
    n = scene["count"]

    bbox_lo, bbox_hi = bounding_box(xyz)
    pbox_lo, pbox_hi = percentile_box(xyz)
    bbox_diag = box_diagonal(bbox_lo, bbox_hi)
    extent = box_diagonal(pbox_lo, pbox_hi)
    if extent <= 0:
        raise ValueError("scene has zero extent (all positions identical)")

    centre = robust_centre(xyz, weights)
    inner_radius = SUBJECT_INNER_RATIO * extent
    d_global = np.linalg.norm(xyz - centre, axis=1)
    inner_share_global = float(weights[d_global <= inner_radius].sum()
                               / weights.sum())
    centre_confidence = clip01(inner_share_global / SUBJECT_INNER_TARGET)

    plane = fit_plane(xyz, axes, opacity, extent)
    up = estimate_up(xyz, weights, plane)
    subject = detect_subject(xyz, axes, opacity, extent, plane)
    needles = count_needles(axes)
    below = count_below_plane(xyz, weights, plane, extent)

    report = {
        "tool": TOOL,
        "tool_version": TOOL_VERSION,
        "source": scene["path"],
        "gaussians": int(n),
        "centre": {
            "xyz": [float(x) for x in centre],
            "confidence": centre_confidence,
            "method": ("coordinate-wise weighted median, weight = "
                       "sigmoid(opacity) x ellipsoid volume; confidence "
                       "= mass share within "
                       f"{SUBJECT_INNER_RATIO} x extent / "
                       f"{SUBJECT_INNER_TARGET}"),
        },
        "extent": {
            "bbox_min": [float(x) for x in bbox_lo],
            "bbox_max": [float(x) for x in bbox_hi],
            "bbox_diag": bbox_diag,
            "percentile_box_min": [float(x) for x in pbox_lo],
            "percentile_box_max": [float(x) for x in pbox_hi],
            "percentile_box_diag": extent,
            "robust_extent": extent,
            "bbox_vs_percentile_ratio": (bbox_diag / extent
                                         if extent > 0 else None),
            "method": (f"robust extent = diagonal of p{PERCENTILE_LO:.0f}"
                       f"..p{PERCENTILE_HI:.0f} percentile box"),
        },
        "plane": {k: v for k, v in plane.items()},
        "up": up,
        "subject": subject,
        "needles": {k: v for k, v in needles.items()
                    if k != "needle_mask"},
        "below_plane": below,
    }
    return report


def format_report(rep):
    L = []
    add = L.append
    add(f"Scene: {rep['source']}")
    add(f"Gaussians: {rep['gaussians']}")
    e = rep["extent"]
    add("")
    add("Extent")
    add(f"  bbox            min ({e['bbox_min'][0]:.2f}, {e['bbox_min'][1]:.2f}, "
        f"{e['bbox_min'][2]:.2f})  max ({e['bbox_max'][0]:.2f}, "
        f"{e['bbox_max'][1]:.2f}, {e['bbox_max'][2]:.2f})  diag {e['bbox_diag']:.2f}")
    add(f"  percentile box  min ({e['percentile_box_min'][0]:.2f}, "
        f"{e['percentile_box_min'][1]:.2f}, {e['percentile_box_min'][2]:.2f})  "
        f"max ({e['percentile_box_max'][0]:.2f}, "
        f"{e['percentile_box_max'][1]:.2f}, {e['percentile_box_max'][2]:.2f})  "
        f"diag {e['percentile_box_diag']:.2f}")
    add(f"  robust extent   {e['robust_extent']:.2f}  "
        f"(bbox / percentile = {e['bbox_vs_percentile_ratio']:.3f})")
    add(f"  method: {e['method']}")

    c = rep["centre"]
    add("")
    add("Robust centre (opacity- and size-weighted median)")
    add(f"  xyz ({c['xyz'][0]:.3f}, {c['xyz'][1]:.3f}, {c['xyz'][2]:.3f})  "
        f"confidence {c['confidence']:.3f}")
    add(f"  method: {c['method']}")

    p = rep["plane"]
    add("")
    add("Dominant plane")
    if p["normal"] is None:
        add(f"  none ({p['method']})")
    else:
        add(f"  normal ({p['normal'][0]:.4f}, {p['normal'][1]:.4f}, "
            f"{p['normal'][2]:.4f})  point ({p['point'][0]:.2f}, "
            f"{p['point'][1]:.2f}, {p['point'][2]:.2f})")
        add(f"  inliers {p['inlier_count']}/{p['candidate_count']} "
            f"({p['inlier_fraction'] * 100:.1f}%)  tol {p['tolerance']:.3f}  "
            f"stability {p['normal_stability']:.3f}  "
            f"confidence {p['confidence']:.3f}")
        add(f"  method: {p['method']}")

    u = rep["up"]
    add("")
    add("Estimated up direction")
    if u["vector"] is None:
        add(f"  none ({u['method']})")
    else:
        add(f"  up ({u['vector'][0]:.4f}, {u['vector'][1]:.4f}, "
            f"{u['vector'][2]:.4f})  confidence {u['confidence']:.3f}")
        add(f"  angle to Brush y axis {u['angle_to_brush_y_deg']:.2f} deg  "
            f"mass on up side {u['mass_share_on_up_side'] * 100:.1f}%")
        add(f"  method: {u['method']}")

    s = rep["subject"]
    add("")
    add("Subject candidate")
    add(f"  centre ({s['centre'][0]:.3f}, {s['centre'][1]:.3f}, "
        f"{s['centre'][2]:.3f})  confidence {s['confidence']:.3f}"
        + ("  LOW" if s.get("low_confidence") else ""))
    if s.get("radius_p90") is not None:
        add(f"  radius p90 {s['radius_p90']:.2f} "
            f"({s['radius_ratio']:.3f} of extent)  cluster "
            f"{s['cluster_count']} candidates  density contrast "
            f"{s['density_contrast']:.3f}")
        add(f"  scores: contrast {s['contrast_score']:.3f}  "
            f"compactness {s['compactness_score']:.3f}")
        add(f"  offset from global centre {s['offset_from_global']:.2f} "
            f"({s['offset_ratio'] * 100:.1f}% of extent)")
    add(f"  method: {s['method']}")

    nd = rep["needles"]
    add("")
    add("Needles (anisotropy AND size outliers)")
    add(f"  needles {nd['needles']}  aniso outliers {nd['aniso_outliers']}  "
        f"size outliers {nd['size_outliers']}")
    add(f"  anisotropy median {nd['median_anisotropy']:.2f}  "
        f"p99 {nd['p99_anisotropy']:.2f}  threshold "
        f"{nd['anisotropy_threshold']:.2f}")
    add(f"  max axis median {nd['median_max_axis']:.4f}  "
        f"p99 {nd['p99_max_axis']:.4f}  threshold {nd['size_threshold']:.4f}")
    add(f"  method: {nd['method']}")

    b = rep["below_plane"]
    add("")
    add("Below dominant plane (candidate mirror content)")
    add(f"  margin {b['margin']:.3f}")
    if not b["reliable"]:
        add(f"  not counted: {b['reason']}")
    else:
        add(f"  count {b['count']}  ({b['count_share'] * 100:.2f}% of "
            f"Gaussians, {b['weight_share'] * 100:.2f}% of weight)")
        add(f"  reason: {b['reason']}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Read-only scene diagnostics (scene-relative)")
    ap.add_argument("--ply", required=True, help="Brush PLY to analyze")
    ap.add_argument("--json", action="store_true",
                    help="print the report as JSON instead of text")
    a = ap.parse_args(argv)
    try:
        rep = analyze(a.ply)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # unreadable/corrupt file: still one line
        print(f"ERROR: could not analyze PLY: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(rep, indent=2))
    else:
        print(format_report(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
