"""Spatially safe Gaussian pruning for Brush-exported PLY scenes.

The baseline prune.py cuts globally on opacity alone, which concentrates
removals in one region and leaves visible holes. smart_prune.py instead:

1. Scores every Gaussian: importance = sigmoid(opacity) * ellipsoid volume.
2. Flags floaters/outliers: isolated (k-NN distance, scipy cKDTree),
   extreme size (robust percentile of exp(scale)), and Gaussians far
   outside a robust percentile bounding box of the main body.
3. Spends a per-voxel removal budget: no occupied voxel ever loses more
   than a configurable fraction of its Gaussians, and none is ever emptied.
   Within a voxel, flagged floaters go first, then removals follow
   ascending importance rank.
4. Prints a removal report: counts per reason, per-voxel removal
   distribution, bbox of removed vs kept, concentration warning.

Attributes are always accessed by name, never by column position.
The input file is never modified; header comments and property order
are preserved in the output.

Usage:
    python -m optimization.pruning.smart_prune --input in.ply --output out.ply
        [--max-voxel-fraction 0.3] [--voxel-size 5.0]
        [--importance-threshold 0.0001] [--knn-k 5] [--iso-percentile 99]
        [--size-percentile 99.9] [--body-percentile-low 0.5]
        [--body-percentile-high 99.5] [--body-margin 20]
        [--remove-singleton-floaters] [--json report.json]

Exit codes: 0 success, 2 bad arguments or paths, 3 failure.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

POS_KEYS = ("x", "y", "z")
SCALE_KEYS = ("scale_0", "scale_1", "scale_2")
ROT_KEYS = ("rot_0", "rot_1", "rot_2", "rot_3")
REQUIRED = POS_KEYS + ("opacity",) + ROT_KEYS + SCALE_KEYS

REASON_OUTSIDE, REASON_ISOLATED, REASON_SIZE, REASON_IMPORTANCE = 0, 1, 2, 3
REASON_NAMES = ("outside_body", "isolated", "size_outlier", "low_importance")
NO_REASON = -1


def sigmoid(x):
    """Overflow-safe logistic function."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def require_properties(v):
    """Raise ValueError naming any missing core property (access by name)."""
    names = v.dtype.names or ()
    missing = [k for k in REQUIRED if k not in names]
    if missing:
        raise ValueError(f"PLY is missing core properties: {missing}")


def renormalize_quaternions(q):
    """Return quaternions with unit norm; all-zero rows become identity.

    q is (..., 4). Used after any operation that may perturb rot_* values
    (e.g. dtype changes in later quantization work).
    """
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    out = np.where(n > 0, q / np.where(n > 0, n, 1.0), 0.0)
    zero = (n == 0).squeeze(-1)
    if np.any(zero):
        out[zero] = np.array([1.0, 0.0, 0.0, 0.0])
    return out


def importance_scores(v):
    """importance = sigmoid(opacity) * ellipsoid volume from exp(scale).

    Ellipsoid semi-axes are exp(scale_i); volume = 4/3 * pi * a * b * c.
    """
    op = sigmoid(v["opacity"])
    ssum = (v["scale_0"].astype(np.float64)
            + v["scale_1"].astype(np.float64)
            + v["scale_2"].astype(np.float64))
    return op * (4.0 / 3.0) * np.pi * np.exp(ssum)


def build_flags(xyz, v, knn_k, iso_percentile, size_percentile,
                body_percentile_low, body_percentile_high, body_margin):
    """Boolean flag arrays plus the thresholds that produced them."""
    n = len(xyz)
    if n == 0:
        raise ValueError("empty vertex data")

    k = min(knn_k + 1, n)  # include self
    d = cKDTree(xyz).query(xyz, k=k)[0]
    kth = d[:, -1]
    iso_thr = float(np.percentile(kth, iso_percentile))
    isolated = kth > iso_thr

    sc = np.exp(np.stack([v[k] for k in SCALE_KEYS], axis=1).astype(np.float64)).max(axis=1)
    size_thr = float(np.percentile(sc, size_percentile))
    size_out = sc > size_thr

    lo = np.percentile(xyz, body_percentile_low, axis=0)
    hi = np.percentile(xyz, body_percentile_high, axis=0)
    blo = lo - body_margin
    bhi = hi + body_margin
    outside = np.any((xyz < blo) | (xyz > bhi), axis=1)

    flags = {"outside_body": outside, "isolated": isolated, "size_outlier": size_out}
    thresholds = {
        "knn_k": int(knn_k),
        "isolation_distance": iso_thr,
        "size_limit": size_thr,
        "body_box_raw": [float(x) for x in np.concatenate([lo, hi])],
        "body_box_used": [float(x) for x in np.concatenate([blo, bhi])],
    }
    return flags, thresholds


def primary_reason(flags, low_importance):
    """Per-Gaussian reason, -1 if not flagged. Precedence: outside >
    isolated > size > low_importance (a Gaussian can carry several flags)."""
    n = len(low_importance)
    r = np.full(n, NO_REASON, dtype=np.int8)
    r[flags["outside_body"]] = REASON_OUTSIDE
    r[flags["isolated"] & (r == NO_REASON)] = REASON_ISOLATED
    r[flags["size_outlier"] & (r == NO_REASON)] = REASON_SIZE
    r[low_importance & (r == NO_REASON)] = REASON_IMPORTANCE
    return r


def voxel_capacities(counts, max_fraction):
    """Max removable per voxel: <= floor(counts * max_fraction), never counts.

    This is the voxel-safety limit: an occupied voxel can never be emptied
    and never loses more than max_fraction of its Gaussians.
    """
    counts = np.asarray(counts, dtype=np.int64)
    cap = np.floor(counts * float(max_fraction)).astype(np.int64)
    cap = np.minimum(cap, counts - 1)
    return np.maximum(cap, 0)


def voxel_ids(xyz, voxel_size):
    """Flat voxel index per point on a grid anchored at the min corner."""
    origin = xyz.min(axis=0)
    idx = np.floor((xyz - origin) / voxel_size).astype(np.int64)
    span = np.floor((xyz.max(axis=0) - origin) / voxel_size).astype(np.int64) + 1
    flat = idx[:, 0] * span[1] * span[2] + idx[:, 1] * span[2] + idx[:, 2]
    return flat, origin, span


def select_removals(xyz, geo, low_importance, importance, max_fraction,
                    voxel_size, allow_singleton_floaters=False):
    """Pick which Gaussians to remove under the voxel-safety budget.

    Candidates (geo-flagged or low importance) are spent in priority
    order: geo-flagged first, then ascending importance, each consuming
    one unit of its voxel's capacity. Returns a dict with the remove
    mask and per-voxel statistics.
    """
    n = len(xyz)
    flat, origin, span = voxel_ids(xyz, voxel_size)
    uniq, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    cap = voxel_capacities(counts, max_fraction)

    singleton_removed = np.zeros(n, dtype=bool)
    if allow_singleton_floaters:
        # Documented exception: a lone geo-flagged floater in its own
        # voxel may be removed (the voxel becomes empty). Off by default.
        alone = counts[inverse] == 1
        solo = alone & geo
        cap[inverse[solo]] = 1
        singleton_removed = solo

    cand = geo | low_importance
    cand_idx = np.nonzero(cand)[0]
    rank = np.where(geo[cand_idx], 0, 1)
    order = cand_idx[np.lexsort((importance[cand_idx], rank))]

    remaining = cap.copy()
    remove = np.zeros(n, dtype=bool)
    for i in order:
        vi = inverse[i]
        if remaining[vi] > 0:
            remove[i] = True
            remaining[vi] -= 1

    vox_removed = np.bincount(inverse[remove], minlength=len(uniq)).astype(np.int64)
    return {
        "remove": remove,
        "voxel_unique": uniq,
        "voxel_counts": counts,
        "voxel_capacity": cap,
        "voxel_removed": vox_removed,
        "voxel_origin": origin,
        "voxel_span": span,
        "voxel_size": float(voxel_size),
        "candidates": int(cand.sum()),
        "singleton_mode": bool(allow_singleton_floaters),
        "singleton_candidates": int(singleton_removed.sum()),
    }


def bbox(coords):
    return {k: [float(coords[:, i].min()), float(coords[:, i].max())]
            for i, k in enumerate(POS_KEYS)}


def analyse(v, xyz, flags, low_importance, sel, thresholds):
    """Assemble the removal report dict from real numbers only."""
    remove = sel["remove"]
    n = len(v)
    reason = primary_reason(flags, low_importance)
    geo = flags["outside_body"] | flags["isolated"] | flags["size_outlier"]
    cand = geo | low_importance

    flagged_counts = {name: int(flags[name].sum())
                      for name in ("outside_body", "isolated", "size_outlier")}
    flagged_counts["low_importance"] = int(low_importance.sum())
    removed_reason = {REASON_NAMES[r]: int(((reason == r) & remove).sum())
                      for r in range(len(REASON_NAMES))}
    blocked_counts = {name: int((flag & cand & ~remove).sum())
                      for name, flag in
                      (("outside_body", flags["outside_body"]),
                       ("isolated", flags["isolated"]),
                       ("size_outlier", flags["size_outlier"]),
                       ("low_importance", low_importance))}

    counts = sel["voxel_counts"]
    rem = sel["voxel_removed"]
    occupied = counts > 0
    frac = np.zeros_like(rem, dtype=np.float64)
    frac[occupied] = rem[occupied] / counts[occupied]
    rem_occ = rem[occupied]
    total_removed = int(rem_occ.sum())

    order = np.argsort(-rem_occ)
    top1 = max(1, int(np.ceil(0.01 * len(rem_occ))))
    share = float(rem_occ[order[:top1]].sum() / total_removed) if total_removed else 0.0
    top_n = min(5, int((rem_occ > 0).sum()))
    top_idx = order[:top_n]
    keys = sel["voxel_unique"][occupied][top_idx] if top_n else np.empty(0, dtype=np.int64)
    span = sel["voxel_span"]
    origin = sel["voxel_origin"]
    vs = sel["voxel_size"]
    top_coords = []
    for kk in keys.tolist():
        iz = kk % span[2]
        iy = (kk // span[2]) % span[1]
        ix = kk // (span[1] * span[2])
        top_coords.append([float(x) for x in origin + np.array([ix, iy, iz]) * vs])

    emptied = int(((rem > 0) & (counts - rem == 0)).sum())
    coords_removed = xyz[remove]
    coords_kept = xyz[~remove]

    return {
        "original_count": int(n),
        "removed_count": total_removed,
        "pruned_count": int(n - total_removed),
        "flagged_counts": flagged_counts,
        "removed_by_reason": removed_reason,
        "blocked_by_safety": blocked_counts,
        "candidates": int(sel["candidates"]),
        "voxel_size": vs,
        "occupied_voxels": int(occupied.sum()),
        "voxels_emptied": emptied,
        "singleton_mode": sel["singleton_mode"],
        "singleton_candidates": sel["singleton_candidates"],
        "removed_count_stats": {
            "max": int(rem_occ.max()) if len(rem_occ) else 0,
            "p95": float(np.percentile(rem_occ, 95)) if len(rem_occ) else 0.0,
            "mean": float(rem_occ.mean()) if len(rem_occ) else 0.0,
        },
        "removed_fraction_stats": {
            "max": float(frac[occupied].max()) if occupied.any() else 0.0,
            "p95": float(np.percentile(frac[occupied], 95)) if occupied.any() else 0.0,
            "mean": float(frac[occupied].mean()) if occupied.any() else 0.0,
        },
        "concentration": {
            "top1pct_voxel_share_of_removals": share,
            "top_voxel_counts": [int(x) for x in rem_occ[order[:top_n]].tolist()],
            "top_voxel_min_corner_xyz": top_coords,
        },
        "bbox_removed": bbox(coords_removed) if len(coords_removed) else None,
        "bbox_kept": bbox(coords_kept) if len(coords_kept) else None,
        "thresholds": thresholds,
    }


def report(info, settings, sizes, elapsed):
    print(f"Input:      {settings['input']}")
    print(f"Output:     {settings['output']}")
    print(f"Settings:   max_voxel_fraction={settings['max_fraction']}, "
          f"voxel_size={settings['voxel_size']}, "
          f"importance_threshold={settings['importance_threshold']}, "
          f"knn_k={settings['knn_k']}, iso_percentile={settings['iso_percentile']}, "
          f"size_percentile={settings['size_percentile']}, "
          f"body=({settings['body_low']}..{settings['body_high']}"
          f"+{settings['body_margin']}), "
          f"singleton_floaters={settings['singleton']}")
    n, k = info["original_count"], info["pruned_count"]
    print(f"Gaussians:  {n} -> {k}  (kept {k / n * 100:.2f}%, "
          f"removed {info['removed_count']})")

    print("\nFlagged (candidates, any reason):")
    for name, c in info["flagged_counts"].items():
        print(f"  {name:15s} {c:7d}")
    print(f"  {'union':15s} {info['candidates']:7d}")
    print("Removed by primary reason:")
    for name, c in info["removed_by_reason"].items():
        print(f"  {name:15s} {c:7d}")
    print("Flagged but kept (voxel safety):")
    for name, c in info["blocked_by_safety"].items():
        print(f"  {name:15s} {c:7d}")

    s, f = info["removed_count_stats"], info["removed_fraction_stats"]
    print(f"\nVoxel grid: size {info['voxel_size']}, "
          f"{info['occupied_voxels']} occupied voxels")
    print(f"Per-voxel removed counts:  max {s['max']}   p95 {s['p95']:.1f}   "
          f"mean {s['mean']:.1f}")
    print(f"Per-voxel removed fraction: max {f['max']:.3f}   p95 {f['p95']:.3f}   "
          f"mean {f['mean']:.3f}")
    if info["singleton_mode"]:
        print(f"Voxels emptied: {info['voxels_emptied']} "
              f"(singleton-floater exception enabled, "
              f"{info['singleton_candidates']} singleton candidates)")
    else:
        print(f"Voxels emptied: {info['voxels_emptied']} (invariant: must be 0)")

    c = info["concentration"]
    print(f"\nTop 1% of voxels hold {c['top1pct_voxel_share_of_removals'] * 100:.1f}% "
          f"of all removals")
    print("Worst voxels (removed count @ min corner):")
    for cnt, xyz in zip(c["top_voxel_counts"], c["top_voxel_min_corner_xyz"]):
        print(f"  {cnt:6d}  @ ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f})")

    print("\nBounding box  min            max")
    for key, label in (("bbox_removed", "removed"), ("bbox_kept", "kept   ")):
        b = info[key]
        if b is None:
            print(f"  {label}   (none)")
            continue
        print(f"  {label}   "
              f"({b['x'][0]:8.2f},{b['y'][0]:8.2f},{b['z'][0]:8.2f})  "
              f"({b['x'][1]:8.2f},{b['y'][1]:8.2f},{b['z'][1]:8.2f})")

    thr = info["thresholds"]
    print(f"\nThresholds: isolation kNN distance > {thr['isolation_distance']:.4f}, "
          f"max axis exp(scale) > {thr['size_limit']:.4f}, "
          f"body box {np.round(thr['body_box_used'], 2).tolist()}")

    flagged = (c["top1pct_voxel_share_of_removals"] > 0.5
               or (info["voxels_emptied"] > 0 and not info["singleton_mode"]))
    if flagged:
        print("\nFLAG: removals concentrate in one area "
              "(check the output in Brush before accepting).")
    else:
        print("\nOK: no voxel was emptied and removals are not one-region dominated.")

    print(f"\nFile size:  {sizes[0] / 1e6:.2f} MB -> {sizes[1] / 1e6:.2f} MB  "
          f"(ratio {sizes[0] / sizes[1]:.2f}x)")
    print(f"Time:       {elapsed:.1f} s")
    return flagged


def prune(src, dst, opts):
    ply = PlyData.read(str(src))
    v = ply["vertex"].data
    require_properties(v)
    n = len(v)
    if n == 0:
        raise ValueError("input PLY has no Gaussians")

    xyz = np.stack([v[k] for k in POS_KEYS], axis=1).astype(np.float64)
    importance = importance_scores(v)
    flags, thresholds = build_flags(
        xyz, v, opts["knn_k"], opts["iso_percentile"], opts["size_percentile"],
        opts["body_low"], opts["body_high"], opts["body_margin"])
    low_importance = importance < opts["importance_threshold"]

    sel = select_removals(xyz,
                          flags["outside_body"] | flags["isolated"] | flags["size_outlier"],
                          low_importance, importance,
                          opts["max_fraction"], opts["voxel_size"],
                          allow_singleton_floaters=opts["singleton"])
    remove = sel["remove"]
    kept = v[~remove]
    if len(kept) == 0:
        raise ValueError("pruning would remove every Gaussian; lower the thresholds")

    out = PlyData([PlyElement.describe(kept, "vertex")],
                  text=False, byte_order="<", comments=list(ply.comments))
    out.write(str(dst))
    info = analyse(v, xyz, flags, low_importance, sel, thresholds)
    return info


def main():
    ap = argparse.ArgumentParser(description="Spatially safe Gaussian pruning")
    ap.add_argument("--input", required=True, help="original PLY (never modified)")
    ap.add_argument("--output", required=True, help="pruned PLY to create")
    ap.add_argument("--max-voxel-fraction", type=float, default=0.3,
                    help="max fraction of any voxel that may be removed (default 0.3)")
    ap.add_argument("--voxel-size", type=float, default=5.0,
                    help="voxel edge length in scene units (default 5.0)")
    ap.add_argument("--importance-threshold", type=float, default=1e-4,
                    help="flag Gaussians with sigmoid(opacity)*volume below this; "
                         "0 disables the low-importance reason (default 1e-4)")
    ap.add_argument("--knn-k", type=int, default=5,
                    help="k for isolation via k-NN distance (default 5)")
    ap.add_argument("--iso-percentile", type=float, default=99.0,
                    help="flag k-NN distances above this percentile (default 99)")
    ap.add_argument("--size-percentile", type=float, default=99.9,
                    help="flag exp(scale) max axis above this percentile (default 99.9)")
    ap.add_argument("--body-percentile-low", type=float, default=0.5,
                    help="low percentile of the robust body box (default 0.5)")
    ap.add_argument("--body-percentile-high", type=float, default=99.5,
                    help="high percentile of the robust body box (default 99.5)")
    ap.add_argument("--body-margin", type=float, default=20.0,
                    help="margin added around the body box, scene units (default 20)")
    ap.add_argument("--remove-singleton-floaters", action="store_true",
                    help="EXCEPTION: allow removing a lone geo-flagged Gaussian "
                         "even though its voxel becomes empty (default off)")
    ap.add_argument("--json", help="also write the report to this JSON file")
    a = ap.parse_args()

    src, dst = Path(a.input), Path(a.output)
    if not src.is_file():
        print(f"ERROR: input not found: {src}", file=sys.stderr)
        return 2
    if src.resolve() == dst.resolve():
        print("ERROR: output must differ from input", file=sys.stderr)
        return 2
    if dst.exists():
        print(f"ERROR: output already exists, not overwriting: {dst}", file=sys.stderr)
        return 2
    if not 0.0 < a.max_voxel_fraction <= 1.0:
        print("ERROR: --max-voxel-fraction must be in (0, 1]", file=sys.stderr)
        return 2
    if a.voxel_size <= 0:
        print("ERROR: --voxel-size must be > 0", file=sys.stderr)
        return 2
    if a.importance_threshold < 0:
        print("ERROR: --importance-threshold must be >= 0", file=sys.stderr)
        return 2
    if a.knn_k < 1:
        print("ERROR: --knn-k must be >= 1", file=sys.stderr)
        return 2
    for name, val in (("iso-percentile", a.iso_percentile),
                      ("size-percentile", a.size_percentile)):
        if not 0.0 < val <= 100.0:
            print(f"ERROR: --{name} must be in (0, 100]", file=sys.stderr)
            return 2
    if not (0.0 <= a.body_percentile_low < a.body_percentile_high <= 100.0):
        print("ERROR: body percentiles must satisfy 0 <= low < high <= 100",
              file=sys.stderr)
        return 2
    if a.body_margin < 0:
        print("ERROR: --body-margin must be >= 0", file=sys.stderr)
        return 2

    opts = {
        "max_fraction": a.max_voxel_fraction,
        "voxel_size": a.voxel_size,
        "importance_threshold": a.importance_threshold,
        "knn_k": a.knn_k,
        "iso_percentile": a.iso_percentile,
        "size_percentile": a.size_percentile,
        "body_low": a.body_percentile_low,
        "body_high": a.body_percentile_high,
        "body_margin": a.body_margin,
        "singleton": bool(a.remove_singleton_floaters),
    }
    settings = {"input": str(src), "output": str(dst), **opts,
                "singleton": opts["singleton"]}

    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        info = prune(src, dst, opts)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"ERROR: pruning failed: {e}", file=sys.stderr)
        if dst.exists():
            dst.unlink()
        return 3
    elapsed = time.time() - t0

    sizes = (src.stat().st_size, dst.stat().st_size)
    report(info, settings, sizes, elapsed)
    if a.json:
        payload = {"settings": {k: (v if not isinstance(v, np.generic) else v.item())
                                for k, v in settings.items()},
                   "sizes": {"input_bytes": sizes[0], "output_bytes": sizes[1]},
                   "time_s": round(elapsed, 3),
                   "info": json.loads(json.dumps(info, default=float))}
        Path(a.json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
