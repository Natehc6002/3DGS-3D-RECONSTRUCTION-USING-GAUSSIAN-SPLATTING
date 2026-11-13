"""Show where in space removed Gaussians were (voxel histogram).

Compares an original PLY with a pruned PLY (produced by prune.py or
smart_prune.py), reconstructs which original Gaussians were removed by
order-preserving row matching, and reports a per-voxel histogram of the
removals. This is a geometric proxy for holes: a voxel whose Gaussians
were all removed is a region that went empty.

Usage:
    python -m optimization.pruning.removal_map --input original.ply \
        --pruned pruned.ply [--voxel-size 5.0] [--json out.json]

Exit codes: 0 success, 2 bad arguments or paths, 3 failure.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

POS_KEYS = ("x", "y", "z")


def _void_view(arr):
    """View a structured array as opaque bytes per row (for exact equality)."""
    contiguous = np.ascontiguousarray(arr)
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize))).ravel()


def removed_mask(orig, pruned):
    """Boolean mask over orig rows that are absent from pruned.

    Pruned rows are a subset of orig rows in the same order (prune.py
    writes v[keep]), so an exact two-pointer walk recovers the removed
    indices even when duplicate rows exist. Raises ValueError if pruned
    is not an order-preserving subset of orig.
    """
    if orig.dtype != pruned.dtype:
        raise ValueError(
            f"dtype mismatch: original {orig.dtype.names and len(orig.dtype.names)} "
            f"properties vs pruned {pruned.dtype.names and len(pruned.dtype.names)}")
    if len(pruned) > len(orig):
        raise ValueError("pruned has more Gaussians than original")

    o = _void_view(orig)
    p = _void_view(pruned)
    keep = np.zeros(len(o), dtype=bool)
    j = 0
    for i in range(len(o)):
        if j < len(p) and o[i] == p[j]:
            keep[i] = True
            j += 1
    if j != len(p):
        raise ValueError(
            f"pruned is not an order-preserving subset of original "
            f"(matched {j} of {len(p)} rows); wrong input pair?")
    return ~keep


def bbox(coords):
    return {k: [float(coords[k].min()), float(coords[k].max())] for k in POS_KEYS}


def voxel_histogram(coords_removed, coords_kept, voxel_size):
    """Per-voxel counts of removed/kept Gaussians on a shared grid.

    Returns a dict with flat voxel indices and counts; grid origin is the
    min corner over both sets so the two share one grid.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    allc = np.concatenate([coords_removed, coords_kept], axis=0)
    origin = allc.min(axis=0)
    idx_r = np.floor((coords_removed - origin) / voxel_size).astype(np.int64)
    idx_k = np.floor((coords_kept - origin) / voxel_size).astype(np.int64)

    def counts(idx):
        if len(idx) == 0:
            return {}, np.empty(0, dtype=np.int64)
        flat = idx[:, 0]
        # guard against off-by-one at the max boundary
        span = np.floor((allc.max(axis=0) - origin) / voxel_size).astype(np.int64) + 1
        flat = flat * span[1] * span[2] + idx[:, 1] * span[2] + idx[:, 2]
        uniq, cnt = np.unique(flat, return_counts=True)
        return dict(zip(uniq.tolist(), cnt.tolist())), uniq

    cr, uniq_r = counts(idx_r)
    ck, _ = counts(idx_k)
    all_keys = np.union1d(np.fromiter(cr.keys(), dtype=np.int64),
                          np.fromiter(ck.keys(), dtype=np.int64))
    removed = np.array([cr.get(int(k), 0) for k in all_keys], dtype=np.int64)
    kept = np.array([ck.get(int(k), 0) for k in all_keys], dtype=np.int64)
    return {
        "voxel_keys": all_keys,
        "removed": removed,
        "kept": kept,
        "total": removed + kept,
        "origin": origin,
        "voxel_size": voxel_size,
    }


def analyse(orig, pruned, voxel_size):
    mask = removed_mask(orig, pruned)
    n_rem = int(mask.sum())
    cr = {k: orig[k].astype(np.float64) for k in POS_KEYS}
    coords_removed = np.stack([cr[k][mask] for k in POS_KEYS], axis=1)
    coords_kept = np.stack([cr[k][~mask] for k in POS_KEYS], axis=1)

    hist = voxel_histogram(coords_removed, coords_kept, voxel_size)
    rem, kep, tot = hist["removed"], hist["kept"], hist["occupied"] if "occupied" in hist else hist["total"]
    occupied = tot > 0
    frac = np.zeros_like(rem, dtype=np.float64)
    frac[occupied] = rem[occupied] / tot[occupied]

    rem_occ = rem[occupied]
    order = np.argsort(-rem_occ)
    total_removed = int(rem_occ.sum())
    top1_pct = max(1, int(np.ceil(0.01 * len(rem_occ))))
    share_top1 = float(rem_occ[order[:top1_pct]].sum() / total_removed) if total_removed else 0.0
    top_n = min(5, len(rem_occ))

    # coordinates of the worst voxels (min corner)
    keys = hist["voxel_keys"][occupied][order[:top_n]]
    span = (np.floor((np.concatenate([coords_removed, coords_kept]).max(axis=0)
                      - hist["origin"]) / voxel_size).astype(np.int64) + 1)
    top_coords = []
    for k in keys.tolist():
        iz = k % span[2]
        iy = (k // span[2]) % span[1]
        ix = k // (span[1] * span[2])
        c = hist["origin"] + np.array([ix, iy, iz]) * voxel_size
        top_coords.append([float(x) for x in c])

    empty_after = int(((rem > 0) & (kep == 0)).sum())
    half_or_more = int((frac >= 0.5).sum())

    return {
        "original_count": len(orig),
        "pruned_count": int(len(orig) - n_rem),
        "removed_count": n_rem,
        "voxel_size": voxel_size,
        "occupied_voxels": int(occupied.sum()),
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
        "voxels_emptied": empty_after,
        "voxels_half_or_more_removed": half_or_more,
        "concentration": {
            "top1pct_voxel_share_of_removals": share_top1,
            "top_voxel_counts": [int(x) for x in rem_occ[order[:top_n]].tolist()],
            "top_voxel_min_corner_xyz": top_coords,
        },
        "bbox_removed": bbox({k: coords_removed[:, i] for i, k in enumerate(POS_KEYS)}),
        "bbox_kept": bbox({k: coords_kept[:, i] for i, k in enumerate(POS_KEYS)}),
    }


def report(r):
    print(f"Original:   {r['original_count']} Gaussians")
    print(f"Pruned:     {r['pruned_count']} Gaussians  (removed {r['removed_count']})")
    print(f"Voxel size: {r['voxel_size']}  ->  {r['occupied_voxels']} occupied voxels")
    s = r["removed_count_stats"]
    f = r["removed_fraction_stats"]
    print("\nPer-voxel removed counts (over occupied voxels):")
    print(f"  max {s['max']}   p95 {s['p95']:.1f}   mean {s['mean']:.1f}")
    print("Per-voxel removed fraction:")
    print(f"  max {f['max']:.3f}   p95 {f['p95']:.3f}   mean {f['mean']:.3f}")
    print(f"Voxels emptied (all Gaussians removed):      {r['voxels_emptied']}")
    print(f"Voxels with >= 50% removed:                  {r['voxels_half_or_more_removed']}")
    c = r["concentration"]
    print(f"\nTop 1% of voxels hold {c['top1pct_voxel_share_of_removals'] * 100:.1f}% of all removals")
    print("Worst voxels (removed count @ min corner):")
    for cnt, xyz in zip(c["top_voxel_counts"], c["top_voxel_min_corner_xyz"]):
        print(f"  {cnt:6d}  @ ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f})")
    print("\nBounding box  min            max")
    for name in ("bbox_removed", "bbox_kept"):
        b = r[name]
        label = "removed" if name == "bbox_removed" else "kept   "
        print(f"  {label}   "
              f"({b['x'][0]:8.2f},{b['y'][0]:8.2f},{b['z'][0]:8.2f})  "
              f"({b['x'][1]:8.2f},{b['y'][1]:8.2f},{b['z'][1]:8.2f})")
    flagged = (c["top1pct_voxel_share_of_removals"] > 0.5
               or r["voxels_emptied"] > 0)
    if flagged:
        print("\nFLAG: removals concentrate in one area and/or empty voxels "
              "(the one-region problem).")
    else:
        print("\nOK: removals are spread out; no voxel was fully emptied.")
    return flagged


def main():
    ap = argparse.ArgumentParser(description="Voxel histogram of removed Gaussians")
    ap.add_argument("--input", required=True, help="original PLY (never modified)")
    ap.add_argument("--pruned", required=True, help="pruned PLY to compare against")
    ap.add_argument("--voxel-size", type=float, default=5.0,
                    help="voxel edge length in scene units (default 5.0)")
    ap.add_argument("--json", help="also write the report to this JSON file")
    a = ap.parse_args()

    src, prn = Path(a.input), Path(a.pruned)
    for p, label in ((src, "input"), (prn, "pruned")):
        if not p.is_file():
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            return 2
    if src.resolve() == prn.resolve():
        print("ERROR: --input and --pruned must differ", file=sys.stderr)
        return 2
    if a.voxel_size <= 0:
        print("ERROR: --voxel-size must be > 0", file=sys.stderr)
        return 2

    try:
        orig = PlyData.read(str(src))["vertex"].data
        pruned = PlyData.read(str(prn))["vertex"].data
        r = analyse(orig, pruned, a.voxel_size)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"ERROR: could not read PLY: {e}", file=sys.stderr)
        return 3

    report(r)
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2))
        print(f"\nWrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
