"""Occupancy coverage check: original PLY vs pruned PLY per voxel.

GEOMETRIC PROXY ONLY. It reports where Gaussians were lost in space
(voxel occupancy drop) so holes can be spotted before opening the
Brush viewer. It does NOT measure image quality: no PSNR/SSIM/FPS is
computed or estimated here; visual quality is judged in Brush.

Usage:
    python -m optimization.pruning.coverage_check --input original.ply \
        --pruned pruned.ply [--voxel-size 5.0] [--loss-threshold 0.5] \
        [--json out.json]

Exit codes: 0 success, 2 bad arguments or paths, 3 failure.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

from optimization.pruning.removal_map import POS_KEYS, removed_mask, voxel_histogram


def coverage(orig, pruned, voxel_size, loss_threshold):
    mask = removed_mask(orig, pruned)
    cr = {k: orig[k].astype(np.float64) for k in POS_KEYS}
    coords_removed = np.stack([cr[k][mask] for k in POS_KEYS], axis=1)
    coords_kept = np.stack([cr[k][~mask] for k in POS_KEYS], axis=1)

    hist = voxel_histogram(coords_removed, coords_kept, voxel_size)
    rem, kep = hist["removed"], hist["kept"]
    tot = rem + kep
    occupied = tot > 0
    loss = np.zeros(len(tot), dtype=np.float64)
    loss[occupied] = rem[occupied] / tot[occupied]

    bad = occupied & (loss >= loss_threshold)
    order = np.argsort(-loss)
    top_n = min(10, int(occupied.sum()))
    span = (np.floor((np.concatenate([coords_removed, coords_kept]).max(axis=0)
                      - hist["origin"]) / voxel_size).astype(np.int64) + 1)

    def voxel_xyz(k):
        iz = k % span[2]
        iy = (k // span[2]) % span[1]
        ix = k // (span[1] * span[2])
        return [float(x) for x in hist["origin"] + np.array([ix, iy, iz]) * voxel_size]

    top = [{"voxel_min_corner_xyz": voxel_xyz(int(hist["voxel_keys"][occupied][i])),
            "original": int(tot[occupied][i]),
            "pruned": int(kep[occupied][i]),
            "loss_fraction": float(loss[occupied][i])}
           for i in order[:top_n]]

    return {
        "original_count": int(len(orig)),
        "pruned_count": int(len(orig) - int(mask.sum())),
        "removed_count": int(mask.sum()),
        "voxel_size": float(voxel_size),
        "loss_threshold": float(loss_threshold),
        "occupied_voxels": int(occupied.sum()),
        "voxels_over_threshold": int(bad.sum()),
        "voxels_emptied": int(((rem > 0) & (kep == 0)).sum()),
        "loss_stats": {
            "max": float(loss[occupied].max()) if occupied.any() else 0.0,
            "p95": float(np.percentile(loss[occupied], 95)) if occupied.any() else 0.0,
            "mean": float(loss[occupied].mean()) if occupied.any() else 0.0,
        },
        "worst_voxels": top,
        "note": "geometric proxy only; image quality not measured",
    }


def report(r):
    print("GEOMETRIC PROXY ONLY - occupancy loss per voxel. "
          "Image quality NOT measured (judge visually in Brush).")
    print(f"Original: {r['original_count']} Gaussians   "
          f"Pruned: {r['pruned_count']}   removed {r['removed_count']}")
    print(f"Voxel size {r['voxel_size']} -> {r['occupied_voxels']} occupied voxels")
    s = r["loss_stats"]
    print(f"Loss fraction per occupied voxel: max {s['max']:.3f}   "
          f"p95 {s['p95']:.3f}   mean {s['mean']:.3f}")
    print(f"Voxels with loss >= {r['loss_threshold']}: {r['voxels_over_threshold']}")
    print(f"Voxels emptied: {r['voxels_emptied']}")
    if r["worst_voxels"]:
        print("Worst voxels (loss @ min corner):")
        for w in r["worst_voxels"]:
            c = w["voxel_min_corner_xyz"]
            print(f"  {w['loss_fraction'] * 100:5.1f}%  {w['pruned']:5d}/{w['original']:<5d}"
                  f" @ ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})")
    if r["voxels_over_threshold"] or r["voxels_emptied"]:
        print(f"\nFLAG: {r['voxels_over_threshold']} voxel(s) lost >= "
              f"{r['loss_threshold'] * 100:.0f}% of their Gaussians"
              + (f", {r['voxels_emptied']} emptied" if r['voxels_emptied'] else "")
              + ". Inspect those areas in Brush.")
    else:
        print("\nOK: no voxel exceeded the loss threshold.")


def main():
    ap = argparse.ArgumentParser(description="Voxel occupancy coverage check")
    ap.add_argument("--input", required=True, help="original PLY (never modified)")
    ap.add_argument("--pruned", required=True, help="pruned PLY to check")
    ap.add_argument("--voxel-size", type=float, default=5.0,
                    help="voxel edge length in scene units (default 5.0)")
    ap.add_argument("--loss-threshold", type=float, default=0.5,
                    help="flag voxels losing at least this fraction (default 0.5)")
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
    if not 0.0 < a.loss_threshold <= 1.0:
        print("ERROR: --loss-threshold must be in (0, 1]", file=sys.stderr)
        return 2

    try:
        orig = PlyData.read(str(src))["vertex"].data
        pruned = PlyData.read(str(prn))["vertex"].data
        r = coverage(orig, pruned, a.voxel_size, a.loss_threshold)
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
