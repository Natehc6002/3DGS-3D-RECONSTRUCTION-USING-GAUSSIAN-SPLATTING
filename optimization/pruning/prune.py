"""Baseline Gaussian pruning for Brush-exported PLY scenes.

Removes Gaussians whose opacity is below a threshold, and optionally
Gaussians with an extreme size. The input file is never modified; the
output is a new file with the same header comments and property order.

Usage:
    python -m optimization.pruning.prune --input in.ply --output out.ply
        [--opacity-threshold 0.05] [--max-scale-percentile 99.9]

Exit codes: 0 success, 2 bad arguments or paths, 3 failure.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

SCALE_KEYS = ("scale_0", "scale_1", "scale_2")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_mask(v, opacity_threshold, max_scale_percentile):
    """Return (keep_mask, info). Attributes are read by name."""
    op = sigmoid(v["opacity"].astype(np.float64))
    keep = op >= opacity_threshold
    info = {"removed_by_opacity": int((~keep).sum())}

    if max_scale_percentile is not None:
        # raw scales are log values; largest axis after exp
        sc = np.exp(np.stack([v[k] for k in SCALE_KEYS], 1).astype(np.float64)).max(1)
        limit = np.percentile(sc, max_scale_percentile)
        big = sc > limit
        info["scale_limit"] = float(limit)
        info["removed_by_scale_only"] = int((big & keep).sum())
        keep &= ~big
    return keep, info


def prune(src, dst, opacity_threshold, max_scale_percentile):
    ply = PlyData.read(str(src))
    v = ply["vertex"].data
    n = len(v)

    keep, info = build_mask(v, opacity_threshold, max_scale_percentile)
    kept = v[keep]
    if len(kept) == 0:
        raise ValueError("pruning would remove every Gaussian; lower the threshold")

    out = PlyData([PlyElement.describe(kept, "vertex")],
                  text=False, byte_order="<", comments=list(ply.comments))
    out.write(str(dst))
    return n, len(kept), info


def main():
    ap = argparse.ArgumentParser(description="Baseline Gaussian pruning")
    ap.add_argument("--input", required=True, help="original PLY (never modified)")
    ap.add_argument("--output", required=True, help="pruned PLY to create")
    ap.add_argument("--opacity-threshold", type=float, default=0.05,
                    help="remove Gaussians with sigmoid(opacity) below this (default 0.05)")
    ap.add_argument("--max-scale-percentile", type=float, default=None,
                    help="optional: also remove Gaussians above this percentile of largest axis size")
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
    if not 0.0 <= a.opacity_threshold < 1.0:
        print("ERROR: --opacity-threshold must be in [0, 1)", file=sys.stderr)
        return 2
    if a.max_scale_percentile is not None and not 0.0 < a.max_scale_percentile <= 100.0:
        print("ERROR: --max-scale-percentile must be in (0, 100]", file=sys.stderr)
        return 2

    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        n, k, info = prune(src, dst, a.opacity_threshold, a.max_scale_percentile)
    except Exception as e:
        print(f"ERROR: pruning failed: {e}", file=sys.stderr)
        if dst.exists():
            dst.unlink()
        return 3

    s_in, s_out = src.stat().st_size, dst.stat().st_size
    print(f"Input:      {src}")
    print(f"Output:     {dst}")
    print(f"Settings:   opacity_threshold={a.opacity_threshold}, "
          f"max_scale_percentile={a.max_scale_percentile}")
    print(f"Gaussians:  {n} -> {k}  (kept {k / n * 100:.2f}%, removed {n - k})")
    print(f"  removed by opacity: {info['removed_by_opacity']}")
    if "scale_limit" in info:
        print(f"  removed by scale only: {info['removed_by_scale_only']} (limit {info['scale_limit']:.5f})")
    print(f"File size:  {s_in / 1e6:.2f} MB -> {s_out / 1e6:.2f} MB  "
          f"(ratio {s_in / s_out:.2f}x)")
    print(f"Time:       {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
