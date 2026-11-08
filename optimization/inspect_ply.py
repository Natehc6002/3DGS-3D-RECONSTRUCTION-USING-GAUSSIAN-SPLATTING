"""Inspect a Gaussian splat PLY (Brush export) without modifying it.

Usage:
    python -m optimization.inspect_ply --ply path/to/scene.ply [--json out.json]

Reports: Gaussian count, file size, header comments, property list,
per-attribute statistics, and an opacity/scale summary.
Attributes are always accessed by name, never by column position,
because Brush writes f_rest_* in text-sort order (0, 1, 10, 11, ...).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

CORE = ["x", "y", "z", "opacity", "rot_0", "rot_1", "rot_2", "rot_3",
        "scale_0", "scale_1", "scale_2"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def stats(a):
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std())}


def inspect(path):
    path = Path(path)
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = list(v.dtype.names)
    n = len(v)

    rest = [k for k in names if k.startswith("f_rest_")]
    dc = [k for k in names if k.startswith("f_dc_")]
    # SH degree from f_rest count: 3 * ((d+1)^2 - 1)
    sh_degree = None
    for d in range(0, 5):
        if 3 * ((d + 1) ** 2 - 1) == len(rest):
            sh_degree = d

    info = {
        "file": str(path),
        "size_bytes": path.stat().st_size,
        "format": ply.header.splitlines()[1] if ply.header else "",
        "comments": list(ply.comments),
        "gaussian_count": n,
        "property_count": len(names),
        "properties": [(k, str(v.dtype[k])) for k in names],
        "f_rest_count": len(rest),
        "f_rest_numeric_order": rest == sorted(rest, key=lambda s: int(s.split("_")[-1])),
        "sh_degree_from_props": sh_degree,
        "has_normals": all(k in names for k in ("nx", "ny", "nz")),
    }

    missing = [k for k in CORE if k not in names]
    info["missing_core"] = missing
    if missing:
        return info

    st = {k: stats(v[k]) for k in CORE + dc}
    info["stats"] = st

    op = sigmoid(v["opacity"].astype(np.float64))
    info["opacity_sigmoid"] = {
        **stats(op),
        "frac_below_0.01": float((op < 0.01).mean()),
        "frac_below_0.05": float((op < 0.05).mean()),
        "frac_below_0.10": float((op < 0.10).mean()),
    }

    sc = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float64))
    info["scale_exp_max_axis"] = stats(sc.max(1))

    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64)
    qn = np.linalg.norm(q, axis=1)
    info["quat_norm"] = stats(qn)
    info["finite_all"] = bool(all(np.isfinite(v[k]).all() for k in names))
    return info


def report(info):
    print(f"File:      {info['file']}")
    print(f"Size:      {info['size_bytes'] / 1e6:.2f} MB")
    print(f"Format:    {info['format']}")
    for c in info["comments"]:
        print(f"Comment:   {c}")
    print(f"Gaussians: {info['gaussian_count']}")
    print(f"Props:     {info['property_count']}  (f_rest: {info['f_rest_count']}, "
          f"SH degree: {info['sh_degree_from_props']}, normals: {info['has_normals']})")
    print(f"f_rest stored in numeric order: {info['f_rest_numeric_order']}")
    if info["missing_core"]:
        print("MISSING core properties:", info["missing_core"])
        return
    print(f"All values finite: {info['finite_all']}")
    print("\nAttribute stats (raw stored values):")
    for k, s in info["stats"].items():
        print(f"  {k:8s} min {s['min']:9.4f}  max {s['max']:9.4f}  mean {s['mean']:9.4f}")
    o = info["opacity_sigmoid"]
    print("\nOpacity after sigmoid:")
    print(f"  min {o['min']:.4f}  max {o['max']:.4f}  mean {o['mean']:.4f}")
    for t in ("0.01", "0.05", "0.10"):
        print(f"  fraction below {t}: {o['frac_below_' + t] * 100:.2f}%")
    s = info["scale_exp_max_axis"]
    print(f"\nLargest axis after exp(scale): min {s['min']:.5f}  max {s['max']:.4f}  mean {s['mean']:.5f}")
    qn = info["quat_norm"]
    print(f"Quaternion norm: min {qn['min']:.4f}  max {qn['max']:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Inspect a Gaussian splat PLY")
    ap.add_argument("--ply", required=True, help="PLY file to inspect")
    ap.add_argument("--json", help="also write the report to this JSON file")
    a = ap.parse_args()
    p = Path(a.ply)
    if not p.is_file():
        print(f"ERROR: file not found: {p}", file=sys.stderr)
        return 2
    try:
        info = inspect(p)
    except Exception as e:
        print(f"ERROR: could not read PLY: {e}", file=sys.stderr)
        return 3
    report(info)
    if a.json:
        Path(a.json).write_text(json.dumps(info, indent=2))
        print(f"\nWrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())