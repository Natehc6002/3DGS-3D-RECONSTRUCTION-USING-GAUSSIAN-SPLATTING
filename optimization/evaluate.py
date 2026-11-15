"""Evaluation table for Gaussian optimization variants (real numbers only).

Builds one CSV plus a short markdown summary comparing variants of one
original scene. Every cell is either measured by this script (Gaussian
count, file size, ratio, coverage proxy, round-trip error) or passed in
with real timings from the tool that produced the file (--time). Nothing
is estimated:

    - Quality columns (PSNR, SSIM, FPS) are always "not measured".
    - Processing time is "not measured" unless supplied with --time.
    - Brush load status comes from the manual verdicts in
      BRUSH_LOAD_STATUS; fill them in after a visual check, everything
      not verified stays "not tested".
    - Coverage proxy is the voxel occupancy loss versus the original
      (geometric proxy only), computed the same way as
      optimization.pruning.coverage_check.

Usage:
    python -m optimization.evaluate --original orig.ply \
        --variant baseline_prune=baseline.ply \
        --variant smart_default=smart.ply \
        --time baseline_prune=0.2 \
        --round-trip route_b_decoded=smart.ply,smart_qback.ply \
        [--voxel-size 5.0] [--out-csv PATH] [--out-md PATH]

The two report files are rewritten on every run (they are regenerable
reports, unlike pipeline outputs).

Exit codes: 0 success, 2 bad arguments/paths.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from plyfile import PlyData

from optimization.pruning import coverage_check

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_CSV = RESULTS_DIR / "evaluation_table.csv"
DEFAULT_MD = RESULTS_DIR / "evaluation_summary.md"

COLUMNS = (
    "variant", "gaussians", "file_size_mb", "size_ratio",
    "processing_time_s", "coverage_proxy",
    "route_b_roundtrip_max_abs_err", "brush_load_status",
    "psnr", "ssim", "fps", "notes",
)

# Manual Brush verdicts - update ONLY after a visual check in Brush.
# Anything not verified must stay "not tested".
BRUSH_LOAD_STATUS = {
    "original": "n/a (Brush export)",
    "baseline_prune": "not tested",
    "smart_default": "not tested",
    "smart_shd2": "not tested",
    "smart_shd1": "not tested",
    "smart_shd0": "not tested",
    "route_b_decoded": "not tested",
}

NOTES = {
    "original": "reference Brush export",
    "baseline_prune": "baseline opacity prune (threshold 0.05)",
    "smart_default": "smart prune, max_voxel_fraction 0.3, no emptied voxels",
    "smart_shd2": "smart default, SH truncated to degree 2",
    "smart_shd1": "smart default, SH truncated to degree 1",
    "smart_shd0": "smart default, SH truncated to degree 0 (no view-dependent colour)",
    "route_b_decoded": "smart default quantized to .npz, decoded back to PLY",
}

QUALITY = "not measured"
NA_NO_REMOVALS = "n/a (no removals)"
NA_OTHER_LAYOUT = "n/a (layout differs from original)"


def load_vertex(path):
    return PlyData.read(str(path))["vertex"].data


def roundtrip_max_abs_err(src_path, decoded_path):
    """Max |delta| over all properties: source PLY vs decoded PLY."""
    a = load_vertex(src_path)
    b = load_vertex(decoded_path)
    if len(a) != len(b):
        raise ValueError(f"round-trip count mismatch: {len(a)} vs {len(b)}")
    if list(a.dtype.names) != list(b.dtype.names):
        raise ValueError("round-trip property lists differ")
    return float(max(np.abs(a[n].astype(np.float64)
                            - b[n].astype(np.float64)).max()
                     for n in a.dtype.names))


def coverage_proxy(orig_v, variant_v, voxel_size, loss_threshold=0.5):
    """One-line voxel occupancy loss summary versus the original.

    Only exact subsets of the original (prune outputs) can be diffed row
    by row; quantized/rewritten variants cannot.
    """
    if len(variant_v) == len(orig_v):
        return NA_NO_REMOVALS
    if variant_v.dtype != orig_v.dtype:
        return NA_OTHER_LAYOUT
    try:
        r = coverage_check.coverage(orig_v, variant_v, voxel_size,
                                    loss_threshold)
    except ValueError:
        return "n/a (not an exact subset of original)"
    except Exception as e:
        return f"n/a (coverage failed: {e})"
    s = r["loss_stats"]
    return (f"voxels emptied {r['voxels_emptied']}; "
            f"over {int(loss_threshold * 100)}% loss "
            f"{r['voxels_over_threshold']}; max loss {s['max']:.3f}")


def collect_rows(original_path, variants, times=None, round_trips=None,
                 voxel_size=5.0):
    """Build one row dict per variant (including the original row).

    variants: ordered {label: path}, times: {label: seconds},
    round_trips: {label: (source_ply, decoded_ply)}.
    """
    times = times or {}
    round_trips = round_trips or {}
    src = Path(original_path)
    if not src.is_file():
        raise FileNotFoundError(f"original not found: {src}")
    orig_v = load_vertex(src)
    if len(orig_v) == 0:
        raise ValueError("original PLY has no Gaussians")
    orig_bytes = src.stat().st_size

    unknown_time = set(times) - set(variants)
    unknown_rt = set(round_trips) - set(variants)
    if unknown_time:
        raise ValueError(f"--time for unknown variant(s): {sorted(unknown_time)}")
    if unknown_rt:
        raise ValueError(
            f"--round-trip for unknown variant(s): {sorted(unknown_rt)}")

    rows = [{
        "variant": "original",
        "gaussians": int(len(orig_v)),
        "file_size_mb": round(orig_bytes / 1e6, 2),
        "size_ratio": 1.0,
        "processing_time_s": "n/a",
        "coverage_proxy": NA_NO_REMOVALS,
        "route_b_roundtrip_max_abs_err": "n/a",
        "brush_load_status": BRUSH_LOAD_STATUS["original"],
        "psnr": QUALITY,
        "ssim": QUALITY,
        "fps": QUALITY,
        "notes": NOTES["original"],
    }]

    for label, path_str in variants.items():
        path = Path(path_str)
        if not path.is_file():
            raise FileNotFoundError(f"variant '{label}' not found: {path}")
        v = load_vertex(path)
        if len(v) == 0:
            raise ValueError(f"variant '{label}' has no Gaussians")
        size = path.stat().st_size

        if label in times:
            t = float(times[label])
            if t < 0:
                raise ValueError(f"--time {label} must be >= 0")
            time_s = round(t, 3)
        else:
            time_s = "n/a" if label == "original" else QUALITY

        if label in round_trips:
            a, b = round_trips[label]
            rt = f"{roundtrip_max_abs_err(a, b):.6g}"
        else:
            rt = "n/a"

        rows.append({
            "variant": label,
            "gaussians": int(len(v)),
            "file_size_mb": round(size / 1e6, 2),
            "size_ratio": round(orig_bytes / size, 3),
            "processing_time_s": time_s,
            "coverage_proxy": coverage_proxy(orig_v, v, voxel_size),
            "route_b_roundtrip_max_abs_err": rt,
            "brush_load_status": BRUSH_LOAD_STATUS.get(label, "not tested"),
            "psnr": QUALITY,
            "ssim": QUALITY,
            "fps": QUALITY,
            "notes": NOTES.get(label, ""),
        })
    return rows


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_markdown(rows, original_path, path, generated_at=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = generated_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    header = ("variant", "gaussians", "size (MB)", "ratio", "time (s)",
              "coverage proxy (voxels)", "Route B round-trip max abs diff",
              "Brush load", "PSNR", "SSIM", "FPS")
    lines = [
        "# Optimization evaluation",
        "",
        f"- Generated: {stamp}",
        f"- Original: `{original_path}`",
        f"- Quality metrics (PSNR, SSIM, FPS) are not measured anywhere in "
        f"this repository yet.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in rows:
        cells = [
            r["variant"],
            str(r["gaussians"]),
            f"{r['file_size_mb']:.2f}",
            f"{r['size_ratio']:.3f}",
            str(r["processing_time_s"]),
            str(r["coverage_proxy"]),
            str(r["route_b_roundtrip_max_abs_err"]),
            str(r["brush_load_status"]),
            r["psnr"],
            r["ssim"],
            r["fps"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Notes",
        "",
        "- Size ratio is original size / variant size (higher = smaller).",
        "- Coverage proxy is voxel occupancy loss versus the original "
        "(geometric proxy only, see optimization/pruning/coverage_check.py).",
        "- Route B round-trip is the max absolute difference over all PLY "
        "properties, source vs decoded PLY.",
        "- Brush load status comes from manual visual checks; "
        "\"not tested\" means nobody has opened that file in Brush yet.",
        "- The .npz archive itself is not readable by Brush - only the "
        "decoded PLY in this table is.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _parse_pairs(items, flag, split_csv=False):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"{flag} expects label=value, got: {item}")
        label, value = item.split("=", 1)
        label = label.strip()
        if not label or not value:
            raise ValueError(f"{flag} expects label=value, got: {item}")
        if split_csv:
            parts = value.split(",")
            if len(parts) != 2:
                raise ValueError(
                    f"{flag} expects label=source.ply,decoded.ply, got: {item}")
            out[label] = tuple(p.strip() for p in parts)
        else:
            out[label] = value.strip()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the optimization evaluation table (real values only)")
    ap.add_argument("--original", required=True, help="original Brush PLY")
    ap.add_argument("--variant", action="append", metavar="LABEL=PATH",
                    help="variant file to add (repeatable)")
    ap.add_argument("--time", action="append", metavar="LABEL=SECONDS",
                    help="real processing time for a variant (repeatable)")
    ap.add_argument("--round-trip", action="append",
                    metavar="LABEL=SRC.ply,DECODED.ply",
                    help="measure Route B round-trip error for a variant")
    ap.add_argument("--voxel-size", type=float, default=5.0,
                    help="voxel edge for the coverage proxy (default 5.0)")
    ap.add_argument("--out-csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out-md", default=str(DEFAULT_MD))
    a = ap.parse_args(argv)

    try:
        variants = _parse_pairs(a.variant, "--variant")
        times = _parse_pairs(a.time, "--time")
        round_trips = _parse_pairs(a.round_trip, "--round-trip",
                                   split_csv=True)
        if a.voxel_size <= 0:
            raise ValueError("--voxel-size must be > 0")
        rows = collect_rows(a.original, variants, times, round_trips,
                            a.voxel_size)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    csv_path = write_csv(rows, a.out_csv)
    md_path = write_markdown(rows, a.original, a.out_md)

    width = max(len(r["variant"]) for r in rows)
    for r in rows:
        print(f"{r['variant']:<{width}}  {r['gaussians']:>7}  "
              f"{r['file_size_mb']:>7.2f} MB  {r['size_ratio']:>5.3f}x  "
              f"{r['processing_time_s']}  {r['brush_load_status']}")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
