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
    - New-stage columns (needles removed, clip removed and weight
      share, normalize centre/up/scale methods with confidence,
      normalize round-trip relative error) are read from the pipeline
      report given with --report LABEL=report.json; without a report
      they are "not measured", with a report whose stage did not run
      they are "n/a (stage off)".

Usage:
    python -m optimization.evaluate --original orig.ply \
        --variant baseline_prune=baseline.ply \
        --variant smart_default=smart.ply \
        --time baseline_prune=0.2 \
        --round-trip route_b_decoded=smart.ply,smart_qback.ply \
        --report smart_default=results/smart_pipeline.json \
        [--voxel-size 5.0] [--out-csv PATH] [--out-md PATH]

The two report files are rewritten on every run (they are regenerable
reports, unlike pipeline outputs).

Exit codes: 0 success, 2 bad arguments/paths.
"""
from __future__ import annotations

import argparse
import csv
import json
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
    "route_b_roundtrip_max_abs_err",
    "needles_removed", "clip_removed", "clip_weight_share",
    "normalize_center", "normalize_up", "normalize_scale",
    "normalize_roundtrip_rel",
    "brush_load_status", "psnr", "ssim", "fps", "notes",
)

# Columns filled from a pipeline report (--report) or left "not measured".
STAGE_FACT_COLUMNS = (
    "needles_removed", "clip_removed", "clip_weight_share",
    "normalize_center", "normalize_up", "normalize_scale",
    "normalize_roundtrip_rel",
)

# Manual Brush verdicts - update ONLY after a visual check in Brush.
# Anything not verified must stay "not tested"; each tested entry also
# records the GPU it was viewed on (see tools/README_brush_gpu.md).
# original: opened in Brush 0.3.0 on the NVIDIA RTX 4060 (launcher
# -Gpu auto; Stats panel "NVIDIA GeForce RTX 4060 Laptop GPU",
# DiscreteGpu, driver NVIDIA 581.86, Splats 258309, SH Degree 3,
# Bytes in use 60.96 MB). Screen recording: figure not centred at first
# view (small, lower right, camera faces the room), subject tiny vs the
# room background, long needle/streak splats whenever the camera leaves
# the training path, fog and ghost splats near the floor, mirrored
# copies of figure and background under the marble floor, general blur
# off the training path. Not judged: up-axis tilt, colour shift versus
# other files. Cause of the off-centre start not proven.
# smart_shd2: opened in Brush 0.3.0 on the Intel iGPU (Vulkan),
# reported SH degree 2 / 182,489 splats, no load error; front and side
# views clean, no holes noticed in the main body. Blur and streaks
# appear only when the camera leaves the training path and were not
# compared against the original under the same path, so they are not
# attributed to pruning or SH truncation. Colour shift / flat shading
# versus the original were NOT judged - that is why SH truncation stays
# off by default. Every other file is untested (no verdict was given).
BRUSH_LOAD_STATUS = {
    "original": ("opens ok; GPU: NVIDIA GeForce RTX 4060 Laptop GPU "
                 "(DiscreteGpu, driver NVIDIA 581.86)"),
    "baseline_prune": "not tested",
    "smart_default": "not tested",
    "smart_shd2": "opens ok (no load error); GPU: Intel iGPU (Vulkan)",
    "smart_shd1": "not tested",
    "smart_shd0": "not tested",
    "route_b_decoded": "not tested",
}

NOTES = {
    "original": ("reference Brush export; not centred at first view "
                 "(small, lower right), subject tiny vs room background; "
                 "needle/streak splats off the training path; fog/ghost "
                 "splats near the floor; mirrored copies under the "
                 "marble floor; blur off path; up-axis tilt and colour "
                 "shift not judged; cause of off-centre start not proven"),
    "baseline_prune": "baseline opacity prune (threshold 0.05)",
    "smart_default": "smart prune, max_voxel_fraction 0.3, no emptied voxels",
    "smart_shd2": ("smart default, SH truncated to degree 2; opens in "
                   "Brush 0.3.0 (SH degree 2, 182,489 splats) on the "
                   "Intel iGPU, front/side views clean, no holes noticed "
                   "in the main body; blur/streaks only off the training "
                   "path (not compared against original); colour shift / "
                   "flat shading not judged"),
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


def report_facts(path):
    """New-stage facts from a pipeline report JSON (real values only).

    Returns {column: str} for STAGE_FACT_COLUMNS. Raises ValueError or
    FileNotFoundError for a missing/foreign file (exit 2 upstream).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or "schema_version" not in data
            or "stages" not in data):
        raise ValueError(f"not a pipeline report JSON: {path}")
    facts = {col: "n/a (stage off)" for col in STAGE_FACT_COLUMNS}

    prune = data.get("smart_prune")
    if prune:
        if prune.get("thresholds", {}).get("needle_enabled"):
            facts["needles_removed"] = str(
                int(prune["removed_by_reason"]["needle"]))
        else:
            facts["needles_removed"] = "n/a (needle cleanup off)"

    clip = data.get("mirror_clip")
    if clip:
        facts["clip_removed"] = str(int(clip["removed_count"]))
        facts["clip_weight_share"] = (
            f"{float(clip['removed_weight_share']):.3f}")

    norm = data.get("normalize")
    if norm:
        plan = norm.get("plan") or {}
        methods = plan.get("methods") or {}
        confidence = plan.get("confidence") or {}

        def _fmt(key):
            if key not in methods:
                return "n/a (not applied)"
            conf = confidence.get(key)
            if conf is None:
                return str(methods[key])
            return f"{methods[key]} (confidence {float(conf):.3f})"

        facts["normalize_center"] = _fmt("center")
        facts["normalize_up"] = _fmt("upright")
        facts["normalize_scale"] = _fmt("scale")
        facts["normalize_roundtrip_rel"] = f"{float(norm['relative']):.3e}"
    return facts


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
                 voxel_size=5.0, reports=None):
    """Build one row dict per variant (including the original row).

    variants: ordered {label: path}, times: {label: seconds},
    round_trips: {label: (source_ply, decoded_ply)},
    reports: {label: pipeline report JSON path}.
    """
    times = times or {}
    round_trips = round_trips or {}
    reports = reports or {}
    src = Path(original_path)
    if not src.is_file():
        raise FileNotFoundError(f"original not found: {src}")
    orig_v = load_vertex(src)
    if len(orig_v) == 0:
        raise ValueError("original PLY has no Gaussians")
    orig_bytes = src.stat().st_size

    unknown_time = set(times) - set(variants)
    unknown_rt = set(round_trips) - set(variants)
    unknown_rep = set(reports) - set(variants)
    if unknown_time:
        raise ValueError(f"--time for unknown variant(s): {sorted(unknown_time)}")
    if unknown_rt:
        raise ValueError(
            f"--round-trip for unknown variant(s): {sorted(unknown_rt)}")
    if unknown_rep:
        raise ValueError(
            f"--report for unknown variant(s): {sorted(unknown_rep)}")

    rows = [{
        "variant": "original",
        "gaussians": int(len(orig_v)),
        "file_size_mb": round(orig_bytes / 1e6, 2),
        "size_ratio": 1.0,
        "processing_time_s": "n/a",
        "coverage_proxy": NA_NO_REMOVALS,
        "route_b_roundtrip_max_abs_err": "n/a",
        **{col: "n/a" for col in STAGE_FACT_COLUMNS},
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

        if label in reports:
            stage_facts = report_facts(reports[label])
        else:
            stage_facts = {col: QUALITY for col in STAGE_FACT_COLUMNS}

        rows.append({
            "variant": label,
            "gaussians": int(len(v)),
            "file_size_mb": round(size / 1e6, 2),
            "size_ratio": round(orig_bytes / size, 3),
            "processing_time_s": time_s,
            "coverage_proxy": coverage_proxy(orig_v, v, voxel_size),
            "route_b_roundtrip_max_abs_err": rt,
            **stage_facts,
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
              "needles removed", "clip removed", "clip weight share",
              "normalize centre (method, confidence)",
              "normalize up (method, confidence)",
              "normalize scale (method, confidence)",
              "normalize round-trip rel",
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
            str(r["needles_removed"]),
            str(r["clip_removed"]),
            str(r["clip_weight_share"]),
            str(r["normalize_center"]),
            str(r["normalize_up"]),
            str(r["normalize_scale"]),
            str(r["normalize_roundtrip_rel"]),
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
        "- Needles removed, clip removed/weight share and the normalize "
        "columns come from the pipeline report passed with "
        "--report LABEL=report.json; without a report they read "
        "\"not measured\", with a report whose stage did not run they read "
        "\"n/a (stage off)\". Needle cleanup off reads "
        "\"n/a (needle cleanup off)\".",
        "- Normalize centre/up/scale show the automatic method and its "
        "confidence from the report; normalize round-trip rel is the "
        "sidecar round-trip relative error recorded by the run.",
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
    ap.add_argument("--report", action="append", metavar="LABEL=REPORT.json",
                    help="pipeline report JSON for a variant; fills the "
                         "needles/clip/normalize columns with the run's "
                         "real numbers (repeatable)")
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
        reports = _parse_pairs(a.report, "--report")
        if a.voxel_size <= 0:
            raise ValueError("--voxel-size must be > 0")
        rows = collect_rows(a.original, variants, times, round_trips,
                            a.voxel_size, reports=reports)
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
