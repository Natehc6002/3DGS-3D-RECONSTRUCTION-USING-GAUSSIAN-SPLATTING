"""End-to-end Gaussian optimization pipeline for Brush-exported PLY scenes.

Stages (in order):
    inspect        -> read the input, report counts/properties/SH degree
    smart_prune    -> spatially safe pruning (voxel budget, floater flags)
    mirror_clip    -> optional: clip content below the dominant plane
    sh_truncate    -> optional Route A: SH degree truncation of the pruned PLY
    normalize      -> optional: centre + scale (+ optional upright rotation)
                      with a sidecar JSON; runs last so every earlier
                      stage keeps working in the input's own units
    route_b        -> optional Route B: uint8/uint16 .npz archive + decode check
    report         -> write <name>_pipeline.json and append one CSV row

Every stage after inspect is opt-in or follows the documented defaults
(smart prune on; mirror clip, SH truncation, normalisation and Route B
off), so a default run produces the same files as before this stage list.

The input PLY is never modified and existing outputs are never overwritten.

Project folder convention (matches core/project_manager.py):
    if <output_dir> contains project.json it is treated as a 3DGS Studio
    project root, optimized PLYs go to <output_dir>/scenes/optimized/ and
    the report + CSV log go to <output_dir>/results/. Otherwise every file
    is written directly into <output_dir>.

Usage:
    python -m optimization.pipeline --input in.ply --output-dir dir
        [--no-prune] [--max-voxel-fraction 0.3] [--voxel-size 5.0]
        [--importance-threshold 1e-4] [--knn-k 5] [--iso-percentile 99]
        [--size-percentile 99.9] [--body-percentile-low 0.5]
        [--body-percentile-high 99.5] [--body-margin 20]
        [--remove-singleton-floaters] [--needle-cleanup] [--mirror-clip]
        [--sh-degree {0,1,2,3}] [--route-b] [--no-coverage]
        [--normalize] [--upright] [--up-axis X Y Z] [--no-normalize-scale]
        [--report-name NAME.json] [--quiet]

Library use (the desktop app calls this):
    from optimization.pipeline import run_pipeline
    result = run_pipeline(input_ply, output_dir, sh_degree=1, route_b=True)
    # result is a plain JSON-serializable dict, same content as the report

Exit codes: 0 success, 2 bad arguments/paths/existing outputs, 3 stage failure.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from plyfile import PlyData

from optimization import analyze_scene, inspect_ply
from optimization.normalize import normalize as normalize_cli
from optimization.pruning import coverage_check, smart_prune
from optimization.pruning import mirror_clip as mirror_clip_mod
from optimization.quantize import core as route_b_core
from optimization.quantize import sh_truncate

SCHEMA_VERSION = 1
CSV_NAME = "pipeline_runs.csv"
CSV_COLUMNS = (
    "generated_at", "status", "input", "project",
    "gaussians_in", "gaussians_out", "size_in_bytes", "size_out_bytes",
    "ratio", "prune", "mirror_clip", "sh_degree", "normalize", "route_b",
    "prune_s", "mirror_clip_s", "sh_truncate_s", "normalize_s", "route_b_s",
    "total_s", "error", "report",
)

# Defaults follow the Brush verdicts only: smart pruning is on (voxel
# safety invariant measured - 0 voxels emptied vs baseline 610); SH
# truncation is off because no verdict reports no visible colour or
# shading loss versus the original (smart_shd2 opened ok but colour
# shift / flat shading were not judged, shd1/shd0 are "not tested"),
# so it stays an optional export; Route B is off because it is
# storage-only (Brush cannot read .npz) and its decoded PLY is
# "not tested".
DEFAULTS = {
    "max_voxel_fraction": 0.3,
    "voxel_size": 5.0,
    "importance_threshold": 1e-4,
    "knn_k": 5,
    "iso_percentile": 99.0,
    "size_percentile": 99.9,
    "body_percentile_low": 0.5,
    "body_percentile_high": 99.5,
    "body_margin": 20.0,
}


class _StageFailure(Exception):
    """A stage failed; the message is written into the report."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _jsonable(obj):
    """Return obj with numpy/Path values replaced by plain Python types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _validate_options(prune, sh_degree, route_b, coverage, opts, extra,
                      normalize=False, upright=False, up_axis=None,
                      no_scale=False, needle_cleanup=False,
                      mirror_clip=False):
    unknown = set(extra) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown option(s): {sorted(unknown)}")
    if sh_degree is not None and sh_degree not in (0, 1, 2, 3):
        raise ValueError(f"sh_degree must be 0..3 or None, got {sh_degree}")
    if not 0.0 < opts["max_voxel_fraction"] <= 1.0:
        raise ValueError("max_voxel_fraction must be in (0, 1]")
    if opts["voxel_size"] <= 0:
        raise ValueError("voxel_size must be > 0")
    if opts["importance_threshold"] < 0:
        raise ValueError("importance_threshold must be >= 0")
    if opts["knn_k"] < 1:
        raise ValueError("knn_k must be >= 1")
    for name in ("iso_percentile", "size_percentile"):
        if not 0.0 < opts[name] <= 100.0:
            raise ValueError(f"{name} must be in (0, 100]")
    if not (0.0 <= opts["body_percentile_low"] < opts["body_percentile_high"] <= 100.0):
        raise ValueError("body percentiles must satisfy 0 <= low < high <= 100")
    if opts["body_margin"] < 0:
        raise ValueError("body_margin must be >= 0")
    for flag in (prune, route_b, coverage, normalize, upright, no_scale,
                 needle_cleanup, mirror_clip):
        if not isinstance(flag, bool):
            raise ValueError("stage flags must be bool")
    if not normalize and (upright or up_axis is not None or no_scale):
        raise ValueError(
            "--upright/--up-axis/--no-normalize-scale require --normalize")
    if up_axis is not None:
        if len(up_axis) != 3 or float(np.linalg.norm(up_axis)) <= 0:
            raise ValueError("up_axis must be a non-zero 3-vector")


def _timings_from_stages(stages, total_s):
    timings = {"total_s": round(total_s, 3)}
    for st in stages:
        timings[st["name"]] = st["time_s"]
    return timings


def run_pipeline(input_ply, output_dir, *,
                 prune=True,
                 sh_degree=None,
                 route_b=False,
                 coverage=True,
                 report_name=None,
                 quiet=False,
                 singleton_floaters=False,
                 needle_cleanup=False,
                 mirror_clip=False,
                 normalize=False,
                 upright=False,
                 up_axis=None,
                 no_scale=False,
                 **kwargs):
    """Run inspect -> smart prune -> clip -> SH -> normalise -> Route B.

    Returns a plain dict (JSON-serializable, schema_version 1) that is also
    written as the report file on every run where at least one stage ran:

        status        "ok" | "error"
        error         None | message
        error_kind    None | "bad_input" | "exists" | "stage"
        exit_code     0 | 2 | 3 (same meaning as the CLI)
        input         {path, size_bytes, gaussians, properties, sh_degree,
                       comments}
        project       project root (str) when output_dir holds project.json
        output_dir    str
        stages        [{name, status: ok|skipped|error, time_s, error, ...}]
        inspect       full inspect_ply report of the input
        smart_prune   smart_prune removal report (when the stage ran)
        mirror_clip   mirror_clip plan (when the stage ran)
        coverage      coverage_check proxy (when computed)
        sh_truncate   sh_truncate info (when the stage ran)
        normalize     {plan, sidecar, roundtrip, relative, sidecar_path}
                      (when the stage ran)
        route_b       {npz, source_ply, q_back_ply, encode_*,
                       dequantize_check}
        counts        {input, pruned, clipped, output}
        sizes         {input_bytes, pruned_bytes, final_bytes, npz_bytes,
                       q_back_bytes} (entries that do not apply are None)
        ratios        {pruned_vs_input, final_vs_input, npz_vs_input}
        timings       {total_s, <stage>: s, ...}
        outputs       {pruned_ply, clipped_ply, sh_ply, normalized_ply,
                       normalize_sidecar, final_ply, npz, q_back_ply,
                       report_json, csv_log}
        warnings      [str]
        generated_at  UTC ISO timestamp

    Expected errors never raise: bad input/paths/existing outputs return
    status "error" with error_kind "bad_input"/"exists" and write nothing;
    a stage failure returns error_kind "stage", removes the files that run
    wrote, and leaves the error in the report.
    """
    t_start = time.perf_counter()
    src = Path(input_ply)
    outdir = Path(output_dir)
    opts = {k: kwargs.get(k, DEFAULTS[k]) for k in DEFAULTS}
    opts["singleton"] = bool(singleton_floaters)
    opts["needle"] = bool(needle_cleanup)

    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "ok",
        "error": None,
        "error_kind": None,
        "error_stage": None,
        "exit_code": 0,
        "input": {"path": str(src), "size_bytes": None, "gaussians": None,
                  "properties": None, "sh_degree": None, "comments": []},
        "project": None,
        "output_dir": str(outdir),
        "stages": [],
        "inspect": None,
        "smart_prune": None,
        "mirror_clip": None,
        "coverage": None,
        "sh_truncate": None,
        "normalize": None,
        "route_b": None,
        "counts": {"input": None, "pruned": None, "clipped": None,
                   "output": None},
        "sizes": {"input_bytes": None, "pruned_bytes": None,
                  "final_bytes": None, "npz_bytes": None,
                  "q_back_bytes": None},
        "ratios": {},
        "timings": {},
        "outputs": {"pruned_ply": None, "clipped_ply": None, "sh_ply": None,
                    "normalized_ply": None, "normalize_sidecar": None,
                    "final_ply": None,
                    "npz": None, "q_back_ply": None, "report_json": None,
                    "csv_log": None},
        "warnings": [],
    }
    warnings = result["warnings"]
    stages = result["stages"]

    def _finish(kind, message, code):
        result["status"] = "error"
        result["error"] = message
        result["error_kind"] = kind
        result["exit_code"] = code
        result["timings"] = _timings_from_stages(stages, time.perf_counter() - t_start)
        return _jsonable(result)

    # ---------------- preflight (nothing written) ----------------
    try:
        _validate_options(prune, sh_degree, route_b, coverage, opts, kwargs,
                          normalize=normalize, upright=upright,
                          up_axis=up_axis, no_scale=no_scale,
                          needle_cleanup=needle_cleanup,
                          mirror_clip=mirror_clip)
    except ValueError as e:
        return _finish("bad_input", str(e), 2)
    if not src.is_file():
        return _finish("bad_input", f"input not found: {src}", 2)
    if outdir.exists() and not outdir.is_dir():
        return _finish("bad_input", f"output-dir is not a directory: {outdir}", 2)

    is_project = (outdir / "project.json").is_file()
    scenes_dir = outdir / "scenes" / "optimized" if is_project else outdir
    results_dir = outdir / "results" if is_project else outdir
    result["project"] = str(outdir) if is_project else None

    stem = src.stem
    prune_stem = f"{stem}_smart" if prune else stem
    clip_stem = f"{prune_stem}_clip" if mirror_clip else prune_stem
    sh_stem = f"{clip_stem}_shd{sh_degree}" if sh_degree is not None else clip_stem
    report_path = results_dir / (report_name or f"{stem}_pipeline.json")
    csv_path = results_dir / CSV_NAME
    result["outputs"]["report_json"] = str(report_path)
    result["outputs"]["csv_log"] = str(csv_path)

    pruned_path = scenes_dir / f"{prune_stem}.ply" if prune else None
    clip_path = scenes_dir / f"{clip_stem}.ply" if mirror_clip else None
    sh_path = scenes_dir / f"{sh_stem}.ply" if sh_degree is not None else None
    # norm_path/sidecar_path depend on the final stem, which is known only
    # after inspect (the SH stage may be skipped); both stay None until then.
    norm_path = None
    sidecar_path = None
    npz_path = None
    qback_path = None

    def _check_paths(paths):
        for p in paths:
            if p is not None and p.resolve() == src.resolve():
                return ("bad_input", f"output would overwrite the input: {p}", 2)
        existing = [str(p) for p in paths if p is not None and p.exists()]
        if existing:
            return ("exists",
                    "refusing to overwrite existing output(s): "
                    + ", ".join(existing), 2)
        return None

    problem = _check_paths([pruned_path, clip_path, report_path])
    if problem:
        return _finish(*problem)

    try:
        scenes_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return _finish("bad_input", f"cannot create output directories: {e}", 2)

    # ---------------- stages ----------------
    def _run_stage(name, fn, skip_reason=None):
        if skip_reason is not None:
            stages.append({"name": name, "status": "skipped",
                           "time_s": 0.0, "error": None,
                           "reason": skip_reason})
            warnings.append(f"{name} skipped: {skip_reason}")
            return None
        t0 = time.perf_counter()
        try:
            if quiet:
                with contextlib.redirect_stdout(io.StringIO()):
                    value = fn()
            else:
                value = fn()
        except Exception as e:
            stages.append({"name": name, "status": "error",
                           "time_s": round(time.perf_counter() - t0, 3),
                           "error": str(e)})
            raise _StageFailure(name, str(e)) from e
        stages.append({"name": name, "status": "ok",
                       "time_s": round(time.perf_counter() - t0, 3),
                       "error": None})
        return value

    try:
        def do_inspect():
            try:
                info = inspect_ply.inspect(src)
            except Exception as e:
                raise ValueError(f"could not inspect PLY: {e}") from e
            if info.get("missing_core"):
                raise ValueError(
                    f"PLY is missing core properties: {info['missing_core']}")
            if info.get("empty") or info.get("gaussian_count", 0) == 0:
                raise ValueError("input PLY has no Gaussians")
            return info

        info = _run_stage("inspect", do_inspect)
        result["inspect"] = info
        result["input"].update({
            "size_bytes": info["size_bytes"],
            "gaussians": info["gaussian_count"],
            "properties": info["property_count"],
            "sh_degree": info["sh_degree_from_props"],
            "comments": info["comments"],
        })
        result["counts"]["input"] = info["gaussian_count"]
        result["sizes"]["input_bytes"] = info["size_bytes"]
        in_deg = info["sh_degree_from_props"]
        last_count = info["gaussian_count"]

        # Phase 2 path check: the SH stage is skipped when the target degree
        # equals the input degree, so final names are known only now (still
        # before anything is written).
        sh_will_run = sh_degree is not None and not (
            in_deg is not None and sh_degree == in_deg)
        final_stem = sh_stem if sh_will_run else clip_stem
        if normalize:
            norm_path = scenes_dir / f"{final_stem}_norm.ply"
            sidecar_path = normalize_cli.default_sidecar_path(norm_path)
        if route_b:
            # Route B encodes the final PLY, so its names follow the norm
            # stem when normalisation is on.
            rb_stem = f"{final_stem}_norm" if normalize else final_stem
            npz_path = scenes_dir / f"{rb_stem}.npz"
            qback_path = scenes_dir / f"{rb_stem}_qback.ply"
        problem = _check_paths([sh_path if sh_will_run else None,
                                norm_path, sidecar_path,
                                npz_path, qback_path])
        if problem:
            return _finish(*problem)

        # ---- smart prune ----
        if not prune:
            _run_stage("smart_prune", None,
                       skip_reason="disabled (prune=False)")
            pruned_src = src
        else:
            prune_opts = {
                "max_fraction": opts["max_voxel_fraction"],
                "voxel_size": opts["voxel_size"],
                "importance_threshold": opts["importance_threshold"],
                "knn_k": opts["knn_k"],
                "iso_percentile": opts["iso_percentile"],
                "size_percentile": opts["size_percentile"],
                "body_low": opts["body_percentile_low"],
                "body_high": opts["body_percentile_high"],
                "body_margin": opts["body_margin"],
                "singleton": opts["singleton"],
                "needle": opts["needle"],
            }

            def do_prune():
                prune_info = smart_prune.prune(src, pruned_path, prune_opts)
                cov = None
                if coverage:
                    try:
                        orig_v = PlyData.read(str(src))["vertex"].data
                        out_v = PlyData.read(str(pruned_path))["vertex"].data
                        cov = coverage_check.coverage(
                            orig_v, out_v, opts["voxel_size"], 0.5)
                    except Exception as e:  # proxy only: never fail the run
                        warnings.append(f"coverage check failed: {e}")
                return prune_info, cov

            prune_info, cov = _run_stage("smart_prune", do_prune)
            result["smart_prune"] = prune_info
            result["coverage"] = cov
            result["counts"]["pruned"] = prune_info["pruned_count"]
            result["sizes"]["pruned_bytes"] = pruned_path.stat().st_size
            pruned_src = pruned_path
            last_count = prune_info["pruned_count"]

        # ---- mirror clip ----
        if not mirror_clip:
            _run_stage("mirror_clip", None, skip_reason="not requested")
            clip_src = pruned_src
        else:
            def do_clip():
                rep = analyze_scene.analyze(str(pruned_src))
                scene = analyze_scene.load_scene(str(pruned_src))
                plan = mirror_clip_mod.build_plan(rep, scene)
                mirror_clip_mod.write_clipped(pruned_src, clip_path, plan)
                plan.pop("remove", None)
                return plan

            clip_plan = _run_stage("mirror_clip", do_clip)
            result["mirror_clip"] = clip_plan
            result["counts"]["clipped"] = clip_plan["kept_count"]
            result["outputs"]["clipped_ply"] = str(clip_path)
            clip_src = clip_path
            last_count = clip_plan["kept_count"]

        # ---- SH truncation ----
        if sh_degree is None:
            _run_stage("sh_truncate", None,
                       skip_reason="not requested (sh_degree=None)")
            sh_out = None
        elif not sh_will_run:
            _run_stage("sh_truncate", None,
                       skip_reason=f"input is already SH degree {in_deg}")
            sh_out = None
        else:
            sh_info = _run_stage(
                "sh_truncate",
                lambda: sh_truncate.truncate(clip_src, sh_path, sh_degree))
            result["sh_truncate"] = sh_info
            sh_out = sh_path
            last_count = sh_info["gaussians"]

        # ---- normalisation (last coordinate change; sidecar for undo) ----
        if not normalize:
            _run_stage("normalize", None, skip_reason="not requested")
            norm_out = None
        else:
            norm_src = sh_out if sh_out is not None else clip_src

            def do_normalize():
                outcome = normalize_cli.normalize_file(
                    str(norm_src), str(norm_path), str(sidecar_path),
                    upright=upright, up_axis=up_axis, no_scale=no_scale)
                outcome["source"] = str(norm_src)
                outcome["sidecar_path"] = str(sidecar_path)
                for w in outcome["plan"].get("warnings", []):
                    warnings.append(f"normalize: {w}")
                return outcome

            norm_outcome = _run_stage("normalize", do_normalize)
            result["normalize"] = norm_outcome
            result["outputs"]["normalized_ply"] = str(norm_path)
            result["outputs"]["normalize_sidecar"] = str(sidecar_path)
            norm_out = norm_path

        # ---- final PLY bookkeeping ----
        final_ply = (norm_out or sh_out
                     or (clip_path if mirror_clip else None)
                     or pruned_path)
        if final_ply is not None:
            result["outputs"]["pruned_ply"] = (str(pruned_path)
                                               if pruned_path else None)
            result["outputs"]["sh_ply"] = str(sh_out) if sh_out else None
            result["outputs"]["final_ply"] = str(final_ply)
            if final_ply == pruned_path:
                result["sizes"]["final_bytes"] = result["sizes"]["pruned_bytes"]
            else:
                result["sizes"]["final_bytes"] = final_ply.stat().st_size
            result["counts"]["output"] = last_count
        else:
            result["counts"]["output"] = result["counts"]["input"]

        # ---- Route B ----
        if not route_b:
            _run_stage("route_b", None, skip_reason="not requested")
        else:
            encode_src = final_ply if final_ply is not None else src

            def do_route_b():
                enc = route_b_core.encode_ply(encode_src, npz_path)
                route_b_core.decode_ply(npz_path, qback_path)
                src_v = PlyData.read(str(encode_src))["vertex"].data
                back_v = PlyData.read(str(qback_path))["vertex"].data
                if len(back_v) != len(src_v):
                    raise ValueError(
                        f"dequantize check: count mismatch "
                        f"{len(src_v)} -> {len(back_v)}")
                if list(back_v.dtype.names) != list(src_v.dtype.names):
                    raise ValueError("dequantize check: property list changed")
                diffs = [np.abs(src_v[n].astype(np.float64)
                                - back_v[n].astype(np.float64)).max()
                         for n in src_v.dtype.names]
                enc_report = enc["report"]
                return {
                    "npz": str(npz_path),
                    "source_ply": str(encode_src),
                    "q_back_ply": str(qback_path),
                    "settings_bits": dict(route_b_core.DEFAULT_OPTS),
                    "encode_max_abs_err": float(
                        max(m["max_abs_err"]
                            for m in enc_report["per_attribute"].values())),
                    "encode_position_max_abs_err":
                        float(enc_report["position"]["max_abs_err"]),
                    "encode_within_step_bound":
                        bool(enc_report["within_step_bound"]),
                    "dequantize_check": {
                        "gaussians_match": True,
                        "properties_match": True,
                        "max_abs_err_vs_source": float(max(diffs)),
                    },
                }

            rb = _run_stage("route_b", do_route_b)
            result["route_b"] = rb
            result["outputs"]["npz"] = str(npz_path)
            result["outputs"]["q_back_ply"] = str(qback_path)
            result["sizes"]["npz_bytes"] = npz_path.stat().st_size
            result["sizes"]["q_back_bytes"] = qback_path.stat().st_size
            warnings.append(
                "Route B .npz is storage/transfer only; Brush opens the "
                "decoded float32 PLY (q_back), which is not smaller than "
                "its source")

        # ---- ratios + report ----
        sizes, counts = result["sizes"], result["counts"]
        if sizes["pruned_bytes"]:
            result["ratios"]["pruned_vs_input"] = round(
                sizes["input_bytes"] / sizes["pruned_bytes"], 4)
        if sizes["final_bytes"]:
            result["ratios"]["final_vs_input"] = round(
                sizes["input_bytes"] / sizes["final_bytes"], 4)
        if sizes["npz_bytes"]:
            result["ratios"]["npz_vs_input"] = round(
                sizes["input_bytes"] / sizes["npz_bytes"], 4)

        stages.append({"name": "report", "status": "ok",
                       "time_s": 0.0, "error": None})
        result["timings"] = _timings_from_stages(
            stages, time.perf_counter() - t_start)
        result = _jsonable(result)
    except _StageFailure as e:
        for p in (pruned_path, clip_path, sh_path, norm_path, sidecar_path,
                  npz_path, qback_path):
            try:
                if p is not None and p.exists():
                    p.unlink()
            except OSError:
                warnings.append(f"could not remove partial output: {p}")
        result["status"] = "error"
        result["error"] = str(e)
        result["error_kind"] = "stage"
        result["error_stage"] = e.stage
        result["exit_code"] = 3
        stages.append({"name": "report", "status": "ok",
                       "time_s": 0.0, "error": None})
        result["timings"] = _timings_from_stages(
            stages, time.perf_counter() - t_start)
        result = _jsonable(result)

    # Report and CSV are written for ok runs and for stage failures.
    try:
        report_path.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        message = f"could not write report: {e}"
        result["warnings"].append(message)
        if result["status"] == "ok":
            result["stages"][-1].update(status="error", error=message)
            result["status"] = "error"
            result["error"] = message
            result["error_kind"] = "stage"
            result["error_stage"] = "report"
            result["exit_code"] = 3
        return result

    try:
        _append_csv(csv_path, result)
    except OSError as e:
        result["warnings"].append(f"could not append CSV row: {e}")
    return result


def _append_csv(csv_path, result):
    row = {
        "generated_at": result["generated_at"],
        "status": result["status"],
        "input": result["input"]["path"],
        "project": result["project"] or "",
        "gaussians_in": result["counts"]["input"],
        "gaussians_out": result["counts"]["output"],
        "size_in_bytes": result["sizes"]["input_bytes"],
        "size_out_bytes": result["sizes"]["final_bytes"],
        "ratio": result["ratios"].get("final_vs_input", ""),
        "prune": "yes" if any(s["name"] == "smart_prune"
                              and s["status"] == "ok"
                              for s in result["stages"]) else "no",
        "mirror_clip": "yes" if any(s["name"] == "mirror_clip"
                                    and s["status"] == "ok"
                                    for s in result["stages"]) else "no",
        "sh_degree": (result["sh_truncate"]["degree_out"]
                      if result.get("sh_truncate") else ""),
        "normalize": "yes" if any(s["name"] == "normalize"
                                  and s["status"] == "ok"
                                  for s in result["stages"]) else "no",
        "route_b": "yes" if result.get("route_b") else "no",
        "prune_s": next((s["time_s"] for s in result["stages"]
                         if s["name"] == "smart_prune"), ""),
        "mirror_clip_s": next((s["time_s"] for s in result["stages"]
                               if s["name"] == "mirror_clip"), ""),
        "sh_truncate_s": next((s["time_s"] for s in result["stages"]
                               if s["name"] == "sh_truncate"), ""),
        "normalize_s": next((s["time_s"] for s in result["stages"]
                             if s["name"] == "normalize"), ""),
        "route_b_s": next((s["time_s"] for s in result["stages"]
                           if s["name"] == "route_b"), ""),
        "total_s": result["timings"].get("total_s", ""),
        "error": result["error"] or "",
        "report": result["outputs"]["report_json"],
    }
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _print_summary(result):
    print(f"\nPipeline:  {result['status'].upper()}"
          + (f" ({result['error']})" if result["error"] else ""))
    if result["project"]:
        print(f"Project:   {result['project']}")
    inp = result["input"]
    if inp["gaussians"] is not None:
        print(f"Input:     {inp['path']}  "
              f"({inp['gaussians']} Gaussians, "
              f"{inp['size_bytes'] / 1e6:.2f} MB, SH degree {inp['sh_degree']})")
    for st in result["stages"]:
        line = f"  {st['name']:12s} {st['status']:7s} {st['time_s']:7.3f} s"
        if st.get("reason"):
            line += f"  ({st['reason']})"
        if st.get("error"):
            line += f"  ERROR: {st['error']}"
        print(line)
    if result["counts"]["output"]:
        print(f"Gaussians: {result['counts']['input']} -> "
              f"{result['counts']['output']}")
    for key in ("pruned_vs_input", "final_vs_input", "npz_vs_input"):
        if key in result["ratios"]:
            print(f"Ratio {key}: {result['ratios'][key]:.3f}x")
    for key in ("pruned_ply", "clipped_ply", "sh_ply", "normalized_ply",
                "normalize_sidecar", "final_ply", "npz", "q_back_ply",
                "report_json", "csv_log"):
        path = result["outputs"].get(key)
        if path:
            print(f"{key:12s} {path}")
    for w in result["warnings"]:
        print(f"WARNING: {w}")
    print(f"Total time: {result['timings'].get('total_s', 0):.3f} s")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Gaussian optimization pipeline "
                    "(inspect -> smart prune -> clip -> SH -> "
                    "normalise -> Route B -> report)")
    ap.add_argument("--input", required=True, help="original PLY (never modified)")
    ap.add_argument("--output-dir", required=True,
                    help="output folder, or a 3DGS Studio project root "
                         "(contains project.json)")
    ap.add_argument("--no-prune", action="store_true",
                    help="skip the smart pruning stage")
    ap.add_argument("--max-voxel-fraction", type=float,
                    default=DEFAULTS["max_voxel_fraction"],
                    help="max fraction of any voxel that may be removed "
                         "(default 0.3)")
    ap.add_argument("--voxel-size", type=float,
                    default=DEFAULTS["voxel_size"],
                    help="voxel edge length in scene units (default 5.0)")
    ap.add_argument("--importance-threshold", type=float,
                    default=DEFAULTS["importance_threshold"],
                    help="flag sigmoid(opacity)*volume below this "
                         "(default 1e-4)")
    ap.add_argument("--knn-k", type=int, default=DEFAULTS["knn_k"])
    ap.add_argument("--iso-percentile", type=float,
                    default=DEFAULTS["iso_percentile"])
    ap.add_argument("--size-percentile", type=float,
                    default=DEFAULTS["size_percentile"])
    ap.add_argument("--body-percentile-low", type=float,
                    default=DEFAULTS["body_percentile_low"])
    ap.add_argument("--body-percentile-high", type=float,
                    default=DEFAULTS["body_percentile_high"])
    ap.add_argument("--body-margin", type=float,
                    default=DEFAULTS["body_margin"])
    ap.add_argument("--remove-singleton-floaters", action="store_true",
                    help="EXCEPTION: allow emptying a voxel that holds a "
                         "lone floater (default off)")
    ap.add_argument("--needle-cleanup", action="store_true",
                    help="also flag and remove needle Gaussians during "
                         "pruning (scene-relative criteria; default off)")
    ap.add_argument("--mirror-clip", action="store_true",
                    help="clip content below the dominant plane (candidate "
                         "mirror geometry); refuses when the plane or the "
                         "subject side looks wrong (default off)")
    ap.add_argument("--sh-degree", type=int, choices=(0, 1, 2, 3),
                    default=None,
                    help="Route A: truncate SH to this degree after pruning "
                         "(default: off)")
    ap.add_argument("--route-b", action="store_true",
                    help="Route B: write a .npz archive plus a decoded "
                         "q_back PLY and check the round trip (default off)")
    ap.add_argument("--no-coverage", action="store_true",
                    help="skip the voxel coverage proxy after pruning")
    ap.add_argument("--normalize", action="store_true",
                    help="centre the subject and scale to unit radius after "
                         "all other stages, writing a sidecar JSON for undo "
                         "(default off)")
    ap.add_argument("--upright", action="store_true",
                    help="with --normalize: rotate the estimated up "
                         "direction to Brush +y")
    ap.add_argument("--up-axis", type=float, nargs=3,
                    metavar=("X", "Y", "Z"),
                    help="with --normalize: manual up direction "
                         "(overrides --upright evidence)")
    ap.add_argument("--no-normalize-scale", action="store_true",
                    help="with --normalize: keep input units (centre only)")
    ap.add_argument("--report-name",
                    help="report file name (default <input stem>_pipeline.json)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-stage stdout (evidence blocks)")
    a = ap.parse_args(argv)

    result = run_pipeline(
        a.input, a.output_dir,
        prune=not a.no_prune,
        sh_degree=a.sh_degree,
        route_b=a.route_b,
        coverage=not a.no_coverage,
        report_name=a.report_name,
        quiet=a.quiet,
        singleton_floaters=a.remove_singleton_floaters,
        needle_cleanup=a.needle_cleanup,
        mirror_clip=a.mirror_clip,
        normalize=a.normalize,
        upright=a.upright,
        up_axis=a.up_axis,
        no_scale=a.no_normalize_scale,
        max_voxel_fraction=a.max_voxel_fraction,
        voxel_size=a.voxel_size,
        importance_threshold=a.importance_threshold,
        knn_k=a.knn_k,
        iso_percentile=a.iso_percentile,
        size_percentile=a.size_percentile,
        body_percentile_low=a.body_percentile_low,
        body_percentile_high=a.body_percentile_high,
        body_margin=a.body_margin,
    )
    _print_summary(result)
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
