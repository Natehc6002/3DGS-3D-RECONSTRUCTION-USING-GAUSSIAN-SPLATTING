"""Write per-variant Brush comparison notes from a pipeline report.

Every number in a generated note comes from the pipeline report JSON.
Every verdict comes only from the verdicts JSON (default
optimization/results/brush_verdicts.json); a label absent from that file
reads "not tested". A generated note therefore cannot claim a Brush
verdict that was not recorded there.

Usage:
    python -m optimization.tools.make_brush_notes \
        --note N1=path/to/n1_pipeline.json=path/to/N1.md \
        [--note LABEL=REPORT.json=OUT.md ...] \
        [--verdicts optimization/results/brush_verdicts.json]

Exit codes: 0 success, 2 bad arguments/paths/report/verdicts file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_VERDICTS = (Path(__file__).resolve().parents[1]
                    / "results" / "brush_verdicts.json")
_NO_INSPECT = ("inspect", "report")


def load_verdicts(path):
    """Read the verdicts JSON -> {label: "not tested"|str|dict}."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"verdicts file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"verdicts file is not valid JSON: {p}: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("verdicts"),
                                                     dict):
        raise ValueError(
            f"verdicts file must be an object with a 'verdicts' object: {p}")
    for label, entry in data["verdicts"].items():
        if isinstance(entry, str):
            continue
        strings_ok = (isinstance(entry, dict)
                      and isinstance(entry.get("status"), str)
                      and all(isinstance(entry.get(k, ""), str)
                              for k in ("gpu", "detail")))
        if not strings_ok:
            raise ValueError(
                f"verdict for {label!r} must be a string or an object with a "
                f"'status' string (optional 'gpu'/'detail' strings): {p}")
    return data["verdicts"]


def format_verdict(verdicts, label):
    """One verdict line body; missing labels read 'not tested'."""
    entry = verdicts.get(label, "not tested")
    if isinstance(entry, str):
        return entry
    if entry["status"] == "not tested":
        return "not tested"
    parts = [entry["status"]]
    if entry.get("gpu"):
        parts.append(f"GPU: {entry['gpu']}")
    if entry.get("detail"):
        parts.append(entry["detail"])
    return "; ".join(parts)


def load_report(path):
    """Read and validate one successful pipeline report."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"pipeline report not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"pipeline report is not valid JSON: {p}: {e}") from e
    if not isinstance(data, dict) or "schema_version" not in data \
            or "stages" not in data:
        raise ValueError(f"not a pipeline report JSON: {p}")
    if data.get("status") != "ok":
        raise ValueError(
            f"pipeline report status is {data.get('status')!r}, not 'ok': {p}")
    if not (data.get("outputs") or {}).get("final_ply"):
        raise ValueError(f"pipeline report has no outputs.final_ply: {p}")
    return data


def _display(path):
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _mb(n):
    return f"{n / 1e6:.2f}"


def _stage_titles(report):
    return " + ".join(
        s["name"] for s in report["stages"]
        if s.get("status") == "ok" and s.get("name") not in _NO_INSPECT)


def _stages_line(report):
    out = []
    for s in report["stages"]:
        piece = f"{s.get('name')} {s.get('status')}"
        if s.get("reason"):
            piece += f" ({s['reason']})"
        piece += f" {s.get('time_s', 0.0)} s"
        out.append(piece)
    return "; ".join(out)


def _counts_line(report):
    c = report.get("counts") or {}

    def fmt(key, off):
        v = c.get(key)
        return off if v is None else str(v)

    return (f"Counts: input {fmt('input', 'n/a')} -> "
            f"pruned {fmt('pruned', 'n/a (prune off)')} -> "
            f"clipped {fmt('clipped', 'n/a (clip off)')} -> "
            f"output {fmt('output', 'n/a')}")


def _file_line(report):
    outputs = report["outputs"]
    counts = report.get("counts") or {}
    sizes = report.get("sizes") or {}
    ratios = report.get("ratios") or {}
    bits = []
    if counts.get("output") is not None:
        bits.append(f"{counts['output']} Gaussians")
    if sizes.get("final_bytes") is not None:
        bits.append(f"{_mb(sizes['final_bytes'])} MB")
    if ratios.get("final_vs_input") is not None:
        bits.append(f"{ratios['final_vs_input']}x vs input")
    return f"File: `{outputs['final_ply']}` ({', '.join(bits)})"


def _input_line(report):
    inp = report.get("input") or {}
    bits = []
    if inp.get("gaussians") is not None:
        bits.append(f"{inp['gaussians']} Gaussians")
    if inp.get("size_bytes") is not None:
        bits.append(f"{_mb(inp['size_bytes'])} MB")
    if inp.get("sh_degree") is not None:
        bits.append(f"SH degree {inp['sh_degree']}")
    return f"Input: `{inp.get('path')}` ({', '.join(bits)})"


def _prune_section(report):
    sp = report.get("smart_prune")
    if not sp:
        return None
    reasons = sp.get("removed_by_reason") or {}
    reason_txt = ", ".join(f"{k} {v}" for k, v in
                           sorted(reasons.items(), key=lambda kv: -kv[1]))
    needle_on = (sp.get("thresholds") or {}).get("needle_enabled")
    needle_txt = (f"on ({reasons.get('needle', 0)} removed)"
                  if needle_on else "off")
    return (f"Smart prune: removed {sp.get('removed_count')} of "
            f"{sp.get('original_count')} (by reason: {reason_txt}); "
            f"voxels emptied {sp.get('voxels_emptied')}; "
            f"needle cleanup {needle_txt}.")


def _clip_section(report):
    mc = report.get("mirror_clip")
    if not mc:
        return None
    return (f"Mirror clip: removed {mc.get('removed_count')} "
            f"(count share {mc['removed_count_share']:.3f}, "
            f"weight share {mc['removed_weight_share']:.3f}); "
            f"plane confidence {mc['plane_confidence']:.3f}; "
            f"normal_flipped {str(mc.get('normal_flipped')).lower()}; "
            f"margin {mc['margin']:.3f}; "
            f"subject checked {str(mc.get('subject_checked')).lower()}; "
            f"weight fraction limit {mc.get('max_weight_fraction')}.")


def _normalize_section(report):
    norm = report.get("normalize")
    if not norm:
        return None
    plan = norm.get("plan") or {}
    methods = plan.get("methods") or {}
    confidence = plan.get("confidence") or {}

    def part(key):
        if key not in methods:
            return f"{key} not applied"
        c = confidence.get(key)
        if c is None:
            return f"{key} {methods[key]}"
        return f"{key} {methods[key]} (confidence {c:.3f})"

    bits = [part("center"), part("upright"), part("scale")]
    if plan.get("up_angle_to_brush_y_deg") is not None:
        bits.append(f"up angle to Brush +y "
                    f"{plan['up_angle_to_brush_y_deg']:.2f} deg")
    if norm.get("relative") is not None:
        bits.append(f"sidecar round-trip rel {norm['relative']:.3e}")
    if norm.get("sidecar_path"):
        bits.append(f"sidecar `{norm['sidecar_path']}`")
    plan_warnings = plan.get("warnings") or []
    bits.append("plan warnings: "
                + ("; ".join(plan_warnings) if plan_warnings else "none"))
    return "Normalize: " + "; ".join(bits) + "."


def build_note(label, report, verdicts, report_display, verdicts_display):
    """Render one note as markdown (numbers from report, verdict from file)."""
    titles = _stage_titles(report) or label
    warnings = report.get("warnings") or []
    total = (report.get("timings") or {}).get("total_s")
    lines = [
        f"# {label} - {titles}",
        "",
        "Generated by `optimization/tools/make_brush_notes.py` from "
        f"`{report_display}`;",
        f"verdicts only from `{verdicts_display}` (a label absent from that "
        "file reads",
        '"not tested").',
        "",
        _file_line(report),
        _input_line(report),
        "",
        f"Stages: {_stages_line(report)}",
        "",
        _counts_line(report),
        "",
    ]
    for section in (_prune_section(report), _clip_section(report),
                    _normalize_section(report)):
        if section:
            lines += [section, ""]
    lines.append("Warnings: " + ("; ".join(warnings) if warnings else "none"))
    if total is not None:
        lines.append(f"Total time: {total} s.")
    lines += [
        "",
        "Open in Brush:",
        "",
        "```powershell",
        ".\\optimization\\tools\\open_in_brush.ps1 -Ply "
        + report["outputs"]["final_ply"],
        "```",
        "",
        "Check: load errors, splat count versus this report, colour shift, "
        "holes and",
        "floaters versus the original under the same camera path; record each "
        "item",
        'separately, "not tested" when not judged.',
        "",
        f"Verdict: {format_verdict(verdicts, label)}",
        "",
    ]
    return "\n".join(lines)


def _parse_note(item):
    parts = item.split("=")
    if len(parts) != 3:
        raise ValueError(
            f"--note expects LABEL=REPORT.json=OUT.md, got: {item}")
    label, report, out = (p.strip() for p in parts)
    if not label or not report or not out:
        raise ValueError(
            f"--note expects LABEL=REPORT.json=OUT.md, got: {item}")
    return label, report, out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Write per-variant Brush notes from pipeline reports "
                    "(verdicts only from the verdicts JSON)")
    ap.add_argument("--note", action="append", required=True,
                    metavar="LABEL=REPORT.json=OUT.md",
                    help="note to write (repeatable)")
    ap.add_argument("--verdicts", default=str(DEFAULT_VERDICTS),
                    help="verdicts JSON (default "
                         "optimization/results/brush_verdicts.json)")
    a = ap.parse_args(argv)

    try:
        verdicts = load_verdicts(a.verdicts)
        entries = []
        for item in a.note:
            label, report_arg, out_arg = _parse_note(item)
            entries.append((label, report_arg, Path(out_arg),
                            load_report(report_arg)))
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    verdicts_display = _display(a.verdicts)
    for label, report_arg, out_path, report in entries:
        note = build_note(label, report, verdicts, _display(report_arg),
                          verdicts_display)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(note, encoding="utf-8")
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
