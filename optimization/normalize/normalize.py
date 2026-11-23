"""Normalise a Brush PLY: centre the subject, scale to a documented
unit, optionally rotate the scene upright - and always write the sidecar
JSON the desktop app needs to map coordinates and undo the transform.

    python -m optimization.normalize.normalize --input in.ply --output out.ply
        [--sidecar out.normalize.json] [--colmap-dir SPARSE_DIR]
        [--center X Y Z] [--upright] [--up-axis X Y Z]
        [--target-radius R] [--scale S] [--no-scale]

    python -m optimization.normalize.normalize --undo side.json
        --input normalized.ply --output restored.ply

Evidence rules (see optimize/core.py): every automatic centre, up
direction and scale reports its method and confidence; low confidence
falls back (centre -> robust centre) or warns (up sign may be flipped).
Manual overrides always win. The input PLY is never modified and
existing outputs are never overwritten.

Exit codes: 0 success, 2 bad arguments/paths, 3 stage failure
(SH rotation verification or writing).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from optimization.normalize import core


def default_sidecar_path(output):
    out = Path(output)
    return out.with_name(out.stem + ".normalize.json")


def _validate_paths(src, dst, sidecar):
    if not Path(src).is_file():
        raise ValueError(f"input not found: {src}")
    if Path(src).suffix.lower() != ".ply":
        raise ValueError(f"input is not a .ply file: {src}")
    if str(Path(src)) == str(Path(dst)):
        raise ValueError("output must differ from input")
    if Path(dst).exists():
        raise ValueError(f"output already exists, not overwriting: {dst}")
    if Path(sidecar).exists():
        raise ValueError(
            f"sidecar already exists, not overwriting: {sidecar}")


def normalize_file(src, dst, sidecar_path, *, center=None, colmap_dir=None,
                   upright=False, up_axis=None, target_radius=1.0,
                   scale=None, no_scale=False):
    """Validate, transform and write <dst> + <sidecar_path>.

    Returns {"plan", "sidecar", "roundtrip", "relative", "gaussians"}
    for the caller's report. Raises ValueError (user error) or
    StageError (verification failure); writes nothing on failure.
    """
    _validate_paths(src, dst, sidecar_path)
    ply, v = core.read_vertex(src)
    plan = core.compute_plan(
        src, v=v,
        center=center,
        colmap_dir=colmap_dir,
        upright=upright,
        up_axis=up_axis,
        target_radius=target_radius,
        scale=scale,
        no_scale=no_scale,
    )
    out = core.apply_plan(v, plan)
    # Round trip through the JSON sidecar so serialization counts.
    sidecar = core.build_sidecar(plan, src, dst)
    rt = core.roundtrip_errors(v, plan, sidecar)
    sidecar["verification"]["roundtrip_max_error"] = rt
    rel = core.roundtrip_worst_relative(v, rt)
    sidecar["verification"]["roundtrip_worst_relative"] = rel
    if rel > core.ROUNDTRIP_REL_TOLERANCE:
        raise core.StageError(
            f"round-trip check failed: worst relative error {rel:.3e} > "
            f"{core.ROUNDTRIP_REL_TOLERANCE:g}; refusing to write")
    core.write_vertex(dst, out, list(ply.comments))
    Path(sidecar_path).write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return {"plan": plan, "sidecar": sidecar, "roundtrip": rt,
            "relative": rel, "gaussians": len(out)}


def _run_forward(a):
    if a.scale is not None and a.target_radius != 1.0:
        raise ValueError("use only one of --scale and --target-radius")
    if a.scale is not None and a.no_scale:
        raise ValueError("use only one of --scale and --no-scale")
    outcome = normalize_file(
        a.input, a.output, a.sidecar,
        center=a.center,
        colmap_dir=a.colmap_dir,
        upright=a.upright,
        up_axis=a.up_axis,
        target_radius=a.target_radius,
        scale=a.scale,
        no_scale=a.no_scale,
    )
    for line in core.format_report(outcome["plan"], sidecar=outcome["sidecar"],
                                   rt=outcome["roundtrip"],
                                   sidecar_path=a.sidecar):
        print(line)
    print(f"  wrote     {a.output}")
    return 0


def _run_undo(a):
    if not Path(a.undo).is_file():
        raise ValueError(f"sidecar not found: {a.undo}")
    if not Path(a.input).is_file():
        raise ValueError(f"input not found: {a.input}")
    if Path(a.input).suffix.lower() != ".ply":
        raise ValueError(f"input is not a .ply file: {a.input}")
    if str(Path(a.input)) == str(Path(a.output)):
        raise ValueError("output must differ from input")
    if Path(a.output).exists():
        raise ValueError(f"output already exists, not overwriting: {a.output}")
    try:
        sidecar = json.loads(Path(a.undo).read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"sidecar is not valid JSON: {err}") from err
    sidecar = core.sidecar_to_transform(sidecar)
    ply, v = core.read_vertex(a.input)
    restored = core.undo_vertex(v, sidecar)
    core.write_vertex(a.output, restored, list(ply.comments))
    print(f"Undo: {a.input} <- {a.undo}")
    print(f"  inverse of centre {sidecar['center']}  s = {sidecar['s']:g}")
    print(f"  wrote     {a.output}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Normalise a Brush PLY (centre, scale, optional "
                    "upright rotation) with a mandatory sidecar JSON")
    ap.add_argument("--input", required=True,
                    help="source PLY (never modified)")
    ap.add_argument("--output", required=True,
                    help="normalised PLY to create")
    ap.add_argument("--sidecar",
                    help="sidecar JSON path (default: "
                         "<output stem>.normalize.json)")
    ap.add_argument("--colmap-dir",
                    help="COLMAP sparse folder (cameras/images .bin|.txt) "
                         "for centre and orientation evidence")
    ap.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="manual subject centre (overrides evidence)")
    ap.add_argument("--upright", action="store_true",
                    help="rotate the estimated up direction to Brush +y")
    ap.add_argument("--up-axis", type=float, nargs=3,
                    metavar=("X", "Y", "Z"),
                    help="manual up direction (implies the upright "
                         "rotation; overrides --upright evidence)")
    ap.add_argument("--target-radius", type=float, default=1.0,
                    help="output subject radius (default 1)")
    ap.add_argument("--scale", type=float,
                    help="explicit uniform scale factor (overrides "
                         "--target-radius)")
    ap.add_argument("--no-scale", action="store_true",
                    help="keep input units (no uniform rescale)")
    ap.add_argument("--undo", metavar="SIDECAR",
                    help="apply the sidecar's inverse instead of "
                         "computing a plan")
    a = ap.parse_args(argv)

    if a.sidecar is None:
        a.sidecar = str(default_sidecar_path(a.output))
    if a.target_radius <= 0:
        print("ERROR: --target-radius must be > 0", file=sys.stderr)
        return 2
    if a.scale is not None and a.scale <= 0:
        print("ERROR: --scale must be > 0", file=sys.stderr)
        return 2
    if a.up_axis is not None:
        import math
        if math.sqrt(sum(x * x for x in a.up_axis)) <= 0:
            print("ERROR: --up-axis must be a non-zero vector",
                  file=sys.stderr)
            return 2

    try:
        if a.undo:
            return _run_undo(a)
        return _run_forward(a)
    except core.StageError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 3
    except ValueError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    except OSError as err:
        print(f"ERROR: writing failed: {err}", file=sys.stderr)
        return 3
    except Exception as err:  # noqa: BLE001 - one-line stage failure
        print(f"ERROR: normalisation failed: {err}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
