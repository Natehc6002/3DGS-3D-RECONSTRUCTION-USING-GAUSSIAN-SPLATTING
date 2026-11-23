"""Remove candidate mirror content below the dominant plane.

The figurine scene has mirrored copies of the figure and room baked in
as geometry under the marble floor. This tool clips that content:

1. Fits the dominant plane with analyze_scene (RANSAC, scene-relative),
   or takes a manual plane via --plane NX NY NZ D (plane n.x + D = 0).
2. Removes every Gaussian whose centre is farther than a scene-relative
   margin (default BELOW_MARGIN_RATIO x robust extent - identical to
   analyze_scene's "below dominant plane" count) on the negative side
   (n.x + D < -margin). For a detected plane the normal is first signed
   so the removed side carries the minority of scene weight (RANSAC
   normal signs are arbitrary; mirror content is the small side).
3. Guards, both scene-relative:
   - plane confidence must be at least MIN_PLANE_CONFIDENCE (0.30);
   - the detected subject centre must NOT lie on the removed side;
   - the removed weight share must not exceed --max-weight-fraction
     (default 0.6).
   Any violated guard refuses the clip (exit 2) instead of guessing.
4. Reports plane method + confidence, margin, removed/kept counts and
   weight share, and the removed bounding box.

Attributes are accessed by name. The input file is never modified;
header comments and property order are preserved in the output.

Usage:
    python -m optimization.pruning.mirror_clip --input in.ply
        --output out.ply [--plane NX NY NZ D] [--margin-ratio R]
        [--max-weight-fraction F] [--json report.json]

Exit codes: 0 success, 2 bad arguments, paths or refused clip, 3 failure.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from optimization import analyze_scene

DEFAULT_MAX_WEIGHT_FRACTION = 0.6


def signed_distances(xyz, normal, point):
    """(x - point) . normal per Gaussian; positive = keep side."""
    n = np.asarray(normal, dtype=np.float64)
    p = np.asarray(point, dtype=np.float64)
    return (np.asarray(xyz, dtype=np.float64) - p) @ n


def build_plan(rep, scene, plane=None, margin_ratio=None,
               max_weight_fraction=DEFAULT_MAX_WEIGHT_FRACTION):
    """Decide what to clip from an analyze report + loaded scene.

    plane = (normal, D) manual override for n.x + D = 0.
    Returns a dict with the remove mask and every reported number.
    Raises ValueError with a one-line user message when a guard fails.
    """
    extent = float(rep["extent"]["robust_extent"])
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("scene has no positive robust extent")

    if plane is not None:
        normal = np.asarray(plane[0], dtype=np.float64)
        d = float(plane[1])
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("--plane normal must be a non-zero vector")
        normal = normal / norm
        d = d / norm
        point = -d * normal
        confidence = 1.0
        method = "manual --plane"
    else:
        pl = rep["plane"]
        if pl.get("normal") is None or \
                float(pl.get("confidence", 0.0)) < analyze_scene.MIN_PLANE_CONFIDENCE:
            got = float(pl.get("confidence", 0.0))
            raise ValueError(
                f"no reliable dominant plane (confidence {got:.3f} < "
                f"{analyze_scene.MIN_PLANE_CONFIDENCE:g}); pass "
                f"--plane NX NY NZ D to clip against an explicit plane")
        normal = np.asarray(pl["normal"], dtype=np.float64)
        point = np.asarray(pl["point"], dtype=np.float64)
        confidence = float(pl["confidence"])
        method = str(pl["method"])

    ratio = analyze_scene.BELOW_MARGIN_RATIO if margin_ratio is None \
        else float(margin_ratio)
    if not np.isfinite(ratio) or ratio < 0:
        raise ValueError("--margin-ratio must be a finite number >= 0")
    margin = ratio * extent

    xyz = scene["xyz"]
    weights = scene["weights"]

    normal_flipped = False
    if plane is None:
        # RANSAC normal signs are arbitrary: sign the detected plane so the
        # removed side is the minority-weight side (mirror content is small).
        dist_raw = signed_distances(xyz, normal, point)
        total_raw = float(weights.sum())
        if total_raw > 0 and \
                float(weights[dist_raw < 0].sum()) / total_raw > 0.5:
            normal = -normal
            normal_flipped = True

    dist = signed_distances(xyz, normal, point)
    remove = dist < -margin

    subject = rep.get("subject") or {}
    subject_checked = subject.get("radius_p90") is not None
    if subject_checked:
        sub_dist = float(signed_distances(
            np.asarray(subject["centre"], dtype=np.float64)[None, :],
            normal, point)[0])
        if sub_dist < -margin:
            hint = ("check the sign of --plane (the removed side is "
                    "n.x + D < -margin)"
                    if plane is not None else
                    "pass --plane with the opposite normal or fix "
                    "the scene")
            raise ValueError(
                f"the subject centre lies {margin - sub_dist:.4g} on the "
                f"removed side of the plane; the plane or its sign looks "
                f"wrong - {hint}")

    total = float(weights.sum())
    weight_share = float(weights[remove].sum() / total) if total > 0 else 0.0
    if weight_share > float(max_weight_fraction):
        raise ValueError(
            f"clip would remove {weight_share * 100:.1f}% of scene weight "
            f"> limit {float(max_weight_fraction) * 100:.1f}%; pass "
            f"--max-weight-fraction to raise it or --plane to set the "
            f"plane explicitly")

    removed_xyz = xyz[remove]
    return {
        "remove": remove,
        "normal": [float(x) for x in normal],
        "point": [float(x) for x in point],
        "d": float(-normal @ point),
        "plane_confidence": confidence,
        "plane_method": method,
        "manual_plane": plane is not None,
        "normal_flipped": normal_flipped,
        "extent": extent,
        "margin_ratio": ratio,
        "margin": margin,
        "removed_count": int(remove.sum()),
        "kept_count": int((~remove).sum()),
        "removed_count_share": float(remove.mean()),
        "removed_weight_share": weight_share,
        "subject_checked": subject_checked,
        "subject_centre": ([float(x) for x in subject["centre"]]
                           if subject_checked else None),
        "max_weight_fraction": float(max_weight_fraction),
        "bbox_removed": ([float(removed_xyz[:, i].min())
                          for i in range(3)] +
                         [float(removed_xyz[:, i].max())
                          for i in range(3)]) if len(removed_xyz) else None,
    }


def write_clipped(src, dst, plan):
    """Write input minus the plan's removals; header/order/comments kept."""
    ply = PlyData.read(str(src))
    v = ply["vertex"].data
    if len(v) != len(plan["remove"]):
        raise ValueError("scene changed between planning and writing")
    kept = v[~plan["remove"]]
    if len(kept) == 0:
        raise ValueError("clipping would remove every Gaussian")
    out = PlyData([PlyElement.describe(kept, "vertex")],
                  text=False, byte_order="<", comments=list(ply.comments))
    out.write(str(dst))
    return len(kept)


def format_report(plan, info_out, sizes, elapsed):
    lines = []
    add = lines.append
    add(f"Input:      {info_out['input']}")
    add(f"Output:     {info_out['output']}")
    side = "manual" if plan["manual_plane"] else "detected"
    add(f"Plane:      n ({plan['normal'][0]:.4f}, {plan['normal'][1]:.4f}, "
        f"{plan['normal'][2]:.4f}), d {plan['d']:.4f}  [{side}]")
    add(f"            confidence {plan['plane_confidence']:.3f}  "
        f"method: {plan['plane_method']}")
    if plan.get("normal_flipped"):
        add("            normal signed so the removed side holds the "
            "minority of scene weight")
    add(f"Margin:     {plan['margin']:.4f} "
        f"({plan['margin_ratio']:g} x robust extent {plan['extent']:.4f}); "
        f"removed side is n.x + d < -margin")
    if plan["subject_checked"]:
        c = plan["subject_centre"]
        add(f"Subject:    centre ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}) is on "
            f"the kept side - protected")
    else:
        add("Subject:    no subject candidate; only the weight guard ran")
    n0 = plan["removed_count"] + plan["kept_count"]
    add(f"Gaussians:  {n0} -> {plan['kept_count']}  "
        f"(removed {plan['removed_count']}, "
        f"{plan['removed_count_share'] * 100:.2f}% of count, "
        f"{plan['removed_weight_share'] * 100:.2f}% of weight)")
    b = plan["bbox_removed"]
    if b is not None:
        add(f"Removed bbox min ({b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f})  "
            f"max ({b[3]:.2f}, {b[4]:.2f}, {b[5]:.2f})")
    add(f"File size:  {sizes[0] / 1e6:.2f} MB -> {sizes[1] / 1e6:.2f} MB  "
        f"(ratio {sizes[0] / sizes[1]:.2f}x)")
    add(f"Time:       {elapsed:.1f} s")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Clip candidate mirror content below the dominant plane")
    ap.add_argument("--input", required=True, help="original PLY (never modified)")
    ap.add_argument("--output", required=True, help="clipped PLY to create")
    ap.add_argument("--plane", type=float, nargs=4, metavar=("NX", "NY", "NZ", "D"),
                    help="manual plane n.x + D = 0; the removed side is "
                         "n.x + D < -margin (default: detected dominant plane)")
    ap.add_argument("--margin-ratio", type=float,
                    default=analyze_scene.BELOW_MARGIN_RATIO,
                    help="margin as a fraction of the robust extent "
                         f"(default {analyze_scene.BELOW_MARGIN_RATIO})")
    ap.add_argument("--max-weight-fraction", type=float,
                    default=DEFAULT_MAX_WEIGHT_FRACTION,
                    help="refuse the clip above this removed weight share "
                         f"(default {DEFAULT_MAX_WEIGHT_FRACTION})")
    ap.add_argument("--json", help="also write the report to this JSON file")
    a = ap.parse_args(argv)

    src, dst = Path(a.input), Path(a.output)
    if not src.is_file():
        print(f"ERROR: input not found: {src}", file=sys.stderr)
        return 2
    if src.resolve() == dst.resolve():
        print("ERROR: output must differ from input", file=sys.stderr)
        return 2
    if dst.exists():
        print(f"ERROR: output already exists, not overwriting: {dst}",
              file=sys.stderr)
        return 2
    if not 0.0 <= a.max_weight_fraction <= 1.0:
        print("ERROR: --max-weight-fraction must be in [0, 1]", file=sys.stderr)
        return 2
    if not np.isfinite(a.margin_ratio) or a.margin_ratio < 0:
        print("ERROR: --margin-ratio must be a finite number >= 0",
              file=sys.stderr)
        return 2

    t0 = time.time()
    try:
        rep = analyze_scene.analyze(str(src))
        scene = analyze_scene.load_scene(str(src))
        manual = (tuple(a.plane[:3]), a.plane[3]) if a.plane is not None else None
        plan = build_plan(rep, scene, plane=manual,
                          margin_ratio=a.margin_ratio,
                          max_weight_fraction=a.max_weight_fraction)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_clipped(src, dst, plan)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: mirror clip failed: {e}", file=sys.stderr)
        if dst.exists():
            dst.unlink()
        return 3
    elapsed = time.time() - t0

    sizes = (src.stat().st_size, dst.stat().st_size)
    info_out = {"input": str(src), "output": str(dst),
                "margin_ratio": a.margin_ratio,
                "max_weight_fraction": a.max_weight_fraction,
                "manual_plane": a.plane}
    for line in format_report(plan, info_out, sizes, elapsed):
        print(line)
    if a.json:
        payload = {"settings": info_out,
                   "sizes": {"input_bytes": sizes[0], "output_bytes": sizes[1]},
                   "time_s": round(elapsed, 3),
                   "info": json.loads(json.dumps(
                       {k: v for k, v in plan.items() if k != "remove"},
                       default=float))}
        Path(a.json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
