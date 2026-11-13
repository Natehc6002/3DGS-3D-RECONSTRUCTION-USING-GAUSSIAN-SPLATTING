"""Route B encode: quantize a float32 PLY to a compressed .npz archive.

Per-attribute uint8/uint16 quantization with stored min/max; rotations
are quaternions normalized then quantized in [-1, 1] (renormalized on
decode). The archive is for storage and transfer ONLY - Brush cannot
read it; run dequantize.py to get a float32 PLY back (same size as the
input PLY, so the compression does not shrink what Brush opens).

Usage:
    python -m optimization.quantize.quantize --input in.ply --output out.npz
        [--pos-bits 16] [--rot-bits 8] [--scale-bits 16]
        [--opacity-bits 16] [--dc-bits 16] [--rest-bits 8] [--json r.json]

Exit codes: 0 success, 2 bad arguments or paths, 3 failure.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from optimization.quantize import core


def main():
    ap = argparse.ArgumentParser(description="Route B: quantize PLY to npz")
    ap.add_argument("--input", required=True, help="original PLY (never modified)")
    ap.add_argument("--output", required=True, help=".npz archive to create")
    ap.add_argument("--pos-bits", type=int, default=16, choices=(8, 16))
    ap.add_argument("--rot-bits", type=int, default=8, choices=(8, 16))
    ap.add_argument("--scale-bits", type=int, default=16, choices=(8, 16))
    ap.add_argument("--opacity-bits", type=int, default=16, choices=(8, 16))
    ap.add_argument("--dc-bits", type=int, default=16, choices=(8, 16))
    ap.add_argument("--rest-bits", type=int, default=8, choices=(8, 16))
    ap.add_argument("--json", help="also write the report to this JSON file")
    a = ap.parse_args()

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
    opts = {"pos": a.pos_bits, "rot": a.rot_bits, "scale": a.scale_bits,
            "opacity": a.opacity_bits, "dc": a.dc_bits, "rest": a.rest_bits}
    try:
        core.validate_opts(opts)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        stats = core.encode_ply(src, dst, opts)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if dst.exists():
            dst.unlink()
        return 3
    except Exception as e:
        print(f"ERROR: encoding failed: {e}", file=sys.stderr)
        if dst.exists():
            dst.unlink()
        return 3
    elapsed = time.time() - t0

    s_in, s_out = src.stat().st_size, dst.stat().st_size
    rep = stats["report"]
    print(f"Input:      {src}")
    print(f"Archive:    {dst}")
    print(f"Settings:   bits pos={opts['pos']} rot={opts['rot']} "
          f"scale={opts['scale']} opacity={opts['opacity']} "
          f"dc={opts['dc']} rest={opts['rest']}")
    print(f"Gaussians:  {stats['count']}   properties: {len(stats['properties'])}")
    print(f"File size:  {s_in / 1e6:.2f} MB -> {s_out / 1e6:.2f} MB  "
          f"(ratio {s_in / s_out:.2f}x)")
    print("\nRound-trip validation (decode back to float32, per attribute):")
    print(f"  {'property':12s} {'bits':>4s} {'max_abs_err':>12s} "
          f"{'mean_abs_err':>12s}  in_step")
    for n, m in rep["per_attribute"].items():
        print(f"  {n:12s} {m['bits']:4d} {m['max_abs_err']:12.6g} "
              f"{m['mean_abs_err']:12.6g}  {'yes' if m['within_step_bound'] else 'NO'}")
    p = rep["position"]
    print(f"\nPosition:   max abs err {p['max_abs_err']:.6g} units, "
          f"mean {p['mean_abs_err']:.6g}")
    print(f"            extent {['%.2f' % x for x in p['extent_xyz']]}, "
          f"diagonal {p['diagonal']:.2f}")
    print(f"            max err / axis extent "
          f"{['%.4f%%' % (x * 100) for x in p['max_rel_to_axis_extent']]}, "
          f"/ diagonal {p['max_rel_to_diagonal'] * 100:.5f}%")
    if rep["quaternion"]:
        q = rep["quaternion"]
        print(f"Quaternion: pre-renorm norm min {q['pre_renorm_norm_min']:.6f} "
              f"max {q['pre_renorm_norm_max']:.6f}; "
              f"post-renorm min {q['post_renorm_norm_min']:.6f} "
              f"max {q['post_renorm_norm_max']:.6f}; "
              f"max component err {q['max_component_err']:.6f}")
    print(f"Finite:     {rep['finite']}   "
          f"all errors within bounds (rot = post-renormalization): "
          f"{rep['within_step_bound']}")
    print(f"\nNOTE: archive is storage/transfer only. Brush cannot read it;")
    print(f"      dequantize.py writes a float32 PLY of the same size as the")
    print(f"      input ({s_in / 1e6:.2f} MB) for viewing.")
    print(f"Time:       {elapsed:.1f} s")
    if a.json:
        payload = {"input": str(src), "archive": str(dst),
                   "size_in": s_in, "size_out": s_out,
                   "time_s": round(elapsed, 3), "settings": opts,
                   "stats": stats}
        Path(a.json).write_text(json.dumps(payload, indent=2, default=float))
        print(f"Wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
