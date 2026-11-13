"""Route B decode: turn a quantized .npz archive into a float32 PLY.

Writes a normal binary little-endian PLY with the original header
comments and property order, so Brush can open it. The result has the
same Gaussian count and (barring header bytes) the same size as the
input PLY - the compression lives in the archive, not in the PLY.

Usage:
    python -m optimization.quantize.dequantize --input out.npz --output back.ply

Exit codes: 0 success, 2 bad arguments or paths, 3 failure.
"""
import argparse
import sys
import time
from pathlib import Path

from optimization.quantize import core


def main():
    ap = argparse.ArgumentParser(description="Route B: decode npz to PLY")
    ap.add_argument("--input", required=True, help=".npz archive (never modified)")
    ap.add_argument("--output", required=True, help="float32 PLY to create")
    a = ap.parse_args()

    src, dst = Path(a.input), Path(a.output)
    if not src.is_file():
        print(f"ERROR: archive not found: {src}", file=sys.stderr)
        return 2
    if src.resolve() == dst.resolve():
        print("ERROR: output must differ from input", file=sys.stderr)
        return 2
    if dst.exists():
        print(f"ERROR: output already exists, not overwriting: {dst}",
              file=sys.stderr)
        return 2

    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        stats = core.decode_ply(src, dst)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if dst.exists():
            dst.unlink()
        return 3
    except Exception as e:
        print(f"ERROR: decoding failed: {e}", file=sys.stderr)
        if dst.exists():
            dst.unlink()
        return 3
    elapsed = time.time() - t0

    s_in, s_out = src.stat().st_size, dst.stat().st_size
    print(f"Archive:    {src}  ({s_in / 1e6:.2f} MB)")
    print(f"Output:     {dst}  ({s_out / 1e6:.2f} MB)")
    print(f"Source PLY: {stats['source']}")
    print(f"Gaussians:  {stats['count']}   properties: {len(stats['properties'])}")
    print(f"Comments:   {stats['comments']}")
    print(f"Time:       {elapsed:.1f} s")
    print("Brush: open the output PLY (float32, same size as the input PLY).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
