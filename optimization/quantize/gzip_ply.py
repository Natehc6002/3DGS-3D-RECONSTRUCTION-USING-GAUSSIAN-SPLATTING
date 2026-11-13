"""Route C: gzip baseline for PLY files.

Lossless byte-for-byte compression of the whole file. Verification
re-gzips nothing: the decompressed bytes must equal the input bytes.
Brush cannot open .gz directly; gunzip restores the exact original.

Usage:
    python -m optimization.quantize.gzip_ply --input in.ply --output out.ply.gz

Exit codes: 0 success, 2 bad arguments or paths, 3 failure.
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Route C: gzip a PLY losslessly")
    ap.add_argument("--input", required=True, help="PLY (never modified)")
    ap.add_argument("--output", required=True, help=".gz to create")
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
    if not str(dst).endswith(".gz"):
        print("ERROR: output name must end with .gz", file=sys.stderr)
        return 2

    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        raw = src.read_bytes()
        packed = gzip.compress(raw, compresslevel=6, mtime=0)
        if gzip.decompress(packed) != raw:
            raise RuntimeError("gzip round trip is not byte-identical")
        dst.write_bytes(packed)
    except Exception as e:
        print(f"ERROR: gzip failed: {e}", file=sys.stderr)
        if dst.exists():
            dst.unlink()
        return 3
    elapsed = time.time() - t0

    s_in, s_out = src.stat().st_size, dst.stat().st_size
    print(f"Input:   {src}")
    print(f"Output:  {dst}")
    print(f"Size:    {s_in / 1e6:.2f} MB -> {s_out / 1e6:.2f} MB  "
          f"(ratio {s_in / s_out:.2f}x)")
    print(f"Verify:  decompressed bytes identical to input: True")
    print(f"Time:    {elapsed:.1f} s")
    print("NOTE: lossless; gunzip restores the exact original file. "
          "Brush opens the original (or gunzipped) PLY, not the .gz.")
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"input": str(src), "output": str(dst),
             "size_in": s_in, "size_out": s_out,
             "ratio": round(s_in / s_out, 3),
             "byte_identical": True,
             "time_s": round(elapsed, 3)}, indent=2))
        print(f"Wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
