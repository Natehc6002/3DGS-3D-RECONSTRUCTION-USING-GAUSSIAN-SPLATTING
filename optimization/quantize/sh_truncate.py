"""Route A: SH degree truncation for Brush-compatible PLY size reduction.

f_rest_* holds 45 of 59 float32 properties (76% of the bytes) in a
degree-3 Brush export. Truncating SH degree 3 -> 2 -> 1 -> 0 removes
them. Remaining properties keep their original relative order; the
header comment "SH degree: N" is updated; other comments are kept.

LAYOUT (verified from Brush v0.3.0 source, crates/brush-serde):
- export.rs: rest_coeffs = [red_rest, green_rest, blue_rest].concat()
  -> channel-major blocks of (degree+1)^2-1 coefficients per channel,
  bands 1,2,3 (3+5+7) in standard SH order inside each channel.
- import.rs: coeffs_per_channel = f_rest_count / 3; blocks are read as
  [ch0 block][ch1 block][ch2 block]. A truncated file must therefore
  contain contiguous channel blocks sized to the NEW count.
- Truncation to degree D keeps, per channel, the first (D+1)^2-1
  coefficients and relabels them into contiguous blocks of the new
  size (values move down; original relative order is preserved).

Evidence (per-index energy, block-structure scores, cross-channel
correlation, band means) is printed from the real data before any
truncation. If the data does not show the channel-major block
structure, the script refuses to write (exit 3).

Usage:
    python -m optimization.quantize.sh_truncate --input in.ply --output out.ply --degree 2
    python -m optimization.quantize.sh_truncate --input in.ply --evidence-only

Exit codes: 0 success, 2 bad arguments or paths, 3 failure.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

# Brush writes exactly this comment form (export.rs splat_to_ply).
SH_COMMENT_PREFIX = "SH degree:"
DEG_FROM_REST_COUNT = {0: 0, 9: 1, 24: 2, 45: 3}
REST_COUNT_FROM_DEG = {v: k for k, v in DEG_FROM_REST_COUNT.items()}

BRUSH_SOURCE_NOTE = (
    "Band order within each channel (3+5+7 for bands 1..3) is taken from\n"
    "  Brush v0.3.0 source: crates/brush-serde/src/export.rs builds\n"
    "  rest_coeffs = [red_rest, green_rest, blue_rest].concat(), and\n"
    "  crates/brush-serde/src/import.rs reads coeffs_per_channel =\n"
    "  f_rest_count / 3 channel-major blocks (ply_gaussian.rs returns\n"
    "  f_rest_* in numeric order regardless of header order)."
)


def rest_names_numeric(v):
    """f_rest property names sorted by numeric suffix, not text order."""
    names = [n for n in v.dtype.names if n.startswith("f_rest_")]
    return sorted(names, key=lambda s: int(s.split("_")[-1]))


def rest_energies(v):
    """Mean-square energy per f_rest property, numeric index order."""
    return np.array([np.mean(v[n].astype(np.float64) ** 2)
                     for n in rest_names_numeric(v)], dtype=np.float64)


def detect_layout(energies):
    """Decide channel-major vs coeff-interleaved from entrywise similarity.

    channel_major: index i, i+cp, i+2cp are the same coefficient in
    different channels -> the three blocks of cp entries are similar.
    coeff_interleaved: indices 3t, 3t+1, 3t+2 are the same coefficient
    in different channels -> consecutive triples are similar.
    Lower normalized dissimilarity wins.
    """
    e = np.asarray(energies, dtype=np.float64)
    k = len(e)
    if k == 0 or k % 3 != 0:
        raise ValueError(f"f_rest count must be a positive multiple of 3, got {k}")
    g = float(e.mean())
    if not np.isfinite(g) or g <= 0:
        raise ValueError("non-positive mean energy; cannot analyse layout")
    cp = k // 3

    blocks = e.reshape(3, cp)
    diffs = [np.abs(blocks[a] - blocks[b]).mean()
             for a, b in ((0, 1), (0, 2), (1, 2))]
    ch_score = float(np.mean(diffs) / g)

    trips = e.reshape(-1, 3)
    trip_diffs = (np.abs(trips[:, 0] - trips[:, 1])
                  + np.abs(trips[:, 0] - trips[:, 2])
                  + np.abs(trips[:, 1] - trips[:, 2])) / 3.0
    ci_score = float(trip_diffs.mean() / g)

    layout = "channel_major" if ch_score < ci_score else "coeff_interleaved"
    return {"layout": layout,
            "channel_major_score": ch_score,
            "coeff_interleaved_score": ci_score,
            "channel_block_size": cp}


def cross_channel_correlation(v, cp):
    """Mean corr(idx, idx+cp) vs mean corr of random same-channel pairs."""
    names = rest_names_numeric(v)
    same = []
    for i in range(cp):
        a = v[names[i]].astype(np.float64)
        b = v[names[i + cp]].astype(np.float64)
        same.append(np.corrcoef(a, b)[0, 1])
    rnd = []
    for i in range(0, cp - 1):
        a = v[names[i]].astype(np.float64)
        b = v[names[i + 1]].astype(np.float64)
        rnd.append(np.corrcoef(a, b)[0, 1])
    return float(np.nanmean(same)), float(np.nanmean(rnd))


def band_means(channel_block):
    """Mean energy of bands 1/2/3 (3+5+7) inside one channel block."""
    b = np.asarray(channel_block, dtype=np.float64)
    if len(b) != 15:
        raise ValueError(f"expected 15 coefficients per channel, got {len(b)}")
    return (float(b[0:3].mean()), float(b[3:8].mean()), float(b[8:15].mean()))


def rest_keep_old_indices(k_rest, target_degree):
    """Old numeric indices, in new-file order, for a target SH degree.

    Channel-major input blocks of k_rest//3; output blocks of
    (target_degree+1)^2-1. Channel j, within-channel coeff i:
    old = j*cp_in + i, new position = j*cp_out + i.
    """
    if k_rest % 3 != 0:
        raise ValueError(f"f_rest count must be a multiple of 3, got {k_rest}")
    if target_degree not in REST_COUNT_FROM_DEG:
        raise ValueError(f"unsupported target degree {target_degree}")
    cp_in = k_rest // 3
    cp_out = REST_COUNT_FROM_DEG[target_degree] // 3
    if cp_out > cp_in:
        raise ValueError(
            f"cannot raise SH degree: input has {cp_in} coeffs/channel "
            f"(degree from count), target needs {cp_out}")
    return [j * cp_in + i for j in range(3) for i in range(cp_out)]


def update_sh_comment(comments, degree):
    """Replace the 'SH degree: N' comment in place; append if missing."""
    out = list(comments)
    new = f"{SH_COMMENT_PREFIX} {degree}"
    for i, c in enumerate(out):
        if c.startswith(SH_COMMENT_PREFIX):
            out[i] = new
            return out
    out.append(new)
    return out


def degree_from_comments(comments):
    for c in comments:
        if c.startswith(SH_COMMENT_PREFIX):
            try:
                return int(c[len(SH_COMMENT_PREFIX):].strip())
            except ValueError:
                return None
    return None


def build_output_names(orig_names, new_rest_names):
    """Original relative order, with the f_rest group swapped in place.

    Non-rest properties keep their positions; the new f_rest names sit
    where the old group sat, text-sorted like Brush's own exports.
    """
    out = []
    inserted = False
    for name in orig_names:
        if name.startswith("f_rest_"):
            if not inserted:
                out.extend(sorted(new_rest_names))
                inserted = True
            continue
        out.append(name)
    if new_rest_names and not inserted:
        raise ValueError("input has no f_rest properties to replace")
    return out


def build_evidence(v):
    """All layout evidence, computed from the real data."""
    names = rest_names_numeric(v)
    k = len(names)
    deg_in = DEG_FROM_REST_COUNT.get(k)
    energies = rest_energies(v)
    det = detect_layout(energies)
    cp = det["channel_block_size"]
    corr_same, corr_rnd = cross_channel_correlation(v, cp)

    bands = None
    if det["layout"] == "channel_major" and cp == 15:
        blocks = energies.reshape(3, cp)
        bands = [band_means(blocks[j]) for j in range(3)]
        monotonic = [b[0] > b[1] > b[2] for b in bands]
    else:
        monotonic = None
    return {
        "rest_count": k,
        "degree_from_count": deg_in,
        "energies": energies,
        "layout": det,
        "corr_same_coeff_across_channels": corr_same,
        "corr_random_within_channel": corr_rnd,
        "band_means": bands,
        "band_energy_monotonic_decreasing": monotonic,
    }


def print_evidence(ev, comments):
    print("=== SH layout evidence ===")
    hdr = degree_from_comments(comments)
    print(f"f_rest count: {ev['rest_count']}  "
          f"(degree from count: {ev['degree_from_count']}, "
          f"header comment: {hdr if hdr is not None else 'absent'})")
    e = ev["energies"]
    cp = ev["layout"]["channel_block_size"]
    print("Per-index mean-square energy (numeric index order):")
    for j in range(3):
        blk = e[j * cp:(j + 1) * cp]
        line = " ".join(f"{x:.4f}" for x in blk)
        print(f"  block {j} (idx {j * cp}-{(j + 1) * cp - 1}): {line}")
    d = ev["layout"]
    print("Block-structure scores (lower = better fit):")
    print(f"  channel_major (blocks of {cp}):    {d['channel_major_score']:.4f}")
    print(f"  coeff_interleaved (triples):       {d['coeff_interleaved_score']:.4f}")
    print(f"  -> detected: {d['layout']}")
    print("Cross-channel correlation: "
          f"mean corr(idx, idx+{cp}) = "
          f"{ev['corr_same_coeff_across_channels']:.3f}  vs  "
          f"mean corr(neighbouring idx) = "
          f"{ev['corr_random_within_channel']:.3f}")
    if ev["band_means"] is not None:
        print("Band means per channel (idx 0-2 / 3-7 / 8-14):")
        for j, (b1, b2, b3) in enumerate(ev["band_means"]):
            mono = ev["band_energy_monotonic_decreasing"][j]
            print(f"  ch{j}: {b1:.5f} / {b2:.5f} / {b3:.5f}   "
                  f"decreasing: {'YES' if mono else 'NO'}")
        if not any(ev["band_energy_monotonic_decreasing"]):
            print("  NOTE: energy does NOT decay with SH band in this scene;")
            print("  the 'lower bands carry more energy' expectation does not")
            print("  hold here, so band grouping is taken from Brush's source")
            print("  (authoritative), backed by the channel-block evidence.")
    print(BRUSH_SOURCE_NOTE)
    print()


def truncate(src, dst, target_degree, evidence_only=False):
    ply = PlyData.read(str(src), mmap=None)
    v = ply["vertex"].data
    comments = list(ply.comments)
    names = rest_names_numeric(v)
    k = len(names)
    if k not in DEG_FROM_REST_COUNT:
        raise ValueError(f"unsupported f_rest count {k} (expected 0/9/24/45)")
    deg_in = DEG_FROM_REST_COUNT[k]
    hdr = degree_from_comments(comments)
    if hdr is not None and hdr != deg_in:
        raise ValueError(
            f"header says SH degree {hdr} but f_rest count {k} implies "
            f"degree {deg_in}; refusing to truncate")
    ev = build_evidence(v)
    print_evidence(ev, comments)
    ev_public = {"rest_count": ev["rest_count"],
                 "degree_from_count": ev["degree_from_count"],
                 "layout": ev["layout"],
                 "corr_same_coeff_across_channels":
                     ev["corr_same_coeff_across_channels"],
                 "corr_random_within_channel": ev["corr_random_within_channel"],
                 "band_means": ev["band_means"],
                 "band_energy_monotonic_decreasing":
                     ev["band_energy_monotonic_decreasing"],
                 "energies": [float(x) for x in ev["energies"]]}
    if evidence_only:
        return {"evidence_only": True, "evidence": ev_public,
                "gaussians": int(len(v))}
    if target_degree > deg_in:
        raise ValueError(
            f"cannot truncate to degree {target_degree}: input is degree {deg_in}")
    if ev["layout"]["layout"] != "channel_major":
        raise ValueError(
            f"data shows {ev['layout']['layout']} layout, not channel_major; "
            f"refusing to truncate (wrong layout would corrupt colour)")

    keep_old = rest_keep_old_indices(k, target_degree)
    removed_old = [i for i in range(k) if i not in set(keep_old)]
    old_names = names

    new_rest_names = [f"f_rest_{m}" for m in range(len(keep_old))]
    out_names = build_output_names(list(v.dtype.names), new_rest_names)

    kept = np.zeros(len(v), dtype=np.dtype([(n, v.dtype[n]) for n in out_names]))
    for n in out_names:
        if n.startswith("f_rest_"):
            m = int(n.split("_")[-1])
            kept[n] = v[old_names[keep_old[m]]]
        else:
            kept[n] = v[n]

    # Error metric for truncation: absent f_rest reads as 0 (Brush
    # serde default), so the error IS the removed coefficient values.
    if removed_old:
        removed_vals = np.concatenate(
            [v[old_names[i]].astype(np.float64).ravel() for i in removed_old])
        all_vals = np.concatenate(
            [v[n].astype(np.float64).ravel() for n in old_names])
        err_max = float(np.abs(removed_vals).max())
        err_mean = float(np.abs(removed_vals).mean())
        energy_removed_pct = float(
            np.sum(removed_vals ** 2) / np.sum(all_vals ** 2) * 100.0)
    else:
        err_max = err_mean = energy_removed_pct = 0.0

    new_comments = update_sh_comment(comments, target_degree)
    out = PlyData([PlyElement.describe(kept, "vertex")],
                  text=False, byte_order="<", comments=new_comments)
    out.write(str(dst))

    return {
        "degree_in": deg_in,
        "degree_out": target_degree,
        "rest_in": k,
        "rest_out": len(keep_old),
        "props_in": len(v.dtype.names),
        "props_out": len(out_names),
        "removed_rest_energy_pct": energy_removed_pct,
        "removed_coeff_max_abs": err_max,
        "removed_coeff_mean_abs": err_mean,
        "layout": ev["layout"]["layout"],
        "comments_in": comments,
        "comments_out": new_comments,
        "gaussians": int(len(v)),
        "evidence": ev_public,
    }


def main():
    ap = argparse.ArgumentParser(description="Route A: SH degree truncation")
    ap.add_argument("--input", required=True, help="original PLY (never modified)")
    ap.add_argument("--output", help="truncated PLY to create")
    ap.add_argument("--degree", type=int, choices=(0, 1, 2, 3),
                    help="target SH degree (must be <= input degree)")
    ap.add_argument("--evidence-only", action="store_true",
                    help="print SH layout evidence and exit without writing")
    ap.add_argument("--json", help="also write the report to this JSON file")
    a = ap.parse_args()

    src = Path(a.input)
    if not src.is_file():
        print(f"ERROR: input not found: {src}", file=sys.stderr)
        return 2
    if not a.evidence_only:
        if not a.output or a.degree is None:
            print("ERROR: --output and --degree required "
                  "(or use --evidence-only)", file=sys.stderr)
            return 2
        dst = Path(a.output)
        if src.resolve() == dst.resolve():
            print("ERROR: output must differ from input", file=sys.stderr)
            return 2
        if dst.exists():
            print(f"ERROR: output already exists, not overwriting: {dst}",
                  file=sys.stderr)
            return 2
        dst.parent.mkdir(parents=True, exist_ok=True)
    else:
        dst = None

    t0 = time.time()
    try:
        info = truncate(src, dst, a.degree, evidence_only=a.evidence_only)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"ERROR: truncation failed: {e}", file=sys.stderr)
        if dst is not None and dst.exists():
            dst.unlink()
        return 3
    elapsed = time.time() - t0

    if info.get("evidence_only"):
        print(f"Time: {elapsed:.1f} s (evidence only, nothing written)")
        if a.json:
            Path(a.json).write_text(json.dumps(info, indent=2, default=float))
            print(f"Wrote {a.json}")
        return 0

    s_in, s_out = src.stat().st_size, dst.stat().st_size
    print(f"Input:      {src}")
    print(f"Output:     {dst}")
    print(f"SH degree:  {info['degree_in']} -> {info['degree_out']}   "
          f"f_rest {info['rest_in']} -> {info['rest_out']}   "
          f"properties {info['props_in']} -> {info['props_out']}")
    print(f"Comments:   {info['comments_out']}")
    print(f"Removed-coefficient error (absent = 0 in Brush): "
          f"max {info['removed_coeff_max_abs']:.5f}, "
          f"mean {info['removed_coeff_mean_abs']:.5f}, "
          f"rest energy removed {info['removed_rest_energy_pct']:.2f}%")
    print(f"Gaussians:  {info['gaussians']} (unchanged)")
    print(f"File size:  {s_in / 1e6:.2f} MB -> {s_out / 1e6:.2f} MB  "
          f"(ratio {s_in / s_out:.2f}x, vs 60.96 MB original "
          f"{60962475 / s_out:.2f}x)")
    print(f"Time:       {elapsed:.1f} s")
    if a.json:
        payload = {"input": str(src), "output": str(dst),
                   "size_in": s_in, "size_out": s_out,
                   "time_s": round(elapsed, 3), "info": info}
        Path(a.json).write_text(json.dumps(payload, indent=2, default=float))
        print(f"Wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
