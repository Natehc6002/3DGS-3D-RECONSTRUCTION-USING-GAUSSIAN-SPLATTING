# Brush checklist: mirror clip and normalize (A-D comparison)

Purpose: record the visual verdicts that decide whether the
`mirror_clip` and `normalize` stage defaults in
`optimization/pipeline.py` (`STAGE_DEFAULTS`) may be flipped from
**off** to **on**. Nothing in this file is a verdict - verdicts are
written down only after a visual check in Brush, one per file, with the
GPU the file was viewed on.

What the automated side already establishes (measured, not visual):

- **Invariance is proven on synthetic scenes with a constructed class
  gap** (`optimization/test_normalize_invariance.py`): under any rigid
  + uniform-scale transform the normalised output agrees to float32
  noise, and the mirror clip removes only the dim cluster with the
  weight guard.
- **Subject detection on real scenes is a heuristic validated on one
  real scene (the figurine).** Nothing here claims more than that.
- **Mirror clip counts depend on the earlier stages** (fresh figurine
  runs): 22,371 removed with smart prune only before the clip (weight
  share 0.445, plane confidence 0.414), 25,535 with needle cleanup and
  `--sh-degree 1` before it (weight share 0.449, plane confidence
  0.424). The pipeline runs the clip before SH truncation, so the SH
  flag does not change the clip count; needle cleanup does.
- **Upright confidence is low without the clip** (fresh figurine runs):
  0.032 (N2) and 0.031 (N3) below the 0.35 warning threshold without
  it, 0.470 (N4) with it - so the runs without the clip warn the scene
  may end up flipped 180 deg.

## The four variants (worked example: `projects/figure_01`)

One pipeline run produces B, C and D as a chain (same input, same
prune), so each step isolates exactly one stage. The run below was
executed from the repo root into an empty folder (runs never overwrite):

```powershell
python -m optimization.pipeline `
    --input projects\figure_01\scenes\original\figure_30k_30000.ply `
    --output-dir projects\figure_01\scenes\normalize_test `
    --mirror-clip --normalize --quiet
```

| # | Variant | File | Open in Brush (exact launcher command) |
|---|---|---|---|
| A | original (reference) | `projects\figure_01\scenes\original\figure_30k_30000.ply` | `.\optimization\tools\open_in_brush.ps1 -Ply projects\figure_01\scenes\original\figure_30k_30000.ply` |
| B | smart prune | `projects\figure_01\scenes\normalize_test\figure_30k_30000_smart.ply` | `.\optimization\tools\open_in_brush.ps1 -Ply projects\figure_01\scenes\normalize_test\figure_30k_30000_smart.ply` |
| C | smart prune + mirror clip | `projects\figure_01\scenes\normalize_test\figure_30k_30000_smart_clip.ply` | `.\optimization\tools\open_in_brush.ps1 -Ply projects\figure_01\scenes\normalize_test\figure_30k_30000_smart_clip.ply` |
| D | smart prune + clip + normalize | `projects\figure_01\scenes\normalize_test\figure_30k_30000_smart_clip_norm.ply` | `.\optimization\tools\open_in_brush.ps1 -Ply projects\figure_01\scenes\normalize_test\figure_30k_30000_smart_clip_norm.ply` |

Per-variant notes and check lists live next to the files:
`projects\figure_01\scenes\normalize_test\A_original.md`,
`B_smart_prune.md`, `C_mirror_clip.md`, `D_normalize.md` (local working
files - `projects/` is not committed; the commands above are the
durable record). Dry-run any launcher command with `-PrintOnly`; the
default `-Gpu auto` must show an NVIDIA name of type `DiscreteGpu`
(`optimization/tools/README_brush_gpu.md`).

## What to compare (same camera path for every pair)

1. **A vs B** - smart prune: holes, floaters, colour shift (the
   existing default; verdicts for this scene's prune history are in
   `optimization/README.md` "Brush verdicts").
2. **B vs C** - mirror clip: the mirrored copies under the marble floor
   are gone; nothing above the floor changed (the guards refuse the
   clip rather than cut the subject side).
3. **C vs D** - normalize: same geometry recentred and rescaled to unit
   subject radius; no colour change (opacity and `f_dc` are untouched
   by design); splat count equals C.
4. **A vs D** - the end-to-end result: first-view framing (the original
   verdict records the figure small and lower-right off-centre),
   overall integrity versus the reference.

## Planned comparison set N1-N4 (SH degree 3, no truncation)

A progressive set built from the ORIGINAL input (smart prune on in all
four; one output dir per run - runs never overwrite). Each step adds
one flag. The notes `planned\N1.md` .. `N4.md` are generated with
`optimization/tools/make_brush_notes.py`, so their verdicts come only
from `optimization/results/brush_verdicts.json` (a label that is absent
reads `not tested`).

| # | Flags added | Final PLY (under `projects\figure_01\scenes\normalize_test\planned\`) |
|---|---|---|
| N1 | `--normalize` | `n1\figure_30k_30000_smart_norm.ply` |
| N2 | `--normalize --upright` | `n2\figure_30k_30000_smart_norm.ply` |
| N3 | `--normalize --upright --needle-cleanup` | `n3\figure_30k_30000_smart_norm.ply` |
| N4 | `--normalize --upright --needle-cleanup --mirror-clip` | `n4\figure_30k_30000_smart_clip_norm.ply` |

```powershell
python -m optimization.pipeline --input projects\figure_01\scenes\original\figure_30k_30000.ply --output-dir projects\figure_01\scenes\normalize_test\planned\n1 --normalize
python -m optimization.pipeline --input projects\figure_01\scenes\original\figure_30k_30000.ply --output-dir projects\figure_01\scenes\normalize_test\planned\n2 --normalize --upright
python -m optimization.pipeline --input projects\figure_01\scenes\original\figure_30k_30000.ply --output-dir projects\figure_01\scenes\normalize_test\planned\n3 --normalize --upright --needle-cleanup
python -m optimization.pipeline --input projects\figure_01\scenes\original\figure_30k_30000.ply --output-dir projects\figure_01\scenes\normalize_test\planned\n4 --normalize --upright --needle-cleanup --mirror-clip
```

Measured from the four run reports (fresh runs; times are wall clock):

- counts: N1-N3 258309 -> 182489 -> output 182489; N4 258309 ->
  182489 -> clipped 156954 (clip removed 25,535, weight share 0.449,
  plane confidence 0.424, normal_flipped true).
- needle cleanup (N3/N4): 83 needles removed (prune by-reason total
  still 75,820).
- upright confidence: 0.032 (N2) and 0.031 (N3) WITHOUT the clip -
  both below the 0.35 warning threshold, so those runs carry the
  "may end up flipped 180 deg" warning; 0.470 (N4, WITH the clip) -
  no warning. Upright confidence is low without the clip.
- centre/scale confidence: 1.000 in all four (scale p90 -> 1).
- sidecar round-trip: 6.233e-08 (N1), 2.384e-07 (N2-N4).
- time / final size: 3.203 s / 43.07 MB (N1), 4.101 s / 43.07 MB
  (N2), 4.212 s / 43.07 MB (N3), 4.800 s / 37.04 MB (N4).

Open in Brush (exact launcher commands, run from the repo root;
dry-run any of them with `-PrintOnly`):

```powershell
.\optimization\tools\open_in_brush.ps1 -Ply projects\figure_01\scenes\original\figure_30k_30000.ply
.\optimization\tools\open_in_brush.ps1 -Ply projects\figure_01\scenes\normalize_test\planned\n1\figure_30k_30000_smart_norm.ply
.\optimization\tools\open_in_brush.ps1 -Ply projects\figure_01\scenes\normalize_test\planned\n2\figure_30k_30000_smart_norm.ply
.\optimization\tools\open_in_brush.ps1 -Ply projects\figure_01\scenes\normalize_test\planned\n3\figure_30k_30000_smart_norm.ply
.\optimization\tools\open_in_brush.ps1 -Ply projects\figure_01\scenes\normalize_test\planned\n4\figure_30k_30000_smart_clip_norm.ply
```

Compare in this order, same camera path every time: original vs N1
(first-view framing/centring), N1 vs N2 (the upright rotation), N2 vs
N3 (needle cleanup), N3 vs N4 (mirror clip: mirrored floor copies gone,
nothing above the floor changed). Per-variant notes with the checklists
live next to the files (`planned\N1.md` .. `N4.md`, local working files
- `projects/` is not committed).

## Record per variant (verdicts in `results/brush_verdicts.json`, table in `BRUSH_LOAD_STATUS`, then regenerate)

Verdicts go in `optimization/results/brush_verdicts.json` (the
generated notes read only that file; an absent label reads `not tested`)
and in `BRUSH_LOAD_STATUS` in `evaluate.py` (the evaluation table) -
keep the two in step.

- load ok / error (exact message if any),
- splat count and SH degree reported by Brush (must match the pipeline
  report),
- colour shift, flat shading, holes, floaters versus the pair above -
  each item separately, `not tested` when not judged,
- the GPU name + type from the Stats panel (part of the verdict),
- verdict wording: only what was seen; never infer a cause that was
  not established.

Then rebuild the evaluation table with the real report:

```powershell
python -m optimization.evaluate `
    --original projects\figure_01\scenes\original\figure_30k_30000.ply `
    --variant smart_default=projects\figure_01\scenes\normalize_test\figure_30k_30000_smart.ply `
    --report smart_default=projects\figure_01\scenes\normalize_test\figure_30k_30000_pipeline.json
```

(`--report` fills `needles_removed`, `clip_removed`,
`clip_weight_share`, `normalize_center/up/scale` and
`normalize_roundtrip_rel` from that report; without it they read
`not measured`, a stage that did not run reads `n/a (stage off)`.)

## Decision rule

- Flip `mirror_clip` / `normalize` in `STAGE_DEFAULTS` **only** after
  C and D carry an `opens ok` verdict for this scene and the B-vs-C /
  C-vs-D comparisons found no regression.
- If a variant fails to load or shows a regression: leave the default
  off, record the verdict, and file the problem - do not flip a default
  to make a command disappear.
- `upright` stays off regardless: without the clip its up confidence
  is 0.032/0.031 (below the 0.35 warning threshold), with it 0.470 -
  the N1-N4 set above records both.
