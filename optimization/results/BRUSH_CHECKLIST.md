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

Optional fifth variant (record separately if checked): re-run the
pipeline command with `--upright` to also rotate the estimated up
direction to Brush +y; it gets its own file and its own verdict.

## Record per variant (in `BRUSH_LOAD_STATUS` in `evaluate.py`, then regenerate)

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
- `upright` stays off regardless (its own evidence threshold is the
  normalize plan's up confidence warning).
