# Gaussian optimization (size reduction for Brush scenes)

Tools for inspecting, pruning, clipping and normalizing Gaussian splat
PLY files exported by Brush, plus an end-to-end pipeline the desktop
app can call. Everything here reads a PLY and writes *new* files; the
input is never modified and existing outputs are never overwritten.

| Module | Purpose |
|---|---|
| `inspect_ply.py` | header/property/statistics report for one PLY |
| `analyze_scene.py` | read-only scene diagnostics (centre, extent, plane, up, subject, needles, mirror) |
| `pruning/prune.py` | baseline opacity-threshold prune |
| `pruning/smart_prune.py` | spatially safe prune (voxel budget, floater flags) |
| `pruning/mirror_clip.py` | clip candidate mirror content below the dominant plane |
| `pruning/removal_map.py` | per-voxel histogram of what a prune removed |
| `pruning/coverage_check.py` | occupancy loss proxy, original vs pruned |
| `normalize/normalize.py`, `core.py` | centre + scale (+ optional upright) with a sidecar JSON and undo |
| `quantize/sh_truncate.py` | Route A: SH degree truncation |
| `quantize/quantize.py`, `dequantize.py`, `core.py` | Route B: uint8/uint16 `.npz` archive + decode |
| `quantize/gzip_ply.py` | Route C: gzip baseline |
| `pipeline.py` | `run_pipeline()` + CLI: inspect -> smart prune -> clip -> SH -> normalize -> Route B -> report |
| `evaluate.py` | comparison table (CSV + markdown) from real files |
| `results/` | small CSV/JSON results committed with the repo |

Tests: `optimization/pruning/test_smart_prune.py`,
`optimization/pruning/test_mirror_clip.py`,
`optimization/quantize/test_quantize.py`,
`optimization/test_pipeline.py`, `optimization/test_evaluate.py`,
`optimization/test_normalize_invariance.py`.

## PLY attribute reference (Brush exports)

Verified from real Brush v0.3.0 exports and the Brush source (see the
module docstrings for the exact files). A degree-3 scene has **59
float32 properties per Gaussian, no normals**:

| Properties | Count | Stored encoding | Meaning |
|---|---|---|---|
| `x y z` | 3 | float32, scene units | centre position (COLMAP/scene units, `Vertical axis: y`) |
| `f_dc_0..2` | 3 | float32 SH coefficient | view-independent colour (SH degree 0) |
| `f_rest_0..44` | 45 | float32 SH coefficients | view-dependent colour bands 1-3 (76% of the bytes) |
| `opacity` | 1 | float32 **logit** | apply sigmoid to get opacity in [0, 1] |
| `rot_0..3` | 4 | float32, unit quaternion | Gaussian orientation (renormalize after any edit) |
| `scale_0..2` | 3 | float32, **natural log** | apply `exp` to get ellipsoid semi-axis lengths in scene units |

Header comments written by Brush: `Exported from Brush`,
`Vertical axis: y`, `SH degree: 3`.

Rules that matter when reading these files:

- **Access every property by name, never by column position.** Brush
  lists `f_rest_*` in text order in the header (`f_rest_0, f_rest_1,
  f_rest_10, ...`), so positional indexing silently reads the wrong
  coefficients.
- `f_rest` count gives the SH degree: 0/9/24/45 properties -> degree
  0/1/2/3.
- Inside the file the coefficients are **channel-major**: 3 blocks of
  (degree+1)^2-1 coefficients (`[red block][green block][blue block]`),
  standard SH order inside each block. Verified from
  `crates/brush-serde/src/export.rs` and `import.rs`
  (`rest_coeffs = [red_rest, green_rest, blue_rest].concat()`), and
  confirmed on the figurine data (block score 0.064 vs interleaved
  0.238, cross-channel correlation 0.873 vs 0.023).
- Truncating to a lower degree must keep, per channel, the *first* N
  coefficients and relabel them into contiguous blocks of the new size.
  `sh_truncate.py` checks the block structure on the real data and
  refuses to write if it does not match.

## Methods

### 1. Inspect

```powershell
python -m optimization.inspect_ply --ply <file.ply> [--json report.json]
```

Counts, property list, SH degree, per-attribute stats, sigmoid opacity
histogram, `exp(scale)` sizes, quaternion norms. Read-only.

### 2. Baseline prune (`pruning/prune.py`)

Removes everything with `sigmoid(opacity) < 0.05` (optionally extreme
sizes). Cheap, but removals concentrate wherever low opacity does -
on the figurine it emptied **610 voxels** and lost >=50% of the
Gaussians in 910 voxels (holes).

### 3. Smart prune (`pruning/smart_prune.py`)

1. Importance = `sigmoid(opacity) * ellipsoid volume`.
2. Flags floaters: isolated (k-NN), extreme size, outside a robust
   bounding box, low importance.
3. Spends a **per-voxel removal budget**: a voxel loses at most
   `max_voxel_fraction` of its Gaussians and is **never emptied**
   (unless you explicitly enable `--remove-singleton-floaters`).

| Parameter | Default | Meaning |
|---|---|---|
| `max_voxel_fraction` | `0.3` | max share of any voxel that may be removed |
| `voxel_size` | `5.0` | voxel edge in scene units |
| `importance_threshold` | `1e-4` | flag `sigmoid(opacity)*volume` below this (0 = off) |
| `knn_k` | `5` | k for the isolation test |
| `iso_percentile` | `99` | flag k-NN distances above this percentile |
| `size_percentile` | `99.9` | flag `exp(scale)` above this percentile |
| `body_percentile_low/high` | `0.5 / 99.5` | robust body box percentiles |
| `body_margin` | `20.0` | margin added around the body box |
| `--remove-singleton-floaters` | off | documented exception: may empty a voxel holding one floater |

### 4. Coverage check (`pruning/coverage_check.py`)

Geometric proxy only: voxel occupancy loss between original and pruned.
It reports where Gaussians disappeared (emptied voxels, >=50% loss
voxels, max loss) so holes can be spotted before opening Brush. **It
does not measure image quality.**

### 5. Scene analysis (`analyze_scene.py`)

```powershell
python -m optimization.analyze_scene --ply <file.ply> [--json]
```

Read-only, scene-relative diagnostics: robust centre and extent
(p1..p99 box), dominant plane (RANSAC on compact high-opacity
Gaussians, refined by PCA on inliers) with confidence, estimated up
direction with confidence, subject candidate centre/radius with
confidence, needle count, and Gaussians below the dominant plane by a
scene-relative margin (candidate mirror content). No threshold is a
fixed number of scene units. Never writes or modifies files.

### 6. Mirror clip (`pruning/mirror_clip.py`)

Removes candidate mirror content below the dominant plane (the figurine
scene has mirrored copies of the figure and room baked in as geometry
under the marble floor):

```powershell
python -m optimization.pruning.mirror_clip --input <in.ply> `
    --output <out.ply> [--plane NX NY NZ D] [--margin-ratio R] `
    [--max-weight-fraction F] [--json <report.json>]
```

The detected plane's normal is signed so the removed side carries the
minority of scene weight (RANSAC signs are arbitrary; mirror content is
the small side - `normal_flipped` in the plan records when this
happened; a manual `--plane` is never flipped). Guards, all
scene-relative: plane confidence >= 0.30, the subject centre must not
lie on the removed side, removed weight share <= 0.6 (configurable).
A violated guard refuses the clip (exit 2) instead of guessing.

### 7. Normalize (`normalize/normalize.py` + `core.py`)

```powershell
python -m optimization.normalize.normalize --input <in.ply> --output <out.ply> `
    [--sidecar <out.normalize.json>] [--upright] [--up-axis X Y Z] `
    [--target-radius R] [--scale S] [--no-scale] [--colmap-dir SPARSE_DIR]

python -m optimization.normalize.normalize --undo <sidecar.json> `
    --input <normalized.ply> --output <restored.ply>
```

Centres the subject (evidence: subject density centre with confidence,
falling back to the robust centre), applies one uniform scale so the
subject radius p90 becomes the target (default 1), and optionally
rotates the estimated up direction to Brush +y. Every automatic choice
reports its method and confidence in the sidecar; manual overrides
always win. `opacity` and `f_dc` are untouched; positions, log-scales,
orientations and SH coefficients are transformed consistently, and the
write is refused unless the sidecar round trip stays within
`ROUNDTRIP_REL_TOLERANCE` (1e-5 worst relative error).

**Sidecar contract** (fields, convention string, undo, consumers):
see `optimization/normalize/SIDECAR_CONTRACT.md`.

### 8. Route A: SH degree truncation (`quantize/sh_truncate.py`)

Drops `f_rest` coefficients (45 -> 24/9/0 for degree 2/1/0), updates the
`SH degree:` header comment, keeps property order otherwise. Biggest
Brush-readable saving because `f_rest` is 76% of the bytes.

```powershell
python -m optimization.quantize.sh_truncate --input in.ply --output out.ply --degree 2
python -m optimization.quantize.sh_truncate --input in.ply --evidence-only
```

### 9. Route B: quantized archive (`quantize/quantize.py` + `dequantize.py`)

Quantizes every attribute to uint8/uint16 with stored per-attribute
min/max (default bits: pos 16, rot 8, scale 16, opacity 16, dc 16,
rest 8) into a compressed `.npz`, then decodes back to a float32 PLY.
Encode validates the round trip (all errors within the quantization
step; quaternions renormalized).

**The `.npz` is storage/transfer only - Brush cannot open it.** The
decoded `q_back` PLY is a normal Brush-readable file with the same size
as its source PLY; the compression lives in the archive, not in the PLY
Brush opens.

### 10. Route C: gzip (`quantize/gzip_ply.py`)

Lossless baseline (byte-identical after `gunzip`). Brush cannot open
`.gz` either; barely shrinks the figurine (60.96 -> 57.07 MB).

### 11. Pipeline (`pipeline.py`)

```powershell
python -m optimization.pipeline --input <scene.ply> --output-dir <dir> `
    [--sh-degree {0,1,2,3}] [--route-b] [--no-prune] [--no-coverage] `
    [--needle-cleanup] [--mirror-clip] `
    [--normalize] [--upright] [--up-axis X Y Z] [--no-normalize-scale] `
    [--report-name NAME.json] [--quiet]
```

Stages: `inspect -> smart_prune -> mirror_clip (optional) ->
sh_truncate (optional) -> normalize (optional) -> route_b (optional)
-> report`. Exit codes: **0** ok, **2** bad arguments/paths or output
already exists, **3** stage failure (error is written into the report;
partial outputs of that run are removed). Normalisation runs last so
every earlier stage keeps working in the input's own units.

Stage defaults live in one block, `STAGE_DEFAULTS` in `pipeline.py`
(behind both the CLI and `run_pipeline()`): smart prune **on**, SH
truncation/Route B/needle cleanup/mirror clip/normalise **off**. They
are chosen only from Brush verdicts: smart pruning on (voxel safety
invariant measured: 0 emptied vs baseline 610, and the one pruned file
with a verdict, `smart_shd2`, opens ok), SH truncation off (no verdict
reports no visible colour/shading loss versus the original - `smart_shd2`
colour shift/flat shading were not judged and shd1/shd0 are "not
tested"), Route B off (storage-only, decoded PLY "not tested"), needle
cleanup / mirror clip / normalise off until their outputs are confirmed
in Brush (the A-D comparison; see `results/BRUSH_CHECKLIST.md`). Flip a
default in `STAGE_DEFAULTS` only after that check.

### 12. Evaluation (`evaluate.py`)

Builds `results/evaluation_table.csv` + `results/evaluation_summary.md`
from real files. Columns that cannot be measured here are explicit:
`processing_time_s` = `not measured` unless you pass `--time`,
`psnr/ssim/fps` = always `not measured`, `brush_load_status` = the manual
verdicts in `BRUSH_LOAD_STATUS` **including the GPU each file was viewed
on** (anything unverified stays `not tested`).

New-stage columns (`needles_removed`, `clip_removed`,
`clip_weight_share`, `normalize_center`, `normalize_up`,
`normalize_scale`, `normalize_roundtrip_rel`) are read from a real
pipeline report given with `--report LABEL=report.json`: without a
report they read `not measured`, with a report whose stage did not run
`n/a (stage off)`.

```powershell
python -m optimization.evaluate --original orig.ply `
    --variant smart_default=smart.ply --time smart_default=1.1 `
    --round-trip route_b_decoded=smart.ply,smart_qback.ply `
    --report smart_default=results/smart_pipeline.json
```

## Results (figurine, one scene)

Scene `projects/figure_01/scenes/original/figure_30k_30000.ply`:
**258,309 Gaussians, 60.96 MB**, 59 float32 properties, SH degree 3.
All numbers below were produced by the commands in this folder; quality
metrics were **not** measured.

### Pruning (`results/smart_prune_figurine.csv`)

| setting | kept | output | ratio | voxels emptied | max voxel loss | voxels >=50% loss |
|---|---|---|---|---|---|---|
| baseline opacity 0.05 | 178,344 (69.04%) | 42.09 MB | 1.45x | 610 | 1.000 | 910 |
| smart frac 0.2 | 207,409 (80.29%) | 48.95 MB | 1.25x | 0 | 0.200 | 0 |
| **smart default frac 0.3** | 182,489 (70.65%) | 43.07 MB | 1.42x | 0 | 0.300 | 0 |
| smart frac 0.5 | 130,286 (50.44%) | 30.75 MB | 1.98x | 0 | 0.500 | 432 |

The baseline empties voxels (visible holes risk); the smart settings
never empty one, but at frac 0.5 they still lose >=50% in 432 voxels.

### Size reduction routes (`results/size_reduction_figurine.csv`)

| source | route | setting | output | ratio vs 60.96 MB |
|---|---|---|---|---|
| original | baseline | - | 60.96 MB | 1.00x |
| original | A SH | degree 2 | 39.26 MB | 1.55x |
| original | A SH | degree 1 | 23.77 MB | 2.57x |
| original | A SH | degree 0 | 14.47 MB | 4.21x |
| original | B npz | pos16/rot8/scale16/op16/dc16/rest8, input SH degree 3 | 14.52 MB | 4.20x |
| original | C gzip | level 6 | 57.07 MB | 1.07x |
| smart default | A SH | degree 2 | 27.74 MB | 2.20x |
| smart default | A SH | degree 1 | 16.79 MB | 3.63x |
| smart default | A SH | degree 0 | 10.22 MB | 5.97x |
| smart default | B npz | pos16/rot8/scale16/op16/dc16/rest8, input SH degree 3 | 10.26 MB | 5.94x |
| smart default | C gzip | level 6 | 40.31 MB | 1.51x |

Route B `.npz` sizes are storage sizes; Brush only ever opens the decoded
PLY (same size as the source: 60.96 MB / 43.07 MB). **Every `.npz`
number in this repository uses the same bit preset** (`pos 16, rot 8,
scale 16, opacity 16, dc 16, rest 8` = `quantize/core.py` `DEFAULT_OPTS`,
`np.savez_compressed`); archive size then depends only on the **input
file's SH degree** (see "npz size discrepancy" below).

### Evaluation table (`results/evaluation_table.csv`)

Measured this session with each tool's own timer:

| variant | gaussians | size | ratio | time (s) | coverage proxy | Route B round-trip |
|---|---|---|---|---|---|---|
| original | 258,309 | 60.96 MB | 1.000x | n/a | n/a (no removals) | n/a |
| baseline_prune | 178,344 | 42.09 MB | 1.448x | 0.3 | emptied 610; >=50% 910; max 1.000 | n/a |
| smart_default | 182,489 | 43.07 MB | 1.415x | 1.1 | emptied 0; >=50% 0; max 0.300 | n/a |
| smart_shd2 | 182,489 | 27.74 MB | 2.198x | 13.1 | n/a (layout differs) | n/a |
| smart_shd1 | 182,489 | 16.79 MB | 3.631x | 12.7 | n/a (layout differs) | n/a |
| smart_shd0 | 182,489 | 10.22 MB | 5.965x | 12.8 | n/a (layout differs) | n/a |
| route_b_decoded | 182,489 | 43.07 MB | 1.415x | 13.7 | n/a (not an exact subset) | 0.00547457 |

`smart_shd*` are smart-pruned scenes, so their coverage versus the
*original* cannot be diffed row by row (different property layout); the
prune itself is the smart_default row above. PSNR/SSIM/FPS: **not
measured** for every row.

### Why the same scene shows two `.npz` sizes (10.26 vs 7.41 MB)

Both numbers use the **same code path and the same preset** (`pos16
rot8 scale16 opacity16 dc16 rest8`, `np.savez_compressed`); the only
difference is the **input file**:

| number | run | input to Route B | input SH | input size |
|---|---|---|---|---|
| 10.26 MB | `size_reduction_figurine.csv` (quantize CLI) | `fig_30k_smart_default.ply` | degree 3 (45 `f_rest`) | 43.07 MB |
| 7.41 MB | pipeline smoke run (`--sh-degree 2 --route-b`) | `fig_30k_smart_shd2.ply` (`final_ply`, after truncation) | degree 2 (24 `f_rest`) | 27.74 MB |

Verified on this machine with identical settings: encoding the same
degree-3 smart PLY through the quantize CLI and through
`run_pipeline(..., route_b=True)` produces **byte-identical** archives
(10,259,142 bytes each); encoding its degree-2 truncation produces
7,408,556 bytes. Pipeline Route B always encodes `final_ply` (after any
`--sh-degree` truncation), which is why the smoke-run archive is smaller.
Pinned by `test_pipeline.py::TestRouteBPresetPinning`.

### Pipeline smoke run (same scene)

```text
inspect 0.395 s | smart_prune 1.811 s | sh_truncate(2) 13.184 s |
route_b 9.372 s | total 24.766 s
258,309 -> 182,489 Gaussians; 60.96 -> 43.07 MB (1.4155x) ->
27.74 MB with SH degree 2 (2.1977x);
npz 7.41 MB (8.2287x vs original), preset pos16/rot8/scale16/op16/dc16/rest8,
input = the SH-degree-2 final PLY (not the degree-3 smart PLY - see the
"npz size discrepancy" section);
coverage: 0 emptied, 0 over 50% loss; round trip within quantization step
```

Report: `<output-dir>/<stem>_pipeline.json`, CSV log:
`<output-dir>/pipeline_runs.csv` (in `results/` for a project folder).

### Brush verdicts

Two files have a recorded verdict (everything else stays `not tested` -
no verdict is guessed). The GPU used per entry is part of
`BRUSH_LOAD_STATUS` in `evaluate.py`:

| file | Brush verdict | GPU used |
|---|---|---|
| **original (`figure_30k_30000.ply`, reference, 258,309 splats, 60.96 MB)** | **opens OK** - Stats panel: Splats `258309`, SH Degree `3`, Bytes in use `60.96 MB`. From the screen recording: the figure is **not centred at first view** (small, lower right, camera faces the room) and is tiny compared with the room background; long needle/streak splats cross the view whenever the camera leaves the training path; fog and ghost splats near the floor; mirrored copies of the figure and background appear under the marble floor (reflection baked in as geometry); general blur off the training path. **Not judged:** up-axis tilt, colour shift versus other files. The cause of the off-centre start is **not proven**. | NVIDIA GeForce RTX 4060 Laptop GPU (`DiscreteGpu`, driver `NVIDIA, 581.86`) |
| baseline (`*_pruned_opacity005.ply`) | not tested | not tested |
| smart default (`*_smart.ply`) | not tested | not tested |
| **shd2 (`fig_30k_smart_shd2.ply`, 182,489 splats, 27.74 MB)** | **opens ok** - Brush reported SH degree 2 and 182,489 splats, no load error. Front and side views clean; no holes noticed in the main body. Blur and streaks appear only when the camera leaves the training path (expected novel-view behaviour, **not** compared against the original under the same path, so not attributable to pruning or SH truncation). Colour shift / flat shading versus the original: **not judged**. | Intel iGPU (Vulkan) |
| shd1 / shd0 | not tested | not tested |
| Route B decoded (`*_qback.ply`) | not tested | not tested |
| pipeline combo shd2 output | not tested | not tested |

Test environment: two Brush 0.3.0 runs are recorded on this machine.
The original-file verdict used the **NVIDIA GeForce RTX 4060 Laptop
GPU** via the launcher's `-Gpu auto` mode (Stats panel: Name
"NVIDIA GeForce RTX 4060 Laptop GPU", Type `DiscreteGpu`, Driver
`NVIDIA, 581.86`); the `smart_shd2` verdict was obtained earlier on the
**Intel integrated GPU** (Stats panel: "Intel(R) RaptorLake-S Mobile
Graphics Controller", Type `IntegratedGpu`, driver 101.7026). Both used
`WGPU_BACKEND=vulkan`; the DirectX 12 backend crashes Brush 0.3.0 on
this machine (FXC X3695). See `tools/README_brush_gpu.md`.

Fill `BRUSH_LOAD_STATUS` in `evaluate.py` (and this table) only after a
visual check.

### Recommended pipeline defaults (chosen only from verdicts)

| default | setting | reason (one sentence) |
|---|---|---|
| smart prune | **on** | the voxel safety invariant is measured (0 voxels emptied vs baseline's 610) and the only pruned file with a verdict, `smart_shd2`, opens ok - while `smart.ply` itself is "not tested". |
| SH truncation | **off** (optional export: 27.74 / 16.79 / 10.22 MB for degree 2/1/0) | no verdict reports no visible colour or shading loss versus the original - `smart_shd2` opened ok but colour shift/flat shading were not judged, and `smart_shd1`/`smart_shd0` are "not tested". |
| Route B | **off** (optional storage-only export) | Brush cannot read the `.npz` and the decoded `smart_qback.ply` is "not tested", so there is no verdict to justify enabling it. |
| mirror clip | **off** | the guards are measured on the figurine run (removed 22,371 Gaussians, weight share 0.445, plane confidence 0.414) but no Brush verdict yet confirms the floor copies are gone and the figure intact - see `results/BRUSH_CHECKLIST.md`. |
| normalise | **off** | invariance is proven on synthetic scenes and the figurine run detected its subject at confidence 1.000, but no Brush verdict exists for a normalised file - flip only after the A-D check. |

## Limits

- **Detection evidence, stated precisely.** Invariance of the
  normalisation under rigid + uniform-scale transforms is proven on
  synthetic scenes with a constructed k-NN class gap
  (`optimization/test_normalize_invariance.py`); subject detection on
  real scenes is a heuristic validated on one real scene (the
  figurine). Nothing here claims more than that.
- **No quality metrics.** PSNR, SSIM and FPS are not computed anywhere in
  this repository; they read `not measured` in every table. The coverage
  columns are geometric proxies only.
- **Brush verdicts: two files only.** `figure_30k_30000.ply` (original)
  and `fig_30k_smart_shd2.ply` open in Brush 0.3.0; the shd2 file has
  no load error, front/side views clean, no holes noticed in the main
  body, and colour shift / flat shading versus the original were not
  judged. The original's verdict records the off-centre first view,
  needle/streak splats, floor fog/ghosts and mirrored copies under the
  floor as observed - their causes are not established here. All other
  optimized variants are `not tested`; size and spatial safety are
  measured, visual quality is not.
- **Test environment limit.** Both recorded verdicts come from Brush
  0.3.0 with the Vulkan backend (DirectX 12 crashes on shader compile
  error FXC X3695): the original on the NVIDIA RTX 4060
  (`DiscreteGpu`, driver `NVIDIA, 581.86`), `smart_shd2` on the Intel
  integrated GPU (`IntegratedGpu`, driver 101.7026). Each verdict
  carries its own GPU; see `tools/README_brush_gpu.md`.
- **Route B shrinks storage, not the in-Brush PLY.** Brush cannot read
  `.npz`; the decoded PLY is the same size as its source.
- **Route C `.gz` is not Brush-readable** either (gunzip restores the
  original exactly).
- **SH truncation loses view-dependent colour** (increasingly at degree
  1 and 0); it only keeps the view-independent term at degree 0.
- **One scene tested.** All numbers come from the single figurine scene
  (258,309 Gaussians). Other scenes/scan types are untested.
- **Coverage/round-trip are not rendering tests** - they bound geometry
  loss and quantization error, nothing about what the images look like.
- Pipeline runs never overwrite: to rerun, use a fresh `--output-dir` or
  delete the previous report/outputs first.

## Calling the pipeline from the desktop app

```python
from optimization.pipeline import run_pipeline

result = run_pipeline(
    project / "scenes" / "original" / "scene_30k_30000.ply",
    project,            # folder containing project.json
    sh_degree=None,     # 0/1/2 once the Brush verdicts are filled in
    route_b=False,
    quiet=True,         # suppress per-stage stdout
)
if result["status"] == "ok":
    viewer.open(result["outputs"]["final_ply"])      # scenes/optimized/...
else:
    show_error(result["error"], stage=result["error_stage"])
```

If `output_dir` contains `project.json` the run follows the team folder
contract (`core/project_manager.py`): PLYs go to
`<project>/scenes/optimized/`, the report and `pipeline_runs.csv` go to
`<project>/results/`. Any other folder receives all files directly.

`result` is a plain JSON-serializable dict, identical to the report file
(`schema_version: 1`). Key fields:

| Field | Content |
|---|---|
| `status`, `error`, `error_kind`, `error_stage`, `exit_code` | `ok`/`error`; `bad_input`/`exists`/`stage`; 0/2/3 |
| `stages[]` | `{name, status: ok\|skipped\|error, time_s, error, reason?}` |
| `counts` | `input`, `pruned`, `clipped`, `output` (each stage's remaining count) |
| `sizes`, `ratios` | bytes in/out; `pruned_vs_input`, `clipped_vs_input`, `final_vs_input`, `normalized_vs_input`, `npz_vs_input` |
| `timings` | `total_s` plus one key per stage |
| `outputs` | `pruned_ply`, `clipped_ply`, `sh_ply`, `normalized_ply`, `normalize_sidecar`, `final_ply`, `npz`, `q_back_ply`, `report_json`, `csv_log` |
| `inspect`, `smart_prune`, `mirror_clip`, `coverage`, `sh_truncate`, `normalize`, `route_b` | full per-stage reports (or `null`) |
| `warnings` | e.g. skipped stages, normalize plan warnings (`normalize: ...`), Route B storage-only notice |

Error contract: missing file / bad option / existing output -> dict with
`error_kind` `bad_input` or `exists`, **nothing written**, no report.
A stage failure (unreadable/empty PLY, layout refusal, ...)` ->
`error_kind: stage`, partial outputs removed, error written into the
report JSON.

## Commands (run from the repo root)

```powershell
# inspect + scene analysis
python -m optimization.inspect_ply --ply <file.ply>
python -m optimization.analyze_scene --ply <file.ply>

# pruning
python -m optimization.pruning.prune --input <in.ply> --output <out.ply>
python -m optimization.pruning.smart_prune --input <in.ply> --output <out.ply> --json <r.json>
python -m optimization.pruning.coverage_check --input <orig.ply> --pruned <pruned.ply>
python -m optimization.pruning.removal_map --input <orig.ply> --pruned <pruned.ply>

# mirror clip + normalize
python -m optimization.pruning.mirror_clip --input <in.ply> --output <out.ply>
python -m optimization.normalize.normalize --input <in.ply> --output <out.ply>
python -m optimization.normalize.normalize --undo <sidecar.json> --input <norm.ply> --output <restored.ply>

# quantization routes
python -m optimization.quantize.sh_truncate --input <in.ply> --output <out.ply> --degree 2
python -m optimization.quantize.quantize --input <in.ply> --output <out.npz>
python -m optimization.quantize.dequantize --input <out.npz> --output <back.ply>
python -m optimization.quantize.gzip_ply --input <in.ply> --output <out.ply.gz>

# pipeline + evaluation
python -m optimization.pipeline --input <in.ply> --output-dir <dir> --sh-degree 2 --route-b
python -m optimization.evaluate --original <orig.ply> --variant smart_default=<smart.ply>

# tests
python -m pytest optimization -q
# or: python -m unittest discover -s optimization -p "test_*.py" -t .
```

Open a result in Brush (sets `VK_DRIVER_FILES` + `WGPU_BACKEND=vulkan`
for the Brush child process only; see `tools/README_brush_gpu.md`):

```powershell
.\optimization\tools\open_in_brush.ps1 -Ply <ply path>            # -Gpu auto (default)
.\optimization\tools\open_in_brush.ps1 -Ply <ply path> -Gpu nvidia # require the NVIDIA ICD
.\optimization\tools\open_in_brush.ps1 -Ply <ply path> -PrintOnly  # dry run, prints the command
```

Check when viewing: Stats panel GPU name/type, colour shift
(view-dependent effects), flat shading, holes, floaters, load errors.
