# Running the optimization pipeline on a new scene

No numbers in this file are measurements - it is a command checklist.
Every run produces its own real numbers; never copy results from the
figurine scene onto another scene.

## Prerequisites

- Repo root as the working directory (where `.venv/` lives):
  `cd C:\Users\DARSHAN I S\Desktop\3dgs_studio\3DGS-3D-RECONSTRUCTION-USING-GAUSSIAN-SPLATTING`
- venv active: `.venv\Scripts\Activate.ps1`
- The new scene must be a 3DGS Studio project folder (created by the
  desktop app or `core/project_manager.py`): it contains `project.json`
  and `scenes/original/` holds the trained Brush PLY (see `backend/SETUP.md`
  section 2). Example layout:

```text
projects/<scene_name>/
    project.json
    scenes/original/<scene>.ply      <- input, never modified
    scenes/optimized/                <- pipeline outputs go here
    results/                         <- report JSON + pipeline_runs.csv
```

## 1. Confirm the input

```powershell
python -m optimization.inspect_ply --ply projects\<scene>\scenes\original\<scene>.ply
```

Record: Gaussian count, file size, property count, SH degree. The file
must have the 59 Brush properties (degree 3) and a nonzero count.

## 2. Run the pipeline (recommended defaults)

Defaults: smart prune ON, SH truncation OFF, Route B OFF (see
`optimization/README.md` "Recommended pipeline defaults").

```powershell
python -m optimization.pipeline `
    --input projects\<scene>\scenes\original\<scene>.ply `
    --output-dir projects\<scene>
```

Because `<output-dir>` contains `project.json`, outputs land in
`projects\<scene>\scenes\optimized\` and the report in
`projects\<scene>\results\`. Exit code 0 = ok, 2 = bad arguments/paths
or output already exists, 3 = stage failure.

Optional variants (use a fresh `--output-dir` or delete the previous
report first - runs never overwrite):

```powershell
# Route A: SH truncation (only after a Brush verdict for this scene)
python -m optimization.pipeline --input <in.ply> --output-dir <dir> --sh-degree 2

# Route B: storage-only .npz archive (decoded PLY is what Brush opens)
python -m optimization.pipeline --input <in.ply> --output-dir <dir> --route-b
```

Record from the report (`results/<scene>_pipeline.json`): input/output
Gaussian counts, sizes, ratios, coverage (`voxels_emptied`,
`voxels_over_threshold`), stage times, warnings.

## 3. Open the optimized PLY in Brush (visual check)

```powershell
.\optimization\tools\open_in_brush.ps1 -Ply projects\<scene>\scenes\optimized\<scene>_smart.ply
```

Check in this order and write down the result for each item (use
"not tested" if you do not check it):

1. **Stats panel GPU name** - the launcher's default `-Gpu auto` points
   Brush at the NVIDIA Vulkan ICD, so this must be an NVIDIA name and
   Type `DiscreteGpu`. If it still shows Intel / `IntegratedGpu`, record
   exactly what it shows and re-check with `-Gpu nvidia -PrintOnly`
   (`optimization/tools/README_brush_gpu.md`).
2. **Load errors** - does the file open at all?
3. **Reported splat count and SH degree** - must match the pipeline
   report.
4. **Colour shift** versus the original under the same camera path.
5. **Flat shading** versus the original.
6. **Holes** in the main body (compare against the original, same path).
7. **Floaters** - stray Gaussians away from the body.
8. Note whether the camera stayed on the training path; blur/streaks off
   the path are expected novel-view behaviour and must be compared
   against the original under the same path before blaming a tool.

Open the original too, for the side-by-side comparison:

```powershell
.\optimization\tools\open_in_brush.ps1 -Ply projects\<scene>\scenes\original\<scene>.ply
```

## 4. Build the evaluation table

```powershell
python -m optimization.evaluate `
    --original projects\<scene>\scenes\original\<scene>.ply `
    --variant smart_default=projects\<scene>\scenes\optimized\<scene>_smart.ply `
    --time smart_default=<seconds from the pipeline report> `
    [--variant baseline_prune=<path> --time baseline_prune=<s>] `
    [--round-trip route_b_decoded=<smart.ply>,<q_back.ply>]
```

Fill `BRUSH_LOAD_STATUS` in `optimization/evaluate.py` **only** with
verdicts from step 3 for this scene's files, then rerun the command so
`results/evaluation_table.csv` and `results/evaluation_summary.md`
regenerate. PSNR/SSIM/FPS stay `not measured` - nothing in this repo
computes them.

## 5. What to record (per scene)

- Scene name, Gaussian count, input size (MB), SH degree.
- Pipeline: pruned count/size/ratio, coverage (emptied voxels, >=50%
  loss voxels, max loss), stage times, exit code.
- Brush: GPU name + type from the Stats panel, load result, splat count
  reported, colour shift, flat shading, holes, floaters - each item
  separately, `not tested` when skipped.
- Any `.npz` size must state its preset (`pos16 rot8 scale16 opacity16
  dc16 rest8`) **and** the input SH degree of the file that was encoded
  (see README "npz size discrepancy").
- Never invent PSNR, SSIM, FPS, or a Brush verdict.

## Do not

- Do not commit `*.ply`, `*.npz`, `projects/`, `*.db`, `*.log` - results
  that belong in Git go in small CSV/JSON/MD files under
  `optimization/results/`.
- Do not run `set_brush_gpu_preference.ps1 -Apply` from a script; it is
  a manual, user-authorized step (and you must restart Brush after it).
- Do not overwrite an existing run - use a fresh `--output-dir`.
