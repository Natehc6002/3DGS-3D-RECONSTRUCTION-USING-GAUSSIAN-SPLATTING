# Reconstruction and Training Pipeline: Setup Guide

Scope: data preparation, COLMAP and Brush training (`backend/preprocessing/`, `backend/colmap/`, `backend/training/`).

## 1. Environment

| Item | Version / value |
|------|-----------------|
| OS | Windows |
| GPU | Quadro RTX 3000 (6 GB) |
| Python | 3.11.9 (virtual environment `.venv`) |
| COLMAP | 4.2.1 |
| Brush | 0.3.0 (`C:\Tools\Brush\brush_app.exe`) |
| opencv-python | 5.0.0.93 |
| numpy | 2.4.6 |

COLMAP install path on the reference machine: `C:\Tools\COLMAP\bin\colmap.exe`

Python setup (from the repo root):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install opencv-python
```

Keep `.venv/`, `*.ply`, `*.db`, logs and executables out of Git.

## 2. Project folder layout

Every runner takes `--project <folder>` and works inside it. Input images are never modified.

```text
<project>/
├── input/
│   ├── images/              photos, or frames extracted from a video
│   └── video/               optional source video
├── colmap/
│   ├── database.db
│   ├── sparse/0/            raw COLMAP sparse model
│   └── undistorted/         Brush input (images/ + sparse/), made by the COLMAP runner
├── checkpoints/             periodic PLY exports from Brush
├── scenes/
│   └── original/            final trained PLY (copied from checkpoints/)
└── logs/                    preprocessing.log, colmap.log, brush.log
```

`colmap/undistorted/`, `checkpoints/` and `scenes/original/` are the three locations written by the runners. The rest follows the team folder contract.

## 3. Pipeline commands

Run from the repo root, in this order.

### 3.1 Prepare images

Validate images already placed in `input/images/`:

```powershell
python -m backend.preprocessing.run --project <project>
```

Or extract frames from a video first:

```powershell
python -m backend.preprocessing.run --project <project> --video "<path\to\clip.mp4>" --fps 2 --max-frames 100
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--project` | required | project folder |
| `--video` | none | extract frames into `input/images/` before validating |
| `--fps` | 2.0 | frames kept per second of video |
| `--max-frames` | 0 (no limit) | stop after N frames |
| `--blur-threshold` | 50 | warn when Laplacian variance is below this value |

Checks: files are readable, resolution mix, blur (warning only), image count (warning below 20). Refuses to extract into a folder that already holds images. Exit code 0 on success, 2 for bad arguments or paths, 3 for failure.

Blur threshold depends on resolution. On the 4096x1844 phone photos of the reference dataset (95 images), 50 flagged 31 images and 20 flagged 4. Tune it per dataset. Warnings do not stop the pipeline.

Video mode has not been tested yet.

### 3.2 COLMAP reconstruction and undistortion

```powershell
python -m backend.colmap.run --project <project> --colmap C:\Tools\COLMAP\bin\colmap.exe --quality medium
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--project` | required | project folder |
| `--colmap` | required | path to `colmap.exe` |
| `--quality` | `medium` | `low`, `medium`, `high` or `extreme` |

Reads `input/images/` and runs two COLMAP commands:

1. `automatic_reconstructor` with `--single_camera 1 --dense 0 --use_gpu 1` and the chosen quality. Output: `colmap/database.db` and `colmap/sparse/0/`.
2. `image_undistorter` from `colmap/sparse/0/` with `--output_type COLMAP`. Output: `colmap/undistorted/`.

Log: `logs/colmap.log`.

Because of `--single_camera 1`, all images are treated as coming from one camera. Use photos of one resolution from one camera. Mixed resolutions (flagged by the preprocessing check) can break the reconstruction.

### 3.3 Brush training

```powershell
python -m backend.training.run --project <project> --brush C:\Tools\Brush\brush_app.exe --steps 30000 --export-every 5000
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--project` | required | project folder |
| `--brush` | required | path to `brush_app.exe` |
| `--steps` | 30000 | training steps |
| `--export-every` | 5000 | write a PLY every N steps |
| `--eval-split-every` | 0 (off) | hold out every Nth image for evaluation |
| `--with-viewer` | off | open the Brush viewer window during training |

Reads `colmap/undistorted/`. Writes PLY files to `checkpoints/` named `scene_<N>k_<iter>.ply` (for example `scene_30k_30000.ply`), then copies the final one to `scenes/original/`. Log: `logs/brush.log`.

The runner refuses to start if matching PLY files already exist in `checkpoints/` or the final file already exists in `scenes/original/`. Move or delete them yourself. Nothing is overwritten.

Exit codes: 0 success, 2 bad arguments or missing input, 3 Brush finished without writing a PLY, 4 training stopped before the final step (nothing is copied to `scenes/original/`), 130 interrupted with Ctrl+C, otherwise Brush's own exit code.

## 4. Observed behaviour

- Brush prints a live progress bar to the console (for example `605/1000 Steps`) when launched from the runner. Whether the same output can be captured through `QProcess` has not been tested.
- The runner reports each new PLY it sees in `checkpoints/`. The progress bar redraw can hide these lines in the console, but they are written to `logs/brush.log`.
- `--with-viewer` opens Brush's viewer window by itself. Tested on a 6,000-step run with checkpoints every 1,000 steps: the scene appears and changes as training progresses, no manual refresh is needed, and orbit, pan and zoom work during training. The viewer is part of the Brush process, so the live preview comes from Brush itself rather than from reloading checkpoint files. The viewer window stays open after training finishes. Closing the window during training stops the training run, so no final scene is written.

## 5. Reference results

Existing experiment results on this hardware. They are not guaranteed for other scenes or machines.

| Run | Result |
|-----|--------|
| Figurine, 30,000 steps | 258,309 splats, about 58.1 MB PLY, PSNR 22.75, SSIM 0.893 |
| Integration test project, 1,000 steps, checkpoints every 500 | training took 46 s, final PLY 2.5 MB |

Known-good scene for integration testing: `figure_30k_30000.ply` (shared outside Git, together with its scene metadata).

## Video mode (tested 7 Nov 2026)

    python -m backend.preprocessing.run --project <project> --video <clip.mp4> --fps 3 --max-frames 150 --blur-threshold 20

- The runner keeps every Kth source frame (30 fps source, `--fps 3` keeps every 10th).
- Frames are written to `input/images.tmp` and moved to `input/images` when extraction finishes. A killed run leaves no half-filled folder.
- If `input/images` already has images, the runner exits with an error and changes nothing. Empty it yourself first.
- `--max-frames` stops extraction early. The runner prints a WARN with how much of the video was read.
- Blur is warn-only. Images are never deleted or modified. Scores of all flagged frames go to `logs/preprocessing.log`.
- Blur threshold depends on resolution. On a 2160x3840 clip, threshold 20 flagged 2 of 265 frames and threshold 50 flagged 90 (34%). Compare scores within one dataset.

## COLMAP on video frames

- The COLMAP runner prints how many images each sparse model holds and warns if `sparse/0` holds under 80% of the images or if several models exist. Brush uses `sparse/0` only.
- `--data-type video` (sequential matching) is available. Default is `individual`. On the test clip both gave the same `sparse/0` (154 of 265 frames).

### Test clip findings (44 s, 4K phone video)
- Frames at every 10th and every 5th gave the same break at 25.7 s, so frame density was not the cause.
- At the break, plain wall filled most of the frame and a mirror reflected a monitor. Few trackable features, and reflections give false matches.
- Training on the result works (6000 steps, 505 s, 15.7 MB PLY, no VRAM error at 4K), but the scene covers only the first 25.7 s.

### Capture advice
- Pick a subject with texture across the whole frame. Keep mirrors, glass and screens out of view.
- Avoid framing that is mostly plain wall.
- Move around the subject with it always in view. No turning in place, no camera roll.
- About 20 s per loop, slow, exposure and focus locked.
- Check blur warnings and the `sparse models:` report before training. Reshoot if sparse/0 holds under 80%.