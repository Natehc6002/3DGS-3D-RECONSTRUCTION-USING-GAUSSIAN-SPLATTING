# 3DGS Studio


A desktop application for preparing image datasets, connecting COLMAP and Brush workflows, reviewing Gaussian scenes, and comparing optimization results.

## Run the desktop app

Use Python 3.11 from the project virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

## Backend setup

See [backend/SETUP.md](backend/SETUP.md) for COLMAP and Brush versions, the project folder layout, and the runner commands.

## Optimization tools

See [optimization/README.md](optimization/README.md) for the PLY inspection, pruning, mirror-clip and normalisation tools plus the end-to-end pipeline, [optimization/results/NEXT_SCENES.md](optimization/results/NEXT_SCENES.md) for the per-scene run checklist, and [optimization/results/BRUSH_CHECKLIST.md](optimization/results/BRUSH_CHECKLIST.md) for the Brush A-D verification of the clip/normalise stages.

The desktop workspace also provides **Inspect PLY**, **Optimize scene**, **Evaluate scenes**, and **View results** actions.

### Optimize and evaluate a scene

1. Click **Optimize scene** and choose an original PLY. Studio runs the default smart-pruning and coverage stages. The original stays unchanged, and each run gets a unique output name under `scenes/optimized/`.
2. Click **Evaluate scenes**, choose the original PLY, then select one or more optimized PLY files. Studio compares Gaussian counts, file sizes, size ratios, processing times when available, and a voxel coverage proxy. It saves CSV and Markdown reports with unique names in the project’s `results/` folder.
3. Click **View results** and choose an evaluation CSV to see the main comparison fields in a table.

Coverage is a geometric proxy, not an image-quality score. PSNR, SSIM, and FPS are reported as not measured. Brush status is recorded only from manual visual checks.

The Brush integration uses the backend runner so the GUI can show checkpoint progress and detect incomplete runs.

## Windows desktop workflow

1. In 3DGS Studio, create or open a project.
2. Import your images and check them.
3. Configure COLMAP and Brush using [backend/SETUP.md](backend/SETUP.md).
4. Run COLMAP to prepare the dataset.
5. Choose **Train with Brush + viewer**, select `brush_app.exe`, and use the prepared dataset when prompted.
6. The final PLY scene is saved in the project’s `scenes/original/` folder.
