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

## Current scope

The initial desktop shell includes a project dashboard, project creation/opening, the shared project folder layout, a workspace view, and a `QProcess` environment check that streams output to the console. COLMAP, Brush training, scene viewing, optimization and evaluation are integration steps for subsequent milestones.
