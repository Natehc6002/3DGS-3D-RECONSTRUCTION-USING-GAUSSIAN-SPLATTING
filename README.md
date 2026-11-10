# 3DGS Studio

<p align="center">
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=32&duration=3000&pause=1000&color=FF4D6D&center=true&vCenter=true&repeat=true&width=500&height=60&lines=monish+loves+deeps" alt="monish loves deeps" />
  </a>
</p>

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

The desktop app provides project creation/opening and a workspace. Its Brush integration uses the backend runner so the GUI can show checkpoint progress and detect incomplete runs.
