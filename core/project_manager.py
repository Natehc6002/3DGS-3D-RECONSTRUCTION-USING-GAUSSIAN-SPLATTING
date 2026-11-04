from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QObject, Signal


PROJECT_FOLDERS = (
    "input/images",
    "input/video",
    "colmap/sparse",
    "checkpoints",
    "scenes/original",
    "scenes/optimized",
    "logs",
    "results",
)


class ProjectError(ValueError):
    """Raised when a project cannot be created or opened."""


class ProjectManager(QObject):
    project_opened = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.current_project: Path | None = None

    def create_project(self, folder: str | Path, name: str) -> Path:
        root = Path(folder).expanduser().resolve()
        clean_name = name.strip()
        if not clean_name or clean_name in {".", ".."}:
            raise ProjectError("Enter a valid project name.")
        if any(char in clean_name for char in '<>:"/\\|?*'):
            raise ProjectError("Project names cannot contain \\ / : * ? \" < > or |.")

        project = root / clean_name
        if project.exists() and any(project.iterdir()):
            raise ProjectError(f"The folder already exists and is not empty: {project}")
        project.mkdir(parents=True, exist_ok=True)
        for relative in PROJECT_FOLDERS:
            (project / relative).mkdir(parents=True, exist_ok=True)
        metadata = {
            "name": clean_name,
            "version": 1,
            "input": {"images": "input/images", "video": "input/video"},
            "colmap": "colmap",
            "checkpoints": "checkpoints",
            "scenes": {"original": "scenes/original", "optimized": "scenes/optimized"},
            "logs": "logs",
            "results": "results",
        }
        (project / "project.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        self.open_project(project)
        return project

    def open_project(self, folder: str | Path) -> Path:
        project = Path(folder).expanduser().resolve()
        metadata_file = project / "project.json"
        if not metadata_file.is_file():
            raise ProjectError("This folder does not contain a 3DGS Studio project.json file.")
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectError(f"Could not read project metadata: {exc}") from exc
        if not isinstance(metadata, dict) or not metadata.get("name"):
            raise ProjectError("The project.json file is missing a project name.")
        self.current_project = project
        self.project_opened.emit(project)
        return project


