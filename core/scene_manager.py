"""Discover and register project Gaussian scene files.

Original scenes are stored in ``scenes/original`` and optimized scenes in
``scenes/optimized``. Imported PLY files are copied into one of those
collections; generated PLY files can be registered after a runner finishes.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


SCENE_FOLDERS = {
    "original": Path("scenes") / "original",
    "optimized": Path("scenes") / "optimized",
}


class SceneError(ValueError):
    """Raised when a scene cannot be discovered or registered."""


@dataclass(frozen=True)
class SceneRecord:
    name: str
    category: str
    path: Path
    size_bytes: int
    modified_ns: int


class SceneManager:
    """Manage scene PLYs inside one Studio project."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).expanduser().resolve()

    def discover(self, category: str | None = None) -> list[SceneRecord]:
        """Return registered PLY scenes, newest first.

        By default, only final original and optimized scenes are returned;
        training checkpoints remain separate from the scene catalog.
        """
        categories = (category,) if category is not None else tuple(SCENE_FOLDERS)
        records: list[SceneRecord] = []
        for item in categories:
            folder = self._folder_for(item)
            if not folder.is_dir():
                continue
            for path in folder.glob("*.ply"):
                if path.is_file():
                    records.append(self._record(path, item))
        return sorted(records, key=lambda record: (-record.modified_ns, record.name.casefold()))

    def register_scene(self, source: str | Path, category: str = "original") -> SceneRecord:
        """Copy an external PLY into a project scene collection without overwriting."""
        folder = self._folder_for(category)
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file() or source_path.suffix.lower() != ".ply":
            raise SceneError("Choose an existing PLY scene file.")

        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / source_path.name
        if source_path == destination.resolve():
            return self._record(destination, category)
        if destination.exists():
            raise SceneError(f"A scene with that name already exists in {category}: {destination.name}")
        try:
            shutil.copy2(source_path, destination)
        except OSError:
            destination.unlink(missing_ok=True)
            raise
        return self._record(destination, category)

    def register_output(self, path: str | Path, category: str) -> SceneRecord:
        """Register a PLY already written directly into its project collection."""
        folder = self._folder_for(category).resolve()
        output = Path(path).expanduser().resolve()
        if output.parent != folder or output.suffix.lower() != ".ply" or not output.is_file():
            raise SceneError(f"Expected an existing PLY directly inside {folder}.")
        return self._record(output, category)

    def _folder_for(self, category: str) -> Path:
        try:
            relative = SCENE_FOLDERS[category]
        except KeyError as exc:
            choices = ", ".join(SCENE_FOLDERS)
            raise SceneError(f"Unknown scene category '{category}'. Choose {choices}.") from exc
        return self.project_root / relative

    @staticmethod
    def _record(path: Path, category: str) -> SceneRecord:
        stat = path.stat()
        return SceneRecord(
            name=path.name,
            category=category,
            path=path.resolve(),
            size_bytes=stat.st_size,
            modified_ns=stat.st_mtime_ns,
        )
