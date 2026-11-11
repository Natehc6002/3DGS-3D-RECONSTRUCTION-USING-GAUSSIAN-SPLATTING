from __future__ import annotations

import sys
import re
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QStackedWidget, QVBoxLayout, QWidget,
    QInputDialog,
)

from core.process_manager import ProcessManager
from core.project_manager import ProjectError, ProjectManager
from frontend.dashboard import Dashboard
from frontend.workspace import Workspace


STYLESHEET = """
QMainWindow, QWidget#dashboard, QWidget#workspace { background: #101419; color: #e8edf2; }
QLabel { color: #e8edf2; }
QLabel#eyebrow { color: #76d7bb; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; }
QLabel#heroTitle { font-size: 38px; font-weight: 700; line-height: 1.2; }
QLabel#mutedText { color: #9aa7b2; font-size: 12px; }
QLabel#cardNumber { color: #76d7bb; font-size: 12px; font-weight: 700; }
QLabel#cardTitle, QLabel#sectionTitle { font-size: 17px; font-weight: 650; }
QLabel#topProject { color: #9aa7b2; font-size: 12px; }
QLabel#pipelineItem { background: #1b2229; border-radius: 7px; padding: 10px 8px; color: #c2cbd2; }
QLabel#metricValue { font-size: 24px; font-weight: 700; }
QLabel#statusPill { background: #252d34; border-radius: 10px; padding: 5px 9px; color: #76d7bb; font-size: 10px; font-weight: 700; }
QFrame#actionCard, QFrame#panel, QFrame#infoStrip { background: #171d23; border: 1px solid #28323b; border-radius: 12px; }
QFrame#metricCard { background: #202830; border-radius: 9px; padding: 5px; }
QPushButton { border: 0; border-radius: 7px; padding: 10px 13px; font-weight: 600; }
QPushButton#accentButton { background: #76d7bb; color: #10221e; }
QPushButton#accentButton:hover { background: #91e4cb; }
QPushButton#secondaryButton { background: #28323b; color: #e8edf2; }
QPushButton#secondaryButton:hover { background: #35424d; }
QPushButton#secondaryButton:disabled { color: #69757e; }
QPushButton#textButton { background: transparent; color: #9aa7b2; padding: 6px; }
QLineEdit { background: #202830; border: 1px solid #35424d; border-radius: 6px; padding: 9px; color: #e8edf2; }
QTextEdit#console { background: #0b0e11; border: 1px solid #28323b; border-radius: 7px; color: #a9e2ca; font-family: Consolas, monospace; font-size: 11px; }
QDialog { background: #171d23; color: #e8edf2; }
"""


class NewProjectDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create a project")
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. Figurine Reconstruction")
        self.parent_folder = QLineEdit(str(Path.home() / "3DGS-Studio"))
        browse = QDialogButtonBox()
        choose = browse.addButton("Browse…", QDialogButtonBox.ButtonRole.ActionRole)
        choose.clicked.connect(self._choose_folder)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.parent_folder, 1)
        folder_row.addWidget(choose)
        folder_widget = QWidget()
        folder_widget.setLayout(folder_row)
        form.addRow("Project name", self.name)
        form.addRow("Create under", folder_widget)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        create = buttons.addButton("Create", QDialogButtonBox.ButtonRole.AcceptRole)
        create.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose parent folder", self.parent_folder.text())
        if folder:
            self.parent_folder.setText(folder)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("3DGS Studio")
        self.resize(1320, 850)
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(STYLESHEET)
        self.projects = ProjectManager(self)
        self.processes = ProcessManager(self)
        self.settings = QSettings()
        self._brush_executable = self.settings.value("brush/executable", "", type=str)
        self._active_task = ""
        self._training_steps = 0
        self._output_tail = ""
        self._last_progress_step = 0
        self.stack = QStackedWidget()
        self.dashboard = Dashboard()
        self.workspace = Workspace()
        self.stack.addWidget(self.dashboard)
        self.stack.addWidget(self.workspace)
        self.setCentralWidget(self.stack)

        self.dashboard.new_project_requested.connect(self._new_project)
        self.dashboard.open_project_requested.connect(self._open_project)
        self.dashboard.import_scene_requested.connect(self._import_scene)
        self.workspace.back_requested.connect(lambda: self.stack.setCurrentWidget(self.dashboard))
        self.workspace.environment_check_requested.connect(self._run_environment_check)
        self.workspace.brush_training_requested.connect(self._run_brush_training)
        self.workspace.open_scene_requested.connect(self._open_scene_in_brush)
        self.workspace.import_images_requested.connect(self._import_images)
        self.workspace.prepare_images_requested.connect(self._run_preprocessing)
        self.workspace.colmap_requested.connect(self._run_colmap)
        self.workspace.stop_requested.connect(self.processes.stop)
        self.projects.project_opened.connect(self._on_project_opened)
        self.processes.output_received.connect(self._handle_process_output)
        self.processes.state_changed.connect(self.workspace.set_process_state)
        self.processes.finished.connect(self._handle_process_finished)

    def _new_project(self) -> None:
        dialog = NewProjectDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            project = self.projects.create_project(dialog.parent_folder.text(), dialog.name.text())
        except (ProjectError, OSError) as exc:
            QMessageBox.warning(self, "Could not create project", str(exc))
            return
        self.stack.setCurrentWidget(self.workspace)
        self.workspace.append_output(f"Project created: {project}\n")

    def _open_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open 3DGS Studio project")
        if not folder:
            return
        try:
            project = self.projects.open_project(folder)
        except ProjectError as exc:
            QMessageBox.warning(self, "Could not open project", str(exc))
            return
        self.stack.setCurrentWidget(self.workspace)
        self.workspace.append_output(f"Project opened: {project}\n")

    def _on_project_opened(self, project: Path) -> None:
        self.workspace.set_project(project)
        self._refresh_scene_count()

    def _refresh_scene_count(self) -> None:
        manager = self.projects.scene_manager
        count = len(manager.discover()) if manager is not None else 0
        self.workspace.set_scene_count(count)

    def _import_scene(self) -> None:
        scene, _ = QFileDialog.getOpenFileName(self, "Import Gaussian scene", "", "PLY scenes (*.ply)")
        if not scene:
            return
        try:
            imported = self.projects.import_scene(scene)
        except (ProjectError, OSError) as exc:
            QMessageBox.warning(self, "Could not import scene", str(exc))
            return
        self.workspace.append_output(f"Scene imported: {imported}\n")
        self._refresh_scene_count()
        self.stack.setCurrentWidget(self.workspace)

    def _run_brush_training(self) -> None:
        if self.projects.current_project is None:
            QMessageBox.information(self, "Open a project", "Create or open a project before starting Brush.")
            return
        if self.processes.is_running:
            QMessageBox.information(self, "Process running", "Stop the current process before starting Brush.")
            return
        brush, _ = QFileDialog.getOpenFileName(
            self, "Choose Brush executable", "", "Applications (*.exe);;All files (*)"
        )
        if not brush:
            return
        self._brush_executable = brush
        self.settings.setValue("brush/executable", brush)
        dataset = QFileDialog.getExistingDirectory(self, "Choose COLMAP undistorted dataset")
        if not dataset:
            return
        dataset_path = Path(dataset)
        if not (dataset_path / "images").is_dir() or not (dataset_path / "sparse").is_dir():
            QMessageBox.warning(
                self,
                "Invalid COLMAP dataset",
                "Choose the undistorted folder itself. It must contain images/ and sparse/.",
            )
            return
        steps, accepted = QInputDialog.getInt(self, "Brush training", "Total steps:", 1000, 1, 1000000, 500)
        if not accepted:
            return
        export_every, accepted = QInputDialog.getInt(
            self, "Brush training", "Export a PLY every N steps:", 500, 1, steps, 100
        )
        if not accepted:
            return
        project = self.projects.current_project
        project_dataset = project / "colmap" / "undistorted"
        try:
            if project_dataset.exists():
                if not (project_dataset / "images").is_dir() or not (project_dataset / "sparse").is_dir():
                    QMessageBox.warning(
                        self, "Project dataset exists",
                        f"The project's dataset folder is incomplete:\n{project_dataset}\n"
                        "Move it aside before importing another dataset.",
                    )
                    return
                answer = QMessageBox.question(
                    self, "Use existing dataset?",
                    f"This project already has a valid dataset:\n{project_dataset}\n\nUse it for training?",
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            else:
                project_dataset.parent.mkdir(parents=True, exist_ok=True)
                project_dataset.mkdir(parents=True)
                shutil.copytree(dataset_path / "images", project_dataset / "images")
                shutil.copytree(dataset_path / "sparse", project_dataset / "sparse")
        except OSError as exc:
            QMessageBox.warning(self, "Could not copy dataset", str(exc))
            return

        self._training_steps = steps
        self._active_task = "training"
        self._last_progress_step = 0
        self._output_tail = ""
        self.workspace.set_training_progress(0, steps, "Training runner starting")
        self.workspace.append_output(
            f"Starting Brush training runner with live viewer.\n"
            f"Project dataset: {project_dataset}\n"
        )
        try:
            self.processes.start(
                sys.executable,
                ["-u", "-m", "backend.training.run", "--project", str(project), "--brush", brush,
                 "--steps", str(steps), "--export-every", str(export_every), "--with-viewer"],
                working_directory=str(Path(__file__).resolve().parents[1]),
            )
        except RuntimeError as exc:
            self.workspace.append_output(f"{exc}\n")

    def _handle_process_output(self, text: str) -> None:
        self.workspace.append_output(text)
        scan = self._output_tail + text
        for match in re.finditer(r"export:\s+\S+?_(\d+)\.ply", scan):
            current = int(match.group(1))
            if self._training_steps and current > self._last_progress_step:
                self._last_progress_step = current
                self.workspace.set_training_progress(current, self._training_steps, match.group(0))
        self._output_tail = scan[-256:]

    def _handle_process_finished(self, exit_code: int, succeeded: bool) -> None:
        task = self._active_task
        self._active_task = ""
        if task != "training":
            if task == "viewer":
                self.workspace.set_latest_update("Brush viewer closed.")
                self.workspace.append_output("Brush viewer closed.\n")
            elif task in {"preprocessing", "colmap"}:
                label = "Image check" if task == "preprocessing" else "COLMAP"
                if succeeded:
                    message = f"{label} completed successfully."
                else:
                    message = f"{label} failed with exit code {exit_code}. See the project logs folder."
                self.workspace.set_latest_update(message)
                self.workspace.append_output(message + "\n")
            return
        if succeeded:
            manager = self.projects.scene_manager
            finals = manager.discover("original") if manager is not None else []
            if finals:
                final = manager.register_output(finals[0].path, "original")
                self.workspace.set_training_progress(
                    self._training_steps, self._training_steps, f"Final scene: {final.name}"
                )
                self.workspace.append_output(f"Final scene: {final.path}\n")
                self._refresh_scene_count()
            else:
                self.workspace.set_latest_update("Runner exited successfully but no final PLY was found.")
            return
        if exit_code == 4:
            message = "Training stopped before the final step; no final scene was registered."
        elif exit_code == 3:
            message = "Brush ended without exporting a PLY scene."
        elif exit_code == 2:
            message = "Runner arguments or dataset are invalid; see the console and logs/brush.log."
        else:
            message = f"Training runner failed with exit code {exit_code}. See logs/brush.log."
        self.workspace.set_latest_update(message)
        self.workspace.append_output(message + "\n")

    def _run_environment_check(self) -> None:
        if self.processes.is_running:
            return
        self._active_task = "environment"
        self.workspace.append_output("$ Python environment check\n")
        code = "import sys; print('Python', sys.version.split()[0]); print('Executable:', sys.executable)"
        try:
            self.processes.start(sys.executable, ["-c", code])
        except RuntimeError as exc:
            self.workspace.append_output(f"{exc}\n")

    def _open_scene_in_brush(self) -> None:
        project = self.projects.current_project
        if project is None:
            QMessageBox.information(self, "Open a project", "Open a project before opening a scene.")
            return
        if self.processes.is_running:
            QMessageBox.information(self, "Process running", "Close or stop the current process first.")
            return
        manager = self.projects.scene_manager
        scenes = manager.discover() if manager is not None else []
        if not scenes:
            QMessageBox.information(self, "No scenes yet", f"No PLY scenes found in:\n{project / 'scenes'}")
            return
        options = [f"{record.category.title()} — {record.name} ({record.size_bytes / 1e6:.2f} MB)" for record in scenes]
        selected, accepted = QInputDialog.getItem(
            self, "Choose Gaussian scene", "Available scenes:", options, 0, False
        )
        if not accepted:
            return
        scene = str(next(record.path for record, option in zip(scenes, options) if option == selected))
        brush = self._brush_executable
        if not brush or not Path(brush).is_file():
            brush, _ = QFileDialog.getOpenFileName(
                self, "Choose Brush executable", "", "Applications (*.exe);;All files (*)"
            )
            if not brush:
                return
            self._brush_executable = brush
            self.settings.setValue("brush/executable", brush)
        self._active_task = "viewer"
        self.workspace.append_output(f"Opening scene in Brush: {scene}\n")
        try:
            self.processes.start(brush, [scene, "--with-viewer"], working_directory=str(project))
        except RuntimeError as exc:
            self._active_task = ""
            self.workspace.append_output(f"{exc}\n")

    def _import_images(self) -> None:
        project = self.projects.current_project
        if project is None:
            QMessageBox.information(self, "Open a project", "Open a project before importing images.")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choose source images", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All files (*)",
        )
        if not files:
            return
        destination = project / "input" / "images"
        destination.mkdir(parents=True, exist_ok=True)
        copied = 0
        skipped = []
        for source in files:
            source_path = Path(source)
            target = destination / source_path.name
            if target.exists():
                skipped.append(source_path.name)
                continue
            try:
                shutil.copy2(source_path, target)
                copied += 1
            except OSError as exc:
                QMessageBox.warning(self, "Image import stopped", f"Copied {copied} image(s).\n{exc}")
                return
        summary = f"Imported {copied} image(s) to {destination}."
        if skipped:
            summary += f" Skipped {len(skipped)} existing file(s) to avoid overwriting."
        self.workspace.append_output(summary + "\n")
        self.workspace.set_latest_update(summary)

    def _run_preprocessing(self) -> None:
        project = self.projects.current_project
        if project is None:
            QMessageBox.information(self, "Open a project", "Open a project before checking images.")
            return
        if self.processes.is_running:
            QMessageBox.information(self, "Process running", "Wait for the current process to finish first.")
            return
        self._active_task = "preprocessing"
        self.workspace.append_output(f"Checking project images: {project / 'input' / 'images'}\n")
        try:
            self.processes.start(
                sys.executable,
                ["-u", "-m", "backend.preprocessing.run", "--project", str(project)],
                working_directory=str(Path(__file__).resolve().parents[1]),
            )
        except RuntimeError as exc:
            self._active_task = ""
            self.workspace.append_output(f"{exc}\n")

    def _run_colmap(self) -> None:
        project = self.projects.current_project
        if project is None:
            QMessageBox.information(self, "Open a project", "Open a project before running COLMAP.")
            return
        if self.processes.is_running:
            QMessageBox.information(self, "Process running", "Wait for the current process to finish first.")
            return
        colmap = self.settings.value("colmap/executable", "", type=str)
        if not colmap or not Path(colmap).is_file():
            colmap, _ = QFileDialog.getOpenFileName(
                self, "Choose COLMAP executable", "", "Applications (colmap.exe);;All files (*)"
            )
            if not colmap:
                return
            self.settings.setValue("colmap/executable", colmap)
        use_gpu = QMessageBox.question(
            self,
            "COLMAP acceleration",
            "Use NVIDIA GPU acceleration? Choose No to run on CPU if COLMAP crashes during startup.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes
        self._active_task = "colmap"
        mode = "GPU" if use_gpu else "CPU"
        self.workspace.append_output(f"Starting COLMAP on {project / 'input' / 'images'} using {mode}.\n")
        try:
            self.processes.start(
                sys.executable,
                ["-u", "-m", "backend.colmap.run", "--project", str(project),
                 "--colmap", colmap, "--quality", "medium", "--use-gpu",
                 "1" if use_gpu else "0"],
                working_directory=str(Path(__file__).resolve().parents[1]),
            )
        except RuntimeError as exc:
            self._active_task = ""
            self.workspace.append_output(f"{exc}\n")


