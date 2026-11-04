from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QStackedWidget, QVBoxLayout, QWidget,
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
        self.workspace.stop_requested.connect(self.processes.stop)
        self.projects.project_opened.connect(self.workspace.set_project)
        self.processes.output_received.connect(self.workspace.append_output)
        self.processes.state_changed.connect(self.workspace.set_process_state)

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

    def _import_scene(self) -> None:
        scene, _ = QFileDialog.getOpenFileName(self, "Import Gaussian scene", "", "PLY scenes (*.ply)")
        if not scene:
            return
        QMessageBox.information(
            self, "Scene selected", f"Selected scene:\n{scene}\n\nOpen or create a project to register it."
        )

    def _run_environment_check(self) -> None:
        if self.processes.is_running:
            return
        self.workspace.append_output("$ Python environment check\n")
        code = "import sys; print('Python', sys.version.split()[0]); print('Executable:', sys.executable)"
        try:
            self.processes.start(sys.executable, ["-c", code])
        except RuntimeError as exc:
            self.workspace.append_output(f"{exc}\n")


