from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHeaderView, QLabel,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)


class EvaluationResultsDialog(QDialog):
    """Show the key comparison fields from a saved evaluation CSV."""

    COLUMNS = (
        ("variant", "Scene"),
        ("gaussians", "Gaussians"),
        ("file_size_mb", "File size"),
        ("size_ratio", "Size ratio"),
        ("processing_time_s", "Time"),
        ("coverage_proxy", "Coverage proxy"),
        ("brush_load_status", "Brush status"),
    )

    def __init__(self, csv_path: str | Path, parent=None) -> None:
        super().__init__(parent)
        self.csv_path = Path(csv_path)
        with self.csv_path.open("r", newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))

        self.setWindowTitle(f"Evaluation results — {self.csv_path.name}")
        self.setMinimumSize(1120, 460)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel(f"Comparison report: {self.csv_path.name}")
        title.setStyleSheet("font-size: 17px; font-weight: 650;")
        layout.addWidget(title)

        table = QTableWidget(len(rows), len(self.COLUMNS), self)
        table.setObjectName("evaluationTable")
        table.setHorizontalHeaderLabels([heading for _, heading in self.COLUMNS])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for column in range(1, 5):
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)

        for row_index, row in enumerate(rows):
            for column_index, (key, _) in enumerate(self.COLUMNS):
                value = self._format_value(key, row.get(key, ""))
                cell = QTableWidgetItem(value or "—")
                cell.setToolTip(value)
                table.setItem(row_index, column_index, cell)
        table.setStyleSheet(
            "QTableWidget#evaluationTable { background: #0b0e11; color: #e8edf2; "
            "alternate-background-color: #171d23; gridline-color: #28323b; }"
            "QHeaderView::section { background: #202830; color: #e8edf2; "
            "padding: 7px; border: 0; }"
        )
        layout.addWidget(table, 1)

        note = QLabel(
            "Coverage is a geometry proxy. PSNR, SSIM, and FPS are not measured; "
            "Brush status records visual checks."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #9aa7b2; font-size: 12px;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _format_value(key: str, value: str) -> str:
        if not value:
            return "—"
        if key == "file_size_mb":
            try:
                return f"{float(value):.2f} MB"
            except ValueError:
                return value
        if key == "size_ratio":
            try:
                return f"{float(value):.3f}×"
            except ValueError:
                return value
        if key == "processing_time_s":
            try:
                return f"{float(value):.3f} s"
            except ValueError:
                return value
        return value
