from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)


class Dashboard(QWidget):
    new_project_requested = Signal()
    open_project_requested = Signal()
    import_scene_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("dashboard")
        root = QVBoxLayout(self)
        root.setContentsMargins(52, 42, 52, 38)
        root.setSpacing(24)

        eyebrow = QLabel("3DGS STUDIO  /  DESKTOP WORKSPACE")
        eyebrow.setObjectName("eyebrow")
        root.addWidget(eyebrow)
        title = QLabel("Build a scene.\nExplore every Gaussian.")
        title.setObjectName("heroTitle")
        root.addWidget(title)
        subtitle = QLabel(
            "Prepare images, reconstruct camera poses, train with Brush, and review your scene."
        )
        subtitle.setObjectName("mutedText")
        root.addWidget(subtitle)

        cards = QHBoxLayout()
        cards.setSpacing(14)
        cards.addWidget(self._action_card(
            "01", "New project", "Create a project with the shared folder layout.",
            "Create project", self.new_project_requested.emit, "accentButton"
        ))
        cards.addWidget(self._action_card(
            "02", "Open project", "Continue work from an existing Studio project.",
            "Browse projects", self.open_project_requested.emit, "secondaryButton"
        ))
        cards.addWidget(self._action_card(
            "03", "Import a scene", "Open a Gaussian PLY in your project workflow.",
            "Choose PLY scene", self.import_scene_requested.emit, "secondaryButton"
        ))
        root.addLayout(cards)
        root.addStretch(1)
        footer = QFrame()
        footer.setObjectName("infoStrip")
        strip = QHBoxLayout(footer)
        strip.setContentsMargins(18, 14, 18, 14)
        strip.addWidget(QLabel("WORKFLOW"))
        flow = QLabel("Images  →  COLMAP  →  Brush training  →  Scene review  →  Optimization")
        flow.setObjectName("mutedText")
        strip.addWidget(flow, 1)
        root.addWidget(footer)

    def _action_card(self, number: str, heading: str, description: str, button_text: str,
                     callback, button_style: str) -> QFrame:
        card = QFrame()
        card.setObjectName("actionCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        index = QLabel(number)
        index.setObjectName("cardNumber")
        layout.addWidget(index)
        name = QLabel(heading)
        name.setObjectName("cardTitle")
        layout.addWidget(name)
        detail = QLabel(description)
        detail.setObjectName("mutedText")
        detail.setWordWrap(True)
        layout.addWidget(detail)
        layout.addStretch(1)
        button = QPushButton(button_text)
        button.setObjectName(button_style)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return card
