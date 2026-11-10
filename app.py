from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from frontend.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("3DGS Studio")
    app.setOrganizationName("3DGS Studio Team")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
