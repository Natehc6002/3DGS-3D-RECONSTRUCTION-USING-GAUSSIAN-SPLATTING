from __future__ import annotations

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal


class ProcessManager(QObject):
    output_received = Signal(str)
    state_changed = Signal(str)
    finished = Signal(int, bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.started.connect(lambda: self.state_changed.emit("RUNNING"))
        self.process.errorOccurred.connect(self._on_error)
        self.process.finished.connect(self._on_finished)
        self._stop_timer = QTimer(self)
        self._stop_timer.setSingleShot(True)
        self._stop_timer.setInterval(2000)
        self._stop_timer.timeout.connect(self._kill_if_running)
        self._stopping = False

    @property
    def is_running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, program: str, arguments: list[str], working_directory: str | None = None) -> None:
        if self.is_running:
            raise RuntimeError("A process is already running.")
        if working_directory:
            self.process.setWorkingDirectory(working_directory)
        self._stopping = False
        self.state_changed.emit("STARTING")
        self.process.start(program, arguments)

    def stop(self) -> None:
        if not self.is_running:
            return
        self._stopping = True
        self.state_changed.emit("STOPPING")
        self.process.terminate()
        self._stop_timer.start()

    def _read_stdout(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        if text:
            self.output_received.emit(text)

    def _read_stderr(self) -> None:
        text = bytes(self.process.readAllStandardError()).decode(errors="replace")
        if text:
            self.output_received.emit(text)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.output_received.emit(f"Could not start process: {self.process.errorString()}\n")
            self.state_changed.emit("FAILED")
            self.finished.emit(-1, False)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._stop_timer.stop()
        succeeded = exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0 and not self._stopping
        state = "COMPLETED" if succeeded else ("STOPPED" if self._stopping else "FAILED")
        self.state_changed.emit(state)
        self.finished.emit(exit_code, succeeded)
        self._stopping = False

    def _kill_if_running(self) -> None:
        if self.is_running:
            self.process.kill()
