from qtpy.QtCore import Signal
from qtpy.QtWidgets import QCheckBox


class ToolCheckBox(QCheckBox):
    checked = Signal()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stateChanged.connect(self.on_state_changed)

    def on_state_changed(self, state: int) -> None:
        if self.isChecked():
            self.checked.emit()