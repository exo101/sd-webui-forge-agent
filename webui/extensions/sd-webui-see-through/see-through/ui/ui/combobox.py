from typing import List, Callable

from qtpy.QtWidgets import QComboBox, QWidget
from qtpy.QtCore import Signal, QSize, Qt
from qtpy.QtGui import QGuiApplication, QContextMenuEvent, QIcon, QCloseEvent, QDoubleValidator

class ComboBox(QComboBox):

    # https://stackoverflow.com/questions/3241830/qt-how-to-disable-mouse-scrolling-of-qcombobox
    def __init__(self, parent: QWidget = None, scrollWidget: QWidget = None, options: List[str] = None) -> None:
        super().__init__(parent)
        self.scrollWidget = scrollWidget
        if options is not None:
            self.addItems(options)

    def setScrollWidget(self, scrollWidget: QWidget):
        self.scrollWidget = scrollWidget

    def wheelEvent(self, *args, **kwargs):
        if self.scrollWidget is None or self.hasFocus():
            return super().wheelEvent(*args, **kwargs)
        else:
            return self.scrollWidget.wheelEvent(*args, **kwargs)
        

class SmallComboBox(ComboBox):
    pass


class SizeComboBox(QComboBox):
    
    param_changed = Signal(str, float)
    def __init__(self, val_range: List = None, param_name: str = '', parent=None, init_value=None, validator_cls=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.param_name = param_name
        self.editTextChanged.connect(self.on_text_changed)
        self.activated.connect(self.on_current_index_changed)
        self.setEditable(True)
        self.min_val = val_range[0]
        self.max_val = val_range[1]
        if validator_cls is None:
            validator_cls = QDoubleValidator
        validator = validator_cls()
        if val_range is not None:
            validator.setTop(val_range[1])
            validator.setBottom(val_range[0])
        if isinstance(validator, QDoubleValidator):
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)

        self.setValidator(validator)
        self._value = 0
        if init_value is not None:
            self.setValue(init_value)

    @property
    def value_valid(self):
        v = self.value()
        if v >= self.min_val and v <= self.max_val:
            return True
        return False

    def on_text_changed(self):
        if self.hasFocus():
            if self.value_valid:
                self.param_changed.emit(self.param_name, self.value())

    def on_current_index_changed(self):
        if self.hasFocus() or self.view().isVisible():
            if self.value_valid:
                self.param_changed.emit(self.param_name, self.value())

    def value(self) -> float:
        txt = self.currentText()
        try:
            val = float(txt)
            self._value = val
            return val
        except:
            return self._value

    def setValue(self, value: float):
        value = min(self.max_val, max(self.min_val, value))
        self.setCurrentText(str(round(value, 2)))

    def changeByDelta(self, delta: float, multiplier = 0.01):
        if isinstance(multiplier, Callable):
            multiplier = multiplier()
        self.setValue(self.value() + delta * multiplier)


class SmallSizeComboBox(SizeComboBox):
    pass