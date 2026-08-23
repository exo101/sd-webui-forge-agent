from PyQt6.QtGui import QFocusEvent
from qtpy.QtWidgets import QComboBox, QWidget, QLineEdit
from qtpy.QtCore import Signal, QSize, Qt
from qtpy.QtGui import QDoubleValidator, QIntValidator


class LineEdit(QLineEdit):

    edit_value_changed = Signal()

    def __init__(self, validator_type=None, validator_range=None, prefix='', suffix='', init_value=0):
        super().__init__()
        if validator_type is not None:
            if validator_type == float:
                validator = QDoubleValidator()
            elif validator_type == int:
                validator = QIntValidator()
            if validator_range is not None:
                validator.setTop(validator_range[0])
                validator.setBottom(validator_range[1])
            # self.setValidator(validator)

        self._valid_range = validator_range
        self._prefix = prefix
        self._suffix = suffix
        self._old_value = init_value
        self._validator_type = validator_type
        self.setText(self._prefix + str(init_value) + self._suffix)
        
        # self.textEdited.connect(self.on_text_edited)
        self.returnPressed.connect(self.on_end_edit)

    def focusOutEvent(self, a0: QFocusEvent | None) -> None:
        self.on_end_edit()
        return super().focusOutEvent(a0)
    
    def focusInEvent(self, a0: QFocusEvent | None) -> None:
        self._old_value = self.get_value()
        return super().focusInEvent(a0)
    
    def set_value(self, v):
        self.setText(self._prefix + str(v) + self._suffix)

    def get_value(self):
        v = self.text()
        v = v.replace(self._prefix, '').replace(self._suffix, '')
        validator_type = self._validator_type
        if validator_type == float:
            try:
                v = float(v)
            except:
                return None
        elif validator_type == int:
            try:
                v = int(v)
            except:
                return None
        if self._valid_range is not None:
            v = max(min(v, self._valid_range[1]), self._valid_range[0])
        return v

    def on_end_edit(self):
        value = self.get_value()
        if value is None:
            print('invalid val')
            value = self._old_value
        cursor_pos = self.cursorPosition()
        self.set_value(value)
        if self.hasFocus():
            self.setCursorPosition(cursor_pos)
        if self._old_value != value:
            print(f'val change: {value}')
            self.edit_value_changed.emit()
        self._old_value = value
        