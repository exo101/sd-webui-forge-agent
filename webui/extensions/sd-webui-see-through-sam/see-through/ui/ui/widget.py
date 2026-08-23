from qtpy.QtWidgets import QWidget, QFrame, QLabel
from qtpy.QtCore import Qt, Signal

class Widget(QWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)


class SeparatorWidget(QFrame):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)


class ClickableLabel(QLabel):

    clicked = Signal()

    def __init__(self, text=None, parent=None, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        if text is not None:
            self.setText(text)

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        return super().mousePressEvent(e)