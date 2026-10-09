from qtpy.QtWidgets import QGraphicsOpacityEffect, QLabel, QColorDialog
from qtpy.QtCore import  Qt, QPropertyAnimation, QEasingCurve, Signal
from qtpy.QtGui import QMouseEvent, QWheelEvent, QColor

from .shared import CONFIG_FONTSIZE_CONTENT
from . import shared

class FadeLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # https://stackoverflow.com/questions/57828052/qpropertyanimation-not-working-with-window-opacity
        effect = QGraphicsOpacityEffect(self, opacity=1.0)
        self.setGraphicsEffect(effect)
        self.fadeAnimation = QPropertyAnimation(
            self,
            propertyName=b"opacity",
            targetObject=effect,
            duration=1200,
            startValue=1.0,
            endValue=0.,
        )
        self.fadeAnimation.setEasingCurve(QEasingCurve.Type.InQuint)
        self.fadeAnimation.finished.connect(self.hide)
        self.setHidden(True)
        self.gv = None

    def startFadeAnimation(self):
        self.show()
        self.fadeAnimation.stop()
        self.fadeAnimation.start()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.gv is not None:
            self.gv.wheelEvent(event)
        return super().wheelEvent(event)
    

class ParamNameLabel(QLabel):
    def __init__(self, param_name: str, alignment = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if alignment is None:
            self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        else:
            self.setAlignment(alignment)

        font = self.font()
        font.setPointSizeF(CONFIG_FONTSIZE_CONTENT-2)
        self.setFont(font)
        self.setText(param_name)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)


class SmallParamLabel(QLabel):
    def __init__(self, param_name: str, alignment = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if alignment is None:
            self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        else:
            self.setAlignment(alignment)

        self.setText(param_name)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)


class SizeControlLabel(QLabel):

    btn_released = Signal()
    size_ctrl_changed = Signal(int)

    def __init__(self, parent=None, direction=0, text='', alignment=None, transparent_bg=True):
        super().__init__(parent)
        if text:
            self.setText(text)
        if direction == 0:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.cur_pos = 0
        self.direction = direction
        self.mouse_pressed = False
        if transparent_bg:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        if alignment is not None:
            self.setAlignment(alignment)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.mouse_pressed = True
            if shared.FLAG_QT6:
                g_pos = e.globalPosition().toPoint()
            else:
                g_pos = e.globalPos()
            self.cur_pos = g_pos.x() if self.direction == 0 else g_pos.y()
        return super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.mouse_pressed = False
            self.btn_released.emit()
        return super().mouseReleaseEvent(e)

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self.mouse_pressed:
            if shared.FLAG_QT6:
                g_pos = e.globalPosition().toPoint()
            else:
                g_pos = e.globalPos()
            if self.direction == 0:
                new_pos = g_pos.x()
                self.size_ctrl_changed.emit(new_pos - self.cur_pos)
            else:
                new_pos = g_pos.y()
                self.size_ctrl_changed.emit(self.cur_pos - new_pos)
            self.cur_pos = new_pos
        return super().mouseMoveEvent(e)
    

class SmallSizeControlLabel(SizeControlLabel):
    pass