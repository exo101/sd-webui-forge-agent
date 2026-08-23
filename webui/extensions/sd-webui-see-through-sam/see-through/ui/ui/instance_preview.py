import numpy as np
from typing import List, Union

from qtpy.QtWidgets import QHBoxLayout, QStackedWidget, QLineEdit, QSizePolicy, QLabel, QScrollArea, QVBoxLayout, QApplication
from qtpy.QtCore import Qt, QPoint, Signal, QSize, QMimeData
from qtpy.QtGui import QIntValidator, QFocusEvent, QKeyEvent, QMouseEvent, QDragEnterEvent, QDropEvent, QPixmap, QDrag, QDragLeaveEvent
import cv2

from .widget import Widget
from .scrollbar import ScrollBar
from .misc import ndarray2pixmap
from .structures import Instance
from . import shared
from .logger import logger as LOGGER

STYLE_TRANSPAIR_CHECKED = "border: 3px solid rgba(30, 147, 229, 100%);"
STYLE_TRANSPAIR_BOTTOM = "border-width: 5px; border-bottom-style: solid; border-color: rgb(30, 147, 229);"
STYLE_TRANSPAIR_TOP = "border-width: 5px; border-top-style: solid; border-color: rgb(30, 147, 229);"


class RowIndexEditor(QLineEdit):

    focus_out = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setValidator(QIntValidator())
        self.setReadOnly(True)
        self.setTextMargins(0, 0, 0, 0)

    def focusOutEvent(self, e: QFocusEvent) -> None:
        super().focusOutEvent(e)
        self.focus_out.emit()

    def minimumSizeHint(self):
        size = super().minimumSizeHint()
        return QSize(1, size.height())
    
    def sizeHint(self):
        size = super().sizeHint()
        return QSize(1, size.height())



class RowIndexLabel(QStackedWidget):

    submmit_idx = Signal(int)

    def __init__(self, text: str = None, parent=None, editable=True):
        super().__init__(parent=parent)
        self.lineedit = RowIndexEditor(parent=self)
        self.lineedit.focus_out.connect(self.on_lineedit_focusout)

        self.show_label = QLabel(self)
        self.text = self.show_label.text

        self.addWidget(self.show_label)
        self.addWidget(self.lineedit)
        self.setCurrentIndex(0)

        if text is not None:
            self.setText(text)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

        self._editable = editable

    def setText(self, text):
        if isinstance(text, int):
            text = str(text)
        self.show_label.setText(text)
        self.lineedit.setText(text)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        super().keyPressEvent(e)

        key = e.key()
        if key == Qt.Key.Key_Return:
            self.try_update_idx()

    def try_update_idx(self):
        idx_str = self.lineedit.text().strip()
        if not idx_str:
            return
        if self.text() == idx_str:
            return
        try:
            idx = int(idx_str)
            self.lineedit.setReadOnly(True)
            self.submmit_idx.emit(idx)
            
        except Exception as e:
            LOGGER.warning(f'Invalid index str: {idx}')

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        if self._editable:
            self.startEdit()
        return super().mouseDoubleClickEvent(e)
    
    def setEditable(self, editable: bool):
        self._editable = editable

    def startEdit(self) -> None:
        self.setCurrentIndex(1)
        self.lineedit.setReadOnly(False)
        self.lineedit.setFocus()

    def on_lineedit_focusout(self):
        edited = not self.lineedit.isReadOnly()
        self.lineedit.setReadOnly(True)
        self.setCurrentIndex(0)
        if edited:
            self.try_update_idx()

    def mousePressEvent(self, e: QMouseEvent) -> None:
        e.ignore()
        return super().mousePressEvent(e)


class InstancePreviewLabel(QLabel):

    def updatePreview(self, cutout = None, preview_size=120):
        if cutout is not None:
            h, w = cutout.shape[:2]
            nw = int(round(preview_size / h * w))
            cutout_resized = cv2.resize(cutout, (nw, preview_size), interpolation=cv2.INTER_AREA)
            cutout_resized = ndarray2pixmap(cutout_resized)
        else:
            cutout_resized = QPixmap(1, preview_size)
            cutout_resized.fill(Qt.GlobalColor.transparent)
        self.setPixmap(cutout_resized)


class InstancePreviewScrollArea(QScrollArea):
    preview_list = []
    remove_textblock = Signal()
    selection_changed = Signal()   # this signal could only emit in on_widget_checkstate_changed, i.e. via user op
    rearrange_blks = Signal(object)
    contextmenu_requested = Signal(QPoint, bool)
    focus_out = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scrollContent = Widget(parent=self)
        self.setWidget(self.scrollContent)

        # ScrollBar(Qt.Orientation.Horizontal, self)
        ScrollBar(Qt.Orientation.Vertical, self)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        vlayout = QVBoxLayout(self.scrollContent)
        # vlayout.setContentsMargins(0, 0, 3, 0)
        vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        vlayout.setSpacing(0)
        vlayout.addStretch(1)
        self.setWidgetResizable(True)
        self.vlayout = vlayout
        self.drag: QDrag = None
        self.dragStartPosition = None

        self.source_visible = True
        self.trans_visible = True

        self.drag_to_pos: int = -1

        self.setSizePolicy(self.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Expanding)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setMinimumWidth(200)

    def mouseReleaseEvent(self, e: QMouseEvent):
        if e.button() == Qt.MouseButton.RightButton:
            pos = self.mapToGlobal(e.position()).toPoint()
            self.contextmenu_requested.emit(pos, True)
        super().mouseReleaseEvent(e)
    
    def handle_drag_pos(self, to_pos: int):
        if self.drag_to_pos != to_pos:
            if self.drag_to_pos is not None:
                self.set_drag_style(self.drag_to_pos, True)
            self.drag_to_pos = to_pos
            self.set_drag_style(to_pos)

    def on_idx_edited(self, src_idx: int, tgt_idx: int):
        src_idx_ori = tgt_idx
        tgt_idx = max(min(tgt_idx, len(self.preview_list) - 1), 0)
        if src_idx_ori != tgt_idx:
            self.preview_list[src_idx].idx_label.setText(str(src_idx + 1).zfill(2))
        if src_idx == tgt_idx:
            return
        ids_ori, ids_tgt = [src_idx], [tgt_idx]
        
        if src_idx < tgt_idx:
            for idx in range(src_idx+1, tgt_idx+1):
                ids_ori.append(idx)
                ids_tgt.append(idx-1)
        else:
            for idx in range(tgt_idx, src_idx):
                ids_ori.append(idx)
                ids_tgt.append(idx+1)
        self.rearrange_blks.emit((ids_ori, ids_tgt, (tgt_idx, src_idx)))


    def on_widget_checkstate_changed(self, pwc, shift_pressed: bool, ctrl_pressed: bool):
        return
        if self.drag is not None:
            return
        
        idx = pwc.idx
        if shift_pressed:
            checked = True
        else:
            checked = not pwc.checked
        pwc._set_checked_state(checked)

        num_sel = len(self.checked_list)
        old_idx_list = [pw.idx for pw in self.checked_list]
        old_idx_set = set(old_idx_list)
        new_check_list = []
        if shift_pressed:
            if num_sel == 0:
                new_check_list.append(idx)
            else:
                tgt_w = self.preview_list[idx]
                if ctrl_pressed:
                    sel_min, sel_max = min(old_idx_list[0], tgt_w.idx), max(old_idx_list[-1], tgt_w.idx)
                else:
                    sel_min, sel_max = min(self.sel_anchor_widget.idx, tgt_w.idx), max(self.sel_anchor_widget.idx, tgt_w.idx)
                new_check_list = list(range(sel_min, sel_max + 1))
        elif ctrl_pressed:
            new_check_set = set(old_idx_list)
            if idx in new_check_set:
                new_check_set.remove(idx)
                if self.sel_anchor_widget is not None and self.sel_anchor_widget.idx == idx:
                    self.sel_anchor_widget = None
            elif checked:
                new_check_set.add(idx)
            new_check_list = list(new_check_set)
            new_check_list.sort()
            if checked:
                self.sel_anchor_widget = self.preview_list[idx]
        else:
            if num_sel > 2:
                if idx in old_idx_set:
                    old_idx_set.remove(idx)
                    checked = True
            if checked:
                new_check_list.append(idx)
        
        new_check_set = set(new_check_list)
        check_changed = False
        for oidx in old_idx_set:
            if oidx not in new_check_set:
                self.preview_list[oidx]._set_checked_state(False)
                check_changed = True

        self.checked_list.clear()
        for nidx in new_check_list:
            pw = self.preview_list[nidx]
            if nidx not in old_idx_set:
                check_changed = True
                pw._set_checked_state(True)
            self.checked_list.append(pw)
            
        num_new = len(new_check_list)
        if num_new == 0:
            self.sel_anchor_widget = None
        elif num_new == 1 or self.sel_anchor_widget is None:
            self.sel_anchor_widget = self.checked_list[0]
        if check_changed:
            self.selection_changed.emit()

    def ensureWidgetVisible(self, *args, **kwargs):
        sb = self.horizontalScrollBar()
        if sb is not None:
            pre_hpos = sb.sliderPosition()
        super().ensureWidgetVisible(*args, **kwargs)
        if sb is not None:
            sb.setSliderPosition(pre_hpos)
    
    def focusOutEvent(self, e: QFocusEvent) -> None:
        self.focus_out.emit()
        super().focusOutEvent(e)

    def clearAllSelected(self):
        pass