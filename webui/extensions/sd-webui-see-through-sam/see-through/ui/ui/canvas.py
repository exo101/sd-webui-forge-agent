from typing import List, Union
import os
import os.path as osp

import numpy as np
from qtpy.QtWidgets import QMenu, QGraphicsScene, QGraphicsView, QGraphicsSceneDragDropEvent, QGraphicsRectItem, QGraphicsItem, QScrollBar, QGraphicsPixmapItem, QGraphicsSceneMouseEvent, QRubberBand
from qtpy.QtCore import Qt, QRectF, QPointF, QPoint, Signal, QSizeF
from qtpy.QtGui import QAction, QPixmap, QHideEvent, QKeyEvent, QWheelEvent, QResizeEvent, QPainter, QPen, QPainterPath, QCursor, QNativeGestureEvent
try:
    from qtpy.QtWidgets import QUndoStack, QUndoCommand
except:
    from qtpy.QtGui import QUndoStack, QUndoCommand

from .search_widget import PageSearchWidget    

from .proj import ProjSeg
from .misc import ndarray2pixmap, QKEY, QNUMERIC_KEYS
from .label import FadeLabel
from .scrollbar import ScrollBar
from .drawable_item import DrawableItem, SceneRectTool, TagPathItem
from . import shared
from .ui_config import pcfg, EditMode

CANVAS_SCALE_MAX = 10.0
CANVAS_SCALE_MIN = 0.01
CANVAS_SCALE_SPEED = 0.1



class CustomGV(QGraphicsView):
    ctrl_pressed = False
    scale_up_signal = Signal()
    scale_down_signal = Signal()
    scale_with_value = Signal(float)
    view_resized = Signal()
    hide_canvas = Signal()
    ctrl_released = Signal()
    canvas: QGraphicsScene = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scrollbar_h = ScrollBar(Qt.Orientation.Horizontal, self, fadeout=True)
        self.scrollbar_v = ScrollBar(Qt.Orientation.Vertical, self, fadeout=True)

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def wheelEvent(self, event : QWheelEvent) -> None:
        # qgraphicsview always scroll content according to wheelevent
        # which is not desired when scaling img
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.scale_up_signal.emit()
            else:
                self.scale_down_signal.emit()
            return
        return super().wheelEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == QKEY.Key_Control:
            self.ctrl_pressed = False
            self.ctrl_released.emit()
        return super().keyReleaseEvent(event)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        key = e.key()
        if key == QKEY.Key_Control:
            self.ctrl_pressed = True

        return super().keyPressEvent(e)

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.view_resized.emit()
        return super().resizeEvent(event)

    def hideEvent(self, event: QHideEvent) -> None:
        self.hide_canvas.emit()
        return super().hideEvent(event)

    def event(self, e):
        if isinstance(e, QNativeGestureEvent):
            if e.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                self.scale_with_value.emit(e.value() + 1)
                e.setAccepted(True)

        return super().event(e)
    # def enterEvent(self, event: QEvent) -> None:
    #   # not sure why i add it
        # self.setFocus()
    #     return super().enterEvent(event)

class Canvas(QGraphicsScene):

    scalefactor_changed = Signal()
    end_create_rect = Signal(QRectF, int)

    scale_tool = Signal(QPointF)
    end_scale_tool = Signal()
    export_cutout = Signal(bool)
    
    proj: ProjSeg = None
    edit_mode = EditMode.NONE

    # weird sometimes qt can't catch altmodifier: arow keys+alt
    alt_pressed = False
    scale_tool_mode = False

    projstate_unsaved = False
    proj_savestate_changed = Signal(bool)
    drop_open_folder = Signal(str)
    context_menu_requested = Signal(QPoint)
    incanvas_selection_changed = Signal(list)

    drawable_selection_changed = Signal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scale_factor = 1.
        self.text_transparency = 0
        self.creating_rect = False
        self.create_rect_origin: QPointF = None
        self.insitem_list: List[DrawableItem] = []
        self.did2drawableitem: dict[str, DrawableItem] = {}

        self.gv = CustomGV(self)
        self.gv.scale_down_signal.connect(self.scaleDown)
        self.gv.scale_up_signal.connect(self.scaleUp)
        self.gv.scale_with_value.connect(self.scaleBy)
        self.gv.view_resized.connect(self.onViewResized)
        self.gv.hide_canvas.connect(self.on_hide_canvas)
        self.gv.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.gv.canvas = self
        self.gv.setAcceptDrops(True)
        self.gv.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.gv.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        
        self.search_widget = PageSearchWidget(self.gv)
        self.search_widget.hide()

        self.saved_undo_step = 0
        self.pushed_undo_step = 0
        self.undo_stack = QUndoStack(self)

        self.ctrl_relesed = self.gv.ctrl_released
        self.vscroll_bar = self.gv.verticalScrollBar()
        self.hscroll_bar = self.gv.horizontalScrollBar()
        # self.default_cursor = self.gv.cursor()
        self.rubber_band = self.addWidget(QRubberBand(QRubberBand.Shape.Rectangle))
        self.rubber_band.hide()
        self.rubber_band_origin = None

        self.scaleFactorLabel = FadeLabel(self.gv)
        self.scaleFactorLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scaleFactorLabel.setText('100%')
        self.scaleFactorLabel.gv = self.gv

        self.rect_tool = SceneRectTool(self.gv)
        
        self.baseLayer = QGraphicsRectItem()
        pen = QPen()
        pen.setColor(Qt.GlobalColor.transparent)
        self.baseLayer.setPen(pen)

        self.imgLayer = QGraphicsPixmapItem()
        self.imgLayer.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.imgLayer.setAcceptDrops(True)

        self.baseLayer.setAcceptDrops(True)
        self.base_pixmap: QPixmap = None

        self.addItem(self.baseLayer)
        self.imgLayer.setParentItem(self.baseLayer)
        # self.instanceLayer.setParentItem(self.baseLayer)
        self.rect_tool.setParentItem(self.baseLayer)

        self.scalefactor_changed.connect(self.onScaleFactorChanged)
        # self.selectionChanged.connect(self.on_selection_changed)     

        self.erase_img_key = None

        self.mid_btn_pressed = False
        self.pan_initial_pos = QPoint(0, 0)

        self.drop_folder: str = None
        self.block_selection_signal = False
        
        im_rect = QRectF(0, 0, shared.SCREEN_W, shared.SCREEN_H)
        self.baseLayer.setRect(im_rect)

        self.tag2tagitem: dict[str, TagPathItem] = {}

    def get_tagitem(self, tag: str):
        if tag is None:
            tag = 'None'
        if tag not in self.tag2tagitem:
            tag = 'None'
        return self.tag2tagitem[tag]

    def setShowTagMask(self, show: bool):
        for tag_path in self.tag2tagitem.values():
            tag_path.show_mask(show)

    def setShowContour(self, show: bool):
        for tag_path in self.tag2tagitem.values():
            tag_path.show_contour(show)


    def setColormskOpacity(self, value):
        for tag_path in self.tag2tagitem.values():
            tag_path.set_msk_opacity(value)

    def update_tag(self, dids, tag_list):
        tag_path_to_update = set()
        for did, t in zip(dids, tag_list):
            d = self.did2drawableitem[did]
            if d.drawable.tag != t:
                tag_path_to_update.add(d.drawable.tag)
                tag_path_to_update.add(t)
                d.set_drawable_tag(t)
                tagpath = self.get_tagitem(t)
                d.setVisible(tagpath.isVisible())
        for tag in tag_path_to_update:
            tag_path = self.get_tagitem(tag)
            tag_path.update_path(self.proj.l2dmodel)

    def update_cls_list(self, cls_list):
        for t in self.tag2tagitem.values():
            self.removeItem(t)
        self.tag2tagitem.clear()
        for c in cls_list + ['None']:
            item = TagPathItem(c)
            self.tag2tagitem[c] = item
            item.setParentItem(self.baseLayer)
            item.setZValue(0.5)

    def selectAll(self):
        self.block_selection_signal = True
        for insitem in self.did2drawableitem.values():
            insitem.setSelected(True)
        self.block_selection_signal = False

    def on_incanvas_selection_changed(self):
        selected_lst = self.selected_drawable_items()
        sel_ids = [item.drawable.did for item in selected_lst]
        self.incanvas_selection_changed.emit(sel_ids)

    def on_switch_item(self, switch_delta: int, key_event: QKeyEvent = None):
        return
        n_blk = len(self.textblk_item_list)
        if n_blk < 1:
            return
        
        editing_blk = None
        if current_editing_widget is None:
            editing_blk = self.editingTextItem()
            if editing_blk is not None:
                tgt_idx = editing_blk.idx + switch_delta
            else:
                sel_blks = self.canvas.selected_text_items(sort=False)
                if len(sel_blks) == 0:
                    return
                sel_blk = sel_blks[0]
                tgt_idx = sel_blk.idx + switch_delta
        else:
            tgt_idx = current_editing_widget.idx + switch_delta

        if tgt_idx < 0:
            tgt_idx += n_blk
        elif tgt_idx >= n_blk:
            tgt_idx -= n_blk
        blk = self.textblk_item_list[tgt_idx]

        if current_editing_widget is None:
            if editing_blk is None:
                self.canvas.block_selection_signal = True
                self.canvas.clearSelection()
                blk.setSelected(True)
                self.canvas.block_selection_signal = False
                self.canvas.gv.ensureVisible(blk)
                self.txtblkShapeControl.setBlkItem(blk)
                edit = self.pairwidget_list[tgt_idx].e_trans
                self.changeHoveringWidget(edit)
                self.instance_preview_area.set_selected_list([blk.idx])
            else:
                editing_blk.endEdit()
                editing_blk.setSelected(False)
                self.txtblkShapeControl.setBlkItem(blk)
                blk.setSelected(True)
                blk.startEdit()
                self.canvas.gv.ensureVisible(blk)
        else:
            self.textblk_item_list[current_editing_widget.idx].setSelected(False)
            current_pw = self.pairwidget_list[tgt_idx]
            is_trans = isinstance(current_editing_widget, TransTextEdit)
            if is_trans:
                w = current_pw.e_trans
            else:
                w = current_pw.e_source

            self.changeHoveringWidget(w)
            w.setFocus()

        if key_event is not None:
            key_event.accept()

    def img_window_size(self):
        if self.proj.inpainted_valid:
            return self.imgLayer.pixmap().size()
        return self.baseLayer.rect().size().toSize()

    def dragEnterEvent(self, e: QGraphicsSceneDragDropEvent):
        
        self.drop_folder = None
        if e.mimeData().hasUrls():
            urls = e.mimeData().urls()
            ufolder = None
            for url in urls:
                furl = url.toLocalFile()
                if os.path.isdir(furl):
                    ufolder = furl
                    break
            if ufolder is not None:
                e.acceptProposedAction()
                self.drop_folder = ufolder

    def dropEvent(self, event) -> None:
        if self.drop_folder is not None:
            self.drop_open_folder.emit(self.drop_folder)
            self.drop_folder = None
        return super().dropEvent(event)

    def drawMode(self) -> bool:
        return self.editor_index == 0

    def scaleUp(self):
        self.scaleImage(1 + CANVAS_SCALE_SPEED)

    def scaleDown(self):
        self.scaleImage(1 - CANVAS_SCALE_SPEED)

    def scaleBy(self, value: float):
        self.scaleImage(value)

    def updateLayers(self):
        if not self.proj.model_valid:
            return
        self.base_pixmap = ndarray2pixmap(np.zeros_like(self.proj.l2dmodel.final))
        self.imgLayer.setPixmap(self.base_pixmap)

    def adjustScrollBar(self, scrollBar: QScrollBar, factor: float):
        scrollBar.setValue(int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep() / 2)))

    def scaleImage(self, factor: float, scale_to=False, emit_changed=True):
        if not self.gv.isVisible() or not self.proj.model_valid:
            return
        
        if scale_to:
            s_f = factor
        else:
            s_f = self.scale_factor * factor
        s_f = np.clip(s_f, CANVAS_SCALE_MIN, CANVAS_SCALE_MAX)

        scale_changed = self.scale_factor != s_f
        self.scale_factor = s_f
        self.baseLayer.setScale(self.scale_factor)
        self.rect_tool.updateScale(self.scale_factor)

        if scale_changed:
            self.adjustScrollBar(self.gv.horizontalScrollBar(), factor)
            self.adjustScrollBar(self.gv.verticalScrollBar(), factor)
            if emit_changed:
                self.scalefactor_changed.emit()
        self.setSceneRect(0, 0, self.baseLayer.sceneBoundingRect().width(), self.baseLayer.sceneBoundingRect().height())

    def onViewResized(self):
        gv_w, gv_h = self.gv.geometry().width(), self.gv.geometry().height()

        x = gv_w - self.scaleFactorLabel.width()
        y = gv_h - self.scaleFactorLabel.height()
        pos_new = (QPointF(x, y) / 2).toPoint()
        if self.scaleFactorLabel.pos() != pos_new:
            self.scaleFactorLabel.move(pos_new)

        x = gv_w - self.search_widget.width()
        pos = self.search_widget.pos()
        pos.setX(x-30)
        self.search_widget.move(pos)

        
    def onScaleFactorChanged(self):
        self.rect_tool.updateBoundingRect()
        self.scaleFactorLabel.setText(f'{self.scale_factor*100:2.0f}%')
        self.scaleFactorLabel.raise_()
        self.scaleFactorLabel.startFadeAnimation()

    # def on_selection_changed(self):
    #     if self.hasFocus() and not self.block_selection_signal:
    #         self.on_incanvas_selection_changed()

    def update_drawable_selection(self, did: str, selected: bool, ensure_visible = False):
        d = self.did2drawableitem[did]
        d.update_selection(selected)
        if ensure_visible:
            self.gv.ensureVisible(d)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == QKEY.Key_Alt:
            self.alt_pressed = False
        return super().keyReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()

        if key == QKEY.Key_Alt:
            self.alt_pressed = True

        modifiers = event.modifiers()
        if (modifiers == Qt.KeyboardModifier.AltModifier or self.alt_pressed) and \
            not key == QKEY.Key_Alt and \
                self.editing_insitem is None:
            if key in {QKEY.Key_W, QKEY.Key_A, QKEY.Key_Left, QKEY.Key_Up}:
                self.on_switch_item(-1, event)
                return
            elif key in {QKEY.Key_S, QKEY.Key_D, QKEY.Key_Right, QKEY.Key_Down}:
                self.on_switch_item(1, event)
                return

        if self.editing_insitem is not None:
            return super().keyPressEvent(event)
        elif key in QNUMERIC_KEYS:
            value = QNUMERIC_KEYS[key]
        return super().keyPressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self.mid_btn_pressed:
            new_pos = event.screenPos()
            delta_pos = new_pos - self.pan_initial_pos
            self.pan_initial_pos = new_pos
            self.hscroll_bar.setValue(int(self.hscroll_bar.value() - delta_pos.x()))
            self.vscroll_bar.setValue(int(self.vscroll_bar.value() - delta_pos.y()))
            
        elif self.creating_rect:
            self.rect_tool.setRect(QRectF(self.create_rect_origin, event.scenePos() / self.scale_factor).normalized())
        
        elif self.scale_tool_mode:
            self.scale_tool.emit(event.scenePos())
        
        elif self.rubber_band.isVisible() and self.rubber_band_origin is not None:
            self.rubber_band.setGeometry(QRectF(self.rubber_band_origin, event.scenePos()).normalized())
            sel_path = QPainterPath(self.rubber_band_origin)
            sel_path.addRect(self.rubber_band.geometry())
            self.setSelectionArea(sel_path, deviceTransform=self.gv.viewportTransform())
        
        return super().mouseMoveEvent(event)

    def selected_drawable_items(self, sort: bool = True):
        sel_items: List[DrawableItem] = []
        selitems = self.selectedItems()
        for sel in selitems:
            if isinstance(sel, DrawableItem):
                sel_items.append(sel)
        if sort:
            sel_items.sort(key = lambda x : x.idx)
        return sel_items

    def scene_cursor_pos(self):
        origin = self.gv.mapFromGlobal(QCursor.pos())
        return self.gv.mapToScene(origin)
    
    @property
    def no_instance_selected(self):
        return len(self.selected_drawable_items()) == 0

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        btn = event.button()
        left_btn_pressed = btn == Qt.MouseButton.LeftButton
        right_btn_pressed = btn == Qt.MouseButton.RightButton
        if btn == Qt.MouseButton.MiddleButton:
            self.mid_btn_pressed = True
            self.pan_initial_pos = event.screenPos()
            return
        
        if self.proj.model_valid:
            if right_btn_pressed:
                if pcfg.edit_mode == EditMode.RectInference and self.no_instance_selected:
                    self.startCreateInferenceRect(event.scenePos(), hide_control=True)
                else:
                    # rubber band selection
                    self.rubber_band_origin = event.scenePos()
                    self.rubber_band.setGeometry(QRectF(self.rubber_band_origin, self.rubber_band_origin).normalized())
                    self.rubber_band.show()
                    self.rubber_band.setZValue(1)

        return super().mousePressEvent(event)
    
    def startCreateInferenceRect(self, pos: QPointF, hide_control=False):
        pos = pos / self.scale_factor
        self.creating_rect = True
        self.create_rect_origin = pos
        self.gv.setCursor(Qt.CursorShape.CrossCursor)
        self.rect_tool.setItem(None)
        self.rect_tool.setPos(0, 0)
        self.rect_tool.setRotation(0)
        self.rect_tool.setRect(QRectF(pos, QSizeF(1, 1)))
        # if hide_control:
        #     self.rect_tool.hideControls()
        self.rect_tool.show()

    def endCreateInferenceRect(self):
        self.creating_rect = False
        self.rect_tool.showControls()
        self.gv.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        btn = event.button()

        rb_valid = self.rubber_band.geometry().height() > 0 or self.rubber_band.geometry().width() > 0
        self.hide_rubber_band()

        Qt.MouseButton.LeftButton
        if btn == Qt.MouseButton.MiddleButton:
            self.mid_btn_pressed = False
        if self.creating_rect:
            self.endCreateInferenceRect()
        elif btn == Qt.MouseButton.RightButton:
            if not self.creating_rect and  not rb_valid:
                self.context_menu_requested.emit(event.screenPos())
        elif btn == Qt.MouseButton.LeftButton:
            if self.scale_tool_mode:
                self.scale_tool_mode = False
                self.end_scale_tool.emit()
        return super().mouseReleaseEvent(event)

    def updateCanvas(self, update_preview=True):
        '''
        clear states and load current image & instance masks
        '''
        self.editing_insitem = None
        self.erase_img_key = None
        self.mid_btn_pressed = False
        self.creating_rect = False
        self.search_widget.reInitialize()

        self.clear_undostack()
        self.clearDrawableItems()
        self.setProjSaveState(False)
        self.updateLayers()
        self.rect_tool.hide()

        if self.base_pixmap is not None:
            im_rect = self.base_pixmap.rect()
            self.baseLayer.setRect(QRectF(im_rect))
            if im_rect != self.sceneRect():
                self.setSceneRect(0, 0, im_rect.width(), im_rect.height())
            self.scaleImage(1)

        for ins in self.proj.l2dmodel.valid_drawables():
            visible = self.get_tagitem(ins.tag).isVisible()
            item = DrawableItem(ins, canvas=self)
            item.selection_changed = self.drawable_selection_changed
            item.setZValue(0)
            item.setVisible(visible)
            item.setParentItem(self.baseLayer)
            self.did2drawableitem[item.drawable.did] = item

        for item in self.tag2tagitem.values():
            # self.removeItem(item)
            # item.setZValue(0.75)
            # self.addItem(item)
            # item.setParentItem(self.baseLayer)
            item.update_path(self.proj.l2dmodel)

    def hide_rubber_band(self):
        if self.rubber_band.isVisible():
            self.rubber_band.hide()
            self.rubber_band_origin = None
    
    def on_hide_canvas(self):
        self.clear_states()

    def on_activation_changed(self):
        self.clear_states()
        for textitem in self.selected_text_items():
            if textitem.isEditing():
                self.editing_insitem = textitem

    def clear_states(self):
        self.alt_pressed = False
        self.scale_tool_mode = False
        self.creating_rect = False
        self.create_rect_origin = None
        self.editing_insitem = None
        self.gv.ctrl_pressed = False

    def setProjSaveState(self, un_saved: bool):
        if un_saved == self.projstate_unsaved:
            return
        else:
            self.projstate_unsaved = un_saved
            self.proj_savestate_changed.emit(un_saved)

    def removeItem(self, item: QGraphicsItem) -> None:
        self.block_selection_signal = True
        super().removeItem(item)
        if isinstance(item, DrawableItem) and hasattr(item, 'path_item'):
            self.removeItem(item.path_item)
        self.block_selection_signal = False

    def push_undo_command(self, command: QUndoCommand, update_pushed_step=True):
        self.undo_stack.push(command)
        if update_pushed_step:
            self.pushed_undo_step += 1
            self.on_undostack_changed()

    def redo(self):
        self.pushed_undo_step += 1
        self.undo_stack.redo()
        self.on_undostack_changed()

    def undo(self):
        if self.pushed_undo_step > 0:
            self.pushed_undo_step -= 1
        self.undo_stack.undo()
        self.on_undostack_changed()

    def clear_undostack(self):
        self.undo_stack.clear()
        self.saved_undo_step = 0
        self.pushed_undo_step = 0

    def prepareClose(self):
        self.blockSignals(True)

    def clearDrawableItems(self):
        self.block_selection_signal = True
        self.clearSelection()
        self.rect_tool.setItem(None)
        for item in self.did2drawableitem.values():
            item._block_select_signal = True
            self.removeItem(item)

        self.did2drawableitem.clear()
        self.block_selection_signal = False

    def on_undostack_changed(self):
        if self.pushed_undo_step != self.saved_undo_step:
            self.setProjSaveState(True)
        else:
            self.setProjSaveState(False)

    def update_saved_undostep(self):
        self.saved_undo_step = self.pushed_undo_step
        self.projstate_unsaved = False

    def get_inference_rect(self):
        '''
        return None if not valid or not in RectInference mode
        '''
        return None
        
        # if pcfg.edit_mode != EditMode.RectInference or self.proj.img_array is None:

    def setTagsVisible(self, tags, visible):
        tags = set(tags)
        for t in tags:
            tagitem = self.get_tagitem(t)
            tagitem.setVisible(visible)
        if 'None' in tags:
            tags.add(None)
        for d in self.did2drawableitem.values():
            if d.drawable.tag in tags:
                d.setVisible(visible)