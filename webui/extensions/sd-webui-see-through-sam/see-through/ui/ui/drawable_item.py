from typing import List
import math

import cv2
import numpy as np
from qtpy.QtWidgets import QGraphicsSceneHoverEvent, QWidget, QStyle, QLabel, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsItem, QGraphicsSceneMouseEvent, QStyleOptionGraphicsItem, QGraphicsPathItem
from qtpy.QtCore import Qt, QPoint, Signal, QRectF, QPointF
from qtpy.QtGui import QColor, QPolygon, QImage, QPainter, QPen, QPainterPath, QPixmap

from .cursor import rotateCursorList, resizeCursorList
from .misc import ndarray2pixmap
from live2d.scrap_model import Drawable, Live2DScrapModel
# from utils.visualize import get_color
from .shared import get_cls_color

# 4 points bbox to 8 points polygon
def xywh2xyxypoly(xywh, to_int=True):
    xyxypoly = np.tile(xywh[:, [0, 1]], 4)
    xyxypoly[:, [2, 4]] += xywh[:, [2]]
    xyxypoly[:, [5, 7]] += xywh[:, [3]]
    if to_int:
        xyxypoly = xyxypoly.astype(np.int64)
    return xyxypoly


CONTOUR_THICKNESS = 2
CONTOUR_SELECTED_THICKNESS = 3


TAGPATH_THICKNESS = 5


class TagPathItem(QGraphicsPixmapItem):

    def __init__(self, tag: str, parent=None):
        super().__init__(parent)
        # self.setPath()
        self.tag = tag
        self.mask = None
        self.img = None
        self._shape = None
        self._contour_list = []
        self._cnt_color = QColor(*get_cls_color(self.tag))
        self._selected = False
        self._show_mask = False
        self._show_contour = False
        self._mask_alpha = 255
        self.is_none_tag = False

    def set_selected(self, selected):
        self._selected = selected
        self.update()

    def show_mask(self, show):
        if self._show_mask != show:
            self._show_mask = show
            self.update()

    def show_contour(self, show):
        if self._show_contour != show:
            self._show_contour = show
            self.update()


    def update_path(self, lmodel: Live2DScrapModel):
        self.is_none_tag = is_none_tag = self.tag.lower() == 'none'
        if is_none_tag:
            tag = None
        else:
            tag = self.tag

        if lmodel is not None:
            img = lmodel.compose_bodypart_drawables(tag).astype(np.uint8)
            mask = lmodel.compose_bodypart_drawables(tag, mask_only=True, final_visible_mask=True).astype(np.uint8)
            x1, y1, w, h = cv2.boundingRect(cv2.findNonZero(mask))
            x2, y2 = x1 + w, y1 + h
            if h > 0 and w > 0:
                mask = mask[y1: y2, x1: x2].copy()
                self.mask = mask
                cons, _ = cv2.findContours(self.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                cnts = []
                for cnt in cons:
                    cnts.append(cv2.approxPolyDP(cnt, 3, True))
            else:
                self.mask = None
                cnts = []
            ix1, iy1, iw, ih = cv2.boundingRect(cv2.findNonZero(img[..., -1]))
            if ih > 0 and iw > 0:
                img = img[iy1: iy1 + ih, ix1: ix1 + iw].copy()
                self.img = img
            else:
                self.img = None
        else:
            self.img = None
            self.mask = None
            cnts = []
            x1 = y1 = 0

        self._contour_list = []
        shape = QPainterPath()
        for cnt in cnts:
            pnt_list = [QPoint(point[0][0], point[0][1]) for point in cnt]
            p = QPolygon(pnt_list)
            self._contour_list.append(p)
            shape.addPolygon(p.toPolygonF())
        self._shape = shape

        self.update_mask_pixmap()
        self.setPos(x1, y1)

    def opaqueArea(self):
        if self._shape is not None:
            return self._shape
        else:
            return super().opaqueArea()

    def shape(self):
        if self._shape is not None:
            return self._shape
        else:
            return super().shape()
        
    def set_msk_opacity(self, value):
        self._mask_alpha = int(round(value / 100 * 255))
        self.update_mask_pixmap()

    def update_mask_pixmap(self):
        if self.mask is not None:
            h, w = self.mask.shape[:2]
            color = np.array(get_cls_color(self.tag), dtype=np.uint8).reshape((1, 1, 3))
            pixmap = np.full((h, w, 3), dtype=np.uint8, fill_value=color)
            pixmap = np.concatenate([pixmap, self.mask[..., None] * self._mask_alpha], axis=2)
            pixmap = ndarray2pixmap(pixmap)
            self.setPixmap(pixmap)
        else:
            pixmap = QPixmap(1, 1)
            pixmap.fill(Qt.GlobalColor.transparent)
            self.setPixmap(pixmap)
        
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        if self.is_none_tag:
            return
        painter.save()
        if self._show_mask:
            super().paint(painter, option, widget)
        if self._contour_list is not None and self._show_contour:
            clist = self._contour_list
            if clist is not None:
                if self._selected:
                    pass
                else:
                    pen = QPen(self._cnt_color, TAGPATH_THICKNESS, Qt.PenStyle.SolidLine)
                pen.setCosmetic(True)
                painter.setPen(pen)
                for polygon in clist:
                    painter.drawPolyline(polygon)
        painter.restore()


class DrawablePathItem(QGraphicsRectItem):

    def __init__(self, _shape, _contour_list, xywh, parent=None):
        super().__init__(parent=parent)
        self._selected = False
        self._hovered = False
        self._contour_list = _contour_list
        self._shape = _shape
        self._cnt_color = get_cls_color(None)
        self.setRect(0, 0, xywh[2], xywh[3])
        self.setZValue(1.)
        self._shape = None

    def select_mode(self):
        return self._selected

    def hover_mode(self):
        return self._hovered
    
    def set_select_mode(self, selected):
        if self._selected != selected:
            self._selected = selected
            if self.should_hide():
                self.setVisible(False)
            else:
                self.setVisible(True)

    def set_hover_mode(self, hovered):
        if self._hovered != hovered:
            self._hovered = hovered
            if self.should_hide():
                self.setVisible(False)
            else:
                self.setVisible(True)

    def should_hide(self):
        return not self._selected and not self._hovered

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        painter.save()
        if self.select_mode() or self.hover_mode():
            clist = self._contour_list
            if clist is not None:
                painter.setCompositionMode(QPainter.CompositionMode.RasterOp_NotDestination)
                pen = QPen(self._cnt_color, CONTOUR_SELECTED_THICKNESS, Qt.PenStyle.SolidLine) if self.select_mode() \
                    else QPen(self._cnt_color, CONTOUR_THICKNESS, Qt.PenStyle.DashLine)
                pen.setCosmetic(True)
                painter.setPen(pen)
                for polygon in clist:
                    painter.drawPolyline(polygon)
        painter.restore()

    def shape(self):
        cnt_list = self._contour_list
        if cnt_list is None or self._shape is None:
            return super().shape()
        return self._shape


class DrawableItem(QGraphicsPixmapItem):

    moved = Signal()

    def __init__(self, drawable: Drawable, canvas, parent=None, opacity=1, draw_contours=True) -> None:
        super().__init__(parent)
        self._draw_contours = draw_contours
        self._contour_list = None
        self._shape = None
        self._cnt_color = None
        self._hover_entered = False
        self.old_pos = None
        self.canvas = canvas
        self.setAcceptHoverEvents(True)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        # self.setCacheMode(QGraphicsItem.CacheMode.ItemCoordinateCache)
        self.setOpacity(opacity)
        self.setDrawable(drawable)
        self.contour_list() # init first
        self.path_item = DrawablePathItem(self._shape, self._contour_list, self.drawable.xywh)
        self.path_item.setPos(self.pos())
        self.path_item._cnt_color = self._cnt_color
        self._block_select_signal = False
        self.selection_changed = None

    def setParentItem(self, parent):
        super().setParentItem(parent)
        self.path_item.setParentItem(parent)

    def shape(self):
        cnt_list = self.contour_list()
        if cnt_list is None:
            return super().shape()
        if self._shape is not None:
            return self._shape
    
    @property
    def idx(self):
        return self.drawable.idx

    @idx.setter
    def idx(self, value: float):
        self.drawable.idx = value

    def setDrawable(self, drawable: Drawable):
        self.drawable = drawable
        x, y, w, h = drawable.xyxy
        w -= x
        h -= y

        _img = ndarray2pixmap(drawable.img, return_qimg=False, img_format=QImage.Format.Format_RGBA8888_Premultiplied)
        self.setPixmap(_img)
        self.on_update_drawable_tag()
        self._contour_list = None
        self._shape = None
        # self.setRect(0, 0, w, h)
        self.setPos(x, y)

    def set_drawable_tag(self, tag: str):
        self.drawable.set_tag(tag)
        self.on_update_drawable_tag()

    def on_update_drawable_tag(self):
        self._cnt_color = QColor(*get_cls_color(self.drawable.tag))
        if hasattr(self, 'path_item'):
            self.path_item._cnt_color = self._cnt_color
        t = self.drawable.tag
        t = 'None' if t is None else t
        self.setToolTip(t + '-' + self.drawable.did)

    def contour_list(self) -> List[QPolygon]:
        if self._contour_list is None:
            contours = self.drawable.get_contours()
            if contours is None:
                self._contour_list = None
                self._shape = None
                return None
            self._contour_list = []
            self._shape = QPainterPath()
            for cnt in contours:
                pnt_list = [QPoint(point[0][0], point[0][1]) for point in cnt]
                p = QPolygon(pnt_list)
                self._contour_list.append(p)
                self._shape.addPolygon(p.toPolygonF())
        return self._contour_list

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        if self.isSelected():
            option.state = option.state & ~QStyle.StateFlag.State_Selected
        super().paint(painter, option, widget)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            selected = self.isSelected()
            self.path_item.set_select_mode(selected)
            if not self._block_select_signal and self.selection_changed is not None:
                self.selection_changed.emit(self.drawable.did, selected)
        elif change == QGraphicsItem.GraphicsItemChange.ItemVisibleHasChanged:
            if not self.isVisible() and self._hover_entered:
                self._hover_entered = False
                self.path_item.set_hover_mode(False)
                self.update()
        return super().itemChange(change, value)
    
    def update_selection(self, selected: bool):
        self._block_select_signal = True
        self.setSelected(selected)
        self._block_select_signal = False

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent | None) -> None:
        self._hover_entered = False
        self.path_item.set_hover_mode(False)
        return super().hoverLeaveEvent(event)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self._hover_entered = True
        self.path_item.set_hover_mode(True)
        # self.setZValue(1)
        return super().hoverEnterEvent(event)
    
    # def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
    #     super().mouseMoveEvent(event)  
    #     if self.textInteractionFlags() != Qt.TextInteractionFlag.TextEditorInteraction:
    #         self.moving.emit(self)

    # # QT 5.15.x causing segmentation fault 
    # def contextMenuEvent(self, event):
    #     return super().contextMenuEvent(event)


CBEDGE_WIDTH = 30

class ControlBlockItem(QGraphicsRectItem):
    DRAG_NONE = 0
    DRAG_RESHAPE = 1
    DRAG_ROTATE = 2
    CURSOR_IDX = -1
    def __init__(self, parent, idx: int):
        super().__init__(parent)
        self.idx = idx
        self.ctrl: SceneRectTool = parent
        self.edge_width = 0
        self.drag_mode = self.DRAG_NONE
        self.setAcceptHoverEvents(True)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.updateEdgeWidth(CBEDGE_WIDTH)
    
    def updateEdgeWidth(self, edge_width: float):
        self.edge_width = edge_width
        self.visible_len = self.edge_width // 2
        self.pen_width = edge_width / CBEDGE_WIDTH * 2 
        offset = self.edge_width // 4 + self.pen_width / 2
        self.visible_rect = QRectF(offset, offset, self.visible_len, self.visible_len)
        self.setRect(0, 0, self.edge_width, self.edge_width)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        rect = QRectF(self.visible_rect)
        rect.setTopLeft(self.boundingRect().topLeft()+rect.topLeft())
        painter.setPen(QPen(QColor(75, 75, 75), self.pen_width, Qt.PenStyle.SolidLine, Qt.SquareCap))
        painter.fillRect(rect, QColor(200, 200, 200, 125))
        painter.drawRect(rect)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:     
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.drag_mode = self.DRAG_RESHAPE
        return super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self.drag_mode = self.DRAG_NONE
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.CURSOR_IDX = -1
        return super().hoverLeaveEvent(event)

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        angle = self.ctrl.rotation() + 45 * self.idx
        idx = self.get_angle_idx(angle)
        if self.visible_rect.contains(event.pos()):
            self.setCursor(resizeCursorList[idx % 4])
        else:
            # self.setCursor(rotateCursorList[idx])
            pass
        self.CURSOR_IDX = idx
        return super().hoverMoveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.ctrl.ctrlblockPressed()
        if event.button() == Qt.MouseButton.LeftButton:
            if self.visible_rect.contains(event.pos()):
                self.ctrl.reshaping = True
                self.drag_mode = self.DRAG_RESHAPE
                self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
            else:
                return
                self.drag_mode = self.DRAG_ROTATE
                self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                preview = self.ctrl.previewPixmap

                preview.setPixmap(blk_item.toPixmap().copy(blk_item.unpadRect(blk_item.boundingRect()).toRect()))
                preview.setOpacity(0.7)
                preview.setVisible(True)
                rotate_vec = event.scenePos() - self.ctrl.sceneBoundingRect().center()
                self.updateAngleLabelPos()
                rotation = np.rad2deg(math.atan2(rotate_vec.y(), rotate_vec.x()))
                self.rotate_start = - rotation + self.ctrl.rotation() 
        event.accept()

    def updateAngleLabelPos(self):
        angleLabel = self.ctrl.angleLabel
        sp = self.scenePos()
        gv = angleLabel.parent()
        pos = gv.mapFromScene(sp)
        x = max(min(pos.x(), gv.width() - angleLabel.width()), 0)
        y = max(min(pos.y(), gv.height() - angleLabel.height()), 0)
        angleLabel.move(QPoint(x, y))
        angleLabel.setText("{:.1f}°".format(self.ctrl.rotation()))
        if not angleLabel.isVisible():
            angleLabel.setVisible(True)
            angleLabel.raise_()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        super().mouseMoveEvent(event)

        if self.drag_mode == self.DRAG_RESHAPE:    
            block_group = self.ctrl.ctrlblock_group
            crect = self.ctrl.rect()
            pos_x, pos_y = 0, 0
            opposite_block = block_group[(self.idx + 4) % 8 ]
            oppo_pos = opposite_block.pos()
            if self.idx % 2 == 0:
                if self.idx == 0:
                    pos_x = min(self.pos().x(), oppo_pos.x())
                    pos_y = min(self.pos().y(), oppo_pos.y())
                    crect.setX(pos_x+self.visible_len)
                    crect.setY(pos_y+self.visible_len)
                elif self.idx == 2:
                    pos_x = max(self.pos().x(), oppo_pos.x())
                    pos_y = min(self.pos().y(), oppo_pos.y())
                    crect.setWidth(pos_x-oppo_pos.x())
                    crect.setY(pos_y+self.visible_len)
                elif self.idx == 4:
                    pos_x = max(self.pos().x(), oppo_pos.x())
                    pos_y = max(self.pos().y(), oppo_pos.y())
                    crect.setWidth(pos_x-oppo_pos.x())
                    crect.setHeight(pos_y-oppo_pos.y())
                else:   # idx == 6
                    pos_x = min(self.pos().x(), oppo_pos.x())
                    pos_y = max(self.pos().y(), oppo_pos.y())
                    crect.setX(pos_x+self.visible_len)
                    crect.setHeight(pos_y-oppo_pos.y())
            else:
                if self.idx == 1:
                    pos_y = min(self.pos().y(), oppo_pos.y())
                    crect.setY(pos_y+self.visible_len)
                elif self.idx == 3:
                    pos_x = max(self.pos().x(), oppo_pos.x())
                    crect.setWidth(pos_x-oppo_pos.x())
                elif self.idx == 5:
                    pos_y = max(self.pos().y(), oppo_pos.y())
                    crect.setHeight(pos_y-oppo_pos.y())
                else:   # idx == 7
                    pos_x = min(self.pos().x(), oppo_pos.x())
                    crect.setX(pos_x+self.visible_len)
            
            self.ctrl.setRect(crect)
            # scale = self.ctrl.current_scale
            # new_center = self.ctrl.sceneBoundingRect().center()
            # new_xy = QPointF(new_center.x() / scale - crect.width() / 2, new_center.y() / scale - crect.height() / 2)
            # rect = QRectF(new_xy.x(), new_xy.y(), crect.width(), crect.height())
            # self.ctrl._abs_rect = rect

        elif self.drag_mode == self.DRAG_ROTATE:   # rotating
            rotate_vec = event.scenePos() - self.ctrl.sceneBoundingRect().center()
            rotation = np.rad2deg(math.atan2(rotate_vec.y(), rotate_vec.x()))
            self.ctrl.setAngle((rotation+self.rotate_start))
            # angle = self.ctrl.rotation()
            angle = self.ctrl.rotation() + 45 * self.idx
            idx = self.get_angle_idx(angle)
            if self.CURSOR_IDX != idx:
                self.setCursor(rotateCursorList[idx])
                self.CURSOR_IDX = idx
            self.updateAngleLabelPos()

    def get_angle_idx(self, angle) -> int:
        idx = int((angle + 22.5) % 360 / 45)
        return idx
    
    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.ctrl.reshaping = False
            # if self.drag_mode == self.DRAG_RESHAPE:
            #     self.ctrl.blk_item.endReshape()
            # if self.drag_mode == self.DRAG_ROTATE:
            #     self.ctrl.blk_item.rotated.emit(self.ctrl.rotation())
            self.drag_mode = self.DRAG_NONE
            
            self.ctrl.previewPixmap.setVisible(False)
            self.ctrl.angleLabel.setVisible(False)
            self.ctrl.updateBoundingRect()
            return super().mouseReleaseEvent(event)



class SceneRectTool(QGraphicsRectItem):
    ins_item: DrawableItem = None
    reshaping: bool = False
    
    def __init__(self, parent) -> None:
        super().__init__()
        self._br = QRectF(0, 0, 1, 1)
        self.gv = parent

        self.ctrlblock_group = [
            ControlBlockItem(self, idx) for idx in range(8)
        ]
        
        self.previewPixmap = QGraphicsPixmapItem(self)
        self.previewPixmap.setVisible(False)
        pen = QPen(QColor(69, 71, 87), 2, Qt.PenStyle.SolidLine)
        pen.setDashPattern([7, 14])
        self.setPen(pen)
        self.setVisible(False)

        self.current_scale = 1.
        self.need_rescale = False
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self._abs_rect = None

        self.angleLabel = QLabel(parent)
        self.angleLabel.setText("{:.1f}°".format(self.rotation()))
        self.angleLabel.setObjectName("angleLabel")
        self.angleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.angleLabel.setHidden(True)

        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

    def ctrlblockPressed(self):
        self.scene().clearSelection()

    def setItem(self, item: DrawableItem):
        if self.ins_item == item and self.isVisible():
            return
        if self.ins_item is not None:
            self.ins_item.under_ctrl = False
            self.ins_item.update()
            
        self.ins_item = item
        if item is None:
            self.hide()
            return
        item.under_ctrl = True
        item.update()
        self.updateBoundingRect()
        self.show()

    def updateBoundingRect(self):
        if self.ins_item is not None:
            abr = self.ins_item.boundingRect()
        else:
            abr = self.sceneBoundingRect()
            scale = self.current_scale
            abr = QRectF(abr.x() / scale, abr.y() / scale, abr.width() / scale, abr.height() / scale)
        br = QRectF(0, 0, abr.width(), abr.height())
        self.setRect(br)
        if self.ins_item is not None:
            self.ins_item.setCenterTransform()
            self.setTransformOriginPoint(self.ins_item.transformOriginPoint())
        self.setPos(abr.x(), abr.y())

    def setRect(self, *args): 
        super().setRect(*args)
        self.updateControlBlocks()

    def updateControlBlocks(self):
        b_rect = self.rect()
        b_rect = [b_rect.x(), b_rect.y(), b_rect.width(), b_rect.height()]
        corner_pnts = xywh2xyxypoly(np.array([b_rect])).reshape(-1, 2)
        edge_pnts = (corner_pnts[[1, 2, 3, 0]] + corner_pnts) / 2
        pnts = [edge_pnts, corner_pnts]
        for ii, ctrlblock in enumerate(self.ctrlblock_group):
            is_corner = not ii % 2
            idx = ii // 2
            pos = pnts[is_corner][idx] -0.5 * ctrlblock.edge_width
            ctrlblock.setPos(pos[0], pos[1])

    def paint(self, painter: QPainter, option: 'QStyleOptionGraphicsItem', widget = ...) -> None:
        # https://stackoverflow.com/a/10986248
        painter.setCompositionMode(QPainter.CompositionMode.RasterOp_NotDestination)
        option.state = option.state & ~QStyle.StateFlag.State_Selected
        super().paint(painter, option, widget)

    def updateScale(self, scale: float):
        if not self.isVisible():
            if scale != self.current_scale:
                self.need_rescale = True
                self.current_scale = scale
            return

        self.current_scale = scale
        scale = 1 / scale
        pen = self.pen()
        pen.setWidthF(2 * scale)
        self.setPen(pen)
        for ctrl in self.ctrlblock_group:
            ctrl.updateEdgeWidth(CBEDGE_WIDTH * scale)

    def show(self) -> None:
        super().show()
        if self.need_rescale:
            self.updateScale(self.current_scale)
            self.need_rescale = False
        self.setZValue(1)

    def hideControls(self):
        for ctrl in self.ctrlblock_group:
            ctrl.hide()

    def showControls(self):
        for ctrl in self.ctrlblock_group:
            ctrl.show()

