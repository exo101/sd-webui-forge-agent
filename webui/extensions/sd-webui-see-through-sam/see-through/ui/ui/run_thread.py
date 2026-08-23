import os.path as osp

import cv2
import numpy as np
from qtpy.QtCore import QThread, Signal
from .proj import ProjSeg
from .ui_config import pcfg, EditMode, SegModel
from .structures import Instance, save_instance_list
from .logger import logger as LOGGER

from .io_thread import ThreadBase


class SegmentationThread(ThreadBase):

    page_finished = Signal(int)
    manual_inference_finished = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self._stop_flag = False

    def runSegmentation(self, proj: ProjSeg, inference_rect=None, inference_points=None):
        if self.job is None:
            if inference_rect is None:
                self.job = lambda : self._run_batch_seg(proj=proj)
            else:
                self.job = lambda: self._run_manual_inference(proj=proj, inference_rect=inference_rect, inference_points=inference_points)
            self.start()

    def _run_manual_inference(self, proj: ProjSeg, inference_rect: list, inference_points: list = None):

        if pcfg.segmentation_model == SegModel.CartoonSeg:
            from annotators.animeinsseg.instance_segmentation import apply_instance_segmentation
            refine_method = 'refinenet_isnet' if pcfg.segmentation_refine else None
            instances = apply_instance_segmentation(
                proj.img_array,
                refine_method=refine_method,
                instance_thresh=pcfg.segmentation_conf_thr,
            )
            if instances.is_empty:
                masks, bboxes, scores = [], [], []
            else:
                masks = instances.masks
                bboxes = instances.bboxes
                scores = instances.scores
        elif pcfg.segmentation_model == SegModel.SAM:
            from annotators.lang_sam.models.sam import SAM
            sam = SAM()
            sam.build_model('sam2.1_hiera_large', device=pcfg.segmentation_device)
            xyxy = np.array(inference_rect)
            pred_masks, scores, _logits = sam.predict(proj.img_array, xyxy)
            # Derive xywh bboxes from each predicted mask
            masks = []
            bboxes = []
            for mi in range(len(pred_masks)):
                m = pred_masks[mi]
                ys, xs = np.where(m)
                if len(xs) == 0:
                    continue
                x, y = int(xs.min()), int(ys.min())
                w, h = int(xs.max()) - x, int(ys.max()) - y
                masks.append(m)
                bboxes.append(np.array([x, y, w, h]))
            scores = scores[:len(masks)]
        num_new_ins = len(masks)
        num_exists_ins = len(proj.current_instance_list)
        if num_new_ins == 0:
            self.manual_inference_finished.emit(0)
            return
        instance_list = []
        for ii in range(len(masks)):
            mask = masks[ii]
            x, y, h, w = bboxes[ii]
            if mask is None or h < 1 or w < 1:
                # invalid masks (h < 1 or w < 1) should be tackled for multi prompts
                LOGGER.info('Skip invalid mask')
                continue
            instance_list.append(Instance(mask=mask, bbox=bboxes[ii], score=scores[ii]))
        for ii, instance in enumerate(instance_list):
            instance.idx = ii + num_exists_ins
        num_new_ins = len(instance_list)
        proj.current_instance_list += instance_list
        self.manual_inference_finished.emit(num_new_ins)

    def _run_batch_seg(self, proj: ProjSeg):

        for page_index, imgname in enumerate(proj.pages):
            imgp = osp.join(proj.directory, imgname)
            img = cv2.cvtColor(cv2.imread(imgp), cv2.COLOR_BGR2RGB)
            from annotators.animeinsseg.instance_segmentation import apply_instance_segmentation
            refine_method = 'refinenet_isnet' if pcfg.segmentation_refine else None
            instances = apply_instance_segmentation(
                img,
                refine_method=refine_method,
                instance_thresh=pcfg.segmentation_conf_thr,
            )
            if instances.is_empty:
                masks, bboxes, scores = [], [], []
            else:
                masks = instances.masks
                bboxes = instances.bboxes
                scores = instances.scores
            instance_list = []
            for ii in range(len(masks)):
                instance_list.append(Instance(mask=masks[ii], bbox=bboxes[ii], score=scores[ii], idx=ii))
            # instance_list = sort_instance_list(instance_list, right_to_left=False)
            for ii, instance in enumerate(instance_list):
                instance.idx = ii
            ins_path = proj.get_instance_path(imgname)
            save_instance_list(instance_list, ins_path)
            self.page_finished.emit(page_index)

            if self._stop_flag and page_index + 1 < proj.num_pages:
                self.early_stop_signal.emit('Stopped a segmentation thread')
                break
