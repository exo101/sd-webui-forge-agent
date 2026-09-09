import numpy as np
import os.path as osp
import os
from pathlib import Path

from qtpy.QtCore import Qt, Signal, QUrl, QThread
from qtpy.QtGui import QImage, QPixmap
from PIL import Image

from .logger import logger as LOGGER
from utils.io_utils import imread, imwrite
from .structures import Instance
from .logger import create_error_dialog
from .message import ProgressMessageBox
from .proj import ProjSeg, load_instance_list

class ThreadBase(QThread):

    _thread_exception_type = None
    _thread_error_msg = 'Thread job failed.'
    
    early_stop_signal = Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job = None
        self._stop_flag = False
        self._job_list_on_finished = []

    def on_exec_failed(self):
        return
    
    def set_stop_flag(self):
        self._stop_flag = True
    
    def run(self):
        if self.job is not None:
            try:
                self.job()
            except Exception as e:
                self.on_exec_failed()
                create_error_dialog(e, self._thread_error_msg, self._thread_exception_type)

        self.on_thread_finished()

    def on_thread_finished(self):
        self.job = None
        if self._stop_flag:
            self._stop_flag = False
            self._job_list_on_finished.clear()
        if len(self._job_list_on_finished) > 0:
            job = self._job_list_on_finished.pop(0)
            self.job = job
            self.start()


class ProjSaveThread(ThreadBase):

    progress = Signal(int)
    export_sucess = Signal(str)

    def exportAllCutout(self, proj: ProjSeg, save_dir, export_mask=False):
        job = lambda : self._export_all_cutout(proj=proj, save_dir=save_dir, export_mask=export_mask)
        if self.isRunning():
            self._stop_flag = True
            self._job_list_on_finished.append(job)
        else:
            self.job = job
            self.start()

    def exportPageCutout(self, img: np.ndarray, instance_list: list[Instance], save_dir, save_prefix='', export_mask=False):
        job = lambda : self._export_page_cutout(img=img, instance_list=instance_list, save_dir=save_dir, save_prefix=save_prefix, export_mask=export_mask)
        if self.isRunning():
            self._stop_flag = True
            self._job_list_on_finished.append(job)
        else:
            self.job = job
            self.start()

    def _export_page_cutout(self, img: np.ndarray, instance_list: list[Instance], save_dir, save_prefix='', export_mask=False):
        os.makedirs(save_dir, exist_ok=True)
        for ins_idx, ins in enumerate(instance_list):
            if export_mask:
                cutout = ins.mask
            else:
                cutout = ins.get_cutout(img)
            if cutout is None:
                continue
            savep = osp.join(save_dir, f'{save_prefix}_{ins_idx+1}.png')
            Image.fromarray(cutout).save(savep)
            if self._stop_flag and ins_idx + 1 < len(instance_list):
                self.early_stop_signal.emit('Stopped a saving thread')
                break
        if not self._stop_flag:
            self.export_sucess.emit(self.tr('All cutouts were exported successfully to ') + save_dir)

    def _export_all_cutout(self, proj: ProjSeg, save_dir, export_mask=False):
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        for page_idx, imgname in enumerate(proj.pages):
            imgp = osp.join(proj.directory, imgname)
            insp = proj.get_instance_path(imgname)
            if osp.exists(insp):
                ins_list = load_instance_list(insp)
            else:
                continue
            img = np.array(Image.open(imgp).convert('RGB'))
            for ins_idx, ins in enumerate(ins_list):
                if export_mask:
                    cutout = ins.mask
                else:
                    cutout = ins.get_cutout(img)
                if cutout is None:
                    continue
                imname = osp.splitext(imgname)[0]
                savep = osp.join(save_dir, f'{imname}_{ins_idx+1}.png')
                Image.fromarray(cutout).save(savep)
            progress = int(round((page_idx + 1) / proj.num_pages * 100))
            self.progress.emit(progress)
            if self._stop_flag and page_idx + 1 < proj.num_pages:
                self.early_stop_signal.emit('Stopped a saving thread')
                break
        if not self._stop_flag:
            self.export_sucess.emit(self.tr('All cutouts were exported successfully to ') + save_dir)


class ImgSaveThread(ThreadBase):

    img_writed = Signal(str)
    _thread_exception_type = 'ImgSaveThread'
    _thread_error_msg = 'Failed to save image.'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.im_save_list = []

    def saveImg(self, save_path: str, img: QImage, pagename_in_proj: str = '', save_params: dict = None):
        self.im_save_list.append((save_path, img, pagename_in_proj, save_params))
        if self.job is None:
            self.job = self._save_img
            self.start()

    def _save_img(self):
        while True:
            if len(self.im_save_list) == 0:
                break
            save_path, img, pagename_in_proj, save_params = self.im_save_list.pop(0)
            if isinstance(img, QImage) or isinstance(img, QPixmap):
                if save_params is not None and save_params['ext'] in {'.jpg', '.webp'}:
                    img.save(save_path, quality=save_params['quality'])
                else:
                    img.save(save_path)
            elif isinstance(img, np.ndarray):
                imwrite(save_path, img)
            self.img_writed.emit(pagename_in_proj)
