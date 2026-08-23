import os, json
import numpy as np
import os.path as osp
from typing import Tuple, Union, List, Dict

from .logger import logger as LOGGER
from utils.io_utils import find_all_imgs, load_image, NumpyEncoder, find_all_files_with_name
from .structures import Instance, load_instance_list, save_instance_list
from .misc import ImgnameNotInProjectException, ProjectLoadFailureException, ProjectDirNotExistException, ProjectNotSupportedException
from .ui_config import pcfg

from live2d.scrap_model import Live2DScrapModel, fix_drawable_rgbs


class ProjSeg:
    def __init__(self, directory: str = None):
        self.type = 'live2d_parsing'
        self.directory: str = None
        self.pages: Dict = {}
        self._pagename2idx = {}
        self._idx2pagename = {}

        self.not_found_pages: Dict = {}
        self.proj_path: str = None

        self.current_model: str = None
        self.l2dmodel: Live2DScrapModel = None
        self.is_current_page_valid = True
        self.is_incomplete = False


    def idx2pagename(self, idx: int) -> str:
        return self._idx2pagename[idx]

    def pagename2idx(self, pagename: str) -> int:
        if pagename in self.pages:
            return self._pagename2idx[pagename]
        return -1

    def proj_name(self) -> str:
        return self.type+'_'+osp.basename(self.directory)

    def load(self, p: str) -> bool:
        self.directory = osp.dirname(p)
        if p.lower().endswith('.json'):
            self.load_from_dict(p)
            self.proj_path = p
        else:
            self.proj_path = osp.join(self.directory, osp.splitext(osp.basename(p))[0] + '.json')
            self.load_from_txt(p)
            
        new_proj = False
        # if not osp.exists(self.proj_path):
        #     new_proj = True
        #     self.new_project()
        # else:
        #     try:
        #         with open(self.proj_path, 'r', encoding='utf8') as f:
        #             proj_dict = json.loads(f.read())
        #     except Exception as e:
        #         raise ProjectLoadFailureException(e)
        #     self.load_from_dict(proj_dict)

        return new_proj

    def load_from_dict(self, proj_dict: dict):
        if isinstance(proj_dict, str):
            with open(proj_dict, 'r', encoding='utf8') as f:
                proj_dict = json.loads(f.read())
        self.set_current_page(None)
        try:
            self.pages = {}
            self._pagename2idx = {}
            self._idx2pagename = {}
            self.not_found_pages = {}
            page_dict = proj_dict['pages']
            for ii, imname in enumerate(page_dict.keys()):
                self._pagename2idx[imname] = ii
                self._idx2pagename[ii] = imname
            self.pages.update(page_dict)
        except Exception as e:
            raise ProjectNotSupportedException(e)
        set_img_failed = False
        if 'current_model' in proj_dict:
            current_model = proj_dict['current_model']
            try:
                self.set_current_page(current_model)
            except ImgnameNotInProjectException:
                set_img_failed = True
        else:
            set_img_failed = True
            # LOGGER.warning(f'{current_model} not found.')
        if set_img_failed:
            if len(self.pages) > 0:
                self.set_current_page_byidx(0)

    def load_from_txt(self, txt_path: str):
        from utils.io_utils import load_exec_list
        flist = load_exec_list(txt_path)
        pages = {}
        self.directory = osp.dirname(txt_path)
        for f in flist:
            if osp.isfile(f):
                f = osp.dirname(f)
            srcd = 'workspace/datasets/' + osp.basename(self.directory) + '/'
            if f.startswith(srcd):
                f = f[len(srcd):]
            # f = osp.relpath(f, self.directory)
            pages[f] = []
        self.load_from_dict({'pages': pages, 'directory': self.directory})

    def set_current_page(self, imgname: str):
        if imgname is not None:
            if imgname not in self.pages:
                raise ImgnameNotInProjectException
            self.current_model = imgname
            if self.l2dmodel is not None:
                del self.l2dmodel
            self.l2dmodel = Live2DScrapModel(self.current_model_path(), pad_to_square=False, seg_type=pcfg.seg_type)
            self.l2dmodel.init_drawable_visible_map()
            parsing_src = pcfg.parsing_src
            valid_parsing_lst = self.l2dmodel.valid_parsing_list()
            if parsing_src not in valid_parsing_lst:
                parsing_src = None
                # return
            self.l2dmodel.load_body_parsing(parsing_src)

            metadata = {}
            if self.l2dmodel._body_parsing is not None:
                metadata = self.l2dmodel._body_parsing.get('metadata', {})
            if metadata is None:
                metadata = {}

            cleaned = metadata.get('cleaned', False)
            self.is_current_page_valid = metadata.get('is_valid', False)
            self.is_incomplete = metadata.get('is_incomplete', False)

            if parsing_src is not None and 'bodyparsingv3' in parsing_src and not metadata.get('v3_cleaned', False):
                for d in self.l2dmodel.drawables:
                    if d.did is None:
                        continue
                    if 'eye_ball' in d.did.lower():
                        d.set_tag('irides')
                    if 'brow' in d.did.lower():
                        d.set_tag('eyebrow')
                    if 'eyewhite' in d.did.lower():
                        d.set_tag('eyebg')
                    if 'mouth' in d.did.lower():
                        d.set_tag('mouth')
                    if 'eyelash' in d.did.lower():
                        d.set_tag('eyelash')
                    if 'tooth' in d.did.lower():
                        d.set_tag('mouth')
                metadata['v3_cleaned'] = True

            if not cleaned:
                did_contain_arms = False
                for d in self.l2dmodel.drawables:
                    if d.did is None:
                        continue
                    if 'arm' in d.did.lower():
                        did_contain_arms = True

                for d in self.l2dmodel.drawables:
                    if d.did is None:
                        continue
                    did_lower = d.did.lower()
                    if d.tag == 'objects':
                        continue
                    if d.tag is None:
                        if 'hair' in did_lower:
                            d.set_tag('hair')
                        elif 'arm' in did_lower:
                            d.set_tag('handwear')
                        elif 'mouth' in did_lower:
                            d.set_tag('mouth')
                        elif 'body' in did_lower:
                            if 'body2' in did_lower:
                                d.set_tag('bottomwear')
                            else:
                                d.set_tag('topwear')
                        elif 'face' in did_lower:
                            d.set_tag('face')
                        elif 'ear' in did_lower:
                            d.set_tag('ears')
                        elif 'eye' in did_lower:
                            d.set_tag('eyes')
                        elif 'leg' in did_lower:
                            d.set_tag('legwear')
                    elif d.tag == 'hair':
                        if 'face' in did_lower:
                            d.set_tag('face')
                        elif 'arm' in did_lower:
                            d.set_tag('handwear')
                        elif 'body' in did_lower and 'hair' not in did_lower:
                            d.set_tag('topwear')
                    elif d.tag == 'handwear':
                        if did_contain_arms:
                            if 'body' in did_lower and 'arm' not in did_lower:
                                if 'body2' in did_lower:
                                    d.set_tag('bottomwear')
                                else:
                                    d.set_tag('topwear')
                        if 'hair' in did_lower:
                            d.set_tag('hair')
                    elif d.tag == 'topwear':
                        if 'hair' in did_lower:
                            d.set_tag('headwear')
                        elif 'arm' in did_lower:
                            d.set_tag('handwear')
                    elif d.tag == 'bottomwear':
                        if 'hair' in did_lower:
                            d.set_tag('headwear')
                    else:
                        if 'arm' in did_lower:
                            d.set_tag('handwear')
                    if d.tag in {'legwear', 'topwear', 'bottomwear', 'hair', 'face', None} and 'ear' in did_lower:
                        d.set_tag('ears')
                    elif d.tag in {'legwear', 'topwear', 'bottomwear', 'hair', 'face', None} and 'neck' in did_lower:
                        d.set_tag('neck')
                    elif d.tag in {'legwear', 'topwear', 'bottomwear', 'hair', None} and ('hand' in did_lower or 'arm' in did_lower):
                        d.set_tag('handwear')
                    elif d.tag in {'legwear', 'topwear', 'bottomwear', 'hair', None} and 'eye' in did_lower:
                        d.set_tag('eyes')
                    elif d.tag in {'legwear', 'topwear', 'bottomwear', 'hair', None} and 'mouth' in did_lower:
                        d.set_tag('mouth')
                    elif d.tag in {'legwear', 'topwear', 'bottomwear', 'hair', 'face', None} and 'nose' in did_lower:
                        d.set_tag('nose')
                    else:
                        model_base_dir = osp.basename(osp.dirname(osp.dirname(self.current_model)))
                        if model_base_dir in ['KONOTSUBA_jxl_crop', 'sekai_jxl_crop']:
                            if d.tag in {'legwear', 'topwear', 'bottomwear', 'hair', 'face', None} and 'eye' in did_lower:
                                d.set_tag('eyes')
                            elif d.tag in {'legwear', 'topwear', 'bottomwear', 'hair', 'face', None} and 'mouth' in did_lower:
                                d.set_tag('mouth')
                        
        else:
            self.current_model = None
            del self.l2dmodel
            self.l2dmodel = None

    def current_model_path(self):
        return osp.join(self.directory, self.current_model)

    def save_current_page(self):
        metadata = {}
        if self.l2dmodel._body_parsing is not None:
            metadata = self.l2dmodel._body_parsing.get('metadata', {})
        if metadata is None:
            metadata = {}
        metadata['is_valid'] = self.is_current_page_valid
        metadata['cleaned'] = True
        metadata['is_incomplete'] = self.is_incomplete
        self.l2dmodel.save_tag_parsing(pcfg.seg_type, pcfg.parsing_src, metadata=metadata)
        LOGGER.debug(f'saved page {self.current_model} - {pcfg.parsing_src}')


    def set_current_page_byidx(self, idx: int):
        num_pages = self.num_pages
        if idx < 0:
            idx = idx + self.num_pages
        if idx < 0 or idx > num_pages - 1:
            self.set_current_page(None)
        else:
            self.set_current_page(self.idx2pagename(idx))

    def get_page_byidx(self, idx: int):
        return self.pages[self.idx2pagename(idx)]

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def current_idx(self) -> int:
        return self.pagename2idx(self.current_model)

    def new_project(self):
        if not osp.exists(self.directory):
            raise ProjectDirNotExistException
        self.set_current_page(None)
        imglist = find_all_imgs(self.directory, abs_path=False, sort=True)
        self.pages = {}
        self._pagename2idx = {}
        self._idx2pagename = {}
        for ii, imgname in enumerate(imglist):
            self.pages[imgname] = []
            self._pagename2idx[imgname] = ii
            self._idx2pagename[ii] = imgname
        self.set_current_page_byidx(0)
        self.save()
        
    def save(self):
        if not osp.exists(self.directory):
            raise ProjectDirNotExistException
        with open(self.proj_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), ensure_ascii=False, cls=NumpyEncoder))
            LOGGER.debug(f'project saved to {self.proj_path}')

    def to_dict(self) -> Dict:
        pages = self.pages.copy()
        pages.update(self.not_found_pages)
        return {
            'directory': self.directory,
            'pages': pages,
            'current_model': self.current_model,
        }

    def current_model_path(self) -> str:
        if self.current_model is None:
            return None
        return osp.join(self.directory, self.current_model)

    @property
    def is_empty(self):
        return len(self.pages) == 0

    @property
    def is_all_pages_no_text(self):
        return all([len(blklist) == 0 for blklist in self.pages.values()])

    @property
    def model_valid(self):
        return self.l2dmodel is not None
    
    def get_did_tag_pairs(self, seg_type='body_part_tag'):
        dids, tag_list = [], []
        if self.model_valid:
            for d in self.l2dmodel.valid_drawables():
                dids.append(d.did)
                tag_list.append(getattr(d, seg_type))
        return dids, tag_list

    def set_next_img(self):
        if self.current_model is not None:
            next_idx = (self.current_idx + 1) % self.num_pages
            self.set_current_page(self.idx2pagename(next_idx))

    def set_prev_img(self):
        if self.current_model is not None:
            next_idx = (self.current_idx - 1 + self.num_pages) % self.num_pages
            self.set_current_page(self.idx2pagename(next_idx))

    def merge_from_proj_dict(self, tgt_dict: Dict) -> Dict:
        if self.pages is None:
            self.pages = {}
        src_dict = self.pages if self.pages is not None else {}
        key_lst = list(dict.fromkeys(list(src_dict.keys()) + list(tgt_dict.keys())))
        key_lst.sort()
        rst_dict = {}
        pagename2idx = {}
        idx2pagename = {}
        page_counter = 0
        for key in key_lst:
            if key in src_dict and not key in tgt_dict:
                rst_dict[key] = src_dict[key]
            else:
                rst_dict[key] = tgt_dict[key]
            pagename2idx[key] = page_counter
            idx2pagename[page_counter] = key
            page_counter += 1
        self.pages.clear()
        self.pages.update(rst_dict)
        self._pagename2idx = pagename2idx
        self._idx2pagename = idx2pagename        