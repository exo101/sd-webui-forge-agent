from typing import Tuple, List, ClassVar, Union, Any, Dict
from dataclasses import dataclass, field, is_dataclass
import copy
import json

import cv2
import numpy as np
import pycocotools.mask as maskUtils

from utils.io_utils import NumpyEncoder, json2dict, dict2json
from utils.cv import mask2rle

# decorator to wrap original __init__
# https://www.geeksforgeeks.org/creating-nested-dataclass-objects-in-python/
def nested_dataclass(*args, **dataclass_kwargs):
    '''
    nested dataclass support \n
    also ignore extra arguments 
    '''
    def wrapper(check_class):
          
        # passing class to investigate
        check_class = dataclass(check_class, **dataclass_kwargs)
        o_init = check_class.__init__
          
        def __init__(self, *args, **kwargs):
              
            store_deprecated = 'deprecated_attributes' in self.__annotations__
            deprecated = {}
            for name in list(kwargs.keys()):
                if name not in self.__annotations__:
                    # print(f'warning: type object \'{self.__class__.__name__}\' has no attribute {name}, might be loading from an older config')
                    val = kwargs.pop(name)
                    if store_deprecated:
                        deprecated[name] = val
                    continue
                value = kwargs[name]
                # getting field type
                ft = check_class.__annotations__.get(name, None)
                  
                if is_dataclass(ft) and isinstance(value, dict):
                    obj = ft(**value)
                    kwargs[name]= obj

            if len(deprecated) > 0:
                kwargs['deprecated_attributes'] = deprecated        
            
            o_init(self, *args, **kwargs)
        check_class.__init__=__init__
          
        return check_class
      
    return wrapper(args[0]) if args else wrapper

@dataclass
class Config:
    
    def update(self, key: str, value):
        assert key in self.__annotations__, f'type object \'{self.__class__.__name__}\' has no attribute {key}'
        self.__setattr__(key, value)

    @classmethod
    def annotations_set(cls):
        return set(list(cls.__annotations__))
    
    def __getitem__(self, key: str):
        assert key in self.__annotations__, f'type object \'{self.__class__.__name__}\' has no attribute {key}'
        return self.__getattribute__(key)
    
    def __setitem__(self, key: str, value):
        self.__setattr__(key, value)

    @classmethod
    def params(cls):
        return cls.__annotations__
    
    def merge(self, target):
        tgt_keys = target.annotations_set()
        for key in tgt_keys:
            self.update(key, target[key])

    def copy(self):
        return copy.deepcopy(self)
    

class Instance:
    def __init__(self, mask: np.ndarray, bbox, score=1., idx=0) -> None:
        self._mask = mask
        self.bbox = bbox
        self.score = score
        self.idx = idx
        self._contours = None

    def get_cutout(self, src_img: np.ndarray):
        cutout = None
        if src_img is not None and self._mask is not None:
            x1, y1, x2, y2 = self.bbox
            x2 += x1
            y2 += y1
            ox1, ox2, oy1, oy2 = x1, x2, y1, y2
            h, w = src_img.shape[:2]
            x1 = np.clip(x1, 0, w)
            x2 = np.clip(x2, 0, w)
            y1 = np.clip(y1, 0, h)
            y2 = np.clip(y2, 0, h)
            if x2 > x1 and y2 > y1:
                img_clip = src_img[y1: y2, x1: x2]
                mh, mw = self.mask.shape[:2]
                if mh != img_clip.shape[0] or mw != img_clip.shape[1]:
                    pad_left = x1 - ox1
                    pad_right = ox2 - x2
                    pad_top = y1 -  oy1
                    pad_btn = oy2 - y2
                    img_clip = cv2.copyMakeBorder(img_clip, pad_top, pad_btn, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=0)
                assert mh == img_clip.shape[0] and mw == img_clip.shape[1]
                cutout = np.concatenate([img_clip, self.mask[..., None].astype(np.uint8) * 255], axis=2)
        return cutout

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask: np.ndarray):
        self._mask = mask
        self._contours = None

    def get_contours(self, simplify=True):
        if self.mask is None:
            return None
        if self._contours is None:
            cons, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            if simplify:
                cnts = []
                for cnt in cons:
                    cnts.append(cv2.approxPolyDP(cnt, 3, True))
            else:
                cnts = cons
            self._contours = cnts
        return self._contours
    
    @property
    def box_center(self):
        return [self.bbox[0] + self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2]

    @property
    def xyxy(self):
        return [self.bbox[0], self.bbox[1], self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3]]
    

def save_instance_list(instance_list: List[Instance], savep: str, compress=None):
    dmp_instance_list = []
    for instance in instance_list:
        mask_rle = mask2rle(instance.mask)
        dmp_instance_list.append(
            {'mask_rle': mask_rle, 'score': instance.score, 'bbox': instance.bbox}
        )
    dict2json(dmp_instance_list, savep, compress=compress)

def load_instance_list(p: str) -> List[Instance]:
    dmp_instance_list = json2dict(p)
    instance_list = []
    for idx, ins in enumerate(dmp_instance_list):
        instance = Instance(
            mask = maskUtils.decode(ins['mask_rle']) > 0,
            score = ins['score'],
            bbox = ins['bbox'],
            idx=idx,
        )
        instance_list.append(instance)
    return instance_list