import json, os, traceback
import os.path as osp

from . import shared
from .structures import List, Dict, Config, field, nested_dataclass
from .logger import logger as LOGGER
from utils.io_utils import json_dump_nested_obj
# from .torch_utils import DEFAULT_DEVICE
from enum import Enum


class SegModel(str, Enum):
    CartoonSeg = 'CartoonSeg'
    SAM = 'SAM'

AVAILABLE_SEGMODELS = list(map(lambda c: c.value, SegModel))


class EditMode(int, Enum):
    NONE = 0
    RectInference = 1
    PointInference = 2


@nested_dataclass
class ProgramConfig(Config):

    cls_path: str = 'assets/tagcluster_bodypart_v2.json'
    show_colorcode: bool = True
    segmentation_model: str = SegModel.CartoonSeg
    segmentation_device: str = 'cuda'
    segmentation_refine: bool = True
    segmentation_conf_thr: float = 0.3
    recent_proj_list: List = field(default_factory=lambda: list())
    edit_mode: int = EditMode.NONE
    mask_opacity: int = 75
    show_colorcode: bool = True
    show_contour: bool = False
    original_transparency: float = 0.
    open_recent_on_startup: bool = True 
    show_page_list: bool = False
    darkmode: bool = True
    display_lang: str = field(default_factory=lambda: shared.DEFAULT_DISPLAY_LANG) # to always apply shared.DEFAULT_DISPLAY_LANG
    imgsave_quality: int = 100
    imgsave_ext: str = '.png'
    parsing_src: str= 'parsinglog_sambody_iter1_step18k_masks.json'

    seg_type = 'body_part_tag'

    @staticmethod
    def load(cfg_path: str):
        
        with open(cfg_path, 'r', encoding='utf8') as f:
            config_dict = json.loads(f.read())
        return ProgramConfig(**config_dict)
    

pcfg: ProgramConfig = None

def load_config():

    if osp.exists(shared.CONFIG_PATH):
        try:
            config = ProgramConfig.load(shared.CONFIG_PATH)
        except Exception as e:
            LOGGER.exception(e)
            LOGGER.warning("Failed to load config file, using default config")
            config = ProgramConfig()
    else:
        LOGGER.info(f'{shared.CONFIG_PATH} does not exist, new config file will be created.')
        config = ProgramConfig()
    
    global pcfg
    pcfg = config
    

def save_config():
    global pcfg
    try:
        with open(shared.CONFIG_PATH, 'w', encoding='utf8') as f:
            f.write(json_dump_nested_obj(pcfg))
        LOGGER.info('Config saved')
        return True
    except Exception as e:
        LOGGER.error(f'Failed save config to {shared.CONFIG_PATH}: {e}')
        LOGGER.error(traceback.format_exc())
        return False