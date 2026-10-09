from typing import Dict, Union
import os
import os.path as osp
import json
import sys
from functools import lru_cache

from utils.visualize import get_color

PROGRAM_PATH = osp.abspath(osp.dirname(__file__))
LOGGING_PATH = osp.join(PROGRAM_PATH, 'logs')

LIBS_PATH = osp.join(PROGRAM_PATH, 'data/libs')

STYLESHEET_PATH = osp.join(PROGRAM_PATH, 'config/stylesheet.css')
THEME_PATH = osp.join(PROGRAM_PATH, 'config/themes.json')
CONFIG_PATH = osp.join(PROGRAM_PATH, 'config/config.json')


CONFIG_FONTSIZE_HEADER = 18
CONFIG_FONTSIZE_TABLE = 16
CONFIG_FONTSIZE_CONTENT = 16

CONFIG_COMBOBOX_HEIGHT = 30 
CONFIG_COMBOBOX_SHORT = 200
CONFIG_COMBOBOX_MIDEAN = 332
CONFIG_COMBOBOX_LONG = 468

HORSLIDER_FIXHEIGHT = 36

WIDGET_SPACING_CLOSE = 8
TEXTEDIT_FIXWIDTH = 350

TEXTEFFECT_FIXWIDTH = 400
TEXTEFFECT_MAXHEIGHT = 500

LEFTBAR_WIDTH = 60
LEFTBTN_WIDTH = 38

LDPI = 96.
DPI = 188.75

SCREEN_H = 2160
SCREEN_W = 3840

DEFAULT_FONT_FAMILY = 'Microsoft YaHei UI'
APP_DEFAULT_FONT = 'Microsoft YaHei UI'

WINDOW_BORDER_WIDTH = 4
BOTTOMBAR_HEIGHT = 32
TITLEBAR_HEIGHT = 30

PAGELIST_THUMBNAIL_MAXNUM = 0
PAGELIST_THUMBNAIL_SIZE = 48

FLAG_QT6 = True

SLIDERHANDLE_COLOR = (85,85,96)
FOREGROUND_FONTCOLOR = (93,93,95)

MAX_NUM_LOG = 7

TRANSLATE_DIR = osp.join(PROGRAM_PATH, 'translate')
DISPLAY_LANGUAGE_MAP = {
    "English": "English",
    # "简体中文": "zh_CN",
    # "Русский": "ru_RU",
    # "Português (Brasil)": "pt_BR",
    # "한국어": "ko_KR"
}
VALID_LANG_SET = set(list(DISPLAY_LANGUAGE_MAP.values()))

# for p in os.listdir(TRANSLATE_DIR):
#     if p.endswith('.qm'):
#         lang = p.replace('.qm', '')
#         if lang not in VALID_LANG_SET:
#             DISPLAY_LANGUAGE_MAP[lang] = lang

DEFAULT_DISPLAY_LANG = 'English'

USE_PYSIDE6 = False
ON_MACOS = sys.platform == 'darwin'
ON_WINDOWS = sys.platform == 'win32'
HEADLESS = False
DEBUG = False

check_local_file_hash = True

FONT_FAMILIES: set = None


showed_exception = set()

# it will be set to ui.mainwindow.create_errdialog.emit after UI initialized
create_errdialog_in_mainthread = lambda *args, **kwargs: None

create_infodialog_in_mainthread = lambda *args, **kwargs: None
close_infodialog = lambda *args, **kwargs: None


config_name_to_view_widget = {}
action_to_view_config_name = {}

runtime_widget_set = set()

# cache for HASH value of downloaded models
cache_data: Dict = None
cache_dir: str = osp.join(PROGRAM_PATH, '.cache')
cache_path: str = osp.join(PROGRAM_PATH, '.cache/cache.json')

def load_cache():
    global cache_data
    if cache_data is None:
        if osp.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf8") as file:
                    cache_data = json.load(file)
            except:
                print(f'cached file {cache_path} is invalid')
                cache_data = {}
        else:
            cache_data = {}

def dump_cache():
    global cache_data
    if cache_data is None:
        return
    
    cache_dir = osp.dirname(cache_path)
    if not osp.exists(cache_dir):
        os.makedirs(cache_dir)

    with open(cache_path, "w", encoding="utf8") as file:
        json.dump(cache_data, file, indent=4)

    global CACHE_UPDATED
    CACHE_UPDATED = False


def add_to_runtime_widget_set(widget):
    runtime_widget_set.add(widget)

def add_to_info_widget_set(widget, info_type):
    if info_type is None or info_type == '':
        return
    info_widget_set[info_type] = widget

info_widget_set = {}

def remove_from_runtime_widget_set(widget):
    if hasattr(widget, 'info_type') and widget.info_type in info_widget_set:
        info_widget_set.pop(widget.info_type)
    if widget in runtime_widget_set:
        runtime_widget_set.remove(widget)

cls_list: list[str] = []


@lru_cache
def get_cls_color(tag: Union[int, str]):
    if isinstance(tag, str):
        try:
            tag = cls_list.index(tag)
        except:
            return (127, 127, 127)
    if tag is None:
        return (127, 127, 127)
    return get_color(tag)