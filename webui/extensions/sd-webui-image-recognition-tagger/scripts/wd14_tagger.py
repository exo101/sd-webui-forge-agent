import sys
from pathlib import Path

# The migrated WD14 package lives at the aesthetic extension root.  Add that
# directory before importing it so the original absolute ``tagger`` imports
# continue to work unchanged.
_extension_root = Path(__file__).resolve().parent.parent
if str(_extension_root) not in sys.path:
    sys.path.insert(0, str(_extension_root))

from PIL import Image, ImageFile

from modules import script_callbacks
from tagger.api import on_app_started


# if you do not initialize the Image object
# Image.registered_extensions() returns only PNG
Image.init()

# PIL spits errors when loading a truncated image by default
# https://pillow.readthedocs.io/en/stable/reference/ImageFile.html#PIL.ImageFile.LOAD_TRUNCATED_IMAGES
ImageFile.LOAD_TRUNCATED_IMAGES = True


script_callbacks.on_app_started(on_app_started)
# The UI is embedded into main.py's 图像识别 tab.  Only the API is registered
# here; registering on_ui_tabs would create a second top-level page.
