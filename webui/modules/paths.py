import os
import sys
from modules.paths_internal import models_path, script_path, data_path, extensions_dir, extensions_builtin_dir, cwd  # noqa: F401


sys.path.insert(0, script_path)

sd_path = os.path.dirname(__file__)
