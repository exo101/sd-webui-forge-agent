from pathlib import Path

from modules import scripts, shared
from modules.paths import extensions_dir, script_path

# Webui root path
FILE_DIR = Path(script_path).absolute()

# The extension base path
EXT_PATH = Path(extensions_dir).absolute()

# Tags base path
TAGS_PATH = Path(scripts.basedir()).joinpath("tags").absolute()

# Embeddings directory (Forge Neo: models/embeddings/)
EMB_PATH = Path(shared.cmd_opts.embeddings_dir).absolute()

# Hypernetworks are removed in Forge Neo — always None
HYP_PATH = None

# LoRA and LyCORIS share one directory in Forge Neo
try:
    LORA_PATH = Path(shared.cmd_opts.lora_dir).absolute()
except (AttributeError, TypeError):
    LORA_PATH = None

# LyCORIS is unified with LoRA in Forge Neo
LYCO_PATH = LORA_PATH

# Wildcards directory (Forge Neo: scripts/wildcards/ under webui root)
WILDCARD_PATH = FILE_DIR.joinpath("scripts/wildcards").absolute()


def find_ext_wildcard_paths():
    """Returns paths to wildcard folders registered by other extensions."""
    found = list(EXT_PATH.glob("*/wildcards/"))

    # Append custom wildcard path from sd-dynamic-prompts if present
    try:
        from modules.shared import opts as _opts
        custom = getattr(_opts, "wildcard_dir", None)
    except ImportError:
        custom = None

    if custom is not None:
        p = Path(custom).absolute()
        if p.exists():
            found.append(p)

    return found


# The path to the extension wildcards folder
WILDCARD_EXT_PATHS = find_ext_wildcard_paths()

# Temporary file paths
STATIC_TEMP_PATH = FILE_DIR.joinpath("tmp").absolute()
TEMP_PATH = TAGS_PATH.joinpath("temp").absolute()

# Make sure these folders exist.
# Use parents=True + exist_ok=True so the extension keeps working after a Forge
# update or reinstall that might briefly remove the tags/temp subdirectory (#302).
TEMP_PATH.mkdir(parents=True, exist_ok=True)
STATIC_TEMP_PATH.mkdir(parents=True, exist_ok=True)
