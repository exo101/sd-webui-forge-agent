import warnings, os, json
from urllib3.exceptions import InsecureRequestWarning
from datetime import datetime

# === WebUI imports ===
from modules.shared import opts


class Colors:
    """ANSI color codes for terminal output"""
    BLACK   = '\033[30m'
    RED     = '\033[31m'
    GREEN   = '\033[32m'
    YELLOW  = '\033[33m'
    BLUE    = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN    = '\033[36m'
    RESET   = '\033[0m'


do_debug_print = getattr(opts, 'civitai_debug_prints', False)
def init():
    warnings.simplefilter('ignore', InsecureRequestWarning)

    config_folder = os.path.join(os.getcwd(), 'config_states')
    if not os.path.exists(config_folder):
        os.mkdir(config_folder)

    global download_queue, last_version, cancel_status, recent_model, last_url, json_data, json_info, main_folder, previous_inputs, download_fail, sortNewest, isDownloading, scan_files, from_update_tab, url_list, print, subfolder_json, organization_backup_file, last_organization_backup, update_mode, update_items, local_browser_fallback_items

    cancel_status = None
    recent_model = None
    json_data = None
    json_info = None
    main_folder = None
    previous_inputs = None
    last_version = None
    url_list = {}
    download_queue = []

    subfolder_json = os.path.join(config_folder, 'civitai_subfolders.json')
    if not os.path.exists(subfolder_json):
        with open(subfolder_json, 'w') as json_file:
            #json.dump({}, json_file)
            json.dump({'created_at': datetime.now().timestamp()}, json_file)

    # Organization backup file
    organization_backup_file = os.path.join(config_folder, 'civitai_organization_backups.json')
    if not os.path.exists(organization_backup_file):
        with open(organization_backup_file, 'w', encoding='utf-8') as json_file:
            # Initialize with created_at (required by Forge config_states system)
            json.dump({'created_at': datetime.now().timestamp(), 'backups': []}, json_file)
    last_organization_backup = None

    from_update_tab = False
    scan_files = False
    download_fail = False
    sortNewest = False
    isDownloading = False
    update_mode = False
    update_items = []
    local_browser_fallback_items = []

_print = print
def print(print_message):
    _print(f"{Colors.BLUE}[CivitAI Browser Neo]{Colors.RESET} - {print_message}")

def debug_print(print_message):
    if do_debug_print:
        _print(f"{Colors.MAGENTA}[DEBUG] {Colors.BLUE}[CivitAI Browser Neo]{Colors.RESET} - {print_message}")