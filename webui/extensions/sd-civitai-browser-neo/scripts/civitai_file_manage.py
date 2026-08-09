import urllib.request
import urllib.error
import requests
import hashlib
import base64
import errno
import json
import time
import re
import os
import io
import shutil
import gradio as gr
from urllib.parse import urlparse
from pathlib import Path
from PIL import Image

# === WebUI imports ===
from modules.shared import cmd_opts, opts

# === Extension imports ===
import scripts.civitai_download as _download
import scripts.civitai_file_manage as _file
import scripts.civitai_global as gl
import scripts.civitai_api as _api
from scripts.civitai_global import print, debug_print


IS_KAGGLE = 'KAGGLE_URL_BASE' in os.environ


try:
    from send2trash import send2trash
except ImportError:
    print('Python module "send2trash" has not been imported correctly, please try to restart or install it manually.')
try:
    from bs4 import BeautifulSoup
except ImportError:
    print('Python module "BeautifulSoup" has not been imported correctly, please try to restart or install it manually.')

gl.init()

css_path = Path(__file__).resolve().parents[1] / 'style_html.css'
no_update = False
last_update_scan = None  # Stores last update scan results for Dashboard summary
last_dashboard_data = None  # Stores last dashboard scan raw data (categories, top files, orphans)


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

# ─────────────────────────────────────────────────────────────────────────────
# Creator management — Favorite & Ban lists
# Pattern from SignalFlagZ/sd-webui-civbrowser
# ─────────────────────────────────────────────────────────────────────────────
_EXT_ROOT = Path(__file__).resolve().parents[1]
_CHECKPOINT_HASH_DB_PATH = _EXT_ROOT / 'lib' / 'models' / 'checkpoint_hashes.json'

class UserInfo:
    """Persistent comma-separated list of creator usernames stored in a .txt file."""
    def __init__(self, path: Path):
        self._path = path
        self._names: list = []
        self.load()

    def load(self) -> list:
        if self._path.exists():
            text = self._path.read_text(encoding='utf-8')
            self._names = [n.strip() for n in text.replace('\n', ',').split(',') if n.strip()]
        else:
            self._names = []
        return self._names

    def save(self):
        lines = []
        for i in range(0, len(self._names), 3):
            lines.append(', '.join(self._names[i:i + 3]))
        self._path.write_text('\n'.join(lines), encoding='utf-8')

    def add(self, name: str) -> bool:
        name = name.strip()
        if not name or name in self._names:
            return False
        self._names.append(name)
        self.save()
        return True

    def remove(self, name: str) -> bool:
        name = name.strip()
        if name in self._names:
            self._names.remove(name)
            self.save()
            return True
        return False

    def get_as_list(self) -> list:
        return list(self._names)

    def get_as_text(self) -> str:
        return ','.join(self._names)


class FavoriteUsers(UserInfo):
    def __init__(self):
        super().__init__(_EXT_ROOT / 'favoriteCreators.txt')


class BanUsers(UserInfo):
    def __init__(self):
        super().__init__(_EXT_ROOT / 'bannedCreators.txt')


FavoriteCreators = FavoriteUsers()
BanCreators = BanUsers()


def _creator_button_updates(username: str):
    """Returns (btn_fav, btn_ban, btn_clear, banned_list_txt) gr.updates."""
    u = (username or '').strip()
    if not u:
        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value=BanCreators.get_as_text()),
        )
    is_fav = u in FavoriteCreators.get_as_list()
    is_ban = u in BanCreators.get_as_list()
    already_set = is_fav or is_ban
    return (
        gr.update(interactive=not is_fav),
        gr.update(interactive=not is_ban),
        gr.update(interactive=already_set),
        gr.update(value=BanCreators.get_as_text()),
    )


def add_favorite_creator(username: str):
    """Add creator to favorites (mutually exclusive with ban)."""
    u = (username or '').strip()
    if not u:
        return _creator_button_updates(u)
    FavoriteCreators.add(u)
    BanCreators.remove(u)
    gr.Info(f"\u2b50 {u} added to favorites")
    return _creator_button_updates(u)


def ban_creator(username: str):
    """Ban creator (mutually exclusive with favorite)."""
    u = (username or '').strip()
    if not u:
        return _creator_button_updates(u)
    BanCreators.add(u)
    FavoriteCreators.remove(u)
    gr.Info(f"\U0001f6ab {u} banned")
    return _creator_button_updates(u)


def clear_creator(username: str):
    """Remove creator from both favorites and ban list."""
    u = (username or '').strip()
    if not u:
        return _creator_button_updates(u)
    removed_fav = FavoriteCreators.remove(u)
    removed_ban = BanCreators.remove(u)
    if removed_fav or removed_ban:
        gr.Info(f"\u21ba {u} status cleared")
    return _creator_button_updates(u)


def get_banned_creators_text() -> str:
    """Comma-joined banned creator list for JS initialisation."""
    return BanCreators.get_as_text()

# ─────────────────────────────────────────────────────────────────────────────
# Companion Files Lookup Table
# Required companion files (VAE, text encoders, etc.) per base model architecture.
# Keys are substrings matched against baseModel.upper().
# ─────────────────────────────────────────────────────────────────────────────
_COMPANION_FILES = {
    'FLUX': [
        {'label': 'VAE',                     'filename': 'ae.safetensors',                                    'folder': 'models/VAE',          'size': '335 MB', 'url': 'https://huggingface.co/black-forest-labs/FLUX.1-dev'},
        {'label': 'Text Encoder (CLIP-L)',   'filename': 'clip_l.safetensors',                               'folder': 'models/text_encoder', 'size': '246 MB', 'url': 'https://huggingface.co/comfyanonymous/flux_text_encoders'},
        {'label': 'Text Encoder (T5-XXL)',   'filename': 't5xxl_fp8_e4m3fn_scaled.safetensors',              'folder': 'models/text_encoder', 'size': '4.9 GB', 'url': 'https://huggingface.co/comfyanonymous/flux_text_encoders'},
    ],
    'WAN': [
        {'label': 'VAE',                     'filename': 'wan_2.1_vae.safetensors',                          'folder': 'models/VAE',          'size': '254 MB', 'url': 'https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged'},
        {'label': 'Text Encoder (UMT5-XXL)', 'filename': 'umt5_xxl_fp8_e4m3fn_scaled.safetensors',          'folder': 'models/text_encoder', 'size': '6.7 GB', 'url': 'https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged'},
    ],
    'QWEN': [
        {'label': 'VAE',                     'filename': 'qwen_image_vae.safetensors',                       'folder': 'models/VAE',          'size': '~0.5 GB','url': 'https://huggingface.co/nunchaku-tech/nunchaku-qwen-image'},
        {'label': 'Text Encoder (Qwen2.5-VL)','filename': 'qwen2.5_vl_7b_instruct_fp8_e4m3fn_scaled.safetensors', 'folder': 'models/text_encoder', 'size': '~8 GB', 'url': 'https://huggingface.co/Comfy-Org/Qwen2.5-VL-7B-Instruct_fp8_scaled'},
    ],
    'Z-IMAGE': [
        {'label': 'VAE',                     'filename': 'ae.safetensors',                                   'folder': 'models/VAE',          'size': '335 MB', 'url': 'https://huggingface.co/black-forest-labs/FLUX.1-dev'},
        {'label': 'Text Encoder (Qwen3-4B)', 'filename': 'qwen_3_4b.safetensors',                           'folder': 'models/text_encoder', 'size': '8.0 GB', 'url': 'https://huggingface.co/Comfy-Org/z_image_turbo'},
    ],
    'LUMINA': [
        {'label': 'VAE',                     'filename': 'ae.safetensors',                                   'folder': 'models/VAE',          'size': '335 MB', 'url': 'https://huggingface.co/black-forest-labs/FLUX.1-dev'},
        {'label': 'Text Encoder (Gemma-2)',  'filename': 'gemma_2_2b.safetensors',                           'folder': 'models/text_encoder', 'size': '~5 GB',  'url': 'https://huggingface.co/Comfy-Org/Lumina-Image-2.0'},
    ],
}

# Per-architecture notes injected into the companion banner
_COMPANION_NOTES = {
    'WAN': (
        '⚠️ <strong>Wan 2.2</strong> uses a two-model MoE (Mixture-of-Experts) architecture. '
        'Load the <code>[HN]</code> (high-noise) checkpoint as the main model and the '
        '<code>[LN]</code> (low-noise) model in the <strong>Refiner</strong> slot. '
        'Enable Refiner in <em>Forge Settings → Refiner</em> first.'
    ),
}


def get_companion_banner(base_model: str, model_filename: str = '', model_name: str = '') -> str:
    """
    Returns an HTML banner listing required companion files (VAE, text encoders, etc.)
    for architectures that need them.  Returns '' if nothing is missing.
    Only checks Checkpoints (not LoRAs) — called from update_model_info.

    For Wan 2.2 MoE models, detects whether the file is HN or LN by inspecting
    model_filename and model_name, and shows a tailored instruction note.
    """
    if not base_model or str(base_model).strip() in ('', 'Not Found', 'Unknown'):
        return ''

    base_upper = str(base_model).upper()

    companions = None
    matched_key = None
    for pattern, files in _COMPANION_FILES.items():
        if pattern in base_upper:
            companions = files
            matched_key = pattern
            break

    if not companions:
        return ''

    # For Wan 2.2 MoE — detect HN vs LN from filename/model name
    note_key = matched_key  # default: use the generic note key
    if matched_key == 'WAN':
        combined = (model_filename + ' ' + model_name).upper()
        # Patterns: [HN], _HN_, -HN-, "HN" standalone, wan_hn, etc.
        import re as _re
        if _re.search(r'(?<![A-Z])HN(?![A-Z])', combined):
            note_key = 'WAN_HN'
        elif _re.search(r'(?<![A-Z])LN(?![A-Z])', combined):
            note_key = 'WAN_LN'

    try:
        from modules.paths import models_path as _mp
        base_path = Path(_mp)
    except Exception:
        base_path = Path('models')

    rows = []
    any_missing = False
    for comp in companions:
        sub = comp['folder'].split('/')[-1]          # e.g. 'VAE', 'text_encoder'
        dest_folder = base_path / sub
        present = False
        if dest_folder.exists():
            present = (dest_folder / comp['filename']).exists()
            if not present:
                present = bool(list(dest_folder.rglob(comp['filename'])))
        if not present:
            any_missing = True
        status_icon = '✅' if present else '⬇️'
        status_class = 'companion-present' if present else 'companion-missing'
        rows.append(
            f'<tr class="{status_class}">'
            f'<td class="companion-status">{status_icon}</td>'
            f'<td class="companion-label">{comp["label"]}</td>'
            f'<td class="companion-filename"><code>{comp["filename"]}</code></td>'
            f'<td class="companion-folder">{comp["folder"]}/</td>'
            f'<td class="companion-size">{comp["size"]}</td>'
            f'<td class="companion-link"><a href="{comp["url"]}" target="_blank">HuggingFace ↗</a></td>'
            f'</tr>'
        )

    if not any_missing:
        return ''  # All files already present

    note_html = ''
    if note_key and note_key in _COMPANION_NOTES:
        note_html = f'<p class="companion-note">{_COMPANION_NOTES[note_key]}</p>'
    elif matched_key and matched_key in _COMPANION_NOTES:
        note_html = f'<p class="companion-note">{_COMPANION_NOTES[matched_key]}</p>'

    table_body = ''.join(rows)
    table_html = (
        f'<table class="companion-table">'
        f'<thead><tr><th></th><th>Type</th><th>Filename</th><th>Destination</th><th>Size</th><th>Source</th></tr></thead>'
        f'<tbody>{table_body}</tbody>'
        f'</table>'
    )
    return (
        f'<div class="companion-files-banner">'
        f'<h3 class="companion-title">📦 Required Companion Files</h3>'
        f'{note_html}'
        f'<div class="companion-table-wrap">{table_html}</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
from_tag = False
from_ver = False
from_installed = False
try:
    queue = not cmd_opts.no_gradio_queue
except AttributeError:
    queue = not cmd_opts.disable_queue
except:
    queue = True


def append_update_audit_log(action, details):
    """
    Append one line to the JSONL audit log at the extension root.
    Each line is a standalone JSON object: { timestamp, action, ...details }
    """
    log_path = Path(__file__).resolve().parents[1] / 'neo_update_audit.jsonl'
    entry = {'timestamp': __import__('datetime').datetime.now().isoformat(), 'action': action, **details}
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f'[Audit] Failed to write audit log: {e}')


def handle_existing_model_file(file_path):
    """
    Apply the retention policy before downloading a new version of a model file.
    Policies (civitai_neo_update_retention setting):
      'keep'         — do nothing; old and new file co-exist (only if filenames differ)
      'move to _Trash' — move old file into a _Trash/ subfolder next to it
      'replace'      — delete old file (historical default)
    """
    if not os.path.exists(file_path):
        return
    policy = getattr(opts, 'civitai_neo_update_retention', 'replace')
    if policy == 'keep':
        return
    elif policy == 'move to _Trash':
        # Send main file to system trash
        send2trash(file_path)
        print(f'[Retention] Moved old model file to system trash: {file_path}')
        append_update_audit_log('retention_trash', {'old_file': file_path})
        # Also send adjacent files to system trash so they don't orphan alongside the new version
        parent_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        delete_associated_files(parent_dir, base_name)
    else:  # 'replace' (default) — delete main model file permanently, send associated files to trash
        os.remove(file_path)
        print(f'[Retention] Replaced model file deleted: {file_path}')
        append_update_audit_log('retention_replace', {'old_file': file_path})
        # Also clean up adjacent files (preview, json, html, api_info, numbered images)
        parent_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        delete_associated_files(parent_dir, base_name)


def delete_model(delete_finish=None, model_filename=None, model_string=None, list_versions=None, sha256=None, selected_list=None, model_ver=None, model_json=None):
    deleted = False
    model_id = None

    if model_string:
        _, model_id = _api.extract_model_info(model_string)

    if not model_ver:
        model_versions = _api.update_model_versions(model_id)
    else:
        model_versions = model_ver

    (model_name, ver_value, ver_choices) = _file.card_update(model_versions, model_string, list_versions, False)
    if not model_json:
        if model_id != None:
            selected_content_type = None
            for item in gl.json_data['items']:
                if int(item['id']) == int(model_id):
                    selected_content_type = item['type']
                    desc = item['description']
                    break

            if selected_content_type == None:
                print('Model ID not found in json_data. (delete_model)')
                return
    else:
        for item in model_json['items']:
            selected_content_type = item['type']
            desc = item['description']

    model_folder = os.path.join(_api.contenttype_folder(selected_content_type, desc))

    # Delete based on provided SHA-256 hash
    if sha256:
        sha256_upper = sha256.upper()
        for root, _, files in os.walk(model_folder, followlinks=True):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    data = _api.safe_json_load(file_path)
                    if data:
                        file_sha256 = (data.get('sha256') or '').upper()
                    else:
                        file_sha256 = '0'

                    if file_sha256 == sha256_upper:
                        use_trash = getattr(opts, 'civitai_neo_delete_to_trash', True)
                        unpack_list = data.get('unpackList', []) if data else []
                        for unpacked_file in unpack_list:
                            unpacked_file_path = os.path.join(root, unpacked_file)
                            if os.path.isfile(unpacked_file_path):
                                if use_trash:
                                    try:
                                        send2trash(unpacked_file_path)
                                        print(f"File moved to trash based on unpackList: {unpacked_file_path}")
                                    except:
                                        os.remove(unpacked_file_path)
                                        print(f"File deleted based on unpackList: {unpacked_file_path}")
                                else:
                                    os.remove(unpacked_file_path)
                                    print(f"File deleted based on unpackList: {unpacked_file_path}")

                        base_name, _ = os.path.splitext(file)
                        if os.path.isfile(file_path):
                            if use_trash:
                                try:
                                    send2trash(file_path)
                                    print(f"Model moved to trash based on SHA-256: {file_path}")
                                except:
                                    os.remove(file_path)
                                    print(f"Model deleted based on SHA-256: {file_path}")
                            else:
                                os.remove(file_path)
                                print(f"Model deleted based on SHA-256: {file_path}")
                            delete_associated_files(root, base_name)
                            deleted = True

    # Fallback to delete based on filename if not deleted based on SHA-256
    filename_to_delete = os.path.splitext(model_filename)[0]
    aria2_file = model_filename + '.aria2'
    if not deleted:
        for root, dirs, files in os.walk(model_folder, followlinks=True):
            for file in files:
                current_file_name = os.path.splitext(file)[0]
                if filename_to_delete == current_file_name or aria2_file == file:
                    path_file = os.path.join(root, file)
                    if os.path.isfile(path_file):
                        use_trash = getattr(opts, 'civitai_neo_delete_to_trash', True)
                        if use_trash:
                            try:
                                send2trash(path_file)
                                print(f"Model moved to trash based on filename: {path_file}")
                            except:
                                os.remove(path_file)
                                print(f"Model deleted based on filename: {path_file}")
                        else:
                            os.remove(path_file)
                            print(f"Model deleted based on filename: {path_file}")
                        delete_associated_files(root, current_file_name)

    number = _download.random_number(delete_finish)

    btnDwn = not selected_list or selected_list == '[]'

    return (
        gr.update(interactive=btnDwn, visible=btnDwn),  # Download Button
        gr.update(interactive=False, visible=False),  # Cancel Button
        gr.update(interactive=False, visible=False),  # Delete Button
        gr.update(value=number),  # Delete Finish Trigger
        gr.update(value=model_name),  # Current Model
        gr.update(value=ver_value, choices=ver_choices)  # Version List
    )

def delete_installed_by_sha256(sha256, delete_finish=None):
    """
    Simplified delete function for installed models using only SHA256.
    Searches all model folders for a match and deletes the model.
    """
    if not sha256:
        print("No SHA256 provided for deletion")
        return gr.update(value=_download.random_number(delete_finish))
    
    sha256_upper = sha256.upper()
    
    # Get all content types to search
    content_types = ['Checkpoint', 'LORA', 'LoCon', 'DoRA', 'VAE', 'Controlnet', 'Poses', 
                     'TextualInversion', 'Upscaler', 'MotionModule', 'Workflows', 'Detection', 'Other', 'Wildcards']
    
    folders_to_check = []
    for content_type in content_types:
        if content_type == 'Upscaler':
            for desc in ['SWINIR', 'REALESRGAN', 'GFPGAN', 'BSRGAN', 'ESRGAN']:
                folder = _api.contenttype_folder('Upscaler', desc)
                if folder and folder not in folders_to_check:
                    folders_to_check.append(folder)
        else:
            folder = _api.contenttype_folder(content_type)
            if folder and folder not in folders_to_check:
                folders_to_check.append(folder)
    
    deleted = False
    for model_folder in folders_to_check:
        if deleted:
            break
        for root, _, files in os.walk(model_folder, followlinks=True):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    data = _api.safe_json_load(file_path)
                    if not data:
                        continue
                    
                    file_sha256 = data.get('sha256', '').upper()
                    if file_sha256 == sha256_upper:
                        # Found matching model!
                        model_name = data.get('model', {}).get('name', 'Unknown Model')
                        print(f"Found model to delete: {model_name} (SHA256: {sha256_upper})")
                        
                        # Find the model file that shares the same base name as this
                        # JSON sidecar. The saved JSON has no 'file.name' key, so we
                        # scan the directory directly using a full path (json_base is
                        # just a filename — joining with root is required for exists()).
                        json_base = os.path.splitext(file)[0]
                        model_extensions = ['.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.zip', '.vae', '.th']
                        model_filename = ''
                        for ext in model_extensions:
                            candidate = os.path.join(root, json_base + ext)
                            if os.path.exists(candidate):
                                model_filename = os.path.basename(candidate)
                                break
                        
                        if model_filename:
                            # Delete model file
                            model_file_path = os.path.join(root, model_filename)
                            if os.path.exists(model_file_path):
                                try:
                                    send2trash(model_file_path)
                                    print(f"Model moved to trash: {model_file_path}")
                                except:
                                    os.remove(model_file_path)
                                    print(f"Model deleted: {model_file_path}")
                                
                                # Delete associated files
                                base_filename = os.path.splitext(model_filename)[0]
                                delete_associated_files(root, base_filename)
                                
                                deleted = True
                                break
                        else:
                            print(f"Could not find model file for JSON: {file_path}")
            
            if deleted:
                break
    
    if deleted:
        print(f"Successfully deleted model with SHA256: {sha256_upper}")
    else:
        print(f"Could not find model with SHA256: {sha256_upper}")
    
    return gr.update(value=_download.random_number(delete_finish))

## === ANXETY EDITs ===
def delete_associated_files(directory, base_name):
    """Deletes related model files in the save directory"""
    # Patterns for associated files
    associated_suffixes = ['', '.preview', '.api_info', '.html']
    image_exts = {'.png', '.jpg', '.jpeg'}

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        name, ext = os.path.splitext(file)

        # Delete associated files by suffix
        if name in [f'{base_name}{sfx}' for sfx in associated_suffixes]:
            try:
                send2trash(file_path)
                print(f"Associated file moved to trash: {file_path}")
            except Exception:
                os.remove(file_path)
                print(f"Associated file deleted: {file_path}")
            continue

        # Delete images matching pattern: <base_name>_<number>.<ext>
        if name.startswith(f'{base_name}_') and ext.lower() in image_exts:
            suffix = name[len(f'{base_name}_'):]
            if suffix.isdigit():
                try:
                    send2trash(file_path)
                    print(f"Image moved to trash: {file_path}")
                except Exception:
                    os.remove(file_path)
                    print(f"Image deleted: {file_path}")


def _trash_associated_files(directory, base_name, trash_dir):
    """Moves related model files to the _Trash folder alongside the main file."""
    associated_suffixes = ['', '.preview', '.api_info', '.html']
    image_exts = {'.png', '.jpg', '.jpeg'}

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        name, ext = os.path.splitext(file)

        # Move associated files by suffix
        if name in [f'{base_name}{sfx}' for sfx in associated_suffixes]:
            dest = os.path.join(trash_dir, file)
            if os.path.exists(dest):
                stamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
                base, ext = os.path.splitext(file)
                dest = os.path.join(trash_dir, f'{base}_{stamp}{ext}')
            shutil.move(file_path, dest)
            print(f'[Retention] Moved adjacent file to _Trash: {dest}')
            continue

        # Move images matching pattern: <base_name>_<number>.<ext>
        if name.startswith(f'{base_name}_') and ext.lower() in image_exts:
            suffix = name[len(f'{base_name}_'):]
            if suffix.isdigit():
                dest = os.path.join(trash_dir, file)
                if os.path.exists(dest):
                    stamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
                    base, ext = os.path.splitext(file)
                    dest = os.path.join(trash_dir, f'{base}_{stamp}{ext}')
                shutil.move(file_path, dest)
                print(f'[Retention] Moved adjacent image to _Trash: {dest}')


def _resize_image_bytes(image_bytes, target_size=512):
    """Resize image bytes to target_size on the longer side, keeping aspect ratio"""
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size

    if width > height:
        new_size = (target_size, int(height * target_size / width))
    else:
        new_size = (int(width * target_size / height), target_size)

    resized_image = image.resize(new_size, Image.LANCZOS)

    output = io.BytesIO()
    resized_image.save(output, format="PNG")
    output.seek(0)
    return output.getvalue()  # Return bytes, not BytesIO object

def save_preview(file_path, api_response, overwrite_toggle=False, sha256=None):
    proxies, ssl = _api.get_proxies()
    file_path = Path(file_path)
    install_path = file_path.parent
    name = file_path.stem
    json_file = file_path.with_suffix('.json')
    image_path = install_path / f"{name}.preview.png"

    if not overwrite_toggle and image_path.exists():
        return

    if not sha256 and json_file.exists():
        data = json.loads(json_file.read_text(encoding='utf-8'))
        if 'sha256' in data and data['sha256']:
            sha256 = data['sha256'].upper()
    elif sha256:
        sha256 = sha256.upper()

    for item in api_response['items']:
        for version in item['modelVersions']:
            for file_entry in version['files']:
                if file_entry['hashes'].get('SHA256') == sha256:
                    for image in version['images']:
                        if image['type'] == 'image':
                            url_with_width = re.sub(r'/width=\d+', f"/width={image['width']}", image['url'])
                            response = requests.get(url_with_width, proxies=proxies, verify=ssl)

                            if response.status_code == 200:
                                # Check if resize is enabled for saved previews
                                resize_saved = getattr(opts, 'resize_preview_on_save', True)
                                if resize_saved:
                                    resize_size = getattr(opts, 'resize_preview_size', 512)
                                    image_data = _resize_image_bytes(response.content, resize_size)
                                else:
                                    # Save original size
                                    image_data = response.content

                                if IS_KAGGLE:
                                    import sd_image_encryption  # Import Module for Encrypt Image
                                    img = Image.open(io.BytesIO(image_data))
                                    imginfo = img.info or {}
                                    if not all(key in imginfo for key in ['Encrypt', 'EncryptPwdSha']):
                                        sd_image_encryption.EncryptedImage.from_image(img).save(image_path)
                                else:
                                    image_path.write_bytes(image_data)

                                print(f"Preview saved at: {image_path}")
                            else:
                                print(f"Failed to save preview. Status code: {response.status_code}")
                            return

                    print(f"No preview images found for '{name}'")
                    return

def get_image_path(install_path, api_response, sub_folder):
    image_location = getattr(opts, 'image_location', '')
    sub_image_location = getattr(opts, 'sub_image_location', True)
    image_path = install_path
    if api_response:
        json_info = api_response['items'][0]
    else:
        json_info = gl.json_info

    if image_location:
        if sub_image_location:
            desc = json_info['description']
            content_type = json_info['type']
            image_path = os.path.join(_api.contenttype_folder(content_type, desc, custom_folder=image_location))

            if sub_folder and sub_folder != 'None' and sub_folder != 'Only available if the selected files are of the same model type':
                image_path = os.path.join(image_path, sub_folder.lstrip('/').lstrip('\\'))
        else:
            image_path = Path(image_location)
    else:
        # For TextualInversion (embeddings), save gallery images in a subfolder so
        # Forge's embedding scanner doesn't log "no embedded information found" for
        # every downloaded preview PNG file.
        content_type = json_info.get('type', '')
        if content_type == 'TextualInversion':
            image_path = os.path.join(install_path, 'images')
    make_dir(image_path)
    return image_path

def save_images(preview_html, model_filename, install_path, sub_folder, api_response=None):
    if not preview_html:
        return
    image_path = get_image_path(install_path, api_response, sub_folder)
    img_urls = re.findall(r'data-sampleimg="true" src=[\'"]?([^\'" >]+)', preview_html)

    if not img_urls:
        print('No images found to download.')
        return

    # Limit number of images to download
    img_count = getattr(opts, 'save_img_count', 16)
    img_count = max(4, min(64, img_count))
    img_urls = img_urls[:img_count]

    name = os.path.splitext(model_filename)[0]

    # Setup download
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    # Download images
    downloaded_count = 0
    for i, img_url in enumerate(img_urls):
        filename = f"{name}_{i}.png"
        img_url = urllib.parse.quote(img_url, safe=':/=')
        try:
            with urllib.request.urlopen(img_url) as url:
                image_data = url.read()

                # Check if resize is enabled for saved images
                resize_saved = getattr(opts, 'resize_preview_on_save', True)
                if resize_saved:
                    resize_size = getattr(opts, 'resize_preview_size', 512)
                    image_data = _resize_image_bytes(image_data, resize_size)

                img = Image.open(io.BytesIO(image_data))

                if img.mode in ('RGBA', 'LA', 'P'):
                    pass  # Keep transparency
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                save_path = os.path.join(image_path, filename)

                if IS_KAGGLE:
                    import sd_image_encryption
                    imginfo = img.info or {}
                    if not all(key in imginfo for key in ['Encrypt', 'EncryptPwdSha']):
                        sd_image_encryption.EncryptedImage.from_image(img).save(save_path)
                    else:
                        img.save(save_path, 'PNG')
                else:
                    img.save(save_path, 'PNG')

                print(f"Downloaded image: {filename}")
                downloaded_count += 1

        except urllib.error.URLError as e:
            print(f"Error downloading {filename}: {e.reason}")
        except Exception as e:
            print(f"Error processing image {filename}: {e}")

    if downloaded_count > 0:
        print(f"Successfully downloaded {downloaded_count} images to: {image_path}")
    else:
        print('No images were downloaded.')

def card_update(gr_components, model_name, list_versions, is_install):
    if gr_components:
        version_choices = gr_components['choices']
    else:
        print("Couldn't retrieve version, defaulting to installed")
        model_name += '.New'
        return model_name, None, None

    if is_install and not gl.download_fail and not gl.cancel_status:
        version_value_clean = list_versions + ' [Installed]'
        version_choices_clean = [
            version if version + ' [Installed]' != version_value_clean else version_value_clean
            for version in version_choices
        ]
    else:
        version_value_clean = list_versions.replace(' [Installed]', '')
        version_choices_clean = [
            version if version.replace(' [Installed]', '') != version_value_clean else version_value_clean
            for version in version_choices
        ]

    if not version_choices_clean:
        return model_name, list_versions, version_choices

    first_version_installed = '[Installed]' in version_choices_clean[0]
    any_later_version_installed = any('[Installed]' in version for version in version_choices_clean[1:])

    if first_version_installed:
        model_name += '.New'
    elif any_later_version_installed:
        model_name += '.Old'
    else:
        model_name += '.None'

    return model_name, version_value_clean, version_choices_clean

def list_files(folders):
    model_files = []
    extensions = ['.pt', '.ckpt', '.pth', '.safetensors', '.th', '.zip', '.vae']

    for folder in folders:
        if folder and os.path.exists(folder):
            for root, _, files in os.walk(folder, followlinks=True):
                for file in files:
                    _, file_extension = os.path.splitext(file)
                    if file_extension.lower() in extensions:
                        model_files.append(os.path.join(root, file))

    model_files = sorted(list(set(model_files)))
    return model_files


def _detect_content_type_from_path(file_path):
    upscaler_folders = [
        _api.contenttype_folder('Upscaler', 'SwinIR'),
        _api.contenttype_folder('Upscaler', 'RealESRGAN'),
        _api.contenttype_folder('Upscaler', 'GFPGAN'),
        _api.contenttype_folder('Upscaler', 'BSRGAN'),
        _api.contenttype_folder('Upscaler', 'ESRGAN')
    ]
    for folder in upscaler_folders:
        if folder and file_path.startswith(str(folder)):
            return 'Upscaler'

    content_types = [
        'Checkpoint', 'TextualInversion', 'LORA', 'Poses', 'Controlnet', 'Detection',
        'VAE', 'Wildcards', 'AestheticGradient', 'MotionModule', 'Workflows', 'Other'
    ]
    for content_type in content_types:
        folder = _api.contenttype_folder(content_type)
        if folder and file_path.startswith(str(folder)):
            return content_type

    return 'Other'


def _build_local_fallback_browser_item(file_path):
    file_name = os.path.basename(file_path)
    model_name = os.path.splitext(file_name)[0]
    content_type = _detect_content_type_from_path(file_path)
    local_id = -(abs(hash(os.path.abspath(file_path))) % 2000000000 + 1)

    file_sha = ''
    json_file = os.path.splitext(file_path)[0] + '.json'
    if os.path.exists(json_file):
        data = _api.safe_json_load(json_file)
        if data:
            file_sha = str(data.get('sha256') or '').upper().strip()

    if not file_sha:
        try:
            file_sha = str(gen_sha256(file_path) or '').upper().strip()
        except Exception:
            file_sha = ''

    try:
        size_kb = max(1, int(os.path.getsize(file_path) / 1024))
        mtime = os.path.getmtime(file_path)
    except Exception:
        size_kb = 1
        mtime = time.time()

    published_at = time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime(mtime))

    return {
        'id': local_id,
        'name': model_name,
        'type': content_type,
        'description': 'Local file without CivitAI match',
        'creator': {'username': 'Local Library', 'image': 'https://rawcdn.githack.com/gist/BlafKing/8d3f7a19e3f72cfddab46ae835037ee6/raw/296e81afbdd268200278beef478f3018b15936de/profile_placeholder.svg'},
        'tags': ['local-only'],
        'allowNoCredit': False,
        'allowCommercialUse': [],
        'allowDerivatives': False,
        'allowDifferentLicense': False,
        'local_only': True,
        'local_file_path': file_path,
        'modelVersions': [{
            'id': local_id,
            'name': 'Local file',
            'baseModel': 'Local',
            'publishedAt': published_at,
            'availability': 'Unknown',
            'trainedWords': [],
            'images': [],
            'downloadUrl': '',
            'files': [{
                'name': file_name,
                'sizeKB': size_kb,
                'downloadUrl': '',
                'hashes': {'SHA256': file_sha},
                'metadata': {'size': 'Unknown', 'format': 'SafeTensor', 'fp': 'Unknown'},
                'primary': True
            }]
        }]
    }

def gen_sha256(file_path):
    json_file = os.path.splitext(file_path)[0] + '.json'

    if os.path.exists(json_file):
        data = _api.safe_json_load(json_file)
        if data and 'sha256' in data and data['sha256']:
            return data['sha256']

    def read_chunks(file, size=io.DEFAULT_BUFFER_SIZE):
        while True:
            chunk = file.read(size)
            if not chunk:
                break
            yield chunk

    blocksize = 1 << 20
    h = hashlib.sha256()
    length = 0
    with open(os.path.realpath(file_path), 'rb') as f:
        for block in read_chunks(f, size=blocksize):
            length += len(block)
            h.update(block)

    hash_value = h.hexdigest()

    if os.path.exists(json_file):
        data = _api.safe_json_load(json_file)
        if data:
            data['sha256'] = hash_value
        else:
            data = {'sha256': hash_value}
    else:
        data = {'sha256': hash_value}

    _api.safe_json_save(json_file, data)

    return hash_value


def _normalize_sha256(sha256_value):
    if not sha256_value:
        return None
    sha = str(sha256_value).strip().lower()
    if len(sha) != 64:
        return None
    if not re.fullmatch(r'[0-9a-f]{64}', sha):
        return None
    return sha


def _load_checkpoint_hash_db():
    data = _api.safe_json_load(_CHECKPOINT_HASH_DB_PATH)
    if not isinstance(data, dict):
        return {'version': 1, 'checkpoints': {}}

    checkpoints = data.get('checkpoints', {})
    if not isinstance(checkpoints, dict):
        checkpoints = {}

    return {
        'version': 1,
        'checkpoints': checkpoints
    }


def _save_checkpoint_hash_db(db_data):
    os.makedirs(_CHECKPOINT_HASH_DB_PATH.parent, exist_ok=True)
    _api.safe_json_save(_CHECKPOINT_HASH_DB_PATH, db_data)


def _is_checkpoint_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    return ext in ('.safetensors', '.ckpt')


def _checkpoint_cache_key(file_path):
    checkpoint_root = _api.contenttype_folder('Checkpoint')
    relative = os.path.basename(file_path)

    if checkpoint_root:
        try:
            relative = os.path.relpath(file_path, checkpoint_root)
        except Exception:
            relative = os.path.basename(file_path)

    relative = relative.replace('\\', '/')
    return f'checkpoint/{relative}'


def _upsert_forge_hash_cache(file_path, sha256_value, add_only=True):
    cache_key = _checkpoint_cache_key(file_path)
    normalized_sha = _normalize_sha256(sha256_value)
    if not normalized_sha:
        return False, cache_key

    try:
        from modules import hashes as _hashes
        cache_store = _hashes.cache('hashes')
        existing = cache_store.get(cache_key)

        if add_only and existing:
            return False, cache_key

        cache_store[cache_key] = {
            'mtime': os.path.getmtime(file_path),
            'sha256': normalized_sha
        }
        _hashes.dump_cache()
        return True, cache_key
    except Exception as e:
        debug_print(f"[SHA256 sync] Failed to update Forge hash cache for '{file_path}': {e}")
        return False, cache_key


def _cleanup_deleted_checkpoints(db_data, existing_paths):
    removed_count = 0
    removed_cache_count = 0
    existing_set = set(existing_paths)

    try:
        from modules import hashes as _hashes
        cache_store = _hashes.cache('hashes')
    except Exception:
        cache_store = None
        _hashes = None

    checkpoints = db_data.get('checkpoints', {})
    for tracked_path in list(checkpoints.keys()):
        if tracked_path in existing_set:
            continue

        removed_count += 1
        entry = checkpoints.pop(tracked_path, {})
        cache_key = entry.get('cache_key')

        if cache_store is not None and cache_key and cache_key in cache_store:
            try:
                del cache_store[cache_key]
                removed_cache_count += 1
            except Exception:
                pass

    if removed_cache_count > 0 and _hashes is not None:
        try:
            _hashes.dump_cache()
        except Exception:
            pass

    return removed_count, removed_cache_count


def sync_checkpoint_sha256_on_download(file_path, sha256_value, model_id=None, model_version_id=None):
    if not file_path or not os.path.exists(file_path) or not _is_checkpoint_file(file_path):
        return

    normalized_sha = _normalize_sha256(sha256_value)
    if not normalized_sha:
        return

    abs_path = os.path.abspath(file_path)
    db_data = _load_checkpoint_hash_db()
    was_added, cache_key = _upsert_forge_hash_cache(abs_path, normalized_sha, add_only=False)

    db_data['checkpoints'][abs_path] = {
        'sha256': normalized_sha,
        'mtime': os.path.getmtime(abs_path),
        'modelId': model_id,
        'modelVersionId': model_version_id,
        'cache_key': cache_key,
        'synced_to_forge': True,
        'last_synced': int(time.time())
    }
    _save_checkpoint_hash_db(db_data)

    if was_added:
        debug_print(f"[SHA256 sync] Updated Forge cache for checkpoint: {os.path.basename(abs_path)}")


def sync_checkpoint_sha256_cache(progress=gr.Progress() if queue else None):
    checkpoint_root = _api.contenttype_folder('Checkpoint')
    if not checkpoint_root or not os.path.exists(checkpoint_root):
        return gr.update(value='<div style="color: var(--error-text-color);">Checkpoint folder not found.</div>')

    checkpoint_files = []
    for root, _, files in os.walk(checkpoint_root, followlinks=True):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if _is_checkpoint_file(file_path):
                checkpoint_files.append(os.path.abspath(file_path))

    if not checkpoint_files:
        return gr.update(value='<div style="color: var(--warning-text-color);">No checkpoints found to sync.</div>')

    checkpoint_files = sorted(list(set(checkpoint_files)))
    db_data = _load_checkpoint_hash_db()

    removed_entries, removed_cache_entries = _cleanup_deleted_checkpoints(db_data, checkpoint_files)

    added_count = 0
    skipped_existing = 0
    missing_sha = 0
    failed_count = 0
    total_files = len(checkpoint_files)

    for idx, file_path in enumerate(checkpoint_files, start=1):
        if progress is not None:
            progress(idx / total_files, desc=f"Syncing checkpoint hashes... {idx}/{total_files}")

        sidecar_path = os.path.splitext(file_path)[0] + '.json'
        sidecar = _api.safe_json_load(sidecar_path) if os.path.exists(sidecar_path) else {}
        sidecar = sidecar if isinstance(sidecar, dict) else {}

        sha_from_json = _normalize_sha256(sidecar.get('sha256'))
        if not sha_from_json:
            missing_sha += 1
            continue

        was_added, cache_key = _upsert_forge_hash_cache(file_path, sha_from_json, add_only=True)
        if was_added:
            added_count += 1
        else:
            skipped_existing += 1

        if not cache_key:
            failed_count += 1
            continue

        db_data['checkpoints'][file_path] = {
            'sha256': sha_from_json,
            'mtime': os.path.getmtime(file_path),
            'modelId': sidecar.get('modelId'),
            'modelVersionId': sidecar.get('modelVersionId'),
            'cache_key': cache_key,
            'synced_to_forge': True,
            'last_synced': int(time.time())
        }

    _save_checkpoint_hash_db(db_data)

    summary = (
        '<div style="line-height: 1.5; padding: 8px 0;">'
        f'<strong>SHA256 cache sync complete.</strong><br>'
        f'Checkpoints scanned: {total_files}<br>'
        f'Added to Forge cache: {added_count}<br>'
        f'Already in Forge cache: {skipped_existing}<br>'
        f'Missing SHA256 in sidecar: {missing_sha}<br>'
        f'Removed missing files from local DB: {removed_entries}<br>'
        f'Removed stale Forge cache entries: {removed_cache_entries}'
    )

    if failed_count > 0:
        summary += f'<br>Failed entries: {failed_count}'

    summary += '</div>'
    return gr.update(value=summary)

def convert_local_images(html):
    soup = BeautifulSoup(html)
    for simg in soup.find_all('img', attrs={'data-sampleimg': 'true'}):
        url = urlparse(simg['src'])
        path = url.path
        if not os.path.exists(path):
            print(f"URL path does not exist: {url.path}")
            # Try the raw url, files can be saved in windows as "C:\..." and
            # that confuses urlparse because people only really test on Linux.
            if os.path.exists(simg['src']):
                path = simg['src']
            else:
                continue
        with open(path, 'rb') as f:
            imgdata = f.read()
        b64img = base64.b64encode(imgdata).decode('utf-8')
        imgtype = Image.open(io.BytesIO(imgdata)).format
        if not imgtype:
            imgtype = 'PNG'
        simg['src'] = f"data:image/{imgtype};base64,{b64img}"
    return str(soup)

def _get_cached_html_stripped(model_file) -> str | None:
    """Return stripped (no <head> section) content from the local .html cache, or None if absent."""
    if not model_file:
        return None
    html_file = os.path.splitext(model_file)[0] + '.html'
    if not os.path.exists(html_file):
        return None
    with open(html_file, 'r', encoding='utf-8') as f:
        html = f.read()
    index = html.find('</head>')
    if index != -1:
        html = html[index + len('</head>'):]
    return html


def _wrap_html_with_css(body: str) -> str:
    """Prepend the overlay stylesheet to a body HTML string."""
    css_path = Path(__file__).resolve().parents[1] / 'style_html.css'
    with open(css_path, 'r', encoding='utf-8') as css_file:
        css = css_file.read()
    return f'<head><style>{css}</style></head>{body}'


def model_from_sent(model_name, content_type):
    modelID_failed = False
    output_html = None
    model_file = None
    use_local_html = getattr(opts, 'use_local_html', False)
    local_path_in_html = getattr(opts, 'local_path_in_html', False)

    model_name = re.sub(r'\.\d{3}$', '', model_name)
    content_type = re.sub(r'\.\d{3}$', '', content_type).lower()
    if 'inversion' in content_type:
        content_type = ['TextualInversion']
    elif 'checkpoint' in content_type:
        content_type = ['Checkpoint']
    elif 'lora' in content_type:
        content_type = ['LORA']
    elif 'detection' in content_type:
        content_type = ['Detection']

    extensions = ['.pt', '.ckpt', '.pth', '.safetensors', '.th', '.zip', '.vae']

    for content_type_item in content_type:
        folder = _api.contenttype_folder(content_type_item)
        for folder_path, _, files in os.walk(folder, followlinks=True):
            for file in files:
                if file.startswith(model_name) and file.endswith(tuple(extensions)):
                    model_file = os.path.join(folder_path, file)

    if not model_file:
        output_html = _api.api_error_msg('path_not_found')
        print(f"Error: Could not find model path for model: '{model_name}'")
        print(f"Content type: '{content_type}'")
        print(f"Main folder path: '{folder}'")
        use_local_html = False

    if use_local_html:
        html_file = os.path.splitext(model_file)[0] + '.html'
        if os.path.exists(html_file):
            with open(html_file, 'r', encoding='utf-8') as html:
                output_html = html.read()
                index = output_html.find('</head>')
                if index != -1:
                    output_html = output_html[index + len('</head>'):]
                if local_path_in_html:
                    output_html = convert_local_images(output_html)

    if not output_html:
        api_response = None
        modelID = get_models(model_file, True)
        if not modelID or modelID == 'Model not found':
            # SHA256 lookup returned 404 — check for local cached HTML before giving up
            cached = _get_cached_html_stripped(model_file)
            if cached is not None:
                return gr.update(value=_wrap_html_with_css(_api.inject_removed_banner(cached)), placeholder=_download.random_number()),  # Preview HTML
            output_html = _api.api_error_msg('not_found')
            modelID_failed = True
        if modelID == 'offline':
            output_html = _api.api_error_msg('offline')
            modelID_failed = True
        if not modelID_failed:
            api_response = _api.request_civit_api(f"https://{_api.get_civitai_domain()}/api/v1/models?ids={modelID}&nsfw=true")
        if modelID_failed or api_response in ['timeout', 'error', 'offline']:
            return gr.update(value='<p>ERROR</p>', placeholder=_download.random_number()),  # Preview HTML
        if api_response == 'not_found':
            # Model was removed from CivitAI after being cached locally — show cached HTML with banner
            cached = _get_cached_html_stripped(model_file)
            body = _api.inject_removed_banner(cached) if cached is not None else _api.api_error_msg('removed')
            return gr.update(value=_wrap_html_with_css(body), placeholder=_download.random_number()),  # Preview HTML

        # Get SHA256 hash for the file to find the specific version
        file_sha256 = None
        json_file = os.path.splitext(model_file)[0] + '.json'
        if os.path.exists(json_file):
            data = _api.safe_json_load(json_file)
            file_sha256 = data.get('sha256') if data else None
        # If SHA256 not cached, compute it now (also saves it to .json for future use)
        if not file_sha256 and os.path.exists(model_file):
            try:
                file_sha256 = gen_sha256(model_file)
            except Exception:
                pass
        # Find the specific model version based on SHA256 or filename
        if file_sha256:
            model_version, item = find_model_version_by_sha256(api_response, file_sha256)
        else:
            model_version, item = find_model_version_by_filename(api_response, model_file)
        if model_version and item:
            # Use the specific model version name for HTML generation
            output_html = _api.update_model_info(None, model_version.get('name'), True, modelID, api_response, True)
        else:
            # Fallback to first version if specific version not found
            model_versions = _api.update_model_versions(modelID, api_response)
            output_html = _api.update_model_info(None, model_versions.get('value'), True, modelID, api_response, True)

    css_path = Path(__file__).resolve().parents[1] / 'style_html.css'
    with open(css_path, 'r', encoding='utf-8') as css_file:
        css = css_file.read()

    style_tag = f'<style>{css}</style>'
    head_section = f'<head>{style_tag}</head>'
    output_html = str(head_section + output_html)

    # debug_print(output_html)
    return gr.update(value=output_html, placeholder=_download.random_number()),  # Preview HTML

def send_to_browser(model_name, content_type, click_first_item):
    modelID_failed = False
    output_html = None
    model_file = None
    number = click_first_item

    model_name = re.sub(r'\.\d{3}$', '', model_name)
    content_type = re.sub(r'\.\d{3}$', '', content_type).lower()
    if 'inversion' in content_type:
        content_type = ['TextualInversion']
    elif 'checkpoint' in content_type:
        content_type = ['Checkpoint']
    elif 'lora' in content_type:
        content_type = ['LORA']
    extensions = ['.pt', '.ckpt', '.pth', '.safetensors', '.th', '.zip', '.vae']

    for content_type_item in content_type:
        folder = _api.contenttype_folder(content_type_item)
        for folder_path, _, files in os.walk(folder, followlinks=True):
            for file in files:
                if file.startswith(model_name) and file.endswith(tuple(extensions)):
                    model_file = os.path.join(folder_path, file)

    if not model_file:
        output_html = _api.api_error_msg('path_not_found')
        print(f"Error: Could not find model path for model: '{model_name}'")
        print(f"Content type: '{content_type}'")
        print(f"Main folder path: '{folder}'")
    if not output_html:
        modelID = get_models(model_file, True)
        if not modelID or modelID == 'Model not found':
            output_html = _api.api_error_msg('not_found')
            modelID_failed = True
        if modelID == 'offline':
            output_html = _api.api_error_msg('offline')
            modelID_failed = True

        if not modelID_failed:
            gl.json_data = _api.request_civit_api(f"https://{_api.get_civitai_domain()}/api/v1/models?ids={modelID}&nsfw=true")
            output_html = _api.model_list_html(gl.json_data)
            number = _download.random_number(click_first_item)

    return (
        gr.update(value=output_html),  # Card HTML
        gr.update(interactive=False),   # Prev Button
        gr.update(interactive=False),   # Next Button
        gr.update(value=1, maximum=1),  # Page Slider
        gr.update(value=number)        # Click first card trigger
    )

def convertCustomFolder(folderValue, basemodel, nsfw, author, modelName, modelId, versionName, versionId):
    replacements = {
        'BASEMODEL': _api.cleaned_name(str(basemodel)),
        'AUTHOR': _api.cleaned_name(str(author)),
        'MODELNAME': _api.cleaned_name(str(modelName)),
        'MODELID': _api.cleaned_name(str(modelId)),
        'VERSIONNAME': _api.cleaned_name(str(versionName)),
        'VERSIONID': _api.cleaned_name(str(versionId))
    }

    if not nsfw:
        segments = folderValue.split(os.sep)
        segments = [seg for seg in segments if "{NSFW}" not in seg]
        folderValue = os.sep.join(segments)
    else:
        replacements['NSFW'] = 'nsfw'

    formatted_value = folderValue.format(**replacements)

    converted_folder = formatted_value.replace('/', os.sep).replace('\\', os.sep)
    converted_folder = os.sep.join(part for part in converted_folder.split(os.sep) if part)

    if not converted_folder.startswith(os.sep):
        converted_folder = os.sep + converted_folder

    return converted_folder

def getSubfolders(model_folder, basemodel=None, nsfw=None, author=None, modelName=None, modelId=None, versionName=None, versionId=None):
    try:
        dot_subfolders = getattr(opts, 'dot_subfolders', True)
        sub_folders = ['None']
        for root, dirs, _ in os.walk(model_folder, followlinks=True):
            if dot_subfolders:
                dirs = [d for d in dirs if not d.startswith('.')]
                dirs = [d for d in dirs if not any(part.startswith('.') for part in os.path.join(root, d).split(os.sep))]
            for d in dirs:
                sub_folder = os.path.relpath(os.path.join(root, d), model_folder)
                if sub_folder:
                    if not sub_folder.startswith(os.sep):
                        sub_folder = os.sep + sub_folder
                    sub_folders.append(sub_folder)

        config_data = _api.safe_json_load(gl.subfolder_json) or {}

        for key, value in config_data.items():
            # Skip timestamp field and non-string values
            if key == 'created_at' or not isinstance(value, str):
                continue

            if basemodel:
                try:
                    converted_value = convertCustomFolder(value, basemodel, nsfw, author, modelName, modelId, versionName, versionId)
                    sub_folders.append(converted_value)
                except Exception as e:
                    print(f"Error: Failed to process custom subfolder: {e}")
            else:
                display_value = value
                if not display_value.startswith(os.sep):
                    display_value = os.sep + display_value
                sub_folders.append(display_value)

        sub_folders.remove('None')
        sub_folders = sorted(sub_folders, key=lambda x: (x.lower(), x))
        sub_folders.insert(0, 'None')

    except Exception as e:
        print(e)
        sub_folders = ['None']

    list = set()
    sub_folders = [x for x in sub_folders if not (x in list or list.add(x))]

    return sub_folders

def updateSubfolder(subfolderInput):
    data = _api.safe_json_load(gl.subfolder_json) or {}
    index, action, value = subfolderInput.split('.', 2)
    index = str(index)

    if action == 'delete':
        data.pop(index, None)
    elif action == 'add':
        data[index] = value

    _api.safe_json_save(gl.subfolder_json, data)

def is_image_url(url):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    parsed = urlparse(url)
    return any(parsed.path.endswith(ext) for ext in image_extensions)

## === ANXETY EDITs ===
def clean_description(desc):
    """This function cleans up HTML descriptions for better readability"""
    try:
        # Flatten to single-line string
        cleaned_lines = [line.strip() for line in desc.splitlines() if line.strip()]
        cleaned_text = ''.join(cleaned_lines)
        cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
        # Begin html processing
        soup = BeautifulSoup(cleaned_text, 'html.parser')
        for a in soup.find_all('a', href=True):
            hyperlink_url = a['href']
            if not is_image_url(hyperlink_url):
                # Add the URL to the text if they are different
                a.replace_with(a.text + (f' ({hyperlink_url})' if a.text != hyperlink_url else ''))
        # Apply markdown-like formatting and newlines for various blocks
        for e in soup.find_all(['br']):
            e.replace_with('\n')
        for e in soup.find_all(['hr']):
            e.replace_with('\n\n')
        for e in soup.find_all(['li']):
            if e.text.strip():
                e.insert_before('- ')
                e.insert_after('\n')
                e.unwrap()
            else:
                e.replace_with('\n')
        for e in soup.find_all(['s']):
            if e.text.strip():
                e.insert_before('~~')
                e.insert_after('~~')
                e.unwrap()
            else:
                e.replace_with('')
        for e in soup.find_all(['p', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if e.text.strip():
                e.insert_after('\n\n')
                e.unwrap()
            else:
                e.replace_with('\n\n')
        # Convert back to plaintext
        cleaned_text = soup.get_text()
        # Clean extra characters
        cleaned_text = re.sub(r'~{3,}', '', cleaned_text)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    except ImportError:
        print('Python module "BeautifulSoup" was not imported correctly, cannot clean description. Please try to restart or install it manually.')
        cleaned_text = desc

    return cleaned_text.strip()

def make_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EACCES:
            try:
                os.makedirs(path, mode=0o777)
            except OSError as e2:
                if e2.errno == errno.EACCES:
                    print('Permission denied even with elevated permissions.')
                else:
                    print(f"Error creating directory: {e2}")
        else:
            print(f"Error creating directory: {e}")
    except Exception as e:
        print(f"Error creating directory: {e}")

## === ANXETY EDITs ===
def save_model_info(install_path, file_name, sub_folder, sha256=None, preview_html=None, overwrite_toggle=False, api_response=None):
    save_path, filename = get_save_path_and_name(install_path, file_name, api_response, sub_folder)
    image_path = get_image_path(install_path, api_response, sub_folder)
    json_file = os.path.join(install_path, f'{filename}.json')
    make_dir(install_path)

    save_api_info = getattr(opts, 'save_api_info', False)
    use_local = getattr(opts, 'local_path_in_html', False)
    save_html_on_save = getattr(opts, 'save_html_on_save', False)

    if not api_response:
        api_response = gl.json_data

    # Try to find SHA256 from existing JSON file if not provided
    if not sha256:
        existing_json_file = os.path.splitext(os.path.join(install_path, file_name))[0] + '.json'
        if os.path.exists(existing_json_file):
            data = _api.safe_json_load(existing_json_file)
            sha256 = data.get('sha256') if data else None

    result = find_and_save(api_response, sha256, file_name, json_file, False, overwrite_toggle)
    if result != 'found':
        result = find_and_save(api_response, sha256, file_name, json_file, True, overwrite_toggle)

    if preview_html and save_html_on_save:
        if use_local:
            img_urls = re.findall(r"data-sampleimg='true' src=[\'\"]?([^\'\" >]+)", preview_html)
            for i, img_url in enumerate(img_urls):
                debug_print(img_url)
                img_name = f'{filename}_{i}.png'
                preview_html = preview_html.replace(img_url, f"{os.path.join(image_path, img_name)}")

        match = re.search(r'(\s*)<div class="main-container">', preview_html)
        if match:
            indentation = match.group(1)
        else:
            indentation = ''
        css_link = f'<link rel="stylesheet" type="text/css" href="{css_path}">'
        utf8_meta_tag = f'{indentation}<meta charset="UTF-8">'
        head_section = f'{indentation}<head>{indentation}    {utf8_meta_tag}{indentation}    {css_link}{indentation}</head>'
        HTML = head_section + preview_html
        path_to_new_file = os.path.join(save_path, f'{filename}.html')
        with open(path_to_new_file, 'wb') as f:
            f.write(HTML.encode('utf8'))
        print(f"HTML saved at: {path_to_new_file}")

    # Always save .api_info.json — this is the source of truth for organization.
    # We fetch the version-specific response via by-hash so that 'baseModel' is
    # at the root level, making extraction simple and reliable.
    # Respects overwrite_toggle: won't overwrite an existing file unless the user
    # explicitly requested it (same behaviour as the .json sidecar).
    # Falls back to gl.json_info (full model object) if the hash lookup fails.
    api_info_path = os.path.join(save_path, f'{filename}.api_info.json')
    if not os.path.exists(api_info_path) or overwrite_toggle:
        version_data = None
        try:
            model_file = os.path.join(save_path, file_name)
            if os.path.exists(model_file):
                file_hash = gen_sha256(model_file)
                if file_hash:
                    normalized = _api.normalize_sha256(file_hash)
                    by_hash_url = f"https://{_api.get_civitai_domain()}/api/v1/model-versions/by-hash/{normalized}"
                    headers = _api.get_headers()
                    proxies, ssl_verify = _api.get_proxies()
                    resp = requests.get(by_hash_url, headers=headers, timeout=(60, 30), proxies=proxies, verify=ssl_verify)
                    if resp.status_code == 200:
                        data = resp.json()
                        if 'error' not in data:
                            version_data = data
        except Exception as e:
            pass  # fall through to gl.json_info below
        _api.safe_json_save(api_info_path, version_data if version_data else gl.json_info)
        print(f"[CivitAI Browser Neo] - API info saved to: {api_info_path}")

def find_model_version_by_sha256(api_response, sha256):
    """Find the specific model version that matches the given SHA256 hash"""
    for item in api_response.get('items', []):
        for model_version in item.get('modelVersions', []):
            for file in model_version.get('files', []):
                file_sha256 = file.get('hashes', {}).get('SHA256', '')
                if _api.normalize_sha256(file_sha256) == _api.normalize_sha256(sha256):
                    return model_version, item
    return None, None

def find_model_version_by_filename(api_response, file_name):
    """Find the specific model version that matches the given filename"""
    for item in api_response.get('items', []):
        for model_version in item.get('modelVersions', []):
            for file in model_version.get('files', []):
                file_name_api = file.get('name', '')
                if file_name == file_name_api:
                    return model_version, item
    return None, None

# ─────────────────────────────────────────────────────────────────────────────
# Trigger Word Consolidation (v0.8.1)
# ─────────────────────────────────────────────────────────────────────────────

def extract_safetensors_metadata(file_path):
    """Extract trigger words/metadata from a .safetensors file."""
    if not file_path or not os.path.exists(file_path):
        return []

    def _split_tags(value):
        # Keep multi-word tags intact; split only on common list separators.
        if not isinstance(value, str):
            return []
        return [t.strip() for t in re.split(r'[,;\n\r]+', value) if t.strip()]

    try:
        # Manual parsing of safetensors format:
        # Header format: first 8 bytes = header length (little-endian uint64)
        # followed by header JSON, then tensor data
        with open(file_path, 'rb') as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                return []
            
            header_size = int.from_bytes(header_size_bytes, byteorder='little')
            header_json = f.read(header_size).decode('utf-8', errors='ignore')
            header_dict = json.loads(header_json)
            
            # Look for metadata in the header
            metadata = header_dict.get('__metadata__', {})
            if isinstance(metadata, dict):
                # Try common keys for trigger words
                for key in ['activation_text', 'activation text', 'trigger_words', 'trigger words', 'tags']:
                    if key in metadata:
                        val = metadata[key]
                        if isinstance(val, list):
                            return [str(t).strip() for t in val if str(t).strip()]
                        tags = _split_tags(val)
                        if tags:
                            return tags
            
            return []
    except Exception:
        return []

def consolidate_trigger_words(safetensors_tags=None, json_tags=None, api_tags=None):
    """Consolidate trigger words from 3 sources (safetensors, local json, api) with deduplication."""
    def _split_tags(value):
        if isinstance(value, list):
            return [str(t).strip() for t in value if str(t).strip()]
        if isinstance(value, str):
            return [t.strip() for t in re.split(r'[,;\n\r]+', value) if t.strip()]
        return []

    # Normalize inputs
    sources = []
    sources.extend(_split_tags(safetensors_tags))
    sources.extend(_split_tags(json_tags))
    sources.extend(_split_tags(api_tags))
    
    # Deduplicate while preserving order: use dict keys (3.7+ Python preserves insertion order)
    seen = {}
    for tag in sources:
        tag_lower = tag.lower()
        if tag_lower not in seen:
            seen[tag_lower] = tag
    
    consolidated = list(seen.values())
    return consolidated

## === ANXETY EDITs ===
def find_and_save(api_response, sha256=None, file_name=None, json_file=None, no_hash=None, overwrite_toggle=None):
    save_desc = getattr(opts, 'model_desc_to_json', True)

    # Find the specific model version based on SHA256 or filename
    if no_hash:
        model_version, item = find_model_version_by_filename(api_response, file_name)
    else:
        model_version, item = find_model_version_by_sha256(api_response, sha256)

    if model_version and item:
        gl.json_info = item
        trained_words = model_version.get('trainedWords', [])

        def _sanitize_group_text(value):
            text = str(value) if value is not None else ''
            text = re.sub(r'<[^>]*:[^>]*>', '', text)
            text = re.sub(r',\s*', ', ', text)
            return text.strip(', ').strip()

        # Preserve original API grouping for UI rows (same shape as CivitAI).
        api_groups = []
        _seen_groups = set()
        if isinstance(trained_words, list):
            for group in trained_words:
                cleaned = _sanitize_group_text(group)
                key = cleaned.lower()
                if cleaned and key not in _seen_groups:
                    _seen_groups.add(key)
                    api_groups.append(cleaned)
        elif isinstance(trained_words, str):
            cleaned = _sanitize_group_text(trained_words)
            if cleaned:
                api_groups = [cleaned]

        if save_desc:
            description = item.get('description', '')
            if description is not None and description.strip():
                ver_description = model_version.get('description', '')
                # Include "About This Version" if available
                if ver_description is not None and ver_description.strip():
                    description += '\n<p>About this version:</p>\n' + ver_description
                description = clean_description(description)

        base_model = model_version.get('baseModel', '')
        
        # Save the RAW CivitAI baseModel value (e.g. "NoobAI", "Pony", "Illustrious")
        # normalize_base_model() is only used for folder placement, never for .json storage
        if not base_model:
            base_model = 'Other'

        # ─────────────────────────────────────────────────────────────────────────────
        # v0.8.1 — Consolidate trigger words from 3 sources
        # ─────────────────────────────────────────────────────────────────────────────
        api_tags = ', '.join(api_groups) if api_groups else ''

        # Load existing json to get any previously saved activation text
        content = _api.safe_json_load(json_file) or {}
        json_tags = content.get('activation text', '')
        
        # Try to extract metadata from local model file (if it exists)
        safetensors_tags = []
        model_file_path = None
        if json_file and file_name:
            model_file_path = os.path.join(os.path.dirname(json_file), file_name)
        if not model_file_path and json_file:
            json_stem = os.path.splitext(json_file)[0]
            for ext in ['.safetensors', '.ckpt', '.pt', '.bin', '.pth']:
                candidate = f'{json_stem}{ext}'
                if os.path.exists(candidate):
                    model_file_path = candidate
                    break
        if model_file_path and model_file_path.lower().endswith('.safetensors'):
            safetensors_tags = extract_safetensors_metadata(model_file_path)
        
        # Consolidate all 3 sources
        consolidated_tags = consolidate_trigger_words(
            safetensors_tags=safetensors_tags,
            json_tags=json_tags,
            api_tags=api_tags
        )
        trained_tags = ', '.join(consolidated_tags) if consolidated_tags else ''

        changed = False
        if overwrite_toggle == False:
            if 'activation text' not in content:
                content['activation text'] = trained_tags
                changed = True
            if api_groups and 'activation text groups' not in content:
                content['activation text groups'] = api_groups
                changed = True
            if save_desc and ('description' not in content):
                content['description'] = description
                changed = True
            if 'sd version' not in content:
                content['sd version'] = base_model
                changed = True
            # Add new fields for model and version information
            if 'modelId' not in content:
                content['modelId'] = item.get('id')
                changed = True
            if 'modelVersionId' not in content:
                content['modelVersionId'] = model_version.get('id')
                changed = True
            if 'modelPageURL' not in content:
                content['modelPageURL'] = f"https://{_api.get_civitai_domain()}/models/{item.get('id')}?modelVersionId={model_version.get('id')}"
                changed = True
        else:
            content['activation text'] = trained_tags
            if api_groups:
                content['activation text groups'] = api_groups
            if save_desc:
                content['description'] = description
            content['sd version'] = base_model
            # Always update these fields when overwrite is enabled
            content['modelId'] = item.get('id')
            content['modelVersionId'] = model_version.get('id')
            content['modelPageURL'] = f"https://{_api.get_civitai_domain()}/models/{item.get('id')}?modelVersionId={model_version.get('id')}"
            changed = True

        _api.safe_json_save(json_file, content)

        if changed:
            print(f"Model info saved to: {json_file}")
        return 'found'

    return 'not found'

def get_models(file_path, gen_hash=None):
    modelId = None
    modelVersionId = None
    sha256 = None
    json_file = os.path.splitext(file_path)[0] + '.json'
    if os.path.exists(json_file):
        data = _api.safe_json_load(json_file)
        if data:
            modelId = data.get('modelId')
            modelVersionId = data.get('modelVersionId')
            sha256 = data.get('sha256')

    if not modelId or not modelVersionId or not sha256:
        if not sha256 and gen_hash:
            sha256 = gen_sha256(file_path)

        if sha256:
            by_hash = f"https://{_api.get_civitai_domain()}/api/v1/model-versions/by-hash/{sha256}"
        else:
            return modelId if modelId else None

    proxies, ssl = _api.get_proxies()
    try:
        if not modelId or not modelVersionId:
            response = requests.get(by_hash, timeout=(60, 30), proxies=proxies, verify=ssl)
            if response.status_code == 200:
                api_response = response.json()
                if 'error' in api_response:
                    print(f"{file_path}: {api_response['error']}")
                    return None
                else:
                    modelId = api_response.get('modelId', '')
                    modelVersionId = api_response.get('id', '')
            elif response.status_code == 503:
                return 'offline'
            elif response.status_code == 404:
                modelId = 'Model not found'
                modelVersionId = 'Model not found'

            if os.path.exists(json_file):
                data = _api.safe_json_load(json_file)
                if not data:
                    data = {}
            else:
                data = {}

            data.update({
                'modelId': modelId,
                'modelVersionId': modelVersionId,
                'modelPageURL': f"https://{_api.get_civitai_domain()}/models/{modelId}?modelVersionId={modelVersionId}",
                'sha256': sha256.upper()
            })
            _api.safe_json_save(json_file, data)

        return modelId
    except requests.exceptions.Timeout:
        print(f"Request timed out for {file_path}. Skipping...")
        return 'offline'
    except requests.exceptions.ConnectionError:
        print('Failed to connect to the API. The CivitAI servers might be offline.')
        return 'offline'
    except Exception as e:
        print(f"An error occurred for {file_path}: {str(e)}")
        return None

## === ANXETY EDITs ===
def extract_version_from_ver_name(filename):
    """
    Extracts the model family name and version from the model name string.
    Returns: (family_name or None, version_parts: list[int])
    """
    version_patterns = [
        r'[_\-]?v\.(\d+\.\d+)$',  # v.1.0, _v.2.1, -v.3.2
        r'[_\-]?v\.(\d+)$',       # v.1, _v.2, -v.3
        r'[_\-]?v(\d+\.\d+)$',   # v1.0, _v2.1, -v3.2
        r'[_\-]?v(\d+)$',         # v1, _v2, -v3
        # r'[_\-]?(\d+\.\d+)$',    # 1.0, _2.1, -3.2
        # r'[_\-]?(\d+)$',         # 1, _2, -3
    ]
    for pattern in version_patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            version_str = match.group(1)
            parts = [int(p) for p in version_str.split('.') if p.isdigit()]
            # Remove the matched part from the end of the string to get the family name
            family = filename[:match.start()].strip("_- .")
            # If family looks like a pure version (for example, 'v3'), treat it as None
            if not family or re.fullmatch(r'v?\d+(\.\d+)?', family, re.IGNORECASE):
                family = None

            return family, parts
    return None, []

def compare_version_parts(a_parts, b_parts):
    """
    Compare two version parts lists.
    Returns:
      -1 if a < b,
      0 if a == b,
      1 if a > b
    """
    max_len = max(len(a_parts), len(b_parts))
    a = a_parts + [0] * (max_len - len(a_parts))
    b = b_parts + [0] * (max_len - len(b_parts))
    return (a > b) - (a < b)

def version_match(file_paths, api_response, log=False):
    """
    Check which installed models have a newer version available on CivitAI.

    Strategy (avoids fragile version-string regex parsing):
    - Uses the CivitAI `baseModel` field per version as the "family" key.
    - Uses the position in `modelVersions` as the version indicator:
      index 0 = newest (CivitAI API guarantee). A higher index means older.
    - If `precise_version_check` is True (default): compares per-baseModel family.
      A model is outdated if any installed version-id is not the newest for its baseModel.
    - If False: a model is outdated if any installed version is not at index 0 overall.

    Falls back to SHA256 matching when modelVersionId is absent from the .json sidecar.
    """
    precise_check = getattr(opts, 'precise_version_check', True)
    updated_models = []
    outdated_models = []

    # === 1. Collect installed SHA256s and optionally cached modelVersionIds ===
    installed_hashes = set()           # str (upper) → present
    hash_to_cached_ver_id = {}         # sha256 → int modelVersionId (from .json, may be absent)

    for path in file_paths:
        json_path = f"{os.path.splitext(path)[0]}.json"
        data = _api.safe_json_load(json_path)
        if data:
            sha = data.get('sha256', '').upper()
            if sha:
                installed_hashes.add(sha)
                vid = data.get('modelVersionId')
                if vid:
                    try:
                        hash_to_cached_ver_id[sha] = int(vid)
                    except (ValueError, TypeError):
                        pass

    if log:
        print(f"[LOG] {len(installed_hashes)} installed hashes, "
              f"{len(hash_to_cached_ver_id)} with cached versionId")

    # === 2. Compare per model ===
    for model in api_response.get('items', []):
        model_id = model.get('id')
        model_name = model.get('name', '')
        model_versions = model.get('modelVersions', [])

        if not model_versions:
            continue

        # Map: versionId → (index, baseModel, ver_name)
        # Index 0 = newest version for the model (CivitAI API contract)
        ver_meta = {}  # ver_id -> {'idx': int, 'base': str, 'name': str}
        for idx, ver in enumerate(model_versions):
            vid = ver.get('id')
            if vid is not None:
                ver_meta[vid] = {
                    'idx': idx,
                    'base': (ver.get('baseModel') or '').strip(),
                    'name': ver.get('name', ''),
                }

        # For each baseModel, the newest available index is the minimum index
        # among all versions sharing that baseModel.
        base_to_newest_idx = {}   # baseModel → int (smallest index = newest)
        base_to_newest_name = {}  # baseModel → version name (for semantic comparison)
        for vm in ver_meta.values():
            bm = vm['base']
            if bm not in base_to_newest_idx or vm['idx'] < base_to_newest_idx[bm]:
                base_to_newest_idx[bm] = vm['idx']
                base_to_newest_name[bm] = vm['name']

        # Find which version IDs of THIS model are installed
        installed_ver_ids = []  # list of (ver_id, baseModel, ver_name)
        for ver in model_versions:
            vid = ver.get('id')
            found = False
            for file_entry in ver.get('files', []):
                sha = file_entry.get('hashes', {}).get('SHA256', '').upper()
                if sha in installed_hashes:
                    bm = (ver.get('baseModel') or '').strip()
                    vname = ver.get('name', '')
                    installed_ver_ids.append((vid, bm, vname))
                    if log:
                        print(f"[LOG] '{model_name}' ver '{vname}' (id={vid}, base='{bm}') is installed")
                    found = True
                    break
            if found:
                continue
            # Fallback: match via cached modelVersionId in .json
            if vid is not None:
                for sha, cached_vid in hash_to_cached_ver_id.items():
                    if cached_vid == vid and sha in installed_hashes:
                        bm = (ver.get('baseModel') or '').strip()
                        vname = ver.get('name', '')
                        installed_ver_ids.append((vid, bm, vname))
                        if log:
                            print(f"[LOG] '{model_name}' ver '{vname}' (id={vid}, base='{bm}') "
                                  f"matched via cached versionId")
                        break

        if not installed_ver_ids:
            continue

        has_outdated = False

        if precise_check:
            # Per-baseModel check: is the installed version the newest for its base?
            # Hybrid strategy: prefer semantic version comparison (V3 > V1) when both
            # version names contain clear numbers; fall back to array index otherwise.
            for vid, bm, vname in installed_ver_ids:
                inst_idx = ver_meta.get(vid, {}).get('idx', 0) if vid is not None else 0
                newest_idx = base_to_newest_idx.get(bm, 0)
                newest_name = base_to_newest_name.get(bm, '')

                # Try semantic comparison first
                _inst_family, inst_parts = extract_version_from_ver_name(vname)
                _newest_family, newest_parts = extract_version_from_ver_name(newest_name)
                semantic_outdated = None
                if inst_parts and newest_parts:
                    cmp = compare_version_parts(inst_parts, newest_parts)
                    semantic_outdated = cmp < 0  # installed < newest semantically
                    if log:
                        print(f"[LOG] '{model_name}' base='{bm}' ver='{vname}' vs '{newest_name}': "
                              f"semantic cmp={cmp}")

                if semantic_outdated is not None:
                    # Semantic comparison succeeded — trust it
                    if semantic_outdated:
                        has_outdated = True
                        if log:
                            print(f"[LOG] '{model_name}' base='{bm}' ver='{vname}': "
                                  f"outdated (semantic: {inst_parts} < {newest_parts})")
                    elif log:
                        print(f"[LOG] '{model_name}' base='{bm}' ver='{vname}': "
                              f"up-to-date (semantic: {inst_parts} >= {newest_parts})")
                else:
                    # Fallback to array index (original behavior)
                    if inst_idx > newest_idx:
                        has_outdated = True
                        if log:
                            print(f"[LOG] '{model_name}' base='{bm}' ver='{vname}': "
                                  f"outdated (idx {inst_idx} > newest {newest_idx})")
                    elif log:
                        print(f"[LOG] '{model_name}' base='{bm}' ver='{vname}': up-to-date")
        else:
            # Global check: is the installed version the globally newest (index 0)?
            for vid, bm, vname in installed_ver_ids:
                inst_idx = ver_meta.get(vid, {}).get('idx', 0) if vid is not None else 0
                if inst_idx > 0:
                    has_outdated = True
                    if log:
                        print(f"[LOG] '{model_name}' ver='{vname}': outdated (idx {inst_idx} > 0)")
                elif log:
                    print(f"[LOG] '{model_name}' ver='{vname}': up-to-date (idx 0)")

        model_type = model.get('type', 'Unknown')
        if has_outdated:
            outdated_models.append((f"&ids={model_id}", model_name, model_type))
        else:
            updated_models.append((f"&ids={model_id}", model_name, model_type))

    return updated_models, outdated_models


def collect_update_items(outdated_set, api_response, file_paths):
    """Build gl.update_items: one entry per outdated baseModel family per model.

    Uses the CivitAI `baseModel` field (reliable) and array index (0 = newest)
    instead of fragile version-string regex parsing.

    Returns a list of dicts:
        {'model_id', 'model_name', 'model_type', 'family',
         'installed_ver', 'installed_ver_id', 'latest_ver', 'latest_ver_id',
         'available_versions', 'preview_url'}
    Multi-family models (e.g. DollFace PONY + IL) produce two entries.
    """
    precise_check = getattr(opts, 'precise_version_check', True)

    # Collect installed SHA256 hashes and map each hash to its file path
    installed_hashes = set()
    sha_to_path = {}
    for path in file_paths:
        json_path = f"{os.path.splitext(path)[0]}.json"
        data = _api.safe_json_load(json_path)
        if data:
            sha = data.get('sha256', '')
            if sha:
                installed_hashes.add(sha.upper())
                sha_to_path[sha.upper()] = path

    outdated_ids = {int(entry[0].replace('&ids=', '')) for entry in outdated_set}

    items = []
    for model in api_response.get('items', []):
        model_id = model.get('id')
        if model_id not in outdated_ids:
            continue

        model_name = model.get('name', '')
        model_type = model.get('type', 'Unknown')
        model_versions = model.get('modelVersions', [])

        # First image URL (preview thumbnail)
        preview_url = None
        for ver in model_versions:
            for img in ver.get('images', []):
                url = img.get('url', '')
                if url:
                    preview_url = url
                    break
            if preview_url:
                break

        # Build index map: versionId → index (0 = newest)
        ver_id_to_idx = {ver.get('id'): idx for idx, ver in enumerate(model_versions)}

        # Find all installed versions for this model
        # installed_versions: list of (idx, baseModel, ver_name, ver_id, old_file)
        installed_versions = []
        for ver in model_versions:
            vid = ver.get('id')
            idx = ver_id_to_idx.get(vid, 0)
            bm = (ver.get('baseModel') or '').strip()
            vname = ver.get('name', '?')
            for file_entry in ver.get('files', []):
                sha = file_entry.get('hashes', {}).get('SHA256', '').upper()
                if sha in installed_hashes:
                    installed_versions.append((idx, bm, vname, vid, sha_to_path.get(sha, '')))
                    break

        # Build available_versions list for the UI (all versions with metadata)
        available_versions = [
            {'id': v.get('id'),
             'name': v.get('name', '?'),
             'baseModel': v.get('baseModel', ''),
             'publishedAt': v.get('publishedAt', '')}
            for v in model_versions
        ]

        if not precise_check:
            # No family grouping — one entry for the whole model
            # Pick the installed version with the highest index (oldest) to show
            oldest_inst = max(installed_versions, key=lambda x: x[0])
            inst_idx, _, inst_ver_name, inst_vid, old_file = oldest_inst
            avail_ver = model_versions[0] if model_versions else None
            avail_ver_name = avail_ver.get('name', '?') if avail_ver else '?'
            avail_ver_id = avail_ver.get('id') if avail_ver else None
            if inst_idx > 0:
                items.append({
                    'model_id': model_id,
                    'model_name': model_name,
                    'model_type': model_type,
                    'family': None,
                    'installed_ver': inst_ver_name,
                    'installed_ver_id': inst_vid,
                    'latest_ver': avail_ver_name,
                    'latest_ver_id': avail_ver_id,
                    'available_versions': available_versions,
                    'preview_url': preview_url,
                    'old_file': old_file,
                })
        else:
            # Per-baseModel: find the newest available version for each baseModel
            # base → (newest_idx, newest_ver_name, newest_ver_id)
            base_newest = {}
            for ver in model_versions:
                bm = (ver.get('baseModel') or '').strip()
                idx = ver_id_to_idx.get(ver.get('id'), 0)
                if bm not in base_newest or idx < base_newest[bm][0]:
                    base_newest[bm] = (idx, ver.get('name', '?'), ver.get('id'))

            # For each installed version, emit a card if it's not the newest for its base
            seen_bases = set()  # avoid duplicate cards for the same base
            for inst_idx, bm, inst_ver_name, inst_vid, old_file in installed_versions:
                if bm in seen_bases:
                    continue
                newest_idx, newest_ver_name, newest_vid = base_newest.get(bm, (0, '?', None))
                if inst_idx > newest_idx:
                    seen_bases.add(bm)
                    items.append({
                        'model_id': model_id,
                        'model_name': model_name,
                        'model_type': model_type,
                        'family': bm or None,
                        'installed_ver': inst_ver_name,
                        'installed_ver_id': inst_vid,
                        'latest_ver': newest_ver_name,
                        'latest_ver_id': newest_vid,
                        'available_versions': available_versions,
                        'preview_url': preview_url,
                        'old_file': old_file,
                    })

    # Sort by local file modification time, most recent first
    def _mtime_key(entry):
        path = entry.get('old_file', '')
        try:
            return -os.path.getmtime(path)
        except (OSError, TypeError):
            return 0

    items.sort(key=_mtime_key)
    return items

def get_content_choices(scan_choices=False):
    content_list = [
        'Checkpoint', 'TextualInversion', 'LORA', 'Poses', 'Controlnet', 'Detection',
        'VAE', 'Upscaler', 'Wildcards', 'AestheticGradient', 'MotionModule', 'Workflows', 'Other'
    ]
    if scan_choices:
        content_list.insert(0, 'All')
        return content_list
    return content_list

def get_save_path_and_name(install_path, file_name, api_response, sub_folder=None):
    save_to_custom = getattr(opts, 'save_to_custom', False)

    name = os.path.splitext(file_name)[0]
    if not sub_folder:
        sub_folder = os.path.normpath(os.path.relpath(install_path, gl.main_folder))
    image_path = _file.get_image_path(install_path, api_response, sub_folder)

    if save_to_custom:
        save_path = image_path
    else:
        save_path = install_path

    return save_path, name

## === ANXETY EDITs ===
def file_scan(folders, tag_finish, ver_finish, installed_finish, preview_finish, organize_finish, overwrite_toggle, tile_count, gen_hash, create_html, progress=gr.Progress() if queue else None):
    global no_update
    proxies, ssl = _api.get_proxies()
    gl.scan_files = True
    no_update = False

    if from_tag:
        number = _download.random_number(tag_finish)
    elif from_ver:
        number = _download.random_number(ver_finish)
    elif from_installed:
        number = _download.random_number(installed_finish)
    elif from_preview:
        number = _download.random_number(preview_finish)
    elif from_organize:
        number = _download.random_number(organize_finish)

    if not folders:
        if progress != None:
            progress(0, desc='No model type selected.')
        no_update = True
        gl.scan_files = False
        time.sleep(2)
        return (
            gr.update(value='<div style="min-height: 0px;"></div>'),
            gr.update(value=number)
        )

    folders_to_check = []
    if 'All' in folders:
        folders = _file.get_content_choices()

    for item in folders:
        if item == 'LORA':
            folder = _api.contenttype_folder('LORA')
            if folder:
                folders_to_check.append(folder)
        elif item == 'Upscaler':
            folder = _api.contenttype_folder(item, 'SwinIR')
            if folder:
                folders_to_check.append(folder)
            folder = _api.contenttype_folder(item, 'RealESRGAN')
            if folder:
                folders_to_check.append(folder)
            folder = _api.contenttype_folder(item, 'GFPGAN')
            if folder:
                folders_to_check.append(folder)
            folder = _api.contenttype_folder(item, 'BSRGAN')
            if folder:
                folders_to_check.append(folder)
            folder = _api.contenttype_folder(item, 'ESRGAN')
            if folder:
                folders_to_check.append(folder)
        else:
            folder = _api.contenttype_folder(item)
            if folder:
                folders_to_check.append(folder)

    total_files = 0
    files_done = 0

    files = list_files(folders_to_check)
    total_files += len(files)

    if total_files == 0:
        if progress != None:
            progress(1, desc='No files in selected folder.')
        no_update = True
        gl.scan_files = False
        time.sleep(2)
        return (
            gr.update(value='<div style="min-height: 0px;"></div>'),
            gr.update(value=number)
        )

    all_model_ids = []
    file_paths = []
    all_ids = []
    local_fallback_items = []

    for file_path in files:
        if gl.cancel_status:
            if progress != None:
                progress(files_done / total_files, desc='Processing files cancelled.')
            no_update = True
            gl.scan_files = False
            time.sleep(2)
            return (
                gr.update(value='<div style="min-height: 0px;"></div>'),
                gr.update(value=number)
            )
        file_name = os.path.basename(file_path)
        if progress != None:
            progress(files_done / total_files, desc=f"Processing file: {file_name}")

        model_id = get_models(file_path, gen_hash)
        if model_id == 'offline':
            print('The CivitAI servers did not respond, unable to retrieve Model ID')
        elif model_id == 'Model not found':
            debug_print(f"model: '{file_name}' not found on CivitAI servers.")
            if from_installed:
                local_fallback_items.append(_build_local_fallback_browser_item(file_path))
        elif model_id != None:
            all_model_ids.append(f"&ids={model_id}")
            all_ids.append(model_id)
            file_paths.append(file_path)
        elif not model_id:
            print(f"model ID not found for: '{file_name}'")
            if from_installed:
                local_fallback_items.append(_build_local_fallback_browser_item(file_path))
        files_done += 1

    gl.local_browser_fallback_items = local_fallback_items

    all_items = []

    all_model_ids = list(set(all_model_ids))

    if not all_model_ids and not local_fallback_items:
        progress(1, desc='No model IDs could be retrieved.')
        print("Could not retrieve any Model IDs, please make sure to turn on the 'One-Time Hash Generation for externally downloaded models.' option if you haven't already.")
        no_update = True
        gl.scan_files = False
        time.sleep(2)
        return (
            gr.update(value='<div style="min-height: 0px;"></div>'),
            gr.update(value=number)
        )

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    if not from_installed:
        model_chunks = list(chunks(all_model_ids, 500))

        base_url = f"https://{_api.get_civitai_domain()}/api/v1/models?limit=100&nsfw=true"
        url_list = [f"{base_url}{''.join(chunk)}" for chunk in model_chunks]

        url_count = len(all_model_ids) // 100
        if len(all_model_ids) % 100 != 0:
            url_count += 1
        url_done = 0
        api_response = {}
        for url in url_list:
            while url:
                try:
                    if progress != None:
                        progress(url_done / url_count, desc=f"Sending API request... {url_done}/{url_count}")
                    response = requests.get(url, timeout=(60, 30), proxies=proxies, verify=ssl)
                    if response.status_code == 200:
                        api_response_json = response.json()
                        all_items.extend(api_response_json['items'])
                        metadata = api_response_json.get('metadata', {})
                        url = metadata.get('nextPage', None)
                    elif response.status_code == 503:
                        print(f"Error: Received status code: {response.status_code} with URL: {url}")
                        print(response.text)
                        return (
                            gr.update(value=_api.api_error_msg('error')),
                            gr.update(value=number)
                        )
                    else:
                        print(f"Error: Received status code {response.status_code} with URL: {url}")
                        url = None
                    url_done += 1
                except requests.exceptions.Timeout:
                    print(f"Request timed out for {url}. Skipping...")
                    url = None
                except requests.exceptions.ConnectionError:
                    print('Failed to connect to the API. The servers might be offline.')
                    url = None
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    url = None

        api_response['items'] = all_items
        if api_response['items'] == []:
            return (
                gr.update(value=_api.api_error_msg('no_items')),
                gr.update(value=number)
            )

    if progress != None:
        progress(1, desc='Processing final results...')

    if from_ver:
        updated_models, outdated_models = version_match(file_paths, api_response)

        updated_set = set(updated_models)
        outdated_set = set(outdated_models)
        outdated_set = {model for model in outdated_set if model[0] not in {updated_model[0] for updated_model in updated_set}}

        # Collect per-family detail for Update Mode cards
        gl.update_items = collect_update_items(outdated_set, api_response, file_paths)

        all_model_ids = [model[0] for model in outdated_set]
        all_model_names = [model[1] for model in outdated_set]

        # Store for Dashboard update summary
        import datetime as _dt
        global last_update_scan
        _by_type = {}
        for _entry in outdated_set:
            _mtype = _entry[2] if len(_entry) > 2 else 'Unknown'
            _by_type.setdefault(_mtype, []).append(_entry[1])
        last_update_scan = {
            'outdated_by_type': _by_type,
            'outdated_count': len(outdated_set),
            'updated_count': len(updated_set),
            'scanned_at': _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        append_update_audit_log('update_scan', {
            'outdated_count': len(outdated_set),
            'updated_count': len(updated_set),
            'outdated_by_type': {k: len(v) for k, v in _by_type.items()},
        })

        for model_name in all_model_names:
            print(f'"{model_name}" is currently outdated.')

        if len(all_model_ids) == 0:
            no_update = True
            gl.scan_files = False
            return (
                gr.update(value='<div style="font-size: 24px; text-align: center; margin: 50px !important;">No updates found for selected models.</div>'),
                gr.update(value=number)
            )

    if all_model_ids:
        model_chunks = list(chunks(all_model_ids, tile_count))
        base_url = f"https://{_api.get_civitai_domain()}/api/v1/models?limit=100&nsfw=true"
        gl.url_list = {i + 1: f"{base_url}{''.join(chunk)}" for i, chunk in enumerate(model_chunks)}
    else:
        gl.url_list = {1: 'local_only://fallback'}

    ## === ANXETY EDITs ===
    if from_ver:
        gl.scan_files = False
        return (
            gr.update(value='<div style="font-size: 24px; text-align: center; margin: 50px !important;">Outdated models have been found.<br>Please press the button above to load the models into the browser tab</div>'),
            gr.update(value=number)
        )

    elif from_installed:
        gl.scan_files = False
        return (
            gr.update(value='<div style="font-size: 24px; text-align: center; margin: 50px !important;">Installed models have been loaded.<br>Please press the button above to load the models into the browser tab</div>'),
            gr.update(value=number)
        )

    elif from_tag:
        completed_tags = 0
        tag_count = len(file_paths)

        for file_path, id_value in zip(file_paths, all_ids):
            install_path, file_name = os.path.split(file_path)

            try:
                save_path, name = get_save_path_and_name(install_path, file_name, api_response)

                # Get SHA256 hash for the file to find the specific version
                file_sha256 = None
                json_file = os.path.splitext(file_path)[0] + '.json'
                if os.path.exists(json_file):
                    data = _api.safe_json_load(json_file)
                    file_sha256 = data.get('sha256') if data else None

                # If SHA256 not cached in .json, compute it now and save it
                # This ensures we always match the exact version (not just by filename)
                if not file_sha256 and os.path.exists(file_path):
                    try:
                        file_sha256 = gen_sha256(file_path)
                    except Exception:
                        pass

                # Find the specific model version based on SHA256 or filename
                if file_sha256:
                    model_version, item = find_model_version_by_sha256(api_response, file_sha256)
                else:
                    model_version, item = find_model_version_by_filename(api_response, file_name)

                html_path = os.path.join(save_path, f'{name}.html')

                if create_html and not os.path.exists(html_path) or create_html and overwrite_toggle:
                    if model_version and item:
                        # Use the specific model version name for HTML generation
                        preview_html = _api.update_model_info(None, model_version.get('name'), True, id_value, api_response, True)
                    else:
                        # Fallback to first version if specific version not found
                        model_versions = _api.update_model_versions(id_value, api_response)
                        preview_html = _api.update_model_info(None, model_versions.get('value'), True, id_value, api_response, True)
                else:
                    preview_html = None

                completed_tags += 1
                if progress != None:
                    progress(
                        completed_tags / tag_count,
                        desc=f"Saving tags{' & HTML' if preview_html else ''}... {completed_tags}/{tag_count} | {name}"
                    )
                sub_folder = os.path.normpath(os.path.relpath(install_path, gl.main_folder))
                save_model_info(install_path, file_name, sub_folder, sha256=file_sha256, preview_html=preview_html, api_response=api_response, overwrite_toggle=overwrite_toggle)

            except Exception as e:
                print(f"Error processing model {file_name}: {e}")
                completed_tags += 1
                if progress != None:
                    progress(
                        completed_tags / tag_count,
                        desc=f"Skipped {name} due to error... {completed_tags}/{tag_count}"
                    )
                continue  # Skip this model and continue with the next

        if progress != None:
            progress(1, desc='All tags succesfully saved!')
        gl.scan_files = False
        time.sleep(2)
        return (
            gr.update(value='<div style="min-height: 0px;"></div>'),
            gr.update(value=number)
        )

    elif from_preview:
        completed_preview = 0
        preview_count = len(file_paths)
        for file in file_paths:
            _, file_name = os.path.split(file)
            name = os.path.splitext(file_name)[0]
            completed_preview += 1
            if progress != None:
                progress(
                    completed_preview / preview_count,
                    desc=f"Saving preview images... {completed_preview}/{preview_count} | {name}"
                )
            save_preview(file, api_response, overwrite_toggle)
        gl.scan_files = False
        return (
            gr.update(value='<div style="min-height: 0px;"></div>'),
            gr.update(value=number)
        )
    
    elif from_organize:
        # Step 1: Analyze organization needs
        if progress != None:
            progress(0, desc='Analyzing models for organization...')
        
        organization_plan = analyze_organization_plan(folders, progress)
        
        # Always show preview first with statistics
        preview_html = generate_organization_preview_html(organization_plan)
        
        if not organization_plan['moves']:
            # No files need organization
            gl.scan_files = False
            return (
                gr.update(value=preview_html),
                gr.update(value=number)
            )
        
        # Files need organization - show preview with detailed stats
        _debug_log(f"Organization plan created: {len(organization_plan['moves'])} files to move")
        for category, info in organization_plan['summary'].items():
            _debug_log(f"  {category}: {info['count']} files ({format_size(info['size'])})")
        
        # Step 2: Save backup before making changes
        if progress != None:
            progress(0.9, desc='Creating backup...')
        
        backup_id = save_organization_backup(organization_plan)
        if not backup_id:
            gl.scan_files = False
            error_html = '''
            <div style="padding: 20px; border: 1px solid var(--error-border-color); border-radius: 8px;">
                <h3 style="color: var(--error-text-color);">⚠️ Backup Failed</h3>
                <p>Unable to create backup. Organization cancelled for safety.</p>
                <p>Please check file permissions and try again.</p>
            </div>
            '''
            return (
                gr.update(value=error_html),
                gr.update(value=number)
            )
        
        # Step 3: Execute organization
        total_moves = len(organization_plan['moves'])
        print(f"[CivitAI Browser Neo] Starting organization of {total_moves} files...")
        print(f"[CivitAI Browser Neo] 💾 Backup ID: {backup_id}")
        
        result = execute_organization(organization_plan, progress)
        
        # Step 4: Generate result message with detailed statistics
        if result['success']:
            # Calculate total size moved
            total_size = sum(info['size'] for info in organization_plan['summary'].values())
            folder_list = ', '.join(sorted(organization_plan['summary'].keys()))
            
            result_html = f'''
            <div style="padding: 20px; text-align: center; color: var(--color-accent);">
                <h2 style="margin: 0 0 15px 0;">✅ Organization Complete!</h2>
                <div style="font-size: 18px; margin-bottom: 20px;">
                    <strong>{result['completed']} files</strong> ({format_size(total_size)}) organized into <strong>{len(organization_plan['summary'])} folders</strong>
                </div>
                <div style="background: var(--background-fill-secondary); padding: 10px; border-radius: 5px; font-size: 14px;">
                    📂 {folder_list}
                </div>
                <div style="margin-top: 15px; padding: 10px; background: var(--color-accent-soft); border-radius: 5px; font-size: 13px;">
                    💾 Backup: {backup_id} | Use "↶ Undo" button to rollback
                </div>
            </div>
            '''
        else:
            error_list = '<br>'.join(result['errors'][:10])
            if len(result['errors']) > 10:
                error_list += f'<br><em>... and {len(result["errors"]) - 10} more errors</em>'
            
            result_html = f'''
            <div style="padding: 20px; border: 1px solid var(--error-border-color); border-radius: 8px;">
                <h3 style="color: var(--error-text-color);">⚠️ Organization Completed with Errors</h3>
                <p>Completed: {result['completed']}/{result['total']} files</p>
                <p>💾 Backup saved: {backup_id}</p>
                <details>
                    <summary style="cursor: pointer;">View errors</summary>
                    <div style="margin-top: 10px; padding: 10px; background: var(--block-background-fill); border-radius: 5px;">
                        {error_list}
                    </div>
                </details>
                <div style="margin-top: 10px;">
                    Use the "Undo Organization" button to rollback changes.
                </div>
            </div>
            '''
        
        print(f"[CivitAI Browser Neo] {result['message']}")
        gl.scan_files = False
        return (
            gr.update(value=result_html),
            gr.update(value=number)
        )

def finish_returns():
    return (
        gr.update(interactive=True, visible=True),
        gr.update(interactive=True, visible=True),
        gr.update(interactive=True, visible=True),
        gr.update(interactive=True, visible=True),
        gr.update(interactive=True, visible=True),  # Organize models button
        gr.update(interactive=False, visible=False)
    )

def start_returns(number):
    return (
        gr.update(value=number),
        gr.update(interactive=False, visible=False),
        gr.update(interactive=True, visible=True),
        gr.update(interactive=False, visible=True),
        gr.update(interactive=False, visible=True),
        gr.update(interactive=False, visible=True),
        gr.update(interactive=False, visible=True),  # Organize models button (keep visible but disabled during scan)
        gr.update(value='<div style="min-height: 100px;"></div>')
    )

## === ANXETY EDITs ===
def set_globals(input_global=None):
    global from_tag, from_ver, from_installed, from_preview, from_organize
    from_tag = from_ver = from_installed = from_preview = from_organize = False
    if input_global == 'reset':
        return
    elif input_global == 'from_tag':
        from_tag = True
    elif input_global == 'from_ver':
        from_ver = True
    elif input_global == 'from_installed':
        from_installed = True
    elif input_global == 'from_preview':
        from_preview = True
    elif input_global == 'from_organize':
        from_organize = True

def save_tag_start(tag_start):
    set_globals('from_tag')
    number = _download.random_number(tag_start)
    return start_returns(number)

def save_preview_start(preview_start):
    set_globals('from_preview')
    number = _download.random_number(preview_start)
    return start_returns(number)

def ver_search_start(ver_start):
    set_globals('from_ver')
    number = _download.random_number(ver_start)
    return start_returns(number)

def installed_models_start(installed_start):
    set_globals('from_installed')
    number = _download.random_number(installed_start)
    return start_returns(number)

def get_model_categories():
    """
    Get model organization categories from settings or default
    Returns dict mapping folder names to detection patterns
    """
    # Default categories based on Forge Neo supported models
    default_categories = {
        'SD': ['SD 1', 'SD1', 'SD 2', 'SD2'],
        'SDXL': ['SDXL'],
        'Pony': ['PONY', 'PONYXL', 'PONY XL', 'PONY V6', 'PONYV6'],
        'Illustrious': ['ILLUSTRIOUS'],
        'NoobAI': ['NOOBAI', 'NOOB AI', 'NOOB', 'NAI'],
        'FLUX': ['FLUX'],
        'Wan': ['WAN'],
        'Qwen': ['QWEN'],
        'Z-Image': ['Z-IMAGE', 'ZIMAGE', 'Z IMAGE'],
        'Lumina': ['LUMINA'],
        'Anima': ['ANIMA'],
        'Cascade': ['CASCADE'],
        'PixArt': ['PIXART', 'PIX ART'],
        'Playground': ['PLAYGROUND'],
        'SVD': ['SVD', 'STABLE VIDEO'],
        'Hunyuan': ['HUNYUAN'],
        'Kolors': ['KOLORS'],
        'AuraFlow': ['AURAFLOW', 'AURA FLOW'],
        'Chroma': ['CHROMA'],
    }
    
    # Try to load custom categories from settings
    try:
        custom_categories = getattr(opts, 'civitai_neo_model_categories', None)
        if custom_categories:
            # Parse JSON if stored as string
            if isinstance(custom_categories, str):
                import json
                custom_categories = json.loads(custom_categories)
            return custom_categories
    except:
        pass
    
    return default_categories

def _debug_log(message):
    """
    Print debug messages for organization system if enabled in settings
    Enable in Settings > CivitAI Browser Neo > Debug Organization Logs
    """
    if getattr(opts, 'civitai_neo_debug_organize', False):
        debug_print(message)

def normalize_base_model(base_model):
    """
    Normalize baseModel from CivitAI to folder-friendly name
    Supports all Forge Neo model types with customizable categories
    """
    _debug_log(f"normalize_base_model() received: '{base_model}'")
    
    if not base_model or base_model == 'Not Found':
        # Check if user wants to create "Other" folder
        if not getattr(opts, 'civitai_neo_create_other_folder', True):
            _debug_log("No baseModel, returning None (no 'Other' folder)")
            return None  # Leave in root
        _debug_log("No baseModel, returning 'Other'")
        return 'Other'
    
    base_model_upper = base_model.upper()
    categories = get_model_categories()
    
    _debug_log(f"Checking '{base_model_upper}' against categories...")
    
    # Check each category's patterns
    for folder_name, patterns in categories.items():
        for pattern in patterns:
            if pattern.upper() in base_model_upper:
                _debug_log(f"MATCH! '{pattern}' found in '{base_model_upper}' → Folder: '{folder_name}'")

                # Optional Wan subtype splitting (I2V / T2V / TI2V)
                if folder_name == 'Wan' and getattr(opts, 'civitai_neo_wan_subfolder_by_type', False):
                    bmu = base_model_upper
                    if 'TI2V' in bmu:
                        subfolder = 'TI2V'
                    elif 'I2V' in bmu:
                        subfolder = 'I2V'
                    elif 'T2V' in bmu:
                        subfolder = 'T2V'
                    else:
                        subfolder = None
                    if subfolder:
                        result = os.path.join('Wan', subfolder)
                        _debug_log(f"Wan subtype split enabled → '{result}'")
                        return result

                return folder_name
    
    # No match found
    _debug_log(f"No match found for '{base_model_upper}'")
    if not getattr(opts, 'civitai_neo_create_other_folder', True):
        _debug_log("Returning None (no 'Other' folder)")
        return None  # Leave in root
    _debug_log("Returning 'Other'")
    return 'Other'

def _fetch_api_info_by_hash(file_path, api_info_file):
    """
    Fetch model version info from CivitAI API using the file's SHA256 hash.

    Uses endpoint: GET /api/v1/model-versions/by-hash/{sha256}
    The response contains 'baseModel' at the root level — the cleanest source.

    On success:
      - Saves (overwrites) the .api_info.json with the fresh API response
      - Also patches the .json file's "sd version" field with the correct raw
        baseModel value (fixes any stale/normalised values from older releases)

    Returns the parsed data dict on success, or None on failure.
    """
    model_name = os.path.basename(file_path)
    _debug_log(f"Fetching .api_info.json by SHA256 for: {model_name}")

    file_hash = gen_sha256(file_path)
    if not file_hash:
        _debug_log(f"Could not compute SHA256 for: {model_name}")
        return None

    normalized = _api.normalize_sha256(file_hash)
    if not normalized:
        _debug_log(f"Invalid SHA256 for: {model_name}")
        return None

    api_url = f"https://{_api.get_civitai_domain()}/api/v1/model-versions/by-hash/{normalized}"
    _debug_log(f"API call: {api_url}")

    try:
        headers = _api.get_headers()
        proxies, ssl = _api.get_proxies()
        response = requests.get(api_url, headers=headers, timeout=(60, 30), proxies=proxies, verify=ssl)

        if response.status_code == 200:
            data = response.json()
            if 'error' in data:
                _debug_log(f"API returned error for {model_name}: {data.get('error')}")
                return None

            # 1. Save fresh data as .api_info.json (overwrites any stale/wrong file)
            _api.safe_json_save(api_info_file, data)
            print(f"[CivitAI Browser Neo] ✅ Fetched and saved .api_info.json for: {model_name}")

            # 2. Also patch "sd version" in the .json sidecar with the correct raw value
            #    so the .json is also self-consistent and usable offline in the future
            base_model = data.get('baseModel', '')
            if base_model:
                json_file = os.path.splitext(file_path)[0] + '.json'
                if os.path.exists(json_file):
                    try:
                        content = _api.safe_json_load(json_file) or {}
                        if content.get('sd version') != base_model:
                            content['sd version'] = base_model
                            _api.safe_json_save(json_file, content)
                            _debug_log(f"Patched 'sd version' → '{base_model}' in {os.path.basename(json_file)}")
                    except Exception as patch_err:
                        _debug_log(f"Could not patch .json for {model_name}: {patch_err}")

            return data

        elif response.status_code == 404:
            _debug_log(f"Model not found on CivitAI for hash {normalized} ({model_name})")
            return None
        else:
            _debug_log(f"API returned HTTP {response.status_code} for: {model_name}")
            return None

    except Exception as e:
        _debug_log(f"Error fetching API info for {model_name}: {e}")
        return None


def _extract_base_model_from_api_data(data, file_path=None):
    """
    Extract baseModel string from a CivitAI API response dict.
    Checks fields in priority order:
      1. data['baseModel']           ← by-hash / model-versions endpoint
      2. data['model']['baseModel']
      3. data['modelVersions']       ← SHA256-matched or first version
      4. data['version']['baseModel']
    Returns the raw baseModel string or '' if not found.
    """
    model_name = os.path.basename(file_path) if file_path else '?'

    base_model = data.get('baseModel', '')
    if base_model:
        _debug_log(f"Found from data['baseModel']: '{base_model}'")
        return base_model

    base_model = data.get('model', {}).get('baseModel', '')
    if base_model:
        _debug_log(f"Found from data['model']['baseModel']: '{base_model}'")
        return base_model

    if 'modelVersions' in data:
        versions = data.get('modelVersions', [])
        _debug_log(f"Found modelVersions array with {len(versions)} versions")
        if versions:
            matched_version = None
            if file_path:
                file_hash = gen_sha256(file_path)
                if file_hash:
                    _debug_log(f"Model SHA256: {file_hash}")
                    for version in versions:
                        for vfile in version.get('files', []):
                            if vfile.get('hashes', {}).get('SHA256', '').upper() == file_hash.upper():
                                matched_version = version
                                _debug_log(f"Found matching version by SHA256: {version.get('name')} (id: {version.get('id')})")
                                break
                        if matched_version:
                            break
            target_version = matched_version if matched_version else versions[0]
            base_model = target_version.get('baseModel', '')
            if base_model:
                label = f"MATCHED modelVersion['{target_version.get('name')}']" if matched_version else "modelVersions[0]"
                _debug_log(f"Found from {label}: '{base_model}'")
                return base_model

    base_model = data.get('version', {}).get('baseModel', '')
    if base_model:
        _debug_log(f"Found from data['version']['baseModel']: '{base_model}'")
        return base_model

    return ''


def get_model_info_for_organization(file_path):
    """
    Get model info for organization purposes.

    Source-of-truth priority:
      1. Existing .api_info.json  — read and extract baseModel
      2. CivitAI API by SHA256    — if .api_info.json is missing or has no
                                    baseModel, fetch via by-hash endpoint,
                                    save as .api_info.json, then extract

    The local .json file is intentionally ignored because its "sd version"
    field may contain stale/normalised values (e.g. "Other") written by older
    extension versions, which would cause correctly-placed files to be flagged.

    Returns tuple: (base_model_type, model_name)
    Returns (None, model_name) when metadata is unavailable even after API call.
    """
    model_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_path)[0]

    _debug_log(f"Checking metadata for: {model_name}")

    api_info_file = base_name + '.api_info.json'

    # --- Step 1: try existing .api_info.json ---
    if os.path.exists(api_info_file):
        _debug_log(f"Found existing .api_info.json for: {model_name}")
        try:
            data = _api.safe_json_load(api_info_file)
            if data:
                base_model = _extract_base_model_from_api_data(data, file_path)
                if base_model:
                    _debug_log(f"SUCCESS! Final baseModel: '{base_model}' from existing .api_info.json")
                    return base_model, model_name
                _debug_log(f"No baseModel in existing .api_info.json — will fetch from API")
        except Exception as e:
            _debug_log(f"Error reading {api_info_file}: {e}")

    # --- Step 2: fetch from CivitAI API by SHA256 ---
    _debug_log(f"No usable .api_info.json for {model_name} — fetching from CivitAI by hash...")
    data = _fetch_api_info_by_hash(file_path, api_info_file)
    if data:
        base_model = _extract_base_model_from_api_data(data, file_path)
        if base_model:
            _debug_log(f"SUCCESS! Final baseModel: '{base_model}' from API (by hash)")
            return base_model, model_name

    # --- Step 3: offline fallback — .json sidecar "sd version" field ---
    # Used only when API is unreachable or the model was deleted from CivitAI.
    # "Other" is explicitly rejected: it was the old buggy default value and
    # does not represent a real/confirmed base model type.
    json_file = os.path.splitext(file_path)[0] + '.json'
    if os.path.exists(json_file):
        try:
            content = _api.safe_json_load(json_file) or {}
            sd_version = content.get('sd version', '')
            if sd_version and sd_version.upper() != 'OTHER':
                _debug_log(f"Offline fallback: using 'sd version'='{sd_version}' from .json")
                print(f"[CivitAI Browser Neo] ⚠️ Using offline .json fallback for: {model_name} (API unavailable)")
                return sd_version, model_name
        except Exception as e:
            _debug_log(f"Error reading .json fallback for {model_name}: {e}")

    print(f"[CivitAI Browser Neo] ⚠️ Could not determine baseModel for: {model_name}")
    return None, model_name

def analyze_organization_plan(folders, progress=None):
    """
    Analyze current model files and create an organization plan
    Returns organization plan with moves grouped by base model
    """
    folders_to_check = []
    
    if 'All' in folders:
        folders = _file.get_content_choices()
    
    for item in folders:
        if item == 'LORA':
            folder = _api.contenttype_folder('LORA')
            if folder:
                folders_to_check.append(folder)
        else:
            folder = _api.contenttype_folder(item)
            if folder:
                folders_to_check.append(folder)
    
    files = list_files(folders_to_check)
    
    organization_plan = {
        'moves': [],
        'summary': {},
        'total_files': 0,
        'files_with_info': 0,
        'files_without_info': 0
    }
    
    files_processed = 0
    total_files = len(files)
    
    for file_path in files:
        files_processed += 1
        if progress is not None:
            file_name = os.path.basename(file_path)
            progress(files_processed / total_files, desc=f"Analyzing: {file_name}")
        
        base_model_raw, model_name = get_model_info_for_organization(file_path)
        
        if not base_model_raw:
            organization_plan['files_without_info'] += 1
            continue
        
        organization_plan['files_with_info'] += 1
        
        # Normalize base model to folder name
        base_model_folder = normalize_base_model(base_model_raw)
        
        # If normalize returns None, skip this file (leave in root)
        if not base_model_folder:
            continue
        
        # Get current directory
        current_dir = os.path.dirname(file_path)
        
        # Check if already in correct subfolder
        # Normalize both sides to handle multi-level folders like 'Wan/I2V'
        norm_current = os.path.normpath(current_dir)
        norm_target_suffix = os.path.normpath(base_model_folder)
        if norm_current.endswith(os.sep + norm_target_suffix) or norm_current == norm_target_suffix:
            # Already organized
            continue
        
        # Create target path
        # Determine root folder from current file location
        # Go up levels until we find a models folder or we're at root model type folder
        root_folder = current_dir
        while os.path.basename(root_folder) not in ['Lora', 'Stable-diffusion', 'embeddings', 'VAE', 'ControlNet']:
            parent = os.path.dirname(root_folder)
            if parent == root_folder:  # We've reached filesystem root
                root_folder = current_dir
                break
            root_folder = parent
        
        target_folder = os.path.join(root_folder, base_model_folder)
        target_path = os.path.join(target_folder, os.path.basename(file_path))
        
        # Add to plan
        organization_plan['moves'].append({
            'from': file_path,
            'to': target_path,
            'base_model': base_model_folder,
            'model_name': model_name,
            'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
        })
        
        # Update summary
        if base_model_folder not in organization_plan['summary']:
            organization_plan['summary'][base_model_folder] = {
                'count': 0,
                'size': 0
            }
        
        organization_plan['summary'][base_model_folder]['count'] += 1
        organization_plan['summary'][base_model_folder]['size'] += organization_plan['moves'][-1]['size']
    
    organization_plan['total_files'] = total_files
    
    return organization_plan

def format_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def generate_organization_preview_html(organization_plan):
    """Generate HTML preview of organization plan"""
    
    if not organization_plan['moves']:
        return '''
        <div style="padding: 20px; text-align: center;">
            <h3>✅ All models are already organized!</h3>
            <p>No files need to be moved.</p>
        </div>
        '''
    
    # Build compact summary table
    summary_rows = ''
    for base_model in sorted(organization_plan['summary'].keys()):
        info = organization_plan['summary'][base_model]
        summary_rows += f'''
        <tr>
            <td style="padding: 5px;">📂 {base_model}</td>
            <td style="text-align: center; padding: 5px;">{info['count']}</td>
            <td style="text-align: right; padding: 5px;">{format_size(info['size'])}</td>
        </tr>
        '''
    
    total_size = sum(info['size'] for info in organization_plan['summary'].values())
    total_moves = len(organization_plan['moves'])
    total_folders = len(organization_plan['summary'])
    files_without_info = organization_plan['files_without_info']
    
    html = f'''
    <div style="padding: 15px; border: 1px solid var(--border-color-primary); border-radius: 8px; margin: 10px 0;">
        <h3 style="margin: 0 0 15px 0;">📁 Organization Plan</h3>
        
        <div style="background: var(--background-fill-secondary); padding: 12px; border-radius: 5px; margin-bottom: 15px; font-size: 15px;">
            <strong>{total_moves} files</strong> ({format_size(total_size)}) → <strong>{total_folders} folders</strong>
        </div>
        
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <thead>
                <tr style="background: var(--background-fill-secondary);">
                    <th style="padding: 8px; text-align: left;">Folder</th>
                    <th style="padding: 8px; text-align: center;">Files</th>
                    <th style="padding: 8px; text-align: right;">Size</th>
                </tr>
            </thead>
            <tbody>
                {summary_rows}
            </tbody>
        </table>
        
        {f'<div style="margin-top: 12px; padding: 10px; background: #fff3cd; border-radius: 5px; font-size: 13px;">⚠️ {files_without_info} of {organization_plan["total_files"]} files without metadata (will be skipped)</div>' if files_without_info > 0 else ''}
        
        <details style="margin-top: 12px;">
            <summary style="cursor: pointer; padding: 8px; background: var(--block-background-fill); border-radius: 5px; font-size: 13px;">
                📋 Show file list ({total_moves} files)
            </summary>
            <div style="max-height: 200px; overflow-y: auto; margin-top: 8px; padding: 8px; background: var(--block-background-fill); border-radius: 5px; font-size: 12px; font-family: monospace;">
                {'<br>'.join([f"• {os.path.basename(m['from'])} → {m['base_model']}/" for m in organization_plan['moves'][:50]])}
                {f'<br><em>... and {total_moves - 50} more</em>' if total_moves > 50 else ''}
            </div>
        </details>
    </div>
    '''
    
    return html

def save_organization_backup(organization_plan):
    """
    Save organization plan as backup before executing
    Returns backup ID (timestamp)
    """
    from datetime import datetime
    
    # Calculate statistics
    total_files = len(organization_plan['moves'])
    total_size = sum(info['size'] for info in organization_plan['summary'].values())
    
    backup_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'date_readable': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'moves': organization_plan['moves'],
        'summary': organization_plan['summary'],
        'stats': {
            'total_files': total_files,
            'total_size': total_size,
            'total_folders': len(organization_plan['summary']),
            'folders': list(organization_plan['summary'].keys())
        }
    }
    
    # Load existing backup file
    backup_file_data = {'created_at': datetime.now().timestamp(), 'backups': []}
    if os.path.exists(gl.organization_backup_file):
        try:
            with open(gl.organization_backup_file, 'r', encoding='utf-8') as f:
                backup_file_data = json.load(f)
                # Handle old format (plain list) - migrate to new format
                if isinstance(backup_file_data, list):
                    backup_file_data = {'created_at': datetime.now().timestamp(), 'backups': backup_file_data}
                elif 'backups' not in backup_file_data:
                    backup_file_data['backups'] = []
        except:
            backup_file_data = {'created_at': datetime.now().timestamp(), 'backups': []}
    
    # Add new backup
    backup_file_data['backups'].append(backup_data)
    
    # Keep only last 5 backups
    if len(backup_file_data['backups']) > 5:
        backup_file_data['backups'] = backup_file_data['backups'][-5:]
    
    # Save backups
    try:
        with open(gl.organization_backup_file, 'w', encoding='utf-8') as f:
            json.dump(backup_file_data, f, indent=2, ensure_ascii=False)
        
        gl.last_organization_backup = backup_data['timestamp']
        print(f"[CivitAI Browser Neo] Backup saved: {backup_data['timestamp']}")
        return backup_data['timestamp']
    except Exception as e:
        print(f"[CivitAI Browser Neo] Failed to save backup: {e}")
        return None

def get_last_organization_backup():
    """
    Get the most recent organization backup
    Returns backup data or None
    """
    if not os.path.exists(gl.organization_backup_file):
        return None
    
    try:
        with open(gl.organization_backup_file, 'r', encoding='utf-8') as f:
            backup_file_data = json.load(f)
        
        # Handle old format (plain list)
        if isinstance(backup_file_data, list):
            backups = backup_file_data
        else:
            backups = backup_file_data.get('backups', [])
        
        if backups:
            return backups[-1]
    except Exception as e:
        debug_print(f"Error loading backup: {e}")
    
    return None

def execute_rollback(progress=None):
    """
    Rollback the last organization operation
    Moves files back to their original locations
    """
    backup = get_last_organization_backup()
    
    if not backup:
        return {
            'success': False,
            'message': 'No backup found to rollback',
            'completed': 0,
            'total': 0,
            'errors': []
        }
    
    moves = backup.get('moves', [])
    total_moves = len(moves)
    moves_completed = 0
    errors = []
    
    print(f"[CivitAI Browser Neo] Starting rollback of {total_moves} files...")
    
    for move_info in moves:
        if gl.cancel_status:
            return {
                'success': False,
                'completed': moves_completed,
                'total': total_moves,
                'errors': errors,
                'message': 'Rollback cancelled by user'
            }
        
        # Reverse: from target back to source
        source_path = move_info['to']  # Where it was moved TO
        target_path = move_info['from']  # Where it came FROM
        model_name = move_info['model_name']
        
        moves_completed += 1
        if progress is not None:
            progress(moves_completed / total_moves, 
                    desc=f"Rolling back: {model_name} ({moves_completed}/{total_moves})")
        
        try:
            # Check if source file exists (file might have been deleted/moved)
            if not os.path.exists(source_path):
                errors.append(f"File not found (may have been moved): {model_name}")
                continue
            
            # Create target directory if it doesn't exist
            target_dir = os.path.dirname(target_path)
            os.makedirs(target_dir, exist_ok=True)
            
            # Check if target already exists
            if os.path.exists(target_path):
                errors.append(f"Target already exists, skipping: {model_name}")
                continue
            
            # Move main file back
            shutil.move(source_path, target_path)
            
            # Move associated files back
            base_name_source = os.path.splitext(source_path)[0]
            base_name_target = os.path.splitext(target_path)[0]
            
            associated_extensions = ['.json', '.png', '.jpg', '.jpeg', '.txt', '.html', '.civitai.info']
            
            # Move exact matches back
            for ext in associated_extensions:
                associated_source = base_name_source + ext
                associated_target = base_name_target + ext
                
                if os.path.exists(associated_source):
                    try:
                        shutil.move(associated_source, associated_target)
                        debug_print(f"Rolled back: {os.path.basename(associated_source)}")
                    except Exception as e:
                        debug_print(f"Could not move associated file back: {e}")
            
            # Move numbered preview images back (_0.png, _1.png, etc.)
            for i in range(20):
                for ext in ['.png', '.jpg', '.jpeg']:
                    numbered_source = f"{base_name_source}_{i}{ext}"
                    if os.path.exists(numbered_source):
                        numbered_target = f"{base_name_target}_{i}{ext}"
                        try:
                            shutil.move(numbered_source, numbered_target)
                            debug_print(f"Rolled back preview: {os.path.basename(numbered_source)}")
                        except Exception as e:
                            debug_print(f"Could not rollback preview {numbered_source}: {e}")
            
            # Move suffixed files back (.preview, .api_info, .civitai)
            suffixes = ['.preview', '.api_info', '.civitai']
            for suffix in suffixes:
                for ext in associated_extensions:
                    suffixed_source = base_name_source + suffix + ext
                    if os.path.exists(suffixed_source):
                        suffixed_target = base_name_target + suffix + ext
                        try:
                            shutil.move(suffixed_source, suffixed_target)
                            debug_print(f"Rolled back {suffix}: {os.path.basename(suffixed_source)}")
                        except Exception as e:
                            debug_print(f"Could not rollback {suffixed_source}: {e}")
            
            print(f"✓ Rolled back: {model_name}")
            
        except Exception as e:
            error_msg = f"Failed to rollback {model_name}: {str(e)}"
            errors.append(error_msg)
            debug_print(error_msg)
    
    # Clean up empty folders created during organization
    try:
        for move_info in moves:
            folder = os.path.dirname(move_info['to'])
            if os.path.exists(folder) and not os.listdir(folder):
                os.rmdir(folder)
                print(f"Removed empty folder: {folder}")
    except Exception as e:
        debug_print(f"Error cleaning up folders: {e}")
    
    return {
        'success': len(errors) == 0,
        'completed': moves_completed,
        'total': total_moves,
        'errors': errors,
        'message': f"Successfully rolled back {moves_completed} files" if len(errors) == 0 else f"Completed with {len(errors)} errors"
    }

def _move_associated_files(source_path, target_path):
    """Move all sidecar files (.json, .png, .html, numbered previews, etc.)
    from source_path location to target_path location."""
    base_name        = os.path.splitext(source_path)[0]
    target_base_name = os.path.splitext(target_path)[0]

    associated_extensions = ['.json', '.png', '.jpg', '.jpeg', '.txt', '.html', '.civitai.info']

    # Exact base name matches
    for ext in associated_extensions:
        associated_file = base_name + ext
        if os.path.exists(associated_file):
            target_associated = target_base_name + ext
            try:
                shutil.move(associated_file, target_associated)
                debug_print(f"Moved associated file: {os.path.basename(associated_file)}")
            except Exception as e:
                debug_print(f"Could not move associated file {associated_file}: {e}")

    # Numbered previews: model_0.png, model_1.png, …
    for i in range(20):
        for ext in ['.png', '.jpg', '.jpeg']:
            numbered_file = f"{base_name}_{i}{ext}"
            if os.path.exists(numbered_file):
                target_numbered = f"{target_base_name}_{i}{ext}"
                try:
                    shutil.move(numbered_file, target_numbered)
                    debug_print(f"Moved numbered preview: {os.path.basename(numbered_file)}")
                except Exception as e:
                    debug_print(f"Could not move numbered preview {numbered_file}: {e}")

    # Compound-suffix files: .preview.png, .api_info.json, .civitai.*
    suffixes = ['.preview', '.api_info', '.civitai']
    for suffix in suffixes:
        for ext in associated_extensions:
            associated_file = base_name + suffix + ext
            if os.path.exists(associated_file):
                target_associated = target_base_name + suffix + ext
                try:
                    shutil.move(associated_file, target_associated)
                    debug_print(f"Moved associated file: {os.path.basename(associated_file)}")
                except Exception as e:
                    debug_print(f"Could not move associated file {associated_file}: {e}")


def _make_progress_bar_html(done, total, label):
    """Return an inline HTML progress bar used by generator functions."""
    pct = int(done / total * 100) if total else 0
    return f'''
    <div style="padding:12px 15px;border:1px solid var(--border-color-primary);border-radius:8px;margin:6px 0;">
        <div style="font-size:13px;margin-bottom:6px;">{label}</div>
        <div style="background:var(--border-color-primary);border-radius:4px;height:10px;">
            <div style="background:#4caf50;width:{pct}%;height:100%;border-radius:4px;transition:width 0.2s;"></div>
        </div>
        <div style="font-size:12px;color:var(--body-text-color-subdued);margin-top:4px;">{done} / {total} files ({pct}%)</div>
    </div>'''


def execute_organization(organization_plan, progress=None):
    """
    Execute the organization plan by moving files
    Moves model files along with associated .json, .png, .txt files
    """
    total_moves = len(organization_plan['moves'])
    moves_completed = 0
    errors = []
    
    for move_info in organization_plan['moves']:
        if gl.cancel_status:
            return {
                'success': False,
                'completed': moves_completed,
                'total': total_moves,
                'errors': errors,
                'message': 'Organization cancelled by user'
            }
        
        source_path = move_info['from']
        target_path = move_info['to']
        model_name = move_info['model_name']
        
        moves_completed += 1
        if progress is not None:
            progress(moves_completed / total_moves, 
                    desc=f"Organizing: {model_name} ({moves_completed}/{total_moves})")
        
        try:
            # Create target directory if it doesn't exist
            target_dir = os.path.dirname(target_path)
            os.makedirs(target_dir, exist_ok=True)
            
            # Check if target file already exists
            if os.path.exists(target_path):
                debug_print(f"Target already exists, skipping: {target_path}")
                continue
            
            # Move main model file
            shutil.move(source_path, target_path)

            # Move all associated sidecar files
            _move_associated_files(source_path, target_path)

            print(f"✓ Organized: {model_name} → {move_info['base_model']}/")
            
        except Exception as e:
            error_msg = f"Failed to move {model_name}: {str(e)}"
            errors.append(error_msg)
            debug_print(error_msg)
    
    return {
        'success': len(errors) == 0,
        'completed': moves_completed,
        'total': total_moves,
        'errors': errors,
        'message': f"Successfully organized {moves_completed} files" if len(errors) == 0 else f"Completed with {len(errors)} errors"
    }

def organize_start(organize_start):
    set_globals('from_organize')
    number = _download.random_number(organize_start)
    return start_returns(number)

def validate_organization(folders, progress=gr.Progress() if queue else None):
    """
    Validate that models are in their correct subfolders based on .json metadata.
    Read-only: does NOT move any files.
    Yields (html_report, fix_btn_update, plan_json_string) — generator so the UI
    shows a status message immediately while the scan runs in the background.
    """
    import json as _json

    if not folders:
        html = '''<div style="padding:20px;text-align:center;">
            <strong>No content types selected.</strong><br>
            <span style="color:var(--body-text-color-subdued);">Select at least one type above.</span>
        </div>'''
        yield gr.update(value=html), gr.update(visible=False), '{}'
        return

    # ── Show scanning status immediately ─────────────────────────────────────
    yield (
        gr.update(value='<div style="padding:12px 15px;border:1px solid var(--border-color-primary);border-radius:8px;margin:6px 0;">'
                        '🔍 <strong>Scanning files…</strong> This may take a while for large collections.'
                        '</div>'),
        gr.update(visible=False),
        '{}'
    )

    if progress is not None:
        progress(0, desc="Scanning files for validation...")

    plan = analyze_organization_plan(folders, progress)

    misplaced  = plan['moves']             # files in wrong folder
    no_meta    = plan['files_without_info']
    total      = plan['total_files']
    correct    = total - len(misplaced) - no_meta

    # ── Summary numbers ──────────────────────────────────────────────────────
    if not misplaced and no_meta == 0:
        html = f'''
        <div style="padding:20px;text-align:center;">
            <div style="font-size:48px;margin-bottom:12px;">✅</div>
            <h3 style="margin:0 0 8px 0;color:var(--body-text-color);">All {total} models are in the correct folders!</h3>
            <p style="color:var(--body-text-color-subdued);margin:0;">Nothing to fix.</p>
        </div>'''
        yield gr.update(value=html), gr.update(visible=False), '{}'
        return

    # ── Build misplaced table ─────────────────────────────────────────────────
    rows_html = ''
    for m in misplaced:
        current_folder = os.path.basename(os.path.dirname(m['from']))
        expected_folder = m['base_model']
        rows_html += f'''
        <tr style="border-bottom:1px solid var(--border-color-primary);">
            <td style="padding:6px 10px;font-family:monospace;font-size:12px;">{m['model_name']}</td>
            <td style="padding:6px 10px;text-align:center;color:#e57373;">{current_folder or '(root)'}</td>
            <td style="padding:6px 10px;text-align:center;">→</td>
            <td style="padding:6px 10px;text-align:center;color:#81c784;">{expected_folder}/</td>
        </tr>'''

    no_meta_note = (
        f'<div style="margin-top:12px;padding:10px;background:#fff3cd;border-radius:5px;font-size:13px;">'
        f'⚠️ {no_meta} file(s) have no .json metadata and were skipped.</div>'
        if no_meta > 0 else ''
    )

    html = f'''
    <div style="padding:15px;border:1px solid var(--border-color-primary);border-radius:8px;margin:10px 0;">
        <h3 style="margin:0 0 12px 0;">🔍 Organization Validation Report</h3>

        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px;">
            <div style="padding:8px 16px;background:rgba(129,199,132,0.15);border-radius:6px;font-size:14px;">
                ✅ <strong>{correct}</strong> correctly placed
            </div>
            <div style="padding:8px 16px;background:rgba(229,115,115,0.15);border-radius:6px;font-size:14px;">
                ❌ <strong>{len(misplaced)}</strong> misplaced
            </div>
            <div style="padding:8px 16px;background:rgba(255,193,7,0.15);border-radius:6px;font-size:14px;">
                ⚠️ <strong>{no_meta}</strong> without metadata
            </div>
        </div>

        <details open>
            <summary style="cursor:pointer;padding:8px;background:var(--block-background-fill);border-radius:5px;font-size:13px;margin-bottom:8px;">
                ❌ Misplaced files ({len(misplaced)})
            </summary>
            <div style="max-height:300px;overflow-y:auto;">
                <table style="width:100%;border-collapse:collapse;font-size:13px;">
                    <thead>
                        <tr style="background:var(--background-fill-secondary);">
                            <th style="padding:6px 10px;text-align:left;">File</th>
                            <th style="padding:6px 10px;text-align:center;">Current folder</th>
                            <th style="padding:6px 10px;"></th>
                            <th style="padding:6px 10px;text-align:center;">Expected folder</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
        </details>
        {no_meta_note}
    </div>'''

    plan_json = _json.dumps(plan, ensure_ascii=False)
    yield gr.update(value=html), gr.update(visible=True, interactive=True), plan_json


def fix_misplaced_files(plan_json, progress=gr.Progress() if queue else None):
    """
    Execute the organization plan produced by validate_organization().
    Moves only the misplaced files; creates a backup before moving.
    Generator: yields inline HTML progress updates to the UI.
    """
    import json as _json

    try:
        plan = _json.loads(plan_json) if plan_json else {}
    except Exception:
        plan = {}

    if not plan or not plan.get('moves'):
        html = '''<div style="padding:20px;text-align:center;">
            <strong>Nothing to fix.</strong> Run validation first.
        </div>'''
        yield gr.update(value=html), gr.update(visible=False), gr.update(visible=False), '{}'
        return

    moves = plan['moves']
    total = len(moves)

    # ── Initial status: saving backup ─────────────────────────────────────────
    yield (
        gr.update(value='<div style="padding:12px 15px;border:1px solid var(--border-color-primary);border-radius:8px;margin:6px 0;">'
                        '💾 <strong>Saving backup…</strong>'
                        '</div>'),
        gr.update(visible=True),
        gr.update(visible=False),
        '{}'
    )

    if progress is not None:
        progress(0, desc="Saving backup before fixing...")

    save_organization_backup(plan)

    if progress is not None:
        progress(0.02, desc="Moving misplaced files...")

    # ── Move files one by one, yielding progress every 25 ────────────────────
    completed = 0
    errors    = []

    for i, move_info in enumerate(moves):
        if gl.cancel_status:
            break

        source_path = move_info['from']
        target_path = move_info['to']
        model_name  = move_info.get('model_name', os.path.basename(source_path))
        base_model  = move_info.get('base_model', '?')

        if progress is not None:
            progress((i + 1) / total, desc=f"Moving: {model_name} ({i + 1}/{total})")

        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            if os.path.exists(source_path) and not os.path.exists(target_path):
                shutil.move(source_path, target_path)
                _move_associated_files(source_path, target_path)
            print(f"[CivitAI Browser Neo] ✓ Organized: {model_name} → {base_model}/")
            completed += 1
        except Exception as e:
            errors.append(f"Failed to move {model_name}: {e}")
            print(f"[CivitAI Browser Neo] ✗ Error moving {model_name}: {e}")

        # ── Yield progress every 25 files (and after the last one) ────────────
        if (i + 1) % 25 == 0 or i == total - 1:
            yield (
                gr.update(value=_make_progress_bar_html(i + 1, total, f'📁 Moving: {model_name} → {base_model}/')),
                gr.update(visible=True),
                gr.update(visible=False),
                '{}'
            )

    # ── Final result ──────────────────────────────────────────────────────────
    success = len(errors) == 0

    if success:
        result_html = f'''
        <div style="padding:20px;text-align:center;">
            <div style="font-size:48px;margin-bottom:12px;">✅</div>
            <h3 style="margin:0 0 8px 0;color:var(--body-text-color);">Fixed! {completed} file(s) moved to correct folders.</h3>
            <p style="color:var(--body-text-color-subdued);margin:0;font-size:13px;">A backup was saved. Use "↶ Undo Fix" below to revert if needed.</p>
        </div>'''
    else:
        error_list = '<br>'.join(errors[:10])
        if len(errors) > 10:
            error_list += f'<br><em>... and {len(errors) - 10} more</em>'
        result_html = f'''
        <div style="padding:20px;">
            <h3 style="color:var(--error-text-color);">⚠️ Fixed with errors — {completed}/{total} files moved</h3>
            <details><summary style="cursor:pointer;">View errors</summary>
                <div style="margin-top:8px;font-size:13px;font-family:monospace;">{error_list}</div>
            </details>
        </div>'''

    print(f"[CivitAI Browser Neo] fix_misplaced_files: moved {completed}/{total}, errors={len(errors)}")
    yield gr.update(value=result_html), gr.update(visible=False), gr.update(visible=True), '{}'


def rollback_organization(progress=gr.Progress() if queue else None):
    """
    Rollback the last organization operation
    """
    backup = get_last_organization_backup()
    
    if not backup:
        return gr.update(value='''
            <div style="padding: 20px; text-align: center;">
                <h3>ℹ️ No Backup Found</h3>
                <p>There is no recent organization to undo.</p>
                <p>Organization backups are only available for operations performed in the current session.</p>
            </div>
        ''')
    
    # Show confirmation with backup details
    total_files = len(backup.get('moves', []))
    timestamp = backup.get('timestamp', 'Unknown')
    
    if progress is not None:
        progress(0, desc=f"Starting rollback of {total_files} files...")
    
    print(f"[CivitAI Browser Neo] Starting rollback (Backup: {timestamp})...")
    
    result = execute_rollback(progress)
    
    # Get backup stats if available
    stats = backup.get('stats', {})
    total_size = stats.get('total_size', 0)
    
    # Generate compact result message
    if result['success']:
        result_html = f'''
        <div style="padding: 20px; text-align: center; color: var(--color-accent);">
            <h2 style="margin: 0 0 15px 0;">✅ Rollback Complete!</h2>
            <div style="font-size: 16px;">
                <strong>{result['completed']} files</strong> {f'({format_size(total_size)})' if total_size > 0 else ''} restored to original locations
            </div>
            <div style="margin-top: 10px; font-size: 13px; opacity: 0.8;">
                Backup: {timestamp}
            </div>
        </div>
        '''
    else:
        error_list = '<br>'.join(result['errors'][:10])
        if len(result['errors']) > 10:
            error_list += f'<br><em>... and {len(result["errors"]) - 10} more errors</em>'
        
        result_html = f'''
        <div style="padding: 20px; text-align: center; color: var(--error-text-color);">
            <h2 style="margin: 0 0 15px 0;">⚠️ Rollback Completed with Errors</h2>
            <div style="font-size: 16px; margin-bottom: 15px;">
                {result['completed']}/{result['total']} files restored | {len(result['errors'])} errors
            </div>
            <details style="text-align: left;">
                <summary style="cursor: pointer; padding: 8px; background: var(--block-background-fill); border-radius: 5px;">
                    View errors
                </summary>
                <div style="margin-top: 10px; padding: 10px; background: var(--block-background-fill); border-radius: 5px; font-size: 13px; max-height: 200px; overflow-y: auto;">
                    {error_list}
                </div>
            </details>
        </div>
        '''
    
    print(f"[CivitAI Browser Neo] {result['message']}")
    return gr.update(value=result_html)

def save_tag_finish():
    set_globals('reset')
    return finish_returns()

def save_preview_finish():
    set_globals('reset')
    return finish_returns()

def generate_dashboard_statistics(selected_types, hide_empty_categories=True, detect_orphans=False, progress=gr.Progress() if queue else None):
    """
    Generate dashboard statistics showing disk usage by model type
    Returns HTML with detailed breakdown of files and sizes per folder
    """
    import math
    import time
    from collections import defaultdict

    format_size = _format_size  # use module-level helper

    scan_start_time = time.time()
    scanned_folder_count = 0
    skipped_files = 0
    read_errors = 0
    per_file_records = []   # [{name, path, size, category}] for top-ranking
    orphans_no_json  = []   # model files without a .json sidecar
    orphans_no_id    = []   # model files with .json but missing modelId

    if progress is not None:
        progress(0, desc="Starting dashboard generation...")
    
    # Get content types to scan
    if not selected_types:
        return gr.update(value='''
            <div style="padding: 20px; text-align: center;">
                <strong>No content types selected.</strong><br>
                <span style="color: var(--body-text-color-subdued);">Please select at least one content type to analyze.</span>
            </div>
        ''')
    
    # Dictionary to store stats: {category: {'count': int, 'size': int}}
    model_stats = defaultdict(lambda: {'count': 0, 'size': 0})
    total_files = 0
    total_size = 0
    
    # Extensions to scan
    MODEL_EXTENSIONS = ('.safetensors', '.ckpt', '.pt', '.pth', '.vae', '.zip', '.th')
    
    # Process each content type
    for content_type in selected_types:
        if progress is not None:
            progress(0.1, desc=f"Scanning {content_type}...")
        
        # Get base folder
        folder = None
        if content_type == 'Upscaler':
            # Upscalers have multiple subfolders - handle separately
            for desc in ['ESRGAN', 'RealESRGAN', 'SwinIR', 'GFPGAN', 'BSRGAN']:
                upscaler_folder = _api.contenttype_folder('Upscaler', desc)
                if upscaler_folder and os.path.isdir(str(upscaler_folder)):
                    scanned_folder_count += 1
                    category = f'Upscaler ({desc})'
                    # Scan all files in this upscaler folder
                    for root, dirs, files in os.walk(str(upscaler_folder)):
                        for file in files:
                            if file.endswith(MODEL_EXTENSIONS):
                                file_path = os.path.join(root, file)
                                try:
                                    file_size = os.path.getsize(file_path)
                                    model_stats[category]['count'] += 1
                                    model_stats[category]['size'] += file_size
                                    total_files += 1
                                    total_size += file_size
                                    per_file_records.append({'name': file, 'path': file_path, 'size': file_size, 'category': category})
                                    if detect_orphans:
                                        _js = os.path.splitext(file_path)[0] + '.json'
                                        if not os.path.exists(_js):
                                            orphans_no_json.append({'name': file, 'path': file_path, 'size': file_size})
                                        elif not (_api.safe_json_load(_js) or {}).get('modelId'):
                                            orphans_no_id.append({'name': file, 'path': file_path, 'size': file_size})
                                except:
                                    read_errors += 1
                            else:
                                skipped_files += 1
            continue
        elif content_type == 'Wildcards':
            folder = _api.contenttype_folder('Wildcards')
            wildcard_extensions = ('.txt', '.yaml', '.yml')
        elif content_type == 'Workflows':
            folder = _api.contenttype_folder('Workflows')
            wildcard_extensions = ('.json', '.workflow')
        else:
            folder = _api.contenttype_folder(content_type)
            wildcard_extensions = MODEL_EXTENSIONS
        
        if not folder or not os.path.isdir(str(folder)):
            continue

        scanned_folder_count += 1
        
        folder_str = str(folder)
        if getattr(opts, 'civitai_neo_debug_organize', False):
            print(f"\n[Dashboard] Scanning {content_type} folder: {folder_str}")
        
        # For Checkpoint and LORA: scan subfolders and categorize
        if content_type in ['Checkpoint', 'LORA']:
            # First, check if there are subfolders
            subfolders = []
            root_files = []
            
            for item in os.listdir(folder_str):
                item_path = os.path.join(folder_str, item)
                if os.path.isdir(item_path):
                    subfolders.append(item)
                    if getattr(opts, 'civitai_neo_debug_organize', False):
                        print(f"[Dashboard]   Found subfolder: {item}")
                elif item.endswith(MODEL_EXTENSIONS):
                    root_files.append(item_path)
                else:
                    skipped_files += 1
            
            if getattr(opts, 'civitai_neo_debug_organize', False):
                print(f"[Dashboard]   Total subfolders: {len(subfolders)}")
                print(f"[Dashboard]   Files in root: {len(root_files)}")
            
            # Process files in root (not in subfolders)
            if root_files:
                category = f'{content_type} → Unorganized'
                for file_path in root_files:
                    try:
                        file_size = os.path.getsize(file_path)
                        model_stats[category]['count'] += 1
                        model_stats[category]['size'] += file_size
                        total_files += 1
                        total_size += file_size
                        fname = os.path.basename(file_path)
                        per_file_records.append({'name': fname, 'path': file_path, 'size': file_size, 'category': category})
                        if detect_orphans:
                            _js = os.path.splitext(file_path)[0] + '.json'
                            if not os.path.exists(_js):
                                orphans_no_json.append({'name': fname, 'path': file_path, 'size': file_size})
                            elif not (_api.safe_json_load(_js) or {}).get('modelId'):
                                orphans_no_id.append({'name': fname, 'path': file_path, 'size': file_size})
                    except:
                        pass
                if getattr(opts, 'civitai_neo_debug_organize', False):
                    print(f"[Dashboard]   Category '{category}': {model_stats[category]['count']} files")
            
            # Process each subfolder
            for subfolder in subfolders:
                subfolder_path = os.path.join(folder_str, subfolder)
                scanned_folder_count += 1
                # Use the actual folder name as the category
                # This shows how the user has actually organized their models
                category = f'{content_type} → {subfolder}'
                
                if getattr(opts, 'civitai_neo_debug_organize', False):
                    print(f"[Dashboard]   Scanning subfolder: {subfolder}")
                    print(f"[Dashboard]   Category key: '{category}'")
                folder_file_count = 0
                folder_size_before = model_stats[category]['size']
                
                # Scan all files in subfolder (including nested subfolders)
                for root, dirs, files in os.walk(subfolder_path):
                    for file in files:
                        if file.endswith(MODEL_EXTENSIONS):
                            file_path = os.path.join(root, file)
                            try:
                                file_size = os.path.getsize(file_path)
                                model_stats[category]['count'] += 1
                                model_stats[category]['size'] += file_size
                                total_files += 1
                                total_size += file_size
                                folder_file_count += 1
                                per_file_records.append({'name': file, 'path': file_path, 'size': file_size, 'category': category})
                                if detect_orphans:
                                    _js = os.path.splitext(file_path)[0] + '.json'
                                    if not os.path.exists(_js):
                                        orphans_no_json.append({'name': file, 'path': file_path, 'size': file_size})
                                    elif not (_api.safe_json_load(_js) or {}).get('modelId'):
                                        orphans_no_id.append({'name': file, 'path': file_path, 'size': file_size})

                                # Debug first 2 files per folder
                                if folder_file_count <= 2 and getattr(opts, 'civitai_neo_debug_organize', False):
                                    print(f"[Dashboard]     File: {file} → {format_size(file_size)}")
                            except Exception as e:
                                read_errors += 1
                                if getattr(opts, 'civitai_neo_debug_organize', False):
                                    print(f"[Dashboard]     ERROR reading {file}: {e}")
                        else:
                            skipped_files += 1
                
                folder_size_after = model_stats[category]['size']
                if getattr(opts, 'civitai_neo_debug_organize', False):
                    print(f"[Dashboard]     → Found {folder_file_count} files in '{subfolder}'")
                    print(f"[Dashboard]     → Total size: {format_size(folder_size_after)}")
                    print(f"[Dashboard]     → Added: {format_size(folder_size_after - folder_size_before)}")
        
        else:
            # For other types: just count all files in folder
            category = content_type
            for root, dirs, files in os.walk(folder_str):
                for file in files:
                    if file.endswith(wildcard_extensions if content_type in ['Wildcards', 'Workflows'] else MODEL_EXTENSIONS):
                        file_path = os.path.join(root, file)
                        try:
                            file_size = os.path.getsize(file_path)
                            model_stats[category]['count'] += 1
                            model_stats[category]['size'] += file_size
                            total_files += 1
                            total_size += file_size
                            per_file_records.append({'name': file, 'path': file_path, 'size': file_size, 'category': category})
                            if detect_orphans and content_type not in ('Wildcards', 'Workflows'):
                                _js = os.path.splitext(file_path)[0] + '.json'
                                if not os.path.exists(_js):
                                    orphans_no_json.append({'name': file, 'path': file_path, 'size': file_size})
                                elif not (_api.safe_json_load(_js) or {}).get('modelId'):
                                    orphans_no_id.append({'name': file, 'path': file_path, 'size': file_size})
                        except:
                            read_errors += 1
                    else:
                        skipped_files += 1
    
    if total_files == 0:
        return gr.update(value='''
            <div style="padding: 20px; text-align: center;">
                <strong>No matching model files found.</strong><br>
                <span style="color: var(--body-text-color-subdued);">Try selecting other content types or verify your model folders.</span>
            </div>
        ''')
    
    # Debug: Show final model_stats before sorting
    if getattr(opts, 'civitai_neo_debug_organize', False):
        print("\n[Dashboard] === FINAL STATS BEFORE SORTING ===")
        for cat, stats in model_stats.items():
            print(f"[Dashboard]   {cat}: {stats['count']} files, {format_size(stats['size'])}")
        print("[Dashboard] ===================================\n")
    
    if progress is not None:
        progress(1.0, desc="Generating dashboard...")

    scan_duration_seconds = time.time() - scan_start_time

    display_stats = [
        (category, stats) for category, stats in model_stats.items()
        if (not hide_empty_categories or stats['count'] > 0)
    ]
    
    # Sort by size (descending)
    sorted_stats = sorted(display_stats, key=lambda x: x[1]['size'], reverse=True)
    
    # Generate HTML
    html_parts = []
    
    # Header with total stats
    html_parts.append(f'''
    <div style="padding: 20px; font-family: Arial, sans-serif;">
        <h2 style="margin: 0 0 20px 0; color: var(--body-text-color); text-align: center;">
            📊 Model Collection Dashboard
        </h2>
        <div style="text-align: center; font-size: 18px; margin-bottom: 30px; padding: 15px; background: var(--block-background-fill); border-radius: 8px;">
            <strong>{total_files} files ({format_size(total_size)}) → {len(sorted_stats)} categories</strong>
        </div>
        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 12px; margin-bottom: 24px;">
            <div style="padding: 8px 12px; border-radius: 8px; background: var(--block-background-fill); font-size: 13px; color: var(--body-text-color);">
                <strong>Folders scanned:</strong> {scanned_folder_count}
            </div>
            <div style="padding: 8px 12px; border-radius: 8px; background: var(--block-background-fill); font-size: 13px; color: var(--body-text-color);">
                <strong>Scan duration:</strong> {scan_duration_seconds:.2f}s
            </div>
            <div style="padding: 8px 12px; border-radius: 8px; background: var(--block-background-fill); font-size: 13px; color: var(--body-text-color);">
                <strong>Skipped files:</strong> {skipped_files}
            </div>
            <div style="padding: 8px 12px; border-radius: 8px; background: var(--block-background-fill); font-size: 13px; color: var(--body-text-color);">
                <strong>Read errors:</strong> {read_errors}
            </div>
        </div>
    ''')

    # Update scan summary banner (populated by the last "Scan for available updates" run)
    if last_update_scan:
        _outdated_count = last_update_scan['outdated_count']
        _updated_count  = last_update_scan['updated_count']
        _scanned_at     = last_update_scan['scanned_at']
        _by_type        = last_update_scan['outdated_by_type']

        if _outdated_count == 0:
            _banner_bg     = 'rgba(75, 192, 75, 0.12)'
            _banner_border = 'rgba(75, 192, 75, 0.4)'
            _banner_icon   = '&#x2705;'
            _banner_summary = f'All {_updated_count} scanned models are up to date.'
            _type_breakdown = ''
        else:
            _banner_bg     = 'rgba(255, 165, 0, 0.12)'
            _banner_border = 'rgba(255, 165, 0, 0.5)'
            _banner_icon   = '&#x26A0;&#xFE0F;'
            _s = '' if _outdated_count == 1 else 's'
            _banner_summary = f'{_outdated_count} model{_s} with updates available ({_updated_count} up to date).'
            _type_pills = ''.join(
                f'<span style="padding: 3px 10px; background: rgba(255,165,0,0.2); border-radius: 12px; font-size: 12px;">'
                f'<strong>{_t}</strong>: {len(_names)}</span>'
                for _t, _names in sorted(_by_type.items())
            )
            _type_breakdown = f'<div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;">{_type_pills}</div>'

        html_parts.append(f'''
        <div style="margin: 0 auto 20px auto; padding: 14px 18px; background: {_banner_bg}; border: 1px solid {_banner_border};
                    border-radius: 10px; max-width: 900px;">
            <div style="display: flex; align-items: flex-start; gap: 12px; font-size: 14px;">
                <span style="font-size: 22px; line-height: 1.2;">{_banner_icon}</span>
                <div>
                    <strong>Update Scan Results</strong>
                    <span style="color: var(--body-text-color-subdued); font-size: 12px; margin-left: 8px;">as of {_scanned_at}</span><br>
                    <span style="color: var(--body-text-color-subdued);">{_banner_summary}</span>
                    {_type_breakdown}
                </div>
            </div>
        </div>
        ''')

    # Charts (pie + horizontal bar) with toggle — Always show when there's data
    if sorted_stats and len(sorted_stats) > 0:
        try:
            if getattr(opts, 'civitai_neo_debug_organize', False):
                print(f"[Dashboard] Generating charts for {len(sorted_stats)} categories")

            pie_colors = [
                '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
                '#FF9F40', '#FF6B9D', '#C9CBCF', '#8DD3C7', '#FFED6F',
                '#BEBADA', '#FB8072', '#80B1D3', '#FDB462', '#B3DE69'
            ]

            # Prepare shared data: top 8 categories + "Others"
            top_categories = sorted_stats[:8]
            other_size  = sum(s['size']  for _, s in sorted_stats[8:])
            other_count = sum(s['count'] for _, s in sorted_stats[8:])
            if len(sorted_stats) > 8 and other_size > 0:
                chart_data = top_categories + [('Others', {'size': other_size, 'count': other_count})]
            else:
                chart_data = sorted_stats

            total = sum(s['size'] for _, s in chart_data)

            # ── Toggle buttons ───────────────────────────────────────────────
            html_parts.append('''
            <div style="display:flex;justify-content:center;gap:8px;margin:24px 0 0 0;">
                <button id="neo-btn-pie"
                    onclick="document.getElementById('neo-chart-pie').style.display='';
                             document.getElementById('neo-chart-bar').style.display='none';
                             this.style.background='var(--primary-500)';this.style.color='#fff';
                             document.getElementById('neo-btn-bar').style.background='var(--block-background-fill)';
                             document.getElementById('neo-btn-bar').style.color='var(--body-text-color)';"
                    style="padding:6px 18px;border-radius:20px;border:1px solid var(--border-color-primary);
                           cursor:pointer;font-size:13px;background:var(--primary-500);color:#fff;
                           transition:background 0.2s;">
                    🥧 Pie Chart
                </button>
                <button id="neo-btn-bar"
                    onclick="document.getElementById('neo-chart-bar').style.display='';
                             document.getElementById('neo-chart-pie').style.display='none';
                             this.style.background='var(--primary-500)';this.style.color='#fff';
                             document.getElementById('neo-btn-pie').style.background='var(--block-background-fill)';
                             document.getElementById('neo-btn-pie').style.color='var(--body-text-color)';"
                    style="padding:6px 18px;border-radius:20px;border:1px solid var(--border-color-primary);
                           cursor:pointer;font-size:13px;background:var(--block-background-fill);
                           color:var(--body-text-color);transition:background 0.2s;">
                    📊 Bar Chart
                </button>
            </div>
            ''')

            # ── Pie chart ────────────────────────────────────────────────────
            angles = []
            current_angle = 0
            for category, stats in chart_data:
                percentage = (stats['size'] / total * 100) if total > 0 else 0
                angle      = (stats['size'] / total * 360) if total > 0 else 0
                angles.append((category, percentage, current_angle, current_angle + angle, stats))
                current_angle += angle

            svg_parts = []
            svg_parts.append('<div id="neo-chart-pie" style="">')
            svg_parts.append('''
            <div style="display: flex; justify-content: center; margin: 20px 0; flex-wrap: wrap; gap: 40px; align-items: center;">
                <div style="position: relative;">
                    <svg viewBox="0 0 200 200" style="width: 280px; height: 280px; transform: rotate(-90deg);">
            ''')

            for i, (category, percentage, start_angle, end_angle, stats) in enumerate(angles):
                if percentage < 0.1:
                    continue
                start_rad = start_angle * math.pi / 180
                end_rad   = end_angle   * math.pi / 180
                x1 = 100 + 90 * math.cos(start_rad)
                y1 = 100 + 90 * math.sin(start_rad)
                x2 = 100 + 90 * math.cos(end_rad)
                y2 = 100 + 90 * math.sin(end_rad)
                large_arc = 1 if (end_angle - start_angle) > 180 else 0
                color = pie_colors[i % len(pie_colors)]
                svg_parts.append(f'''
                    <path d="M 100 100 L {x1:.2f} {y1:.2f} A 90 90 0 {large_arc} 1 {x2:.2f} {y2:.2f} Z"
                          fill="{color}" stroke="#ffffff" stroke-width="2"
                          style="transition: opacity 0.3s; cursor: pointer;"
                          onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
                        <title>{category}: {format_size(stats['size'])} ({percentage:.1f}%)</title>
                    </path>
                ''')

            svg_parts.append('</svg></div>')
            svg_parts.append('<div style="display:flex;flex-direction:column;justify-content:center;gap:8px;max-width:300px;">')
            for i, (category, percentage, _, _, stats) in enumerate(angles):
                if percentage < 0.1:
                    continue
                color = pie_colors[i % len(pie_colors)]
                svg_parts.append(f'''
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div style="width:16px;height:16px;background:{color};border-radius:3px;flex-shrink:0;"></div>
                        <div style="color:var(--body-text-color);font-size:13px;">
                            <strong>{category}</strong><br>
                            <span style="color:var(--body-text-color-subdued);font-size:12px;">
                                {format_size(stats['size'])} ({percentage:.1f}%)
                            </span>
                        </div>
                    </div>
                ''')
            svg_parts.append('</div></div></div>')  # legend / flex row / pie wrapper
            html_parts.append(''.join(svg_parts))

            # ── Horizontal bar chart ─────────────────────────────────────────
            max_size = chart_data[0][1]['size'] if chart_data else 1
            bar_rows = []
            for i, (category, stats) in enumerate(chart_data):
                pct      = (stats['size'] / total    * 100) if total    > 0 else 0
                bar_pct  = (stats['size'] / max_size * 100) if max_size > 0 else 0
                color    = pie_colors[i % len(pie_colors)]
                short_cat = category if len(category) <= 30 else category[:27] + '\u2026'
                bar_rows.append(f'''
                    <div style="display:grid;grid-template-columns:220px 1fr 120px;
                                align-items:center;gap:10px;padding:8px 0;
                                border-bottom:1px solid var(--border-color-primary);">
                        <div style="font-size:13px;color:var(--body-text-color);font-weight:bold;
                                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                             title="{category}">{short_cat}</div>
                        <div style="background:var(--block-background-fill);border-radius:4px;
                                    height:20px;overflow:hidden;">
                            <div style="background:{color};width:{bar_pct:.1f}%;height:100%;
                                        border-radius:4px;transition:width 0.4s ease;">
                            </div>
                        </div>
                        <div style="font-size:12px;color:var(--body-text-color-subdued);
                                    text-align:right;white-space:nowrap;">
                            {format_size(stats['size'])}
                            <span style="opacity:0.6;">({pct:.1f}%)</span>
                        </div>
                    </div>
                ''')

            html_parts.append(f'''
            <div id="neo-chart-bar" style="display:none;">
                <div style="max-width:900px;margin:20px auto;padding:0 8px;">
                    {''.join(bar_rows)}
                </div>
            </div>
            ''')

        except Exception as e:
            error_msg = f'Chart generation failed: {str(e)}'
            if getattr(opts, 'civitai_neo_debug_organize', False):
                print(f"[Dashboard ERROR] {error_msg}")
                import traceback
                traceback.print_exc()
            html_parts.append(f'<div style="color:red;padding:10px;text-align:center;">{error_msg}</div>')
    
    # Table with breakdown
    if sorted_stats:
        html_parts.append('''
        <table style="width: 100%; border-collapse: collapse; margin: 0 auto; max-width: 900px;">
            <thead>
                <tr style="background: var(--block-title-background-fill); border-bottom: 2px solid var(--border-color-primary);">
                    <th style="padding: 12px; text-align: left; font-size: 14px;">MODEL TYPE</th>
                    <th style="padding: 12px; text-align: center; font-size: 14px;">FILES</th>
                    <th style="padding: 12px; text-align: right; font-size: 14px;">TOTAL SIZE</th>
                    <th style="padding: 12px; text-align: right; font-size: 14px;">% OF TOTAL</th>
                </tr>
            </thead>
            <tbody>
        ''')
        
        for category, stats in sorted_stats:
            percentage = (stats['size'] / total_size * 100) if total_size > 0 else 0
            
            # Create visual bar for percentage
            bar_width = int(percentage)
            bar_html = f'''
            <div style="background: linear-gradient(90deg, var(--primary-500) 0%, var(--primary-400) 100%); 
                        height: 6px; width: {bar_width}%; border-radius: 3px; margin-top: 4px;"></div>
            '''
            
            html_parts.append(f'''
                <tr style="border-bottom: 1px solid var(--border-color-primary);">
                    <td style="padding: 12px; font-weight: bold; color: var(--body-text-color);">
                        {category}
                        {bar_html}
                    </td>
                    <td style="padding: 12px; text-align: center; color: var(--body-text-color-subdued);">
                        {stats['count']}
                    </td>
                    <td style="padding: 12px; text-align: right; color: var(--body-text-color);">
                        {format_size(stats['size'])}
                    </td>
                    <td style="padding: 12px; text-align: right; color: var(--body-text-color-subdued);">
                        {percentage:.1f}%
                    </td>
                </tr>
            ''')
        
        html_parts.append('''
            </tbody>
        </table>
        ''')

    # Top 10 largest individual files and top 10 categories by file count
    if per_file_records:
        top_by_size  = sorted(per_file_records, key=lambda x: x['size'], reverse=True)[:10]
        top_by_count = sorted(sorted_stats, key=lambda x: x[1]['count'], reverse=True)[:10]

        html_parts.append('''
        <h3 style="color:var(--body-text-color);margin:32px 0 12px 0;text-align:center;">
            &#127942; Top 10 Largest Individual Files
        </h3>
        <table style="width:100%;border-collapse:collapse;margin:0 auto 8px auto;max-width:900px;">
            <thead><tr style="background:var(--block-title-background-fill);border-bottom:2px solid var(--border-color-primary);">
                <th style="padding:10px;text-align:left;font-size:13px;">#</th>
                <th style="padding:10px;text-align:left;font-size:13px;">FILENAME</th>
                <th style="padding:10px;text-align:left;font-size:13px;">CATEGORY</th>
                <th style="padding:10px;text-align:right;font-size:13px;">SIZE</th>
            </tr></thead><tbody>
        ''')
        for i, rec in enumerate(top_by_size, 1):
            html_parts.append(f'''
                <tr style="border-bottom:1px solid var(--border-color-primary);">
                    <td style="padding:10px;color:var(--body-text-color-subdued);font-weight:bold;">#{i}</td>
                    <td style="padding:10px;font-family:monospace;font-size:12px;color:var(--body-text-color);">{rec['name']}</td>
                    <td style="padding:10px;font-size:12px;color:var(--body-text-color-subdued);">{rec['category']}</td>
                    <td style="padding:10px;text-align:right;font-weight:bold;color:var(--body-text-color);">{format_size(rec['size'])}</td>
                </tr>
            ''')
        html_parts.append('</tbody></table>')

        html_parts.append('''
        <h3 style="color:var(--body-text-color);margin:32px 0 12px 0;text-align:center;">
            &#128230; Top 10 Categories by File Count
        </h3>
        <table style="width:100%;border-collapse:collapse;margin:0 auto 8px auto;max-width:900px;">
            <thead><tr style="background:var(--block-title-background-fill);border-bottom:2px solid var(--border-color-primary);">
                <th style="padding:10px;text-align:left;font-size:13px;">#</th>
                <th style="padding:10px;text-align:left;font-size:13px;">CATEGORY</th>
                <th style="padding:10px;text-align:center;font-size:13px;">FILES</th>
                <th style="padding:10px;text-align:right;font-size:13px;">TOTAL SIZE</th>
            </tr></thead><tbody>
        ''')
        for i, (cat, stats) in enumerate(top_by_count, 1):
            html_parts.append(f'''
                <tr style="border-bottom:1px solid var(--border-color-primary);">
                    <td style="padding:10px;color:var(--body-text-color-subdued);font-weight:bold;">#{i}</td>
                    <td style="padding:10px;font-weight:bold;color:var(--body-text-color);">{cat}</td>
                    <td style="padding:10px;text-align:center;color:var(--body-text-color-subdued);">{stats['count']}</td>
                    <td style="padding:10px;text-align:right;color:var(--body-text-color);">{format_size(stats['size'])}</td>
                </tr>
            ''')
        html_parts.append('</tbody></table>')

    # Orphan detection results
    if detect_orphans:
        total_orphans = len(orphans_no_json) + len(orphans_no_id)
        if total_orphans:
            html_parts.append(f'''
            <h3 style="color:#e57373;margin:32px 0 12px 0;text-align:center;">
                &#9888;&#65039; Orphan Files &mdash; {total_orphans} found
            </h3>
            <p style="text-align:center;color:var(--body-text-color-subdued);font-size:13px;margin-bottom:16px;">
                These model files have no CivitAI metadata. They can still be used,
                but won&apos;t show info in the overlay.
            </p>
            ''')
            def _orphan_table(records, heading):
                rows = ''.join(
                    f'<tr style="border-bottom:1px solid var(--border-color-primary);">'
                    f'<td style="padding:8px 12px;font-family:monospace;font-size:12px;color:var(--body-text-color);">{r["name"]}</td>'
                    f'<td style="padding:8px 12px;text-align:right;color:var(--body-text-color-subdued);">{format_size(r["size"])}</td>'
                    f'</tr>'
                    for r in records[:50]
                )
                extra = (f'<tr><td colspan="2" style="padding:8px 12px;color:var(--body-text-color-subdued);'
                         f'font-style:italic;">... and {len(records)-50} more</td></tr>') if len(records) > 50 else ''
                return (
                    f'<h4 style="color:var(--body-text-color);margin:20px 0 8px 20px;">{heading} ({len(records)} files)</h4>'
                    f'<table style="width:100%;border-collapse:collapse;margin:0 auto 8px auto;max-width:900px;">'
                    f'<tbody>{rows}{extra}</tbody></table>'
                )
            if orphans_no_json:
                html_parts.append(_orphan_table(orphans_no_json, 'No .json sidecar'))
            if orphans_no_id:
                html_parts.append(_orphan_table(orphans_no_id, 'Has .json but no modelId'))
        else:
            html_parts.append('<p style="text-align:center;color:rgba(75,192,75,0.9);margin:24px 0;font-size:14px;">&#9989; No orphan files found.</p>')

    # Store raw data for CSV / JSON export
    global last_dashboard_data
    last_dashboard_data = {
        'generated_at': __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_files': total_files,
        'total_size': total_size,
        'categories': [
            {'name': cat, 'count': s['count'], 'size_bytes': s['size'], 'size_human': format_size(s['size'])}
            for cat, s in sorted_stats
        ],
        'top_by_size': [
            {'name': r['name'], 'path': r['path'], 'size_bytes': r['size'],
             'size_human': format_size(r['size']), 'category': r['category']}
            for r in sorted(per_file_records, key=lambda x: x['size'], reverse=True)[:25]
        ],
        'orphans_no_json': [{'name': r['name'], 'path': r['path'], 'size_bytes': r['size']} for r in orphans_no_json],
        'orphans_no_id':   [{'name': r['name'], 'path': r['path'], 'size_bytes': r['size']} for r in orphans_no_id],
    }
    html_parts.append(f'''
        <div style="margin-top: 20px; text-align: center; font-size: 13px; color: var(--body-text-color-subdued);">
            <em>Dashboard generated on {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em>
        </div>
    </div>
    ''')
    
    return gr.update(value=''.join(html_parts))


def export_dashboard_csv():
    """Return the last dashboard scan data as a CSV string for Blob download."""
    if not last_dashboard_data:
        return gr.update(value='')
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['category', 'files', 'size_bytes', 'size_human'])
    for cat in last_dashboard_data['categories']:
        w.writerow([cat['name'], cat['count'], cat['size_bytes'], cat['size_human']])
    return gr.update(value=buf.getvalue())


def export_dashboard_json():
    """Return the last dashboard scan data as a JSON string for Blob download."""
    if not last_dashboard_data:
        return gr.update(value='')
    import json
    return gr.update(value=json.dumps(last_dashboard_data, indent=2, ensure_ascii=False))


def scan_finish():
    set_globals('reset')
    gl.update_mode = False
    return (
        gr.update(interactive=True, visible=True),
        gr.update(interactive=True, visible=True),
        gr.update(interactive=True, visible=True),
        gr.update(interactive=True, visible=True),
        gr.update(interactive=True, visible=True),
        gr.update(interactive=False, visible=False),
        gr.update(interactive=not no_update, visible=not no_update)
    )


def _render_update_mode_banner(count):
    """Return the full HTML for the Update Mode banner + mode switcher."""
    retention = getattr(opts, 'civitai_neo_update_retention', 'replace')
    return f'''
<div class="civupdate-bar" id="civupdate-bar">
  <div class="civupdate-switcher">
    <button class="mode-pill" onclick="exitUpdateMode()">🔍 Search CivitAI</button>
    <button class="mode-pill mode-pill-active">🔄 Update Local Models ({count})</button>
  </div>
  <div class="civupdate-action-bar">
    <span class="civupdate-count">🔄 <strong>{count}</strong> update{"s" if count != 1 else ""} available</span>
    <button class="civupdate-btn-all" id="civupdate-update-btn" onclick="updateOrSelectedModels()">⬆️ Update All ({count})</button>
    <span class="civupdate-retention">Retention: {retention}</span>
  </div>
</div>'''


def enter_update_mode():
    """Called via .then() after load_to_browser_outdated — activates the Update Mode banner."""
    gl.update_mode = True
    count = len(gl.update_items)
    if count == 0:
        return gr.update(value='')
    return gr.update(value=_render_update_mode_banner(count))


def exit_update_mode(content_type, sort_type, period_type, use_search_term, search_term,
                     tile_count, base_filter, nsfw, exact_search):
    """Deactivates Update Mode, clears banner, and returns to a normal browser state."""
    gl.update_mode = False
    gl.update_items = []
    placeholder = '<div style="font-size: 24px; text-align: center; margin: 50px;">Click the search icon to load models.<br>Use the filter icon to filter results.</div>'
    return (
        gr.update(value=''),           # update_mode_banner cleared
        gr.update(value=placeholder),  # list_html reset
        gr.update(interactive=False),  # prev page
        gr.update(interactive=False),  # next page
        gr.update(value=1, maximum=1), # page slider
    )

## === ANXETY EDITs ===
def _resolve_browser_local_folders(content_type):
    folders_to_check = []

    selected_types = content_type if content_type else ['Checkpoint', 'LORA']
    if isinstance(selected_types, str):
        selected_types = [selected_types]

    for item in selected_types:
        if item == 'LORA':
            folder = _api.contenttype_folder('LORA')
            if folder:
                folders_to_check.append(folder)
        elif item == 'Upscaler':
            for sub in ['SwinIR', 'RealESRGAN', 'GFPGAN', 'BSRGAN', 'ESRGAN']:
                folder = _api.contenttype_folder(item, sub)
                if folder:
                    folders_to_check.append(folder)
        else:
            folder = _api.contenttype_folder(item)
            if folder:
                folders_to_check.append(folder)

    folders_to_check = sorted(list(set(folders_to_check)))
    return folders_to_check


def prepare_local_browser_url_list(content_type, tile_count, use_search_term=None, search_term=None):
    folders_to_check = _resolve_browser_local_folders(content_type)
    if not folders_to_check:
        gl.url_list = {}
        return False

    files = list_files(folders_to_check)
    if not files:
        gl.url_list = {}
        return False

    local_term = (search_term or '').strip().lower()
    if use_search_term == 'Model name' and local_term:
        filtered_files = []
        for file_path in files:
            stem = os.path.splitext(os.path.basename(file_path))[0].lower()
            if local_term in stem:
                filtered_files.append(file_path)
        files = filtered_files

    all_model_ids = []
    for file_path in files:
        model_id = get_models(file_path, gen_hash=False)
        if model_id in [None, 'offline', 'Model not found']:
            continue
        all_model_ids.append(f'&ids={model_id}')

    all_model_ids = sorted(list(set(all_model_ids)))
    if not all_model_ids:
        gl.url_list = {}
        return False

    tile_count = int(tile_count) if tile_count else 27
    tile_count = max(1, tile_count)

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    model_chunks = list(chunks(all_model_ids, tile_count))
    base_url = f"https://{_api.get_civitai_domain()}/api/v1/models?limit=100&nsfw=true"
    gl.url_list = {i + 1: f"{base_url}{''.join(chunk)}" for i, chunk in enumerate(model_chunks)}
    return True


def load_to_browser(content_type, sort_type, period_type, use_search_term, search_term, tile_count, base_filter, nsfw, exact_search):
    global from_ver, from_installed

    model_list_return = _api.initial_model_page(
        content_type,
        sort_type,
        period_type,
        use_search_term,
        search_term,
        1,
        base_filter,
        False,
        nsfw,
        exact_search,
        tile_count,
        True
    )
    from_ver, from_installed =  False, False
    return (
        *model_list_return,
        gr.update(interactive=True, visible=True),
        gr.update(interactive=True, visible=True),
        gr.update(interactive=True, visible=True),
        gr.update(interactive=True, visible=True),
        gr.update(interactive=False, visible=False),
        gr.update(interactive=False, visible=False),
        gr.update(value='<div style="min-height: 0px;"></div>')
    )

def cancel_scan():
    gl.cancel_status = True

    while True:
        if not gl.scan_files:
            gl.cancel_status = False
            return
        else:
            time.sleep(0.5)
            continue
