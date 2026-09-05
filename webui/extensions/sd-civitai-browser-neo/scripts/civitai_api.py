import urllib.parse
import time
import requests
import platform
import json
import os
import re
import gradio as gr
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from html import escape
from io import BytesIO
from PIL import Image

# === WebUI imports ===
from modules.paths import models_path, extensions_dir, data_path
from modules.images import read_info_from_image
from modules.shared import cmd_opts, opts

# === Extension imports ===
import scripts.civitai_download as _download
import scripts.civitai_file_manage as _file
import scripts.civitai_global as gl
from scripts.civitai_global import print, debug_print


gl.init()


## === ANXETY EDITs ===
# Mapping for short/clear display names for model types
MODEL_TYPE_DISPLAY_NAMES = {
    'TextualInversion': 'Embedding',
    'AestheticGradient': 'Aesthetic',
    'MotionModule': 'Motion',
    'Workflows': 'Workflow',
    'Wildcards': 'Wildcard'
}

def get_display_type(type_name):
    """Return short/clear display name for model type"""
    return MODEL_TYPE_DISPLAY_NAMES.get(type_name, type_name)

def is_early_access(version_data):
    """Check if the model is an early access"""
    avail = version_data.get('availability')
    return isinstance(avail, str) and avail == 'EarlyAccess'

# Short abbreviations for base model names used in card badges
BASE_MODEL_SHORT = {
    'illustrious':          'IL',
    'illustrious xl':       'IL',
    'noobai':               'Nai',
    'noobai xl':            'Nai',
    'pony':                 'Pony',
    'sdxl 1.0':             'XL',
    'sdxl':                 'XL',
    'sdxl turbo':           'XL',
    'sd 1.5':               'SD1',
    'sd 1.4':               'SD1',
    'sd 2.0':               'SD2',
    'sd 2.1':               'SD2',
    'flux.1 d':             'F1',
    'flux.1 s':             'F1',
    'flux.1':               'F1',
    'flux.2 klein 4b':      'F2',
    'flux.2 klein 9b-base': 'F2',
    'flux.2 klein 9b':      'F2',
    'flux.2 d':             'F2',
    'flux.2':               'F2',
    'flux':                 'F1',
    'wan video 1.3b t2v':       'T2V',
    'wan video 14b t2v':        'T2V',
    'wan video 14b i2v 480p':   'I2V',
    'wan video 14b i2v 720p':   'I2V',
    'wan video 2.2 t2v-a14b':   'T2V',
    'wan video 2.2 i2v-a14b':   'I2V',
    'wan video 2.2 ti2v-5b':    'TI2V',
    'wan video 2.5 t2v':        'T2V',
    'wan video 2.5 i2v':        'I2V',
    'wan video 1.3':            'Wan',
    'wan':                      'Wan',
    'qwen':                 'Qwen',
    'z-image':              'ZiT',
    'lumina':               'Lum',
    'hunyuanvideo':         'HYV',
    'hunyuan video':        'HYV',
    'ltxv':                 'LTXV',
    'cosmos':               'Cosm',
    'other':                'Other',
}

def get_base_model_short(base_model: str) -> str:
    """Return short abbreviation for a base model name, or '' if unknown"""
    if not base_model:
        return ''
    key = base_model.strip().lower()
    if key in BASE_MODEL_SHORT:
        return BASE_MODEL_SHORT[key]
    # Prefix fallback (e.g. "Illustrious XL v0.1" → IL)
    for k, v in BASE_MODEL_SHORT.items():
        if key.startswith(k):
            return v
    return ''


def get_civitai_domain():
    """Return the configured CivitAI domain based on SFW-only toggle."""
    return 'civitai.com' if getattr(opts, 'civitai_sfw_only', False) else 'civitai.red'

# ─────────────────────────────────────────────────────────────────────────────
# v0.8.1 — Local Trigger Words Lookup
# ─────────────────────────────────────────────────────────────────────────────

def get_local_trigger_words(content_type, model_filename, sha256_value=None, allow_legacy=False):
    """Try to load trigger words from local .json sidecar file.

    Priority is the grouped field used to preserve CivitAI rows.
    Legacy flat field can be enabled as fallback with allow_legacy=True.
    """
    try:
        if not content_type or not model_filename:
            return None

        def _extract_groups(data):
            if not isinstance(data, dict):
                return None

            raw_groups = data.get('activation text groups')
            if raw_groups is None:
                raw_groups = data.get('activation_text_groups')

            groups = []
            if isinstance(raw_groups, list):
                groups = [str(g).strip() for g in raw_groups if str(g).strip()]
            elif isinstance(raw_groups, str) and raw_groups.strip():
                # Accept manually edited JSON where groups might be serialized as text.
                parsed = None
                try:
                    parsed = json.loads(raw_groups)
                except Exception:
                    parsed = None
                if isinstance(parsed, list):
                    groups = [str(g).strip() for g in parsed if str(g).strip()]
                else:
                    groups = [g.strip() for g in re.split(r'[\n\r]+', raw_groups) if g.strip()]

            if groups:
                return groups

            if allow_legacy and data.get('activation text'):
                text = data.get('activation text', '')
                return [t.strip() for t in re.split(r'[,;\n\r]+', text) if t.strip()]

            return None

        model_folder = contenttype_folder(content_type)
        if not model_folder:
            return None

        model_folder = Path(model_folder)
        name_stem = Path(model_filename).stem
        candidate_names = [f'{name_stem}.json', f'{model_filename}.json']

        # Fast path: direct files in root folder
        for candidate in candidate_names:
            direct = model_folder / candidate
            if direct.exists():
                data = safe_json_load(str(direct))
                groups = _extract_groups(data)
                if groups:
                    return groups

        # Fallback: recursive search for nested organization paths (e.g. Wan/I2V)
        for candidate in candidate_names:
            matches = list(model_folder.rglob(candidate))
            if not matches:
                continue
            # Prefer shortest path first (usually closest to root / primary install path)
            matches.sort(key=lambda p: len(str(p)))
            for json_file in matches:
                data = safe_json_load(str(json_file))
                groups = _extract_groups(data)
                if groups:
                    return groups

        return None
    except Exception:
        return None

STATUS_BADGE_DAYS = 14  # days window for "New" / "Updated" badges

def get_status_badge_type(item) -> str:
    """Return 'new', 'updated', or '' based on how recently the latest version was published.
    Mirrors CivitAI's logic: 'new' if model has only 1 version (just created),
    'updated' if model has multiple versions and the latest is recent."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    threshold = timedelta(days=STATUS_BADGE_DAYS)

    def parse_dt(s):
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace('Z', '+00:00'))
        except Exception:
            return None

    versions = item.get('modelVersions', [])
    if not versions:
        return ''

    latest_published = parse_dt(versions[0].get('publishedAt', ''))
    if not latest_published or (now - latest_published) >= threshold:
        return ''  # Latest version is not recent

    # Recent version: New = only 1 version (just created), Updated = multiple versions
    return 'new' if len(versions) == 1 else 'updated'

# This nsfwlevel system is not accurate...
def is_model_nsfw(model_data, nsfw_level=12):
    """Determine if a model is NSFW based on its metadata and first image"""
    if model_data.get('nsfw'):
        return True
    model_versions = model_data.get('modelVersions')
    if model_versions and model_versions[0].get('images'):
        first_image = model_versions[0]['images'][0]
        if first_image.get('nsfwLevel', 0) >= nsfw_level:
            return True
    return False

def normalize_sha256(sha256_hash):
    """Normalize SHA256 hash to uppercase and validate format"""
    if not sha256_hash:
        return None
    return sha256_hash.strip().upper()

def safe_json_load(file_path):
    """Safely load JSON file with error handling"""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return None

def safe_json_save(file_path, data):
    """Safely save JSON file with error handling"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")
        return False


def contenttype_folder(content_type, desc=None, custom_folder=None):
    """
    Returns the appropriate folder path for a given content type.
    Args:
        content_type (str): The type of content/model.
        desc (str, optional): Description or additional info for type-specific logic.
        custom_folder (str or Path, optional): Custom base folder to use instead of defaults.
    Returns:
        Path: The resolved folder path for the content type, or None if not found.
    """
    desc_upper  = (desc or 'PLACEHOLDER').upper()                               # Uppercase description for type checks
    main_models = Path(custom_folder) if custom_folder else Path(models_path)   # Main models folder path
    main_data   = Path(custom_folder) if custom_folder else Path(data_path)     # Main data folder path (WebUI root)
    ext_dir     = Path(extensions_dir)                                          # Extensions directory path

    def resolve_path(attr, fallback):
        # Returns a Path from cmd_opts if set, otherwise fallback
        if getattr(cmd_opts, attr, None) and not custom_folder:
            return Path(getattr(cmd_opts, attr))
        return fallback

    # Mapping for content types
    content_type_map = {
        'modelFolder': lambda: main_models,
        'Checkpoint': lambda: resolve_path('ckpt_dir', resolve_path('ckpt_dirs', main_models / 'Stable-diffusion')),
        'TextualInversion': lambda: resolve_path('embeddings_dir', _resolve_embeddings_folder(main_models, main_data)),
        'AestheticGradient': lambda: (Path(custom_folder) if custom_folder else ext_dir / 'stable-diffusion-webui-aesthetic-gradients') / 'aesthetic_embeddings',
        'LORA': lambda: resolve_path('lora_dir', main_models / 'Lora'),
        'LoCon': lambda: resolve_path('lora_dir', main_models / 'Lora'), # 💩
        'DoRA': lambda: resolve_path('lora_dir', main_models / 'Lora'),  # 💩
        'VAE': lambda: resolve_path('vae_dir', main_models / 'VAE'),
        'Controlnet': lambda: resolve_path('controlnet_dir', main_models / 'ControlNet'),
        'Poses': lambda: main_models / 'Poses',
        'MotionModule': lambda: ext_dir / 'sd-webui-animatediff' / 'model',
        'Workflows': lambda: main_models / 'Workflows',
        'Detection': lambda: main_models / 'adetailer',
        'Other': lambda: main_models / 'adetailer' if 'ADETAILER' in desc_upper else main_models / 'Other',
        'Wildcards': lambda: ext_dir / 'sd-dynamic-prompts' / 'wildcards',
        'Upscaler': lambda: _resolve_upscaler_folder(desc_upper, main_models, resolve_path)
    }

    def _resolve_embeddings_folder(main_models, main_data):
        """Detect embeddings folder: new Neo puts it inside models/, old Forge at webui root.
        If both exist with content, prefer new layout (models/embeddings) and warn the user."""
        new_path = main_models / 'embeddings'   # Forge Neo (new): models/embeddings
        old_path = main_data / 'embeddings'     # Forge classic / old: <webui_root>/embeddings
        new_exists = new_path.exists()
        old_exists = old_path.exists()
        if new_exists and old_exists:
            new_has_files = any(new_path.iterdir())
            old_has_files = any(old_path.iterdir())
            if new_has_files and old_has_files:
                debug_print(
                    f"[Embeddings] Both '{new_path}' and '{old_path}' exist and have files. "
                    f"Using '{new_path}' (Forge Neo layout). "
                    f"Consider moving files from '{old_path}' to '{new_path}'."
                )
            return new_path  # always prefer new layout when both exist
        if new_exists:
            return new_path
        if old_exists:
            return old_path
        return new_path  # default to new layout when neither exists yet

    def _resolve_upscaler_folder(desc, main_models, resolve_path):
        """Detect upscaler folder: new Neo consolidates everything under ESRGAN/.
        Falls back to ESRGAN/ when the specific subfolder doesn't exist on disk."""
        esrgan = resolve_path('esrgan_models_path', main_models / 'ESRGAN')
        if 'SWINIR' in desc:
            specific = resolve_path('swinir_models_path', main_models / 'SwinIR')
            return specific if specific.exists() else esrgan
        if 'REALESRGAN' in desc:
            specific = resolve_path('realesrgan_models_path', main_models / 'RealESRGAN')
            return specific if specific.exists() else esrgan
        if 'GFPGAN' in desc:
            specific = resolve_path('gfpgan_models_path', main_models / 'GFPGAN')
            return specific if specific.exists() else esrgan
        if 'BSRGAN' in desc:
            specific = resolve_path('bsrgan_models_path', main_models / 'BSRGAN')
            return specific if specific.exists() else esrgan
        return esrgan

    # Get the folder resolver function for the content type
    folder_resolver = content_type_map.get(content_type)
    if folder_resolver:
        try:
            result = folder_resolver()
            if result is None:
                debug_print(f"Warning: Folder resolver returned None for content_type '{content_type}'")
                return None
            return result
        except Exception as e:
            debug_print(f"Error resolving folder for content_type '{content_type}': {e}")
            return None

    debug_print(f"Warning: Unknown content_type '{content_type}', no folder mapping found")
    return None


def update_mode_page_html(content_type_filter, base_filter, tile_count, current_page):
    """Render the update-mode card grid from gl.update_items (no API call)."""

    def _fam_matches(fam, bf_lower):
        if fam is None:
            return False
        fam_l = fam.lower()
        return any(fam_l in b or b in fam_l for b in bf_lower)

    def _type_short(model_type):
        _map = {
            'Checkpoint': 'CKPT', 'LORA': 'LORA', 'LoCon': 'LORA', 'DoRA': 'LORA',
            'TextualInversion': 'TI', 'Controlnet': 'CTRL', 'VAE': 'VAE',
            'Upscaler': 'UPSCL', 'Wildcards': 'WILD', 'Workflows': 'WFLOW',
        }
        return _map.get(model_type, (model_type or 'UNK')[:4].upper())

    items = list(gl.update_items)

    if not items:
        return ('<div style="font-size:24px;text-align:center;margin:50px">'
                'No updates found.</div>', 1, 1, False, False)

    # Content-type filter
    if content_type_filter:
        ct_list = content_type_filter if isinstance(content_type_filter, list) else [content_type_filter]
        if ct_list:
            items = [i for i in items if i['model_type'] in ct_list]

    # Base-model / family filter
    if base_filter:
        bf_list = base_filter if isinstance(base_filter, list) else [base_filter]
        if bf_list:
            bf_lower = [b.lower() for b in bf_list]
            items = [i for i in items if _fam_matches(i.get('family'), bf_lower)]

    if not items:
        return ('<div style="font-size:24px;text-align:center;margin:50px">'
                'No updates match the current filters.</div>', 1, 1, False, False)

    # Pagination
    tile_count = int(tile_count or 27)
    total = len(items)
    total_pages = max(1, (total + tile_count - 1) // tile_count)
    current_page = max(1, min(int(current_page or 1), total_pages))
    page_items = items[(current_page - 1) * tile_count: current_page * tile_count]

    cards_html = []
    for item in page_items:
        model_id  = item['model_id']
        model_name = item['model_name']
        model_type = item['model_type']
        family     = item.get('family') or ''
        inst_ver   = item.get('installed_ver', '?')
        new_ver    = item.get('latest_ver', '?')
        preview_url = item.get('preview_url') or ''
        inst_ver_id = item.get('installed_ver_id')
        latest_ver_id = item.get('latest_ver_id')
        available_versions = item.get('available_versions', [])

        type_badge  = _type_short(model_type)
        fam_up      = family.upper()
        fam_slug    = family.lower().replace(' ', '-') if family else ''

        thumb_html = (
            f'<img src="{preview_url}" loading="lazy" onerror="this.style.display=\'none\'">'
            if preview_url
            else '<div class="update-card-no-thumb">🖼</div>'
        )
        family_badge_html = (
            f'<span class="update-badge update-badge-family {fam_slug}">{fam_up}</span>'
            if family else ''
        )
        type_badge_html = f'<span class="update-badge update-badge-type">{type_badge}</span>'
        js_update = f"updateSingleModel('{model_id}','{fam_up}')"
        chk_id    = f"upchk-{model_id}-{fam_up}"
        model_str = f"{model_name} ({model_id})"

        # Build version checkbox list (revamp: user-selectable versions)
        ver_options = []
        for ver in available_versions:
            ver_id = ver.get('id')
            ver_name = ver.get('name', '?')
            is_checked = 'checked' if ver_id == latest_ver_id else ''
            badge_html = ''
            if ver_id == inst_ver_id:
                badge_html = '<span class="ver-badge installed">installed</span>'
            elif ver_id == latest_ver_id:
                badge_html = '<span class="ver-badge latest">latest</span>'
            ver_options.append(
                f'<label class="ver-option">'
                f'<input type="checkbox" class="ver-checkbox" data-ver-id="{ver_id}" {is_checked}> '
                f'{ver_name}{badge_html}</label>'
            )
        ver_list_html = (
            '<div class="update-card-ver-list">' + ''.join(ver_options) + '</div>'
            if ver_options else ''
        )

        cards_html.append(f'''<figure class="civmodelcard update-mode-card" data-model-id="{model_id}" data-family="{fam_up}">
  <input type="checkbox" class="model-checkbox" id="{chk_id}" onchange="multi_model_select('{model_str}', '{model_type}', this.checked); syncUpdateBtn()">
  <label for="{chk_id}" class="custom-checkbox"><span class="checkbox-checkmark"></span></label>
  <div class="civmodelcard-img-wrapper update-card-thumb">{thumb_html}</div>
  <figcaption class="update-card-caption">
    <div class="update-card-name" title="{model_name}">{model_name}</div>
    <div class="update-card-badges">{type_badge_html}{family_badge_html}</div>
    <div class="update-card-versions"><span class="ver-old">{inst_ver}</span><span class="ver-arrow"> → </span><span class="ver-new">{new_ver}</span></div>
    {ver_list_html}
    <button class="update-card-btn" onclick="{js_update}" title="Update this model">⬆</button>
  </figcaption>
</figure>''')

    html = f'<div class="civmodelcards update-mode-grid">{"".join(cards_html)}</div>'
    return (html, total_pages, current_page,
            current_page > 1, current_page < total_pages)


def model_list_html(json_data):
    def filter_versions(item, hide_early_access, current_time):
        """Filter model versions based on file presence and early access status"""
        versions = []
        for version in item.get('modelVersions', []):
            if not version.get('files'):
                continue
            if hide_early_access and is_early_access(version):
                continue
            versions.append(version)
        return versions

    def collect_existing_files(model_folders):
        """Collect existing file names and SHA256 hashes from model folders"""
        files_set = set()
        sha256_set = set()
        for folder in model_folders:
            if folder is None:
                continue
            for root, _, files in os.walk(folder, followlinks=True):
                for file in files:
                    files_set.add(file.lower())
                    if file.endswith('.json'):
                        json_path = os.path.join(root, file)
                        json_data = safe_json_load(json_path)
                        if json_data and isinstance(json_data, dict):
                            sha256 = normalize_sha256(json_data.get('sha256'))
                            if sha256:
                                sha256_set.add(sha256)
        return files_set, sha256_set

    ## === ANXETY EDITs ===
    def get_model_card(item, existing_files, existing_files_sha256, playback, favorite_creators):
        """Build HTML for a single model card (civmodelcard - Browser Card)"""
        model_id = item.get('id')
        model_name = item.get('name', '')
        is_nsfw = is_model_nsfw(item)
        nsfw_class = 'civcardnsfw' if is_nsfw else ''

        # Creator info for favorite/ban display
        _creator_data = item.get('creator', {}) or {}
        model_uploader_card = (_creator_data.get('username', '') or '').strip()
        fav_class = ' civcard-favorite' if model_uploader_card in favorite_creators else ''

        # Find the first installed version or fallback to the first version
        display_version = None
        for version in item.get('modelVersions', []):
            for file in version.get('files', []):
                file_name = file['name']
                file_sha256 = normalize_sha256(file.get('hashes', {}).get('SHA256', ''))
                name_match = file_name.lower() in existing_files
                sha256_match = file_sha256 and file_sha256 in existing_files_sha256
                if name_match or sha256_match:
                    display_version = version
                    break
            if display_version:
                break

        # Fallback to first version if no installed version found
        if not display_version and item['modelVersions']:
            display_version = item['modelVersions'][0]

        base_model = display_version.get('baseModel', 'Not Found') if display_version else 'Not Found'
        if display_version and 'publishedAt' in display_version:
            published_at = display_version.get('publishedAt')
            if published_at:
                date = published_at.split('T')[0]
            else:
                date = 'Not Found'
        else:
            date = 'Not Found'

        early_access = is_early_access(display_version) if display_version else False
        early_access_class = 'early-access' if early_access else ''

        # Status badges: New / Updated + base model abbreviation (optional setting)
        show_status_badges = getattr(opts, 'show_civitai_status_badges', True)
        if show_status_badges:
            base_model_short = get_base_model_short(base_model)
            status_badge_type = get_status_badge_type(item)
        else:
            base_model_short = ''
            status_badge_type = ''

        # Image or video preview
        images = display_version.get('images', []) if display_version else []
        if images:
            media_type = images[0].get('type')
            image_url = images[0].get('url')

            # Apply resize if enabled
            resize_preview = getattr(opts, 'resize_preview_cards', True)
            resize_size = getattr(opts, 'resize_preview_size', 512)

            if resize_preview and media_type == 'image':
                # For images, modify the URL to request specific size
                image_url = re.sub(r'/width=\d+', f"/width={resize_size}", image_url)

            if media_type == 'video':
                if resize_preview:
                    # For videos, replace or add width parameter
                    if '/width=' in image_url:
                        image_url = re.sub(r'/width=\d+', f"/width={resize_size}", image_url)
                    else:
                        image_url = image_url.replace('transcode=true,', f"transcode=true,width={resize_size},")
                else:
                    image_url = image_url.replace('width=', 'transcode=true,width=')
                imgtag = f'<video class="video-bg" loop muted playsinline><source src="{image_url}" type="video/mp4"></video>'
            else:
                imgtag = f'<img src="{image_url}"></img>'
        else:
            # Try PNG first, then fallback to JPEG if PNG does not exist
            imgtag = '<img src="./file=html/card-no-preview.png" onerror="this.onerror=null;this.src=\'./file=html/card-no-preview.jpg\';"></img>'

        # Install status - check if model is installed and determine if it's outdated
        # Dual-strategy: API order (primary) + regex fallback (failsafe).
        # API order is the source of truth (index 0 = newest, per baseModel).
        # Regex is used as a secondary check to catch edge cases the API might miss.
        installstatus = ''
        installed_file_sha256 = None  # Track SHA256 of installed file for delete functionality
        model_versions = item.get('modelVersions', [])
        if model_versions:
            precise_check = getattr(opts, 'precise_version_check', True)
            installed_versions_found = set()

            # === PRIMARY: API order + baseModel ===
            # Build: baseModel -> newest index among all versions (API guarantees index 0 = newest)
            base_to_newest_idx = {}
            for idx, ver in enumerate(model_versions):
                bm = (ver.get('baseModel') or '').strip()
                if bm and bm not in base_to_newest_idx:
                    base_to_newest_idx[bm] = idx

            has_installed = False
            has_outdated_api = False
            installed_bases = set()

            for idx, version in enumerate(model_versions):
                for file in version.get('files', []):
                    file_name = file['name'].lower()
                    file_sha256 = normalize_sha256(file.get('hashes', {}).get('SHA256', ''))
                    name_match = file_name in existing_files
                    sha_match = file_sha256 and file_sha256 in existing_files_sha256

                    if sha_match or name_match:
                        has_installed = True
                        if not installed_file_sha256:
                            installed_file_sha256 = file_sha256
                        installed_versions_found.add(version.get('name', ''))

                        bm = (version.get('baseModel') or '').strip()
                        newest_idx = base_to_newest_idx.get(bm, idx)
                        if idx > newest_idx:
                            has_outdated_api = True

                        installed_bases.add(bm)
                        break  # this version is installed, move to next version

            installed_versions_count = len(installed_versions_found)

            # === FALLBACK: regex-based semantic comparison (catches edge cases) ===
            has_outdated_regex = False
            has_latest_regex = False
            installed_map = {}
            available_map = {}
            installed_all = []
            available_all = []

            for version in model_versions:
                version_name = version.get('name', '')
                family, version_parts = _file.extract_version_from_ver_name(version_name)

                if precise_check and family:
                    available_map.setdefault(family, []).append(version_parts)
                else:
                    available_all.append(version_parts)

                # Only add to installed maps for versions we detected as installed above
                if version_name in installed_versions_found:
                    if precise_check and family:
                        installed_map.setdefault(family, []).append(version_parts)
                    else:
                        installed_all.append(version_parts)

            if installed_map or installed_all:
                def _is_outdated(inst, avail):
                    """Compare max installed and available versions"""
                    max_inst = max(inst, key=lambda x: x or [0])
                    max_avail = max(avail, key=lambda x: x or [0])
                    cmp = _file.compare_version_parts(max_inst, max_avail)
                    return cmp < 0

                if precise_check and available_map:
                    for fam, avail in available_map.items():
                        inst = installed_map.get(fam)
                        if not inst:
                            continue
                        if _is_outdated(inst, avail):
                            has_outdated_regex = True
                        else:
                            has_latest_regex = True
                elif installed_all and available_all:
                    has_outdated_regex = _is_outdated(installed_all, available_all)
                    has_latest_regex = not has_outdated_regex

            # === Combine: API order OR regex says outdated → mark as outdated ===
            has_outdated = has_outdated_api or has_outdated_regex

            if has_installed:
                if has_outdated:
                    installstatus = 'civmodelcardoutdated'
                elif precise_check and installed_bases:
                    # Check cross-family: are there available baseModels not installed?
                    has_cross_family = False
                    all_bases = set((v.get('baseModel') or '').strip() for v in model_versions if v.get('baseModel'))
                    for bm in all_bases:
                        if bm not in installed_bases:
                            has_cross_family = True
                            break
                    installstatus = 'civmodelcardcrossfamily' if has_cross_family else 'civmodelcardinstalled'
                else:
                    installstatus = 'civmodelcardinstalled'

            # Multi-family badge: when multiple distinct baseModels are installed on the same model
            # (e.g. Pony V1 AND Illustrious V1), show all abbreviations: "PONY · IL"
            if show_status_badges and len(installed_bases) > 1:
                shorts = []
                seen_shorts = set()
                for ver in model_versions:
                    bm = ver.get('baseModel', '')
                    if bm and bm in installed_bases:
                        short = get_base_model_short(bm)
                        if short and short not in seen_shorts:
                            shorts.append(short)
                            seen_shorts.add(short)
                if len(shorts) > 1:
                    base_model_short = ' · '.join(shorts)

        # Model name for JS and HTML
        model_name_js = model_name.replace("'", "\\'")
        model_string = escape(f"{model_name_js} ({model_id})")
        display_name = escape(model_name[:35] + '...' if len(model_name) > 35 else model_name)
        full_name = escape(model_name)

        ## Badges
        # Base model suffix for type badge (e.g. "| IL")
        bm_suffix = (
            f' <span class="base-model-sep">|</span>'
            f' <span class="base-model-short">{base_model_short}</span>'
        ) if base_model_short else ''

        # Model Type Badge ( + Early Access + base model abbreviation)
        if early_access:
            # Gold badge with a lightning icon
            model_type_badge = (
                f'<div class="model-type-badge {item["type"].lower()} early-access-badge">'
                '<svg class="early-access-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor">'
                '<path d="M13 2L3 14h9l-1 8 10-12h-8z"/>'
                '</svg>'
                f'{get_display_type(item["type"])}{bm_suffix}'
                '</div>'
            )
        else:
            model_type_badge = f'<div class="model-type-badge {item["type"].lower()}">{get_display_type(item["type"])}{bm_suffix}</div>'

        # Status Badge (New / Updated)
        if status_badge_type:
            status_badge = f'<div class="status-badge {status_badge_type}">{status_badge_type.capitalize()}</div>'
        else:
            status_badge = ''

        # NSFW Badge - only show for nsfw cards and if setting is enabled
        show_nsfw_badge = getattr(opts, 'show_nsfw_badge', True)
        if is_nsfw and show_nsfw_badge:
            nsfw_badge = (
                '<div class="nsfw-badge">'
                '<svg class="nsfw-badge-icon" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg" fill="currentColor">'
                '<circle cx="10" cy="10" r="10"/>'
                '<text x="10" y="11" font-size="12" text-anchor="middle" dominant-baseline="middle" font-family="Arial" font-weight="bold" fill="#fff">!</text>'
                '</svg>'
                'NSFW'
                '</div>'
            )
        else:
            nsfw_badge = ''

        # ModelCard HTML (Header)
        card_html = (
            f'<figure class="civmodelcard {nsfw_class} {early_access_class} {installstatus}{fav_class}" '
            f'base-model="{base_model}" date="{date}" data-model-id="{model_id}" data-creator="{escape(model_uploader_card)}" '
            f'onclick="select_model(\'{model_string}\', event)">'
            f'<div class="card-header">'
            f'<div class="badges-container">{model_type_badge}{status_badge}{nsfw_badge}</div>'
        )

        # Show delete button for up-to-date installed models;
        # For outdated: both delete (hidden below tile size 11) + checkbox stacked
        # For non-installed: checkbox only
        if installstatus == 'civmodelcardinstalled':
            sha256_attr = f'data-sha256="{installed_file_sha256}"' if installed_file_sha256 else ''
            card_html += (
                f'<div class="delete-button-container">'
                f'<button class="delete-model-btn" {sha256_attr} data-model-name="{model_name_js}" data-installed-count="{installed_versions_count}" '
                f'onclick="deleteInstalledModel(event, \'{model_string}\', \'{installed_file_sha256 or ""}\', {installed_versions_count})" title="Delete model">'
                f'<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor">'
                f'<path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>'
                f'</svg>'
                f'</button>'
                f'</div>'
            )
        elif installstatus == 'civmodelcardoutdated':
            # Both delete (hides at tile < 11) + checkbox for batch update selection
            sha256_attr = f'data-sha256="{installed_file_sha256}"' if installed_file_sha256 else ''
            card_html += (
                f'<div class="outdated-card-actions">'
                f'<button class="delete-model-btn" {sha256_attr} data-model-name="{model_name_js}" data-installed-count="{installed_versions_count}" '
                f'onclick="deleteInstalledModel(event, \'{model_string}\', \'{installed_file_sha256 or ""}\', {installed_versions_count})" title="Delete model">'
                f'<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor">'
                f'<path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>'
                f'</svg>'
                f'</button>'
                f'<div class="checkbox-container">'
                f'<input type="checkbox" class="model-checkbox" id="checkbox-{model_string}" '
                f'onchange="multi_model_select(\'{model_string}\', \'{item["type"]}\', this.checked)">'
                f'<label for="checkbox-{model_string}" class="custom-checkbox">'
                f'<span class="checkbox-checkmark"></span>'
                f'</label>'
                f'</div>'
                f'</div>'
            )
        else:
            # Non-installed: checkbox for batch download
            card_html += (
                f'<div class="checkbox-container">'
                f'<input type="checkbox" class="model-checkbox" id="checkbox-{model_string}" '
                f'onchange="multi_model_select(\'{model_string}\', \'{item["type"]}\', this.checked)">'
                f'<label for="checkbox-{model_string}" class="custom-checkbox">'
                f'<span class="checkbox-checkmark"></span>'
                f'</label>'
                f'</div>'
            )

        # ModelCard HTML (Footer)
        card_html += (
            f'</div>'
            f'{imgtag}'
            f'<figcaption title="{full_name}">{display_name}</figcaption></figure>'
        )
        return card_html, date

    # Main function logic
    video_playback = getattr(opts, 'video_playback', True)
    playback = 'autoplay loop' if video_playback else ''
    hide_early_access = getattr(opts, 'hide_early_access', True)
    current_time = datetime.now(timezone.utc)

    # Filter model versions and items
    filtered_items = []
    for item in json_data.get('items', []):
        versions = filter_versions(item, hide_early_access, current_time)
        if versions:
            item['modelVersions'] = versions
            filtered_items.append(item)
    json_data['items'] = filtered_items

    # Collect model folders
    model_folders = set()
    for item in json_data['items']:
        folder = contenttype_folder(item['type'], item['description'])
        if folder is not None:
            model_folders.add(str(folder))
    existing_files, existing_files_sha256 = collect_existing_files(model_folders)

    # Build HTML
    HTML = '<div class="column civmodellist">'
    sorted_models = {} if gl.sortNewest else None
    favorite_creators = set(_file.FavoriteCreators.get_as_list())

    for item in json_data['items']:
        model_card, date = get_model_card(item, existing_files, existing_files_sha256, playback, favorite_creators)
        if gl.sortNewest:
            if date not in sorted_models:
                sorted_models[date] = []
            sorted_models[date].append(model_card)
        else:
            HTML += model_card

    if gl.sortNewest:
        HTML += '<div class="date-sections-container">'
        for date, cards in sorted(sorted_models.items(), reverse=True):
            if not cards:
                continue

            if date == 'Not Found':
                formatted_date = 'Unknown Date'
            else:
                try:
                    date_obj = datetime.strptime(date, '%Y-%m-%d')
                    formatted_date = date_obj.strftime('%B %d, %Y')
                except:
                    formatted_date = date  # Fallback to original format

            # Add card counter (only show if more than 1 card)
            card_count = len(cards)
            counter_html = f' <span class="card-counter">{card_count}</span>' if card_count > 1 else ''
            HTML += (
                f'<div class="date-section">'
                f'<h4>{formatted_date}{counter_html}</h4>'
                '<div class="card-row">'
            )
            for card in cards:
                HTML += card
            HTML += '</div></div>'
        HTML += '</div>'
    HTML += '</div>'

    return HTML

def _search_by_sha256(sha256_hash):
    """Search for a model by SHA256 hash"""
    # Normalize and validate hash format
    normalized_hash = normalize_sha256(sha256_hash)
    if not normalized_hash or not re.match(r'^[A-F0-9]{64}$', normalized_hash):
        return 'invalid_hash'

    # Search for model version by hash across both civitai.com and civitai.red
    headers = get_headers()
    proxies, ssl = get_proxies()

    candidates = []
    domains = ['https://civitai.com', 'https://civitai.red']
    try:
        for domain in domains:
            api_url = f"{domain}/api/v1/model-versions/by-hash/{normalized_hash}"
            try:
                response = requests.get(api_url, headers=headers, timeout=(60, 30), proxies=proxies, verify=ssl)
            except requests.exceptions.RequestException:
                continue

            if response.status_code == 200:
                data = response.json()
                if not data or 'error' in data:
                    continue

                # Validate returned version contains file with matching SHA
                files = data.get('files', []) or []
                for f in files:
                    file_sha = (f.get('hashes', {}) or {}).get('SHA256', '')
                    if file_sha and file_sha.strip().upper() == normalized_hash:
                        candidates.append({
                            'domain': domain,
                            'modelId': data.get('modelId'),
                            'versionId': data.get('id'),
                            'version_name': data.get('name'),
                            'file_name': f.get('name'),
                            'downloadUrl': f.get('downloadUrl') or data.get('downloadUrl')
                        })
                        break
            elif response.status_code == 404:
                continue
            elif response.status_code == 503:
                return 'offline'

    except Exception:
        return 'error'

    # Interpret results
    if not candidates:
        return 'sha256_not_found'
    if len(candidates) == 1:
        candidate = candidates[0]
        model_id = candidate.get('modelId')
        if not model_id:
            return 'not_found'
        model_url = f"https://{get_civitai_domain()}/api/v1/models/{model_id}"
        try:
            model_response = requests.get(model_url, headers=headers, timeout=(60, 30), proxies=proxies, verify=ssl)
            if model_response.status_code == 200:
                model_data = model_response.json()
                return {
                    'items': [model_data],
                    'metadata': {
                        'totalItems': 1,
                        'currentPage': 1,
                        'pageSize': 1,
                        'totalPages': 1
                    }
                }
            else:
                return 'not_found'
        except requests.exceptions.RequestException:
            return 'error'

    # Multiple candidates -> return list for disambiguation
    return {'ambiguous': candidates}

def create_api_url(content_type=None, sort_type=None, period_type=None, use_search_term=None, base_filter=None, only_liked=None, tile_count=None, search_term=None, nsfw=None, exact_search=None, isNext=None):
    base_url = f'https://{get_civitai_domain()}/api/v1/models'
    version_url = f'https://{get_civitai_domain()}/api/v1/model-versions'

    if isNext != None:
        api_url = gl.json_data['metadata']['nextPage' if isNext else 'prevPage']
        debug_print(api_url)
        return api_url

    params = {'limit': tile_count, 'sort': sort_type, 'period': period_type.replace(' ', '') if period_type else None}

    if content_type:
        params['types'] = content_type

    ## === ANXETY EDITs ===
    if use_search_term != 'None' and search_term:
        search_term = search_term.replace('\\', '\\\\').lower()

        # Apply exact search logic - wrap search term in quotes if exact_search is True
        # NOTE: CivitAI API only supports exact search (quoted term) for Model name.
        # Tag and User name searches do not support quoting and will return no results.
        if exact_search and use_search_term == 'Model name':
            # Only wrap in quotes if not already wrapped and contains spaces
            if not (search_term.startswith('"') and search_term.endswith('"')) and ' ' in search_term:
                search_term = f'"{search_term}"'

        if 'civitai.com' in search_term or 'civitai.red' in search_term:
            if '/api/download/models' in search_term:
                # Extract version ID from download URL
                version_match = re.search(r'models/(\d+)', search_term)
                if version_match:
                    version_id = version_match.group(1)
                    # Make API request to get model version information
                    version_api_url = f"https://{get_civitai_domain()}/api/v1/model-versions/{version_id}"
                    version_data = request_civit_api(version_api_url, skip_error_check=True)

                    if isinstance(version_data, dict) and 'modelId' in version_data:
                        model_id = version_data['modelId']
                        params = {'ids': str(model_id)}
            else:
                model_match = re.search(r'models/(\d+)', search_term)
                if model_match:
                    model_number = model_match.group(1)
                    params = {'ids': model_number}
        elif use_search_term == 'SHA256':
            # SHA256 search is handled separately in initial_model_page
            pass
        else:
            key_map = {'User name': 'username', 'Tag': 'tag'}
            search_key = key_map.get(use_search_term, 'query')
            params[search_key] = search_term

    if base_filter:
        params['baseModels'] = base_filter

    if only_liked:
        params['favorites'] = 'true'

    params['nsfw'] = 'true' if nsfw else 'false'

    query_parts = []
    for key, value in params.items():
        if isinstance(value, list):
            for item in value:
                query_parts.append((key, item))
        else:
            query_parts.append((key, value))

    query_string = urllib.parse.urlencode(query_parts, doseq=True, quote_via=urllib.parse.quote)
    api_url = f"{base_url}?{query_string}"

    debug_print(api_url)
    return api_url


## === ANXETY EDITs ===
def initial_model_page(content_type=None, sort_type=None, period_type=None, use_search_term=None, search_term=None, current_page=None, base_filter=None, only_liked=None, nsfw=None, exact_search=None, tile_count=None, from_update_tab=False):
    current_inputs = (content_type, sort_type, period_type, use_search_term, search_term, tile_count, base_filter, nsfw, exact_search)
    if current_inputs != gl.previous_inputs and gl.previous_inputs != None or not current_page:
        current_page = 1
    gl.previous_inputs = current_inputs

    # ── Update Mode: render from gl.update_items, no API call ──
    if gl.update_mode:
        # When triggered from outside the Update tab (e.g. pressRefresh/page slider),
        # ignore Browser-tab filters so they don't cross-filter Update Mode results.
        _ct = content_type if from_update_tab else None
        _bf = base_filter if from_update_tab else None
        html, max_page, current_page, hasPrev, hasNext = update_mode_page_html(
            _ct, _bf, tile_count, current_page)
        return (
            gr.update(choices=[], value='', interactive=True),
            gr.update(choices=[], value=''),
            gr.update(value=html),
            gr.update(interactive=hasPrev),
            gr.update(interactive=hasNext),
            gr.update(value=current_page, maximum=max_page),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False, visible=False if gl.isDownloading else True),
            gr.update(interactive=False, visible=False),
            gr.update(interactive=False, value=None, visible=True),
            gr.update(choices=[], value='', interactive=False),
            gr.update(choices=[], value='', interactive=False),
            gr.update(value='<div style="min-height: 0px;"></div>'),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None)
        )

    if not from_update_tab:
        gl.from_update_tab = False
        gl.update_mode = False

        if current_page == 1:
            # Handle SHA256 search specially
            if use_search_term == 'SHA256' and search_term:
                debug_print(f"Performing SHA256 search for hash: {search_term}")
                gl.json_data = _search_by_sha256(search_term)
                gl.url_list = {1: f"sha256_search_{search_term.strip().upper()}" if isinstance(gl.json_data, dict) else 'error'}
            else:
                api_url = create_api_url(content_type, sort_type, period_type, use_search_term, base_filter, only_liked, tile_count, search_term, nsfw, exact_search)
                gl.url_list = {1: api_url}
                gl.json_data = request_civit_api(api_url)
        else:
            api_url = gl.url_list.get(current_page)
    else:
        api_url = gl.url_list.get(current_page)
        gl.from_update_tab = True
        if api_url and api_url.startswith('local_only://'):
            gl.json_data = {'items': [], 'metadata': {}}
        elif api_url and not api_url.startswith('sha256_search_'):
            gl.json_data = request_civit_api(api_url)

        fallback_items = getattr(gl, 'local_browser_fallback_items', [])
        if isinstance(gl.json_data, dict) and fallback_items and current_page == 1:
            existing_ids = {item.get('id') for item in gl.json_data.get('items', [])}
            merged_items = list(gl.json_data.get('items', []))
            for fallback_item in fallback_items:
                if fallback_item.get('id') not in existing_ids:
                    merged_items.append(fallback_item)
            gl.json_data['items'] = merged_items
            if 'metadata' not in gl.json_data or not isinstance(gl.json_data['metadata'], dict):
                gl.json_data['metadata'] = {}

    max_page = 1
    model_list = []
    hasPrev, hasNext = False, False

    if not isinstance(gl.json_data, dict) or 'items' not in gl.json_data or 'metadata' not in gl.json_data:
        # Defensive: _search_by_sha256 may return {'ambiguous': ...} or an error string
        err_key = gl.json_data if not isinstance(gl.json_data, dict) else 'sha256_not_found'
        HTML = api_error_msg(err_key)
    else:
        gl.json_data = insert_metadata(1)

        metadata = gl.json_data['metadata']
        hasNext = 'nextPage' in metadata
        hasPrev = 'prevPage' in metadata

        # Check for empty results when searching by User Name
        if use_search_term == 'User name' and (not gl.json_data.get('items') or len(gl.json_data['items']) == 0):
            HTML = api_error_msg('user_not_found')
        else:
            for item in gl.json_data['items']:
                if len(item['modelVersions']) > 0:
                    model_list.append(f"{item['name']} ({item['id']})")

            max_page = max(gl.url_list.keys())
            HTML = model_list_html(gl.json_data)

    return (
        gr.update(choices=model_list, value='', interactive=True),     # Model List
        gr.update(choices=[], value=''),                               # Version List
        gr.update(value=HTML),                                             # HTML Tiles
        gr.update(interactive=hasPrev),                                  # Prev Page Button
        gr.update(interactive=hasNext),                                  # Next Page Button
        gr.update(value=current_page, maximum=max_page),                 # Page Slider
        gr.update(interactive=False),                                    # Save Tags
        gr.update(interactive=False),                                    # Save Images
        gr.update(interactive=False, visible=False if gl.isDownloading else True),  # Download Button
        gr.update(interactive=False, visible=False),                     # Delete Button
        gr.update(interactive=False, value=None, visible=True),         # Install Path
        gr.update(choices=[], value='', interactive=False),            # Sub Folder List
        gr.update(choices=[], value='', interactive=False),            # File List
        gr.update(value='<div style="min-height: 0px;"></div>'),           # Preview HTML
        gr.update(value=None),                                          # Trained Tags
        gr.update(value=None),                                          # Base Model
        gr.update(value=None)                                           # Model Filename
    )

def prev_model_page(content_type, sort_type, period_type, use_search_term, search_term, current_page, base_filter, only_liked, nsfw, exact_search, tile_count):
    return next_model_page(content_type, sort_type, period_type, use_search_term, search_term, current_page, base_filter, only_liked, nsfw, exact_search, tile_count, isNext=False)

def next_model_page(content_type, sort_type, period_type, use_search_term, search_term, current_page, base_filter, only_liked, nsfw, exact_search, tile_count, isNext=True):

    current_inputs = (content_type, sort_type, period_type, use_search_term, search_term, tile_count, base_filter, nsfw, exact_search)
    if current_inputs != gl.previous_inputs and gl.previous_inputs != None:
        return initial_model_page(content_type, sort_type, period_type, use_search_term, search_term, current_page, base_filter, only_liked, nsfw, exact_search, tile_count)

    api_url = create_api_url(isNext=isNext)
    gl.json_data = request_civit_api(api_url)

    next_page = current_page
    model_list = []
    max_page = 1
    hasPrev, hasNext = False, False

    if not isinstance(gl.json_data, dict):
        HTML = api_error_msg(gl.json_data)
    else:
        next_page = current_page + 1 if isNext else current_page - 1

        gl.json_data = insert_metadata(next_page, api_url)

        metadata = gl.json_data['metadata']
        hasNext = 'nextPage' in metadata
        hasPrev = 'prevPage' in metadata

        for item in gl.json_data['items']:
            if len(item['modelVersions']) > 0:
                model_list.append(f"{item['name']} ({item['id']})")

        max_page = max(gl.url_list.keys())
        HTML = model_list_html(gl.json_data)

    return (
        gr.update(choices=model_list, value='', interactive=True),  # Model List
        gr.update(choices=[], value=''),  # Version List
        gr.update(value=HTML),  # HTML Tiles
        gr.update(interactive=hasPrev),  # Prev Page Button
        gr.update(interactive=hasNext),  # Next Page Button
        gr.update(value=next_page, maximum=max_page),  # Current Page
        gr.update(interactive=False),  # Save Tags
        gr.update(interactive=False),  # Save Images
        gr.update(interactive=False, visible=False if gl.isDownloading else True),  # Download Button
        gr.update(interactive=False, visible=False),  # Delete Button
        gr.update(interactive=False, value=None),  # Install Path
        gr.update(choices=[], value='', interactive=False),  # Sub Folder List
        gr.update(choices=[], value='', interactive=False),  # File List
        gr.update(value='<div style="min-height: 0px;"></div>'),  # Preview HTML
        gr.update(value=None),  # Trained Tags
        gr.update(value=None),  # Base Model
        gr.update(value=None)  # Model Filename
    )

def insert_metadata(page_nr, api_url=None):
    metadata = gl.json_data['metadata']

    if not metadata.get('prevPage', None) and page_nr > 1:
        metadata['prevPage'] = gl.url_list.get((page_nr - 1))

    if gl.from_update_tab:
        if gl.url_list.get((page_nr + 1), None):
            metadata['nextPage'] = gl.url_list.get((page_nr + 1))

    elif page_nr not in gl.url_list:
        gl.url_list[page_nr] = api_url

    return gl.json_data

## === ANXETY EDITs ===
def update_model_versions(model_id, json_input=None, base_filter=None):
    if json_input:
        api_json = json_input
    else:
        api_json = gl.json_data

    if not api_json or 'items' not in api_json:
        return None

    for item in api_json['items']:
        if int(item['id']) == int(model_id):
            content_type = item['type']
            desc = item.get('description', 'None')

            versions_dict = defaultdict(list)
            installed_versions = set()

            model_folder = os.path.join(contenttype_folder(content_type, desc))
            gl.main_folder = model_folder
            versions = item['modelVersions']

            version_files = set()
            for version in versions:
                versions_dict[version['name']].append(item['name'])
                for version_file in version['files']:
                    file_sha256 = normalize_sha256(version_file.get('hashes', {}).get('SHA256', ''))
                    version_filename = version_file['name']
                    version_files.add((version['name'], version_filename, file_sha256))

            for root, _, files in os.walk(model_folder, followlinks=True):
                for file in files:
                    if file.endswith('.json'):
                        try:
                            json_path = os.path.join(root, file)
                            with open(json_path, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                                if isinstance(json_data, dict):
                                    sha256 = normalize_sha256(json_data.get('sha256'))
                                    if sha256:
                                        for version_name, _, file_sha256 in version_files:
                                            if sha256 == file_sha256:
                                                installed_versions.add(version_name)
                                                break
                        except Exception as e:
                            print(f"failed to read: '{file}': {e}")

                    # filename_check
                    for version_name, version_filename, _ in version_files:
                        if file.lower() == version_filename.lower():
                            installed_versions.add(version_name)
                            break

            version_names = list(versions_dict.keys())
            # Build display names with [Installed] and (Early Access) if applicable
            display_version_names = []
            for v in version_names:
                # Find the version object for this name
                version_obj = next((ver for ver in versions if ver['name'] == v), None)
                name = v
                installed = v in installed_versions
                early_access = is_early_access(version_obj) if version_obj else False
                if installed:
                    name += ' [Installed]'
                if early_access:
                    name += ' (Early Access)'
                display_version_names.append(name)
            default_installed = next((name for name in display_version_names if '[Installed]' in name), None)

            # Always prioritize an installed version as default so delete actions remain available.
            if default_installed:
                default_value = default_installed
            elif base_filter:
                filter_normalized = [b.lower() for b in base_filter]
                default_value = None
                for i, v in enumerate(version_names):
                    version_obj = next((ver for ver in versions if ver['name'] == v), None)
                    if version_obj and version_obj.get('baseModel', '').lower() in filter_normalized:
                        default_value = display_version_names[i]
                        break
                if default_value is None:
                    default_value = display_version_names[0] if display_version_names else None
            else:
                default_value = display_version_names[0] if display_version_names else None

            return gr.update(choices=display_version_names, value=default_value, interactive=True)  # Version List

    return gr.update(choices=[], value=None, interactive=False)  # Version List

def cleaned_name(file_name):
    if platform.system() == "Windows":
        illegal_chars_pattern = r'[\\/:*?"<>|]'
    else:
        illegal_chars_pattern = r'/'

    name, extension = os.path.splitext(file_name)
    clean_name = re.sub(illegal_chars_pattern, '', name)
    clean_name = re.sub(r'\s+', ' ', clean_name.strip())

    # Limit to 246 bytes (UTF-8) to avoid filesystem limits on Linux (ext4 max: 255 bytes)
    # Reserve bytes for the extension
    ext_bytes = len(extension.encode('utf-8'))
    max_name_bytes = 246 - ext_bytes
    name_encoded = clean_name.encode('utf-8')
    if len(name_encoded) > max_name_bytes:
        clean_name = name_encoded[:max_name_bytes].decode('utf-8', errors='ignore').rstrip()

    return f"{clean_name}{extension}"

def fetch_and_process_image(image_url):
    proxies, ssl = get_proxies()
    try:
        parsed_url = urllib.parse.urlparse(image_url)
        if parsed_url.scheme and parsed_url.netloc:
            response = requests.get(image_url, proxies=proxies, verify=ssl)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                geninfo, _ = read_info_from_image(image)
                return geninfo
        else:
            image = Image.open(image_url)
            geninfo, _ = read_info_from_image(image)
            return geninfo
    except:
        return None

def extract_model_info(input_string):
    last_open_parenthesis = input_string.rfind('(')
    last_close_parenthesis = input_string.rfind(')')

    name = input_string[:last_open_parenthesis].strip()
    id_number = input_string[last_open_parenthesis + 1:last_close_parenthesis]

    return name, int(id_number)

def update_model_info(model_string=None, model_version=None, only_html=False, input_id=None, json_input=None, from_preview=False):
    video_playback = getattr(opts, 'video_playback', True)
    meta_btn = getattr(opts, 'individual_meta_btn', True)
    playback = ''
    if video_playback:
        playback = 'autoplay loop'

    if json_input:
        api_data = json_input
    else:
        api_data = gl.json_data

    BtnDownInt = True
    BtnDel = False
    BtnImage = False
    model_id = None

    if not input_id:
        _, model_id = extract_model_info(model_string)
    else:
        model_id = input_id

    if model_version and '[Installed]' in model_version:
        model_version = model_version.replace(' [Installed]', '')
    if model_id:
        output_html = ''
        output_training = ''
        output_basemodel = ''
        img_html = ''
        dl_dict = {}
        is_LORA = False
        file_list = []
        file_dict = []
        default_file = None
        model_filename = None
        sha256_value = None
        model_folder = None
        model_name = ''
        content_type = ''
        desc = ''
        is_nsfw = False
        model_uploader = 'Unknown'
        version_name = ''
        version_id = None
        if not isinstance(api_data, dict) or 'items' not in api_data:
            # gl.json_data may be in an invalid state (e.g. after a SHA256 ambiguity result)
            return (
                gr.update(value=None),
                gr.update(value=None, interactive=False),
                gr.update(value=''),
                gr.update(visible=True, value='Download model'),
                gr.update(interactive=False),
                gr.update(visible=False, interactive=False),
                gr.update(choices=None, value=None, interactive=False),
                gr.update(value=None, interactive=False),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(interactive=False, value=None),
                gr.update(choices=None, value=None, interactive=False)
            )
        for item in api_data['items']:
            if int(item['id']) == int(model_id):
                is_local_only = bool(item.get('local_only'))
                content_type = item['type']
                if content_type == 'LORA':
                    is_LORA = True
                desc = item['description']

                model_name = item['name']
                model_folder = os.path.join(contenttype_folder(content_type, desc))
                model_uploader = None
                uploader_avatar = None

                # Use a dedicated function to check if the model is NSFW
                is_nsfw = is_model_nsfw(item)

                creator = item.get('creator', None)
                if creator:
                    model_uploader = creator.get('username', None)
                    uploader_avatar = creator.get('image', None)
                if not model_uploader:
                    model_uploader = 'User not found'
                    uploader_avatar = 'https://rawcdn.githack.com/gist/BlafKing/8d3f7a19e3f72cfddab46ae835037ee6/raw/296e81afbdd268200278beef478f3018b15936de/profile_placeholder.svg'
                uploader_avatar = (f'<div class="avatar"><img src={uploader_avatar}></div>')
                tags = item.get('tags', '')
                model_desc = item.get('description', '')
                if model_desc:
                    model_desc = model_desc.replace('<img', '<img style="max-width: -webkit-fill-available;"')
                    model_desc = model_desc.replace('<code>', '<code style="text-wrap: wrap">')
                if model_version is None:
                    selected_version = item['modelVersions'][0]
                else:
                    selected_version = None
                    for model in item['modelVersions']:
                        if model['name'] == model_version:
                            selected_version = model
                            break
                    if selected_version == None and item['modelVersions']:
                        selected_version = item['modelVersions'][0]  # fallback to first version

                model_availability = selected_version.get('availability', 'Unknown')
                published_at = selected_version.get('publishedAt')
                model_date_published = published_at.split('T')[0] if published_at else 'Unknown'
                version_name = selected_version['name']
                version_id = selected_version['id']
                version_about = selected_version.get('description', '')
                if version_about is not None and version_about.strip():
                    version_about = version_about.replace('<code>', '<code style="text-wrap: wrap">')
                    if model_desc:
                        model_desc += '\n<hr>\n<h3>About this version:</h3>\n' + version_about.strip()
                    else:
                        model_desc = '<h3>About this version:</h3>\n' + version_about.strip()

                if selected_version.get('trainedWords'):
                    output_training = ','.join(selected_version['trainedWords'])
                    output_training = re.sub(r'<[^>]*:[^>]*>', '', output_training)
                    output_training = re.sub(r', ?', ', ', output_training)
                    output_training = output_training.strip(', ')
                if selected_version['baseModel']:
                    output_basemodel = selected_version['baseModel']
                for file in selected_version['files']:
                    dl_dict[file['name']] = file['downloadUrl']

                    if not model_filename:
                        model_filename = file['name']
                        dl_url = file['downloadUrl']
                        gl.json_info = item
                        sha256_value = normalize_sha256(file['hashes'].get('SHA256')) or 'Unknown'

                    size = file['metadata'].get('size', 'Unknown')
                    format = file['metadata'].get('format', 'Unknown')
                    fp = file['metadata'].get('fp', 'Unknown')
                    sizeKB = file.get('sizeKB', 0) * 1024
                    filesize = _download.convert_size(sizeKB)

                    unique_file_name = f"{size} {format} {fp} ({filesize})"
                    is_primary = file.get('primary', False)
                    file_list.append(unique_file_name)
                    file_dict.append({
                        'format': format,
                        'sizeKB': sizeKB
                    })
                    if is_primary:
                        default_file = unique_file_name
                        model_filename = file['name']
                        dl_url = file['downloadUrl']
                        gl.json_info = item
                        sha256_value = normalize_sha256(file['hashes'].get('SHA256')) or 'Unknown'

                safe_tensor_found = False
                pickle_tensor_found = False
                if is_LORA and file_dict:
                    for file_info in file_dict:
                        file_format = file_info.get('format', '')
                        if 'SafeTensor' in file_format:
                            safe_tensor_found = True
                        if 'PickleTensor' in file_format:
                            pickle_tensor_found = True

                    if safe_tensor_found and pickle_tensor_found:
                        if 'PickleTensor' in file_dict[0].get('format', ''):
                            if file_dict[0].get('sizeKB', 0) <= 100:
                                model_folder = os.path.join(contenttype_folder('TextualInversion'))

                model_url = selected_version.get('downloadUrl', '')
                model_main_url = f"https://{get_civitai_domain()}/models/{item['id']}" if not is_local_only else ''

                if is_local_only:
                    api_version = {'images': []}
                else:
                    url = f"https://{get_civitai_domain()}/api/v1/model-versions/{selected_version['id']}"
                    api_version = request_civit_api(url)

                ## === ANXETY EDITs ===
                # --- HTML Generation ---
                BtnImage = True
                # Build image block
                img_html = '<div class="sampleimgs">'

                key_map = {
                    'prompt': 'Prompt',
                    'negativePrompt': 'Negative Prompt',
                    'Model': 'Model',
                    'sampler': 'Sampler',
                    'steps': 'Steps',
                    'cfgScale': 'CFG Scale',
                    'clipSkip': 'Clip Skip',
                    'seed': 'Seed',
                    'Size': 'Size',
                }
                preferred_order = ['prompt', 'negativePrompt', 'Model', 'sampler', 'steps', 'cfgScale', 'Clip skip', 'seed', 'Size']

                # Check if api_version is a valid dictionary (not an error string)
                if isinstance(api_version, dict) and 'images' in api_version:
                    for idx, pic in enumerate(api_version['images']):
                        index = f"preview_{idx}" if from_preview else idx
                        prompt_dict = pic.get('meta', {}) or {}
                        image_url = re.sub(r'/width=\d+', f"/width={pic.get('width', '')}", pic['url'])
                        is_video = pic.get('type') == 'video'

                        img_html += (
                            f'<div class="image-block">'
                            f'<div class="civitai-image-container">'
                        )

                        if is_video:
                            video_url = image_url.replace('width=', 'transcode=true,width=')
                            img_html += (
                                f'<video class="preview-media" data-sampleimg="true" {playback} muted playsinline onclick="openImageViewer(\'{escape(video_url)}\', \'video\')">'
                                f'<source src="{video_url}" type="video/mp4"></video>'
                            )
                            meta_button = False
                            prompt_dict = {}
                        else:
                            img_html += (
                                f'<img class="preview-media" data-sampleimg="true" src="{image_url}" alt="Model preview" onclick="openImageViewer(\'{escape(image_url)}\', \'image\')">'
                            )
                            meta_button = bool(prompt_dict.get('prompt'))

                        if meta_button:
                            img_html += (
                                '<div class="civitai_txt2img">'
                                f'<label onclick="sendImgUrl(\'{escape(image_url)}\')" class="civitai-txt2img-btn">Send to txt2img</label>'
                                '</div>'
                            )
                        img_html += '</div>'  # close .civitai-image-container

                        if prompt_dict:
                            img_html += (
                                '<div id="image_info">'
                                '<dl>'
                            )
                            for key in preferred_order:
                                if key in prompt_dict:
                                    value = prompt_dict[key]
                                    key_disp = key_map.get(key, key)
                                    if meta_btn:
                                        img_html += (
                                            f'<div class="civitai-meta-btn" data-key="{key}" title="Click to replace \u00b7 Shift+click to append" onclick="metaToTxt2Img(event, \'{escape(str(key_disp))}\', this)">'
                                            f'<dt>{escape(str(key_disp))}</dt><dd>{escape(str(value))}</dd></div>'
                                        )
                                    else:
                                        img_html += (
                                            f'<div class="civitai-meta" data-key="{key}"><dt>{escape(str(key_disp))}</dt><dd>{escape(str(value))}</dd></div>'
                                        )
                            # Check if there are remaining keys in meta
                            remaining_keys = [k for k in prompt_dict if k not in preferred_order]

                            # Add the rest
                            if remaining_keys:
                                img_html += (
                                    '<div class="tabs">'
                                    '<div class="tab">'
                                    f'<input type="checkbox" class="accordionCheckbox" id="chck{index}">'
                                    f'<label class="tab-label" for="chck{index}">More details...</label>'
                                    '<div class="tab-content">'
                                )
                                for key in remaining_keys:
                                    value = prompt_dict[key]
                                    img_html += (
                                        f'<div class="civitai-meta" data-key="{key}"><dt>{escape(str(key).capitalize())}</dt><dd>{escape(str(value))}</dd></div>'
                                    )
                                img_html += '</div></div></div>'
                            img_html += '</dl></div>'
                        else:
                            # Show beautiful empty state when no metadata is available
                            if is_video:
                                no_meta_type = "video"
                                icon_svg = (
                                    '<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
                                    '<rect x="3" y="5" width="18" height="14" rx="2" ry="2"></rect>'
                                    '<polygon points="10,9 16,12 10,15"></polygon>'
                                    '</svg>'
                                )
                            else:
                                no_meta_type = "image"
                                icon_svg = (
                                    '<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
                                    '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path>'
                                    '<polyline points="14,2 14,8 20,8"></polyline>'
                                    '<path d="M12 18v-4"></path>'
                                    '<path d="M12 10h.01"></path>'
                                    '</svg>'
                                )
                            img_html += (
                                '<div class="image-metadata-empty">'
                                '<div class="empty-state-icon">'
                                f'{icon_svg}'
                                '</div>'
                                '<div class="empty-state-text">'
                                f'<h4>No metadata available</h4>'
                                f'<p>No generation settings are available for this {no_meta_type}.</p>'
                                '</div>'
                                '</div>'
                            )
                        img_html += '</div>'  # close .image-block
                    img_html += '</div>'
                else:
                    # Handle API error case - show error message instead of images
                    img_html += (
                        '<div class="image-block-empty">'
                            '<div class="image-metadata-empty">'
                                '<div class="empty-state-icon">'
                                    '<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
                                    '<circle cx="12" cy="12" r="10"></circle>'
                                    '<line x1="12" y1="8" x2="12" y2="12"></line>'
                                    '<line x1="12" y1="16" x2="12.01" y2="16"></line>'
                                    '</svg>'
                                '</div>'
                                '<div class="empty-state-text">'
                                    '<h4>Unable to load preview images</h4>'
                                    '<p>There was an error loading the model preview. The model information may still be available.</p>'
                                '</div>'
                            '</div>'
                        '</div>'
                    )
                img_html += '</div>'

                # (Image viewer overlay is created dynamically in JavaScript)

                tags_html = ''.join([f'<span class="civitai-tag">{escape(str(tag))}</span>' for tag in tags])

                # Build permissions block
                allow_svg = '<svg width="16" height="16" viewBox="0 1.5 24 24" stroke-width="4" stroke-linecap="round" stroke="lime"><path d="M5 12l5 5l10 -10"></path></svg>'
                deny_svg = '<svg width="16" height="16" viewBox="0 1.5 24 24" stroke-width="4" stroke-linecap="round" stroke="red"><path d="M18 6l-12 12"></path><path d="M6 6l12 12"></path></svg>'
                allowCommercialUse = item.get('allowCommercialUse', [])

                perms_html = (
                    '<p>'
                        f'{allow_svg if item.get("allowNoCredit") else deny_svg} Use the model without crediting the creator<br/>'
                        f'{allow_svg if "Image" in allowCommercialUse else deny_svg} Sell images they generate<br/>'
                        f'{allow_svg if "Rent" in allowCommercialUse else deny_svg} Run on services that generate images for money<br/>'
                        f'{allow_svg if "RentCivit" in allowCommercialUse else deny_svg} Run on Civitai<br/>'
                        f'{allow_svg if item.get("allowDerivatives") else deny_svg} Share merges using this model<br/>'
                        f'{allow_svg if "Sell" in allowCommercialUse else deny_svg} Sell this model or merges using this model<br/>'
                        f'{allow_svg if item.get("allowDifferentLicense") else deny_svg} Have different permissions when sharing merges'
                    '</p>'
                )

                # Build header block
                if is_local_only:
                    model_page = (
                        '<div class="model-page-line">'
                            '<span class="page-label">Model Source:</span>'
                            f'<span>{escape(str(model_name))} (Local file only)</span>'
                        '</div>'
                    )
                else:
                    model_page = (
                        '<div class="model-page-line">'
                            '<span class="page-label">Model Page:</span>'
                            f'<a href={model_main_url}?modelVersionId={selected_version["id"]} target="_blank">{escape(str(model_name))}</a>'
                        '</div>'
                    )

                if not creator or model_uploader == 'User not found':
                    uploader_page = (
                        '<div class="model-uploader-line">'
                            '<span class="uploader-label">Uploaded Unknown:</span>'
                            f'<span>{escape(str(model_uploader))}</span>'
                            f'{uploader_avatar}'
                        '</div>'
                    )
                else:
                    uploader_page = (
                        '<div class="model-uploader-line">'
                            '<span class="uploader-label">Uploaded by:</span>'
                            f'<a href="https://{get_civitai_domain()}/user/{escape(str(model_uploader))}" target="_blank">{escape(str(model_uploader))}</a>'
                            f'{uploader_avatar}'
                        '</div>'
                    )

                # Build version info block
                _sha256_row = (
                    f'<dt>SHA256</dt>'
                    f'<dd><span style="font-family:monospace;font-size:11px;word-break:break-all;user-select:all;">{escape(sha256_value)}</span></dd>'
                ) if sha256_value and sha256_value != 'Unknown' else ''
                version_info = (
                    '<div class="version-info-block">'
                        '<h3 class="block-header">Version Information</h3>'
                        '<dl>'
                            '<dt>Type</dt>'
                            f'<dd>{get_display_type(content_type)}</dd>'
                            '<dt>Version</dt>'
                            f'<dd>{escape(str(model_version))}</dd>'
                            '<dt>Base Model</dt>'
                            f'<dd>{escape(str(output_basemodel))}</dd>'
                            '<dt>Availability</dt>'
                            f'<dd>{model_availability}</dd>'
                            '<dt>Published</dt>'
                            f'<dd>{model_date_published}</dd>'
                            '<dt>CivitAI Tags</dt>'
                            '<dd>'
                                '<div class="civitai-tags-container">'
                                    f'{tags_html}'
                                '</div>'
                            '</dd>'
                            f'{_sha256_row}'
                            f'{"<dt>Download Link</dt>" if model_url else ""}'
                            f'{f"<dd><a href={model_url} target=_blank>{model_url}</a></dd>" if model_url else ""}'
                        '</dl>'
                    '</div>'
                )

                # Build permissions block
                version_permissions = (
                    '<div class="permissions-block">'
                        '<h3 class="block-header">Permissions</h3>'
                        f'{perms_html}'
                    '</div>'
                )

                # Build description section
                prefix = "preview-" if from_preview else ""
                description_section = (
                    '<div class="description-block">'
                        '<h2 class="block-header">Model Description</h2>'
                        '<div class="description-wrapper">'
                            f'<div class="description-content" id="{prefix}description-content">'
                                f'{model_desc}'
                            '</div>'
                            f'<div class="description-overlay" id="{prefix}description-overlay"></div>'
                            f'<button class="description-toggle-btn" id="{prefix}description-toggle-btn" onclick="toggleDescription(\'{prefix}\')">Show More</button>'
                        '</div>'
                    '</div>'
                )

                # Build trigger words block — per-group rows with individual copy/add buttons
                # v0.8.1: Try to load local consolidated trigger words from .json first, fallback to API
                local_trigger_words = None
                if model_filename and content_type:
                    local_trigger_words = get_local_trigger_words(content_type, model_filename, sha256_value)
                
                # Use local consolidated words if available, otherwise fall back to API
                if local_trigger_words is not None:
                    raw_trained_words = local_trigger_words
                else:
                    raw_trained_words = selected_version.get('trainedWords', [])
                
                def _sanitize_tw(s):
                    s = re.sub(r'<[^>]*:[^>]*>', '', s)
                    s = re.sub(r', ?', ', ', s)
                    return s.strip(', ')
                sanitized_groups = [_sanitize_tw(g) for g in raw_trained_words if g and _sanitize_tw(g)]

                if sanitized_groups or is_LORA:
                    rows_html = ''
                    all_onclick_parts = []

                    if is_LORA and model_filename:
                        lora_stem = os.path.splitext(model_filename)[0]
                        safe_stem_js = lora_stem.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace("'", '&#39;')
                        lora_tag_display = f'&lt;lora:{escape(lora_stem)}:1&gt;'
                        lora_tag_onclick = f'&lt;lora:{safe_stem_js}:1&gt;'
                        rows_html += (
                            f'<div class="trigger-word-row lora-tag-row">'
                                f'<div class="trigger-word-actions">'
                                    f'<button class="tw-copy-btn" onclick="copyTriggerWord(\'{lora_tag_onclick}\', this)" title="Copy">📋</button>'
                                    f'<button class="tw-add-btn" onclick="sendTagsToPrompt(\'{lora_tag_onclick}\')" title="Add to prompt">➕</button>'
                                f'</div>'
                                f'<span class="trigger-word-text">{lora_tag_display}</span>'
                            f'</div>'
                        )
                        all_onclick_parts.append(lora_tag_onclick)

                    for group in sanitized_groups:
                        safe_group = escape(group).replace("'", '&#39;')
                        rows_html += (
                            f'<div class="trigger-word-row">'
                                f'<div class="trigger-word-actions">'
                                    f'<button class="tw-copy-btn" onclick="copyTriggerWord(\'{safe_group}\', this)" title="Copy">📋</button>'
                                    f'<button class="tw-add-btn" onclick="sendTagsToPrompt(\'{safe_group}\')" title="Add to prompt">➕</button>'
                                f'</div>'
                                f'<span class="trigger-word-text">{escape(group)}</span>'
                            f'</div>'
                        )
                        all_onclick_parts.append(safe_group)

                    all_onclick = ', '.join(all_onclick_parts)
                    add_all_label = '➕ Add all to prompt' if len(all_onclick_parts) > 1 else '➕ Add to prompt'
                    trained_words_section = (
                        '<div class="trained-words-block">'
                            '<h3 class="block-header">Trigger Words</h3>'
                            f'{rows_html}'
                            f'<button class="add-to-prompt-btn" onclick="sendTagsToPrompt(\'{all_onclick}\')">{add_all_label}</button>'
                        '</div>'
                    )
                else:
                    trained_words_section = ''

                # Companion files banner — shown when required VAE/text_encoder files are missing
                companion_banner = _file.get_companion_banner(
                    output_basemodel,
                    model_filename=model_filename or '',
                    model_name=model_name or '',
                )

                # Build main HTML structure
                output_html = (
                    '<div class="main-container">'
                        '<div class="info-section">'
                            '<div class="header-block">'
                                f'{model_page}'
                                '<div class="uploader-divider"></div>'
                                f'{uploader_page}'
                            '</div>'
                            '<div class="info-permissions-container">'
                                f'{version_info}'
                                f'{version_permissions}'
                            '</div>'
                            f'{companion_banner}'
                            f'{trained_words_section}'
                            f'{description_section}'
                        '</div>'
                        '<div class="images-section">'
                            f'{img_html}'
                        '</div>'
                    '</div>'
                )

        if only_html:
            return output_html

        if model_folder is None:
            # Model not found in api_data (e.g. lazy fetch with stale gl.json_data)
            return (
                gr.update(value=None),
                gr.update(value=None, interactive=False),
                gr.update(value=''),
                gr.update(visible=True, value='Download model'),
                gr.update(interactive=False),
                gr.update(visible=False, interactive=False),
                gr.update(choices=None, value=None, interactive=False),
                gr.update(value=None, interactive=False),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(interactive=False, value=None),
                gr.update(choices=None, value=None, interactive=False)
            )

        folder_location = 'None'
        default_subfolder = 'None'
        sub_folders = _file.getSubfolders(
            model_folder,
            output_basemodel,
            is_nsfw,
            model_uploader,
            model_name,
            model_id,
            version_name,
            version_id
        )

        # === ANXETY EDIT ===
        installed_model_filename = None
        extensions = ['.pt', '.ckpt', '.pth', '.safetensors', '.th', '.zip', '.vae']

        for root, dirs, files in os.walk(model_folder, followlinks=True):
            for filename in files:
                if filename.endswith('.json'):
                    json_file_path = os.path.join(root, filename)
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            sha256 = normalize_sha256(data.get('sha256'))
                            if sha256 and sha256 == sha256_value:
                                folder_location = root
                                BtnDownInt = False
                                BtnDel = True

                                # Find the actual model file with same base name
                                base_name = os.path.splitext(filename)[0]
                                for model_file in files:
                                    if os.path.splitext(model_file)[0] == base_name:
                                        file_ext = os.path.splitext(model_file)[1].lower()
                                        if file_ext in extensions:
                                            installed_model_filename = model_file
                                            break
                                break
                        except Exception as e:
                            print(f"Error decoding JSON: {str(e)}")
            else:
                # filename_check
                for filename in files:
                    if filename.lower() == model_filename.lower() or filename.lower() == cleaned_name(model_filename).lower():
                        folder_location = root
                        BtnDownInt = False
                        BtnDel = True
                        installed_model_filename = filename
                        break

            if folder_location != 'None':
                break

        # Check if auto-organization is enabled
        auto_organize = getattr(opts, 'civitai_neo_auto_organize', False)
        # Detect wildcards by content_type OR by resolved folder path (double protection
        # in case content_type is None or uses an unexpected casing/variant)
        _is_wildcard_path = (content_type == 'Wildcards') or ('wildcard' in str(model_folder).lower())
        _wildcard_by_base = getattr(opts, 'civitai_neo_wildcard_organize_by_base', False)

        if auto_organize and output_basemodel and folder_location == 'None' and (not _is_wildcard_path or _wildcard_by_base):
            # Use auto-organization: determine folder from baseModel
            from scripts.civitai_file_manage import normalize_base_model
            base_folder = normalize_base_model(output_basemodel)
            
            if base_folder:
                # Create subfolder path for organized download
                if not base_folder.startswith(os.sep):
                    base_folder = os.sep + base_folder
                folder_path = str(model_folder) + base_folder
                default_subfolder = base_folder
            else:
                # No folder (user disabled "Other" folder and model is unrecognized)
                folder_path = str(model_folder)
                default_subfolder = 'None'
        else:
            # Original behavior: use custom subfolders or default
            default_subfolder = sub_folder_value(content_type, desc)
            if default_subfolder != 'None':
                default_subfolder = _file.convertCustomFolder(default_subfolder, output_basemodel, is_nsfw, model_uploader, model_name, model_id, version_name, version_id)
            if folder_location == 'None':
                folder_location = model_folder
                if default_subfolder != 'None':
                    folder_path = str(folder_location) + default_subfolder
                else:
                    folder_path = str(folder_location)
            else:
                folder_path = folder_location

            relative_path = os.path.relpath(folder_location, model_folder)
            default_subfolder = f"{os.sep}{relative_path}" if relative_path != '.' else default_subfolder if BtnDel == False else 'None'

        # Use installed filename if model is installed
        if installed_model_filename and BtnDel:
            display_model_filename = installed_model_filename
        else:
            display_model_filename = cleaned_name(model_filename)

        if gl.isDownloading:
            item = gl.download_queue[0]
            if int(model_id) == int(item['model_id']):
                BtnDel = False
        BtnDownTxt = 'Download model'
        if len(gl.download_queue) > 0:
            BtnDownTxt = 'Add to queue'
            for item in gl.download_queue:
                if item['version_name'] == model_version and int(item['model_id']) == int(model_id):
                    BtnDownInt = False
                    break

        return (
            gr.update(value=output_html),                                                      # Preview HTML
            gr.update(value=output_training, interactive=True),                             # Trained Tags
            gr.update(value=output_basemodel),                                              # Base Model Number
            gr.update(visible=False if BtnDel else True, interactive=BtnDownInt, value=BtnDownTxt),  # Download Button
            gr.update(interactive=BtnImage),                                                 # Images Button
            gr.update(visible=BtnDel, interactive=BtnDel),                                   # Delete Button
            gr.update(choices=file_list, value=default_file, interactive=True),            # File List
            gr.update(value=display_model_filename, interactive=True),                      # Model File Name
            gr.update(value=dl_url),                                                        # Download URL
            gr.update(value=model_id),                                                      # Model ID
            gr.update(value=sha256_value),                                                  # SHA256
            gr.update(interactive=True, value=folder_path if model_name else None),         # Install Path
            gr.update(choices=sub_folders, value=default_subfolder, interactive=True)      # Sub Folder List
        )
    else:
        return (
            gr.update(value=None),                                         # Preview HTML
            gr.update(value=None, interactive=False),                   # Trained Tags
            gr.update(value=''),                                        # Base Model Number
            gr.update(visible=False if BtnDel else True, value='Download model'),  # Download Button
            gr.update(interactive=False),                                # Images Button
            gr.update(visible=BtnDel, interactive=BtnDel),               # Delete Button
            gr.update(choices=None, value=None, interactive=False),    # File List
            gr.update(value=None, interactive=False),                   # Model File Name
            gr.update(value=None),                                      # Download URL
            gr.update(value=None),                                      # Model ID
            gr.update(value=None),                                      # SHA256
            gr.update(interactive=False, value=None),                   # Install Path
            gr.update(choices=None, value=None, interactive=False)     # Sub Folder List
        )

def sub_folder_value(content_type, desc=None):
    if content_type == 'LORA':
        folder = getattr(opts, 'LORA_default_subfolder', 'None')
    elif content_type == 'Upscaler':
        folder = getattr(opts, 'ESRGAN_default_subfolder', 'None')  # default
        desc_upper = (desc or '').upper()
        for upscale_type in ['SWINIR', 'REALESRGAN', 'GFPGAN', 'BSRGAN']:
            if upscale_type in desc_upper:
                folder = getattr(opts, f"{upscale_type}_default_subfolder", folder)
                break  # stop at first match — was missing before, caused ESRGAN to always win
    else:
        folder = getattr(opts, f"{content_type}_default_subfolder", 'None')
    if folder is None:
        return 'None'
    return folder

def update_file_info(model_string, model_version, file_metadata):
    file_list = []
    is_LORA = False
    embed_check = False
    model_name = None
    model_id = None
    model_name, model_id = extract_model_info(model_string)

    if model_version and '[Installed]' in model_version:
        model_version = model_version.replace(' [Installed]', '')
    if model_id and model_version:
        for item in gl.json_data['items']:
            if int(item['id']) == int(model_id):
                content_type = item['type']
                if content_type == 'LORA':
                    is_LORA = True
                desc = item['description']
                for model in item['modelVersions']:
                    if model['name'] == model_version:
                        for file in model['files']:
                            size = file['metadata'].get('size', 'Unknown')
                            format = file['metadata'].get('format', 'Unknown')
                            unique_file_name = f"{size} {format}"
                            file_list.append(unique_file_name)
                            pass

                        if is_LORA and file_list:
                            extracted_formats = [file.split(' ')[1] for file in file_list]
                            if 'SafeTensor' in extracted_formats and 'PickleTensor' in extracted_formats:
                                embed_check = True

                        for file in model['files']:
                            model_id = item['id']
                            file_name = file.get('name', 'Unknown')
                            sha256 = normalize_sha256(file['hashes'].get('SHA256')) or 'Unknown'
                            metadata = file.get('metadata', {})
                            file_size = metadata.get('size', 'Unknown')
                            file_format = metadata.get('format', 'Unknown')
                            file_fp = metadata.get('fp', 'Unknown')
                            sizeKB = file.get('sizeKB', 0)
                            sizeB = sizeKB * 1024
                            filesize = _download.convert_size(sizeB)

                            if f"{file_size} {file_format} {file_fp} ({filesize})" == file_metadata:
                                installed = False
                                folder_location = 'None'
                                model_folder = os.path.join(contenttype_folder(content_type, desc))
                                if embed_check and file_format == 'PickleTensor':
                                    if sizeKB <= 100:
                                        model_folder = os.path.join(contenttype_folder('TextualInversion'))
                                dl_url = file['downloadUrl']
                                gl.json_info = item
                                for root, _, files in os.walk(model_folder, followlinks=True):
                                    if file_name in files:
                                        installed = True
                                        folder_location = root
                                        break

                                if not installed:
                                    for root, _, files in os.walk(model_folder, followlinks=True):
                                        for filename in files:
                                            if filename.endswith('.json'):
                                                with open(os.path.join(root, filename), 'r', encoding='utf-8') as f:
                                                    try:
                                                        data = json.load(f)
                                                        sha256_value = normalize_sha256(data.get('sha256'))
                                                        if sha256_value and sha256_value == sha256:
                                                            folder_location = root
                                                            installed = True
                                                            break
                                                    except Exception as e:
                                                        print(f"Error decoding JSON: {str(e)}")
                                default_sub = sub_folder_value(content_type, desc)
                                if folder_location == 'None':
                                    folder_location = model_folder
                                    if default_sub != 'None':
                                        folder_path = str(folder_location) + default_sub
                                    else:
                                        folder_path = str(folder_location)
                                else:
                                    folder_path = folder_location
                                relative_path = os.path.relpath(folder_location, model_folder)
                                default_subfolder = f"{os.sep}{relative_path}" if relative_path != '.' else default_sub if installed == False else 'None'
                                BtnDownInt = not installed
                                BtnDownTxt = 'Download model'
                                if len(gl.download_queue) > 0:
                                    BtnDownTxt = 'Add to queue'
                                    for item in gl.download_queue:
                                        if item['version_name'] == model_version:
                                            BtnDownInt = False
                                            break

                                return (
                                    gr.update(value=cleaned_name(file['name']), interactive=True),  # Model File Name Textbox
                                    gr.update(value=dl_url),  # Download URL Textbox
                                    gr.update(value=model_id),  # Model ID Textbox
                                    gr.update(value=sha256),  # sha256 textbox
                                    gr.update(interactive=BtnDownInt, visible=False if installed else True, value=BtnDownTxt),  # Download Button
                                    gr.update(interactive=True if installed else False, visible=True if installed else False),  # Delete Button
                                    gr.update(interactive=True, value=folder_path if model_name else None),  # Install Path
                                    gr.update(value=default_subfolder, interactive=True)  # Sub Folder List
                                )

    return (
        gr.update(value=None, interactive=False),  # Model File Name Textbox
        gr.update(value=None),  # Download URL Textbox
        gr.update(value=None),  # Model ID Textbox
        gr.update(value=None),  # sha256 textbox
        gr.update(interactive=False, visible=True),  # Download Button
        gr.update(interactive=False, visible=False),  # Delete Button
        gr.update(interactive=False, value=None),  # Install Path
        gr.update(choices=None, value=None, interactive=False)  # Sub Folder List
    )

def get_proxies():
    custom_proxy = getattr(opts, 'custom_civitai_proxy', '')
    disable_ssl = getattr(opts, 'disable_sll_proxy', False)
    cabundle_path = getattr(opts, 'cabundle_path_proxy', '')

    ssl = True
    proxies = {}
    if custom_proxy:
        if not disable_ssl:
            if cabundle_path:
                ssl = os.path.exists(cabundle_path)  # Check if cabundle_path is a valid file
        else:
            ssl = False
        proxies = {
            'http': custom_proxy,
            'https': custom_proxy,
        }
    return proxies, ssl

def get_headers(referer=None, no_api=None):
    api_key = getattr(opts, 'custom_api_key', '')
    headers = {
        'Connection': 'keep-alive',
        'Sec-Ch-Ua-Platform': 'Windows',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Content-Type': 'application/json'
    }
    if referer:
        headers['Referer'] = f"https://{get_civitai_domain()}/models/{referer}"
    if api_key and not no_api:
        headers['Authorization'] = f"Bearer {api_key}"

    return headers

def request_civit_api(api_url=None, skip_error_check=False):
    headers = get_headers()
    proxies, ssl = get_proxies()
    max_attempts = 3
    base_backoff_seconds = 2

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(api_url, headers=headers, timeout=(60, 30), proxies=proxies, verify=ssl)
            if not response.text or response.text.strip() == '':
                print(f"CivitAI API returned empty response for: {api_url}")
                return 'error'

            if skip_error_check:
                response.encoding = 'utf-8'
                try:
                    data = json.loads(response.text)
                    return data
                except json.JSONDecodeError as e:
                    print(f"CivitAI API: JSON decode error - {e}")
                    return 'error'

            response.raise_for_status()
            response.encoding = 'utf-8'
            try:
                data = json.loads(response.text)
            except json.JSONDecodeError:
                print(response.text)
                print('The CivitAI servers are currently offline. Please try again later.')
                return 'offline'
            return data

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"Model version not found (404): {api_url}")
                return 'not_found'
            
            if e.response.status_code in [500, 502, 503, 504]:
                if attempt < max_attempts:
                    wait_time = base_backoff_seconds * (2 ** (attempt - 1))
                    print(f"[CivitAI Browser Neo] - HTTP {e.response.status_code} Error (attempt {attempt}/{max_attempts}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            
            print(f"HTTP Error {e.response.status_code}: {e}")
            return 'error'

        except requests.exceptions.Timeout:
            if attempt < max_attempts:
                wait_time = base_backoff_seconds * attempt
                print(f"Request timed out (attempt {attempt}/{max_attempts}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            print('The request timed out. Please try again later.')
            return 'timeout'

        except requests.exceptions.RequestException as e:
            error_text = str(e)
            dns_resolution_error = (
                'NameResolutionError' in error_text
                or 'Failed to resolve' in error_text
                or 'Temporary failure in name resolution' in error_text
                or ('Max retries exceeded' in error_text and 'NameResolutionError' in error_text)
                or ('Max retries exceeded' in error_text and 'Failed to resolve' in error_text)
            )

            if dns_resolution_error and attempt < max_attempts:
                wait_time = base_backoff_seconds * attempt
                print(f"[CivitAI Browser Neo] - DNS resolution failed (attempt {attempt}/{max_attempts}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            print(f"[CivitAI Browser Neo] - Error: {e}")
            if dns_resolution_error:
                print(f"[CivitAI Browser Neo] - DNS resolution failed (attempt {max_attempts}/{max_attempts}). No more retries.")
                return 'dns_error'
            return 'error'

    return 'error'

## === ANXETY EDITs ===
def inject_removed_banner(html: str) -> str:
    """Prepend a 'removed by owner' warning banner to existing model HTML."""
    banner = (
        '<div style="background:rgba(229,115,115,0.15);border:1px solid #e57373;border-radius:8px;'
        'padding:12px 16px;margin:0 0 16px 0;display:flex;align-items:center;gap:10px;">'
        '<span style="font-size:20px;">&#9888;&#65039;</span>'
        '<div>'
        '<strong style="color:#e57373;">This resource has been removed by its owner</strong><br>'
        '<span style="font-size:12px;color:var(--body-text-color-subdued);">'
        'Showing cached local data. The model file is still available locally.'
        '</span></div></div>'
    )
    target = '<div class="info-section"'
    if target in html:
        return html.replace(target, banner + target, 1)
    # fallback: prepend to body content
    return banner + html


def api_error_msg(input_string):
    div = '<div style="color: white; font-family: var(--font); font-size: 24px; text-align: center; margin: 50px !important;">'
    if input_string == 'not_found':
        return div + 'Model ID not found on CivitAI.<br>Maybe the model doesn\'t exist on CivitAI?</div>'
    elif input_string == 'removed':
        return div + 'This resource has been removed by its owner.<br>No local cached data was found for this model.</div>'
    elif input_string == 'path_not_found':
        return div + 'Local model not found.<br>Could not locate the model path.</div>'
    elif input_string == 'timeout':
        return div + 'The CivitAI-API has timed out, please try again.<br>The servers might be too busy or down if the issue persists.</div>'
    elif input_string == 'offline':
        return div + 'The CivitAI servers are currently offline.<br>Please try again later.</div>'
    elif input_string == 'no_items':
        return div + 'Failed to retrieve any models from CivitAI<br>The servers might be too busy or down if the issue persists.</div>'
    elif input_string == 'invalid_hash':
        return div + 'Invalid SHA256 hash format.<br>Please enter a valid 64-character hexadecimal hash.</div>'
    elif input_string == 'sha256_not_found':
        return div + 'No model found with this SHA256 hash.<br>The model might not exist on CivitAI or the hash might be incorrect.</div>'
    elif input_string == 'user_not_found':
        return div + 'No models found for this user on CivitAI.<br>Please check the correctness of the user name.</div>'
    elif input_string == 'dns_error':
        return div + 'Temporary DNS resolution failure while contacting CivitAI.<br>Please check your network/DNS and try again in a few seconds.</div>'
    else:
        return div + 'The CivitAI-API failed to respond due to an error.<br>Check the logs for more details.</div>'