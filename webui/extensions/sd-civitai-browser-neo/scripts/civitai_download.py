import subprocess
import threading
import hashlib
import requests
import urllib.parse
import platform
import random
import stat
import json
import time
import re
import os
import gradio as gr
from pathlib import Path

# === WebUI imports ===
from modules.shared import opts, cmd_opts

# === Extension imports ===
import scripts.civitai_file_manage as _file
import scripts.civitai_global as gl
import scripts.civitai_api as _api
import scripts.download_log as _dl_log
from scripts.civitai_api import is_early_access, is_model_nsfw
from scripts.civitai_global import print, debug_print

try:
    from zip_unicode import ZipHandler
except ImportError:
    print('Python module "ZipUnicode" has not been imported correctly, please try to restart or install it manually.')


total_count = 0
current_count = 0
dl_manager_count = 0

# threading.Event: SET = not downloading (safe to cancel/cleanup), CLEARED = download in progress
_not_downloading = threading.Event()
_not_downloading.set()

def random_number(prev=None):
    number = str(random.randint(10000, 99999))
    while number == prev:
        number = str(random.randint(10000, 99999))

    return number


def resolve_ambiguity(choice_index):
    """Resolve a queued item's ambiguous SHA choice and trigger resume of download thread.

    Returns outputs: download_progress HTML, current_model, download_finish trigger, queue_trigger
    """
    try:
        idx = int(str(choice_index).strip())
    except Exception:
        return gr.update(value=''), gr.update(), gr.update(), gr.update()

    for item in gl.download_queue:
        if item.get('ambiguous_candidates') and not item.get('ambiguous_selected'):
            candidates = item.get('ambiguous_candidates')
            if idx < 0 or idx >= len(candidates):
                break
            c = candidates[idx]
            # Apply chosen candidate
            try:
                item['dl_url'] = c.get('downloadUrl') or item.get('dl_url')
                item['model_id'] = int(c.get('modelId')) if c.get('modelId') else item.get('model_id')
            except Exception:
                pass
            item['version_name'] = c.get('version_name') or item.get('version_name')
            # Update sha256 and filename if the chosen candidate differs (defensive:
            # candidate may reference a different file within the same version)
            cand_sha = c.get('sha256')
            if cand_sha and cand_sha.upper() != (item.get('model_sha256') or '').upper():
                item['model_sha256'] = cand_sha.upper()
                debug_print(f"[Ambiguity] Updated SHA256 for '{item.get('model_name')}': {cand_sha[:12]}…")
            cand_fname = c.get('file_name')
            if cand_fname and cand_fname != item.get('model_filename'):
                item['model_filename'] = _api.cleaned_name(cand_fname)
                debug_print(f"[Ambiguity] Updated filename for '{item.get('model_name')}': {item['model_filename']}")
            item['ambiguous_selected'] = True

            # Clear ambiguity HTML and resume download by nudging download_finish
            return gr.update(value=''), gr.update(value=item.get('model_name')), gr.update(value=random_number()), gr.update()

    return gr.update(value=''), gr.update(), gr.update(), gr.update()


gl.init()


rpc_secret = "R7T5P2Q9K6"
try:
    queue = not cmd_opts.no_gradio_queue
except AttributeError:
    queue = not cmd_opts.disable_queue
except:
    queue = True


## === ANXETY EDITs ===

def start_aria2_rpc():
    start_file = os.path.join(aria2path, '_')
    running_file = os.path.join(aria2path, 'running')
    null = open(os.devnull, 'w')

    if os.path.exists(running_file):
        try:
            if os_type == 'Linux':
                env = os.environ.copy()
                env['PATH'] = '/usr/bin:' + env['PATH']
                subprocess.Popen('pkill aria2', shell=True, env=env)
            else:
                subprocess.Popen(stop_rpc, stdout=null, stderr=null)
            time.sleep(1)
        except Exception as e:
            print(f"Failed to stop Aria2 RPC : {e}")
    else:
        if os.path.exists(start_file):
            os.rename(start_file, running_file)
            return
        else:
            with open(start_file, 'w', encoding='utf-8'):
                pass

    try:
        show_log = getattr(opts, 'show_log', False)
        aria2_flags = getattr(opts, 'aria2_flags', '')
        cmd = f'"{aria2}" --enable-rpc --rpc-listen-all --rpc-listen-port=24000 --rpc-secret {rpc_secret} --check-certificate=false --ca-certificate=" " --file-allocation=none {aria2_flags}'
        subprocess_args = {'shell': True}
        if not show_log:
            subprocess_args.update({'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL})

        subprocess.Popen(cmd, **subprocess_args)
        if os.path.exists(running_file):
            print('Aria2 RPC restarted')
        else:
            print('Aria2 RPC started')
    except Exception as e:
        print(f"Failed to start Aria2 RPC server: {e}")

aria2path = Path(__file__).resolve().parents[1] / 'aria2'
os_type = platform.system()

if os_type == 'Windows':
    aria2 = os.path.join(aria2path, 'win', 'aria2.exe')
    stop_rpc = 'taskkill /im aria2.exe /f'
    start_aria2_rpc()
elif os_type == 'Linux':
    aria2 = os.path.join(aria2path, 'lin', 'aria2')
    st = os.stat(aria2)
    os.chmod(aria2, st.st_mode | stat.S_IEXEC)
    stop_rpc = 'pkill aria2'
    start_aria2_rpc()

class TimeOutFunction(Exception):
    pass

def create_model_item(dl_url, model_filename, install_path, model_name, version_name, model_sha256, model_id, create_json, from_batch=False, old_file_path=None, version_id=None):
    global dl_manager_count
    if model_id:
        model_id = int(model_id)
    if model_sha256:
        model_sha256 = model_sha256.upper()
    if model_sha256 == 'UNKNOWN':
        model_sha256 = None

    filtered_items = []

    for item in gl.json_data['items']:
        if int(item['id']) == int(model_id):
            filtered_items.append(item)
            content_type = item['type']
            desc = item['description']
            main_folder = _api.contenttype_folder(content_type, desc)
            break

    sub_folder = os.path.normpath(os.path.relpath(install_path, main_folder))

    model_json = {'items': filtered_items}

    # Duplicate guard
    for item in gl.download_queue:
        if item['dl_url'] == dl_url:
            return None

    dl_manager_count += 1

    item = {
        'dl_id': dl_manager_count,
        'dl_url': dl_url,
        'model_filename': model_filename,
        'install_path': install_path,
        'model_name': model_name,
        'version_name': version_name,
        'model_sha256': model_sha256,
        'model_id': model_id,
        'version_id': version_id,
        'create_json': create_json,
        'model_json': model_json,
        'model_versions': None,        # fetched lazily in download_create_thread
        'preview_html': '',            # fetched lazily in download_create_thread
        'existing_path': install_path, # fetched lazily in download_create_thread
        'from_batch': from_batch,
        'sub_folder': sub_folder,
        '_api_ready': False,
        'old_file_path': old_file_path,  # retention: path of old file being replaced (may differ from new filename)
    }

    _dl_log.log_queued(item)
    return item


def _resolve_versions_to_download(versions_list, model_folder):
    """Return the list of model versions to download for a batch update.

    For models with multiple installed families (e.g. Pony V1 AND Illustrious V1
    on the same page), returns the latest available version for each installed
    family so every family is updated in one batch action.

    Falls back to [versions_list[0]] (original behaviour) when:
    - The model folder cannot be scanned, or
    - No family-tagged installed versions are detected.
    """
    if not versions_list:
        return []

    # Collect SHA256 hashes from local JSON files in the model folder
    installed_hashes = set()
    if model_folder and os.path.isdir(str(model_folder)):
        for root, _, files in os.walk(str(model_folder), followlinks=True):
            for fname in files:
                if fname.endswith('.json'):
                    try:
                        data = _api.safe_json_load(os.path.join(root, fname))
                        if data:
                            sha = data.get('sha256', '')
                            if sha:
                                installed_hashes.add(sha.upper())
                    except Exception:
                        pass

    if not installed_hashes:
        return [versions_list[0]]

    # Build two parallel maps:
    #  - family  (extracted from version name) — more precise when it works
    #  - baseModel (CivitAI field)             — reliable fallback when author renames
    latest_by_family = {}
    latest_by_base   = {}
    installed_families = set()
    installed_bases    = set()
    installed_ver_ids  = set()   # version IDs that are already on disk

    for ver in versions_list:
        ver_name = ver.get('name', '')
        family, _ = _file.extract_version_from_ver_name(ver_name)
        bm = (ver.get('baseModel') or '').strip()
        vid = ver.get('id')

        if family and family not in latest_by_family:
            latest_by_family[family] = ver
        if bm and bm not in latest_by_base:
            latest_by_base[bm] = ver

        for file_entry in ver.get('files', []):
            sha = file_entry.get('hashes', {}).get('SHA256', '').upper()
            if sha and sha in installed_hashes:
                installed_ver_ids.add(vid)
                if family:
                    installed_families.add(family)
                if bm:
                    installed_bases.add(bm)
                break

    # --- Phase 1: family-based resolution (preferred, more precise) ---
    seen_ids = set()
    family_result = []
    for fam in installed_families:
        ver = latest_by_family.get(fam)
        if ver:
            vid = ver.get('id')
            if vid not in seen_ids:
                family_result.append(ver)
                seen_ids.add(vid)

    # If family resolution finds something NEW (different from installed), use it.
    family_updates_anything = any(
        ver.get('id') not in installed_ver_ids for ver in family_result
    )
    if family_updates_anything:
        return family_result

    # --- Phase 2: baseModel fallback (author renamed version scheme) ---
    seen_ids = set()
    base_result = []
    for bm in installed_bases:
        ver = latest_by_base.get(bm)
        if ver:
            vid = ver.get('id')
            if vid not in seen_ids:
                base_result.append(ver)
                seen_ids.add(vid)

    return base_result if base_result else [versions_list[0]]


def selected_to_queue(model_list, subfolder, download_start, create_json, current_html, forced_version_ids=None):
    """Enqueue models for download.

    Args:
        forced_version_ids: Optional dict mapping model_id (str) -> list of version IDs.
            When provided, those specific versions are used instead of auto-resolution.
    """
    global total_count, current_count
    if gl.download_queue:
        number = download_start
    else:
        number = random_number(download_start)
        total_count = 0
        current_count = 0

    model_list = json.loads(model_list)
    total_queued = len(model_list)
    debug_print(f"[Queue] Batch enqueue started: {total_queued} model(s) selected")
    skipped_names = []

    ## === ANXETY EDITs ===
    for idx, model_string in enumerate(model_list, 1):
        model_name, model_id = _api.extract_model_info(model_string)
        item_found = None
        for item in gl.json_data['items']:
            if int(item['id']) == int(model_id):
                item_found = item
                break

        if not item_found:
            skipped_names.append(model_name)
            debug_print(f"[Queue] ({idx}/{total_queued}) Skipped model {model_name} ({model_id}) — not found in json_data")
            continue

        desc = item_found['description']
        content_type = item_found['type']
        is_nsfw = is_model_nsfw(item_found)
        creator = item_found.get('creator', {})
        model_uploader = creator.get('username', 'Unknown')
        versions_list = item_found.get('modelVersions', [])

        if not versions_list:
            skipped_names.append(model_name)
            continue

        model_folder = _api.contenttype_folder(content_type, desc)

        # Resolve which versions to download
        if forced_version_ids and str(model_id) in forced_version_ids:
            # Revamp: user selected specific versions from the update-mode card
            selected_ids = forced_version_ids[str(model_id)]
            versions_to_download = [v for v in versions_list if v.get('id') in selected_ids]
        else:
            # Auto-resolve: one per installed family for multi-family models
            versions_to_download = _resolve_versions_to_download(versions_list, model_folder)

        for version in versions_to_download:
            version_name = version.get('name')
            version_id = version.get('id')
            debug_print(f"[Queue] ({idx}/{total_queued}) Enqueuing: {model_name} — version '{version_name}' (id={version_id})")
            output_basemodel = version.get('baseModel')
            files = version.get('files', [])
            primary_file = next((f for f in files if f.get('primary', False)), None)

            if primary_file:
                model_filename = _api.cleaned_name(primary_file.get('name'))
                model_sha256 = primary_file.get('hashes', {}).get('SHA256')
                dl_url = primary_file.get('downloadUrl')
            elif files:
                model_filename = _api.cleaned_name(files[0].get('name'))
                model_sha256 = files[0].get('hashes', {}).get('SHA256')
                dl_url = files[0].get('downloadUrl')
            else:
                skipped_names.append(f"{model_name} ({version_name})")
                debug_print(f"[Queue] ({idx}/{total_queued}) Skipped {model_name} ({version_name}) — no files in version")
                continue

            # Check if auto-organization is enabled
            auto_organize = getattr(opts, 'civitai_neo_auto_organize', False)
            # Detect wildcards by content_type OR by resolved folder path (double protection
            # in case content_type is None or uses an unexpected casing/variant)
            is_wildcard = (content_type == 'Wildcards') or ('wildcard' in str(model_folder).lower())
            wildcard_by_base = getattr(opts, 'civitai_neo_wildcard_organize_by_base', False)
            from_batch = True  # default: treat as batch (no manual subfolder)

            if auto_organize and output_basemodel and (not is_wildcard or wildcard_by_base):
                # Use auto-organization: determine folder from baseModel
                from scripts.civitai_file_manage import normalize_base_model
                base_folder = normalize_base_model(output_basemodel)
                if base_folder:
                    if not base_folder.startswith(os.sep):
                        base_folder = os.sep + base_folder
                    install_path = str(model_folder) + base_folder
                else:
                    install_path = str(model_folder)
            else:
                # Original behavior: use custom subfolders or default
                default_subfolder = _api.sub_folder_value(content_type, desc)
                if default_subfolder != 'None':
                    default_subfolder = _file.convertCustomFolder(default_subfolder, output_basemodel, is_nsfw, model_uploader, model_name, model_id, version_name, version_id)

                if subfolder and subfolder != 'None' and subfolder != 'Only available if the selected files are of the same model type':
                    from_batch = False
                    if platform.system() == 'Windows':
                        subfolder = re.sub(r'[/:*?"<>|]', '', subfolder)
                    if not subfolder.startswith(os.sep):
                        subfolder = os.sep + subfolder
                    install_path = str(model_folder) + subfolder
                else:
                    from_batch = True
                    if default_subfolder != 'None':
                        install_path = str(model_folder) + default_subfolder
                    else:
                        install_path = str(model_folder)

            # Wildcards: place each download into its own subfolder
            # (e.g. wildcards/emotion-pack/emotion-pack.txt)
            # compatible with sd-dynamic-prompts __subfolder/name__ syntax
            if is_wildcard and getattr(opts, 'civitai_neo_wildcard_own_folder', True):
                safe_name = re.sub(r'[<>:"/\\|?*]', '', model_name).strip()
                safe_name = re.sub(r'_{2,}', '_', safe_name)  # avoid __ delimiter collision with sd-dynamic-prompts
                if safe_name:
                    install_path = os.path.join(install_path, safe_name)

            # For update-mode queuing: find the old installed file path from gl.update_items
            # so retention can be applied even when old and new filenames differ.
            old_file_path = None
            if gl.update_items:
                for _upd in gl.update_items:
                    if _upd.get('model_id') == int(model_id):
                        upd_family = (_upd.get('family') or '').upper()
                        new_family = (output_basemodel or '').upper()
                        if not upd_family or not new_family or upd_family == new_family:
                            old_file_path = _upd.get('old_file', '') or None
                            break

            model_item = create_model_item(dl_url, model_filename, install_path, model_name, version_name, model_sha256, model_id, create_json, from_batch, old_file_path=old_file_path, version_id=version_id)
            if model_item:
                gl.download_queue.append(model_item)
                total_count += 1
            else:
                skipped_names.append(f"{model_name} ({version_name})")
                debug_print(f"Skipped model {model_name} ({model_id}){f' version {version_name}' if version_name else ''} due to processing error")

    html = download_manager_html(current_html)
    if skipped_names:
        names_str = ', '.join(skipped_names[:5]) + ('…' if len(skipped_names) > 5 else '')
        html = html.rsplit('</div>', 1)[0] + (
            f'<div style="background:#3a1a1a;border:1px solid #8b3030;border-radius:6px;'
            f'padding:8px 12px;margin:4px 0;color:#ff9999;font-size:13px;">'
            f'⚠️ Skipped {len(skipped_names)} model(s) — could not load info: {names_str}'
            f'</div>'
        ) + '</div>'

    return (
        gr.update(interactive=False, visible=False),  # Download Button
        gr.update(interactive=True, visible=True),  # Cancel Button
        gr.update(interactive=True if len(gl.download_queue) > 1 else False, visible=True),  # Cancel All Button
        gr.update(value=number),  # Download Start Trigger
        gr.update(value='<div style="min-height: 100px;"></div>'),  # Download Progress
        gr.update(value=html)  # Download Manager HTML
    )


def _build_model_list_for_update(items):
    """Convert gl.update_items entries into a JSON model_list for selected_to_queue."""
    model_list = []
    for item in items:
        # Each entry needs to look like "Model Name (model_id)"
        model_list.append(f"{item['model_name']} ({item['model_id']})")
    return json.dumps(model_list)


def update_all_models(download_start, create_json, current_html):
    """Enqueue all models in gl.update_items for update (respects retention policy)."""
    items = list(gl.update_items)
    if not items:
        html = download_manager_html(current_html)
        number = download_start
        return (
            gr.update(interactive=False, visible=False),
            gr.update(interactive=False, visible=False),
            gr.update(interactive=False, visible=False),
            gr.update(value=number),
            gr.update(value='<div style="min-height: 100px;"></div>'),
            gr.update(value=html)
        )
    model_list_json = _build_model_list_for_update(items)
    return selected_to_queue(model_list_json, None, download_start, create_json, current_html)


def update_selected_models(trigger_value, download_start, create_json, current_html):
    """Enqueue only the checked/selected models (by model string list) from Update Mode."""
    try:
        model_list = json.loads(trigger_value)  # plain list of "Name (id)" strings
    except Exception:
        return update_all_models(download_start, create_json, current_html)
    if not model_list:
        return update_all_models(download_start, create_json, current_html)
    return selected_to_queue(json.dumps(model_list), None, download_start, create_json, current_html)


def download_single_update(trigger_value, download_start, create_json, current_html):
    """Enqueue a single model update.

    trigger_value formats:
      - Legacy:  'model_id|family'
      - Revamp:  'model_id|family|[456, 789]'  (JSON array of selected version IDs)
    """
    if not trigger_value or '|' not in trigger_value:
        html = download_manager_html(current_html)
        number = download_start
        return (
            gr.update(interactive=False, visible=False),
            gr.update(interactive=False, visible=False),
            gr.update(interactive=False, visible=False),
            gr.update(value=number),
            gr.update(value='<div style="min-height: 100px;"></div>'),
            gr.update(value=html)
        )

    # Parse trigger value: model_id|family|json_version_ids
    try:
        parts = trigger_value.split('|', 2)
        model_id_str = parts[0].strip()
        family_str   = parts[1].strip().upper() if len(parts) > 1 else ''
        version_ids_json = parts[2] if len(parts) > 2 else '[]'
        model_id_int = int(model_id_str)
        selected_version_ids = json.loads(version_ids_json) if version_ids_json else []
    except Exception:
        return update_all_models(download_start, create_json, current_html)  # fallback

    matched = [i for i in gl.update_items
               if i['model_id'] == model_id_int and
               (i.get('family') or '').upper() == family_str]
    if not matched:
        # Try without family match (no-family models)
        matched = [i for i in gl.update_items if i['model_id'] == model_id_int]

    if not matched:
        html = download_manager_html(current_html)
        return (
            gr.update(interactive=False, visible=False),
            gr.update(interactive=False, visible=False),
            gr.update(interactive=False, visible=False),
            gr.update(value=download_start),
            gr.update(value='<div style="min-height: 100px;"></div>'),
            gr.update(value=html)
        )

    model_list_json = _build_model_list_for_update(matched[:1])

    # If user selected specific versions (revamp), pass them to selected_to_queue
    forced_version_ids = None
    if selected_version_ids:
        forced_version_ids = {str(model_id_int): selected_version_ids}

    return selected_to_queue(model_list_json, None, download_start, create_json, current_html, forced_version_ids=forced_version_ids)


def gr_progress_threadable():
    """
    Gradio progress bars can no longer be updated from a separate thread,
    so we need to use this to update them from the main thread.
    """

    # Gradio 3 does not need a wrapper
    if not hasattr(gr, '__version__') or int(gr.__version__.split('.')[0]) <= 3:
        return gr.Progress()

    gr_progress = gr.Progress()
    value = [0, None, False]  # progress, desc, has_update

    def progress(p, desc=None):
        value[0] = p
        value[1] = desc
        value[2] = True

    def _update_progress():
        if value[2]:
            try:
                gr_progress(value[0], desc=value[1])
            except Exception:
                # SSE/WebSocket connection dropped (e.g. screen lock, browser lost focus).
                # Silently ignore — the download thread continues unaffected.
                pass
            value[2] = False

    def join(thread):
        _update_progress()
        while thread.is_alive():
            thread.join(timeout=0.1)
            _update_progress()

    progress.join = join

    return progress


def download_start(download_start, dl_url, model_filename, install_path, model_string, version_name, model_sha256, model_id, create_json, current_html):
    global total_count, current_count
    if model_string:
        model_name, _ = _api.extract_model_info(model_string)
    model_item = create_model_item(dl_url, model_filename, install_path, model_name, version_name, model_sha256, model_id, create_json)

    if model_item:
        gl.download_queue.append(model_item)
    else:
        return (
            gr.update(interactive=True, visible=True),  # Download Button
            gr.update(interactive=False, visible=False),  # Cancel Button
            gr.update(interactive=False, visible=False),  # Cancel All Button
            gr.update(value=download_start),  # Download Start Trigger
            gr.update(value='<div style="min-height: 100px;"></div>'),  # Download Progress
            gr.update(value=current_html)  # Download Manager HTML
        )

    if len(gl.download_queue) > 1:
        number = download_start
        total_count += 1
    else:
        number = random_number(download_start)
        total_count = 1
        current_count = 0

    html = download_manager_html(current_html)

    return (
        gr.update(interactive=False, visible=True),  # Download Button
        gr.update(interactive=True, visible=True),  # Cancel Button
        gr.update(interactive=True if len(gl.download_queue) > 1 else False, visible=True),  # Cancel All Button
        gr.update(value=number),  # Download Start Trigger
        gr.update(value='<div style="min-height: 100px;"></div>'),  # Download Progress
        gr.update(value=html)  # Download Manager HTML
    )

def download_finish(model_filename, version, model_id):
    if model_id:
        model_id = int(model_id)
        model_versions = _api.update_model_versions(model_id)
    else:
        model_versions = None
    if model_versions:
        version_choices = model_versions.get('choices', [])
    else:
        version_choices = []
    prev_version = gl.last_version + " [Installed]"

    if prev_version in version_choices:
        version = prev_version
        Del = True
        Down = False
    else:
        Del = False
        Down = True

    if gl.cancel_status:
        Del = False
        Down = True

    gl.download_fail = False
    gl.cancel_status = False

    # In Update Mode, remove the updated model from gl.update_items so the card disappears.
    if gl.update_mode and model_id:
        gl.update_items = [i for i in gl.update_items if i.get('model_id') != model_id]

    # In Update Mode, do not touch the version dropdown — it can reset the user's selection.
    version_update = gr.update(value=version, choices=version_choices) if not gl.update_mode else gr.update()

    return (
        gr.update(interactive=model_filename, visible=Down, value="Download model"),  # Download Button
        gr.update(interactive=False, visible=False),  # Cancel Button
        gr.update(interactive=False, visible=False),  # Cancel All Button
        gr.update(interactive=Del, visible=Del),  # Delete Button
        gr.update(value='<div style="min-height: 0px;"></div>'),  # Download Progress
        version_update  # Version Dropdown
    )

def download_cancel():
    gl.cancel_status = True
    # Do NOT set gl.download_fail — cancel is not a failure.
    # Setting it would mark the item as failed and could bleed into the next queue item.
    if gl.download_queue:
        item = gl.download_queue[0]
    else:
        item = None

    _not_downloading.wait(timeout=60)
    if item:
        model_string = f"{item['model_name']} ({item['model_id']})"
        _file.delete_model(0, item['model_filename'], model_string, item['version_name'], False, model_ver=item['model_versions'], model_json=item['model_json'])
    return

def download_cancel_all():
    gl.cancel_status = True
    # Do NOT set gl.download_fail — cancel is not a failure.
    # Setting it would mark the item as failed and could bleed into the next queue item.

    if gl.download_queue:
        item = gl.download_queue[0]
    else:
        item = None

    _not_downloading.wait(timeout=60)
    if item:
        model_string = f"{item['model_name']} ({item['model_id']})"
        _file.delete_model(0, item['model_filename'], model_string, item['version_name'], False, model_ver=item['model_versions'], model_json=item['model_json'])
    _dl_log.log_all_cancelled()
    gl.download_queue = []
    return

def convert_size(size):
    for unit in ['bytes', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GB"


def _is_signed_civitai_download(url):
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return False
    return 'civitai.red' in host or 'b2.civitai.com' in host or 'b2.civitai.red' in host

def get_download_link(url, model_id):
    headers = _api.get_headers(model_id)
    proxies, ssl = _api.get_proxies()

    response = requests.get(url, headers=headers, allow_redirects=False, proxies=proxies, verify=ssl, timeout=30)

    if 300 <= response.status_code <= 308:
        if 'login?returnUrl' in response.text and 'reason=download-auth' in response.text:
            return 'NO_API'

        download_link = response.headers['Location']
        return download_link
    else:
        return None

def _get_download_link_with_retry(url, model_id, file_name, progress=None):
    """Fetch download link with 1 retry on timeout.
    Returns the link, 'NO_API', or None on failure.
    Sets gl.download_fail and updates progress on failure so the caller
    can simply return and let the queue advance to the next item."""
    for attempt in range(2):
        try:
            link = get_download_link(url, model_id)
            return link
        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
            if attempt == 0:
                msg = f"Timeout getting download link for '{file_name}', retrying in 3s..."
                print(msg)
                debug_print(f"[Download] {msg}")
                if progress is not None:
                    progress(0, desc=msg)
                time.sleep(3)
            else:
                msg = f"Timeout getting download link for '{file_name}' after retry — skipping item."
                print(msg)
                debug_print(f"[Download] {msg}")
                if progress is not None:
                    progress(0, desc=msg)
                gl.download_fail = True
                return None
        except Exception as e:
            msg = f"Error getting download link for '{file_name}': {e}"
            print(msg)
            debug_print(f"[Download] {msg}")
            if progress is not None:
                progress(0, desc=f"Error downloading '{file_name}' — skipped.")
            gl.download_fail = True
            return None

def download_file(url, file_path, install_path, model_id, progress=gr.Progress() if queue else None):
    try:
        disable_dns = getattr(opts, 'disable_dns', False)
        split_aria2 = getattr(opts, 'split_aria2', 64)
        max_retries = 5
        gl.download_fail = False
        aria2_rpc_url = "http://localhost:24000/jsonrpc"

        file_name = os.path.basename(file_path)
        debug_print(f"[Download] aria2 start: {file_name}")

        ## === ANXETY EDITs ===
        # Find the model item in the download queue (by model_id and file_name)
        early_access = False
        for item in gl.download_queue:
            if int(item.get('model_id', -1)) == int(model_id):
                model_json = item.get('model_json', {})
                items = model_json.get('items', [])
                if items and 'modelVersions' in items[0]:
                    version = items[0]['modelVersions'][0]
                    if is_early_access(version):
                        early_access = True
                break

        if early_access:
            msg = f"File: '{file_name}' is marked as Early Access on CivitAI. You need to purchase this model to download it."
            print(msg)
            gl.download_fail = 'EARLY_ACCESS'
            if progress is not None:
                progress(0, desc=msg)
                time.sleep(5)
            return

        download_link = _get_download_link_with_retry(url, model_id, file_name, progress)
        if download_link is None:
            return
        if not download_link:
            msg = f"File: '{file_name}' not found on CivitAI servers, it looks like the file is not available for download."
            debug_print(f"[Download] Failed to get download link for {file_name}")
            print(msg)
            gl.download_fail = True
            return
        elif download_link == 'NO_API':
            msg = f"File: '{file_name}' requires a personal CivitAI API to be downloaded, you can set your own API key in the CivitAI Browser+ settings in the SD-WebUI settings tab"
            print(msg)
            gl.download_fail = 'NO_API'
            if progress is not None:
                progress(0, desc=msg)
                time.sleep(5)
            return
        else:
            debug_print(f"[Download] Got download link for {file_name}")

        signed_download = _is_signed_civitai_download(download_link)

        if os.path.exists(file_path):
            _file.handle_existing_model_file(file_path)
        if signed_download:
            aria2_sidecar = file_path + '.aria2'
            if os.path.exists(aria2_sidecar):
                try:
                    os.remove(aria2_sidecar)
                except Exception:
                    pass

        if disable_dns:
            dns = 'false'
        else:
            dns = 'true'

        options = {
            'dir': install_path,
            'max-connection-per-server': '1' if signed_download else str(f"{split_aria2}"),
            'split': '1' if signed_download else str(f"{split_aria2}"),
            'async-dns': dns,
            'continue': 'false' if signed_download else 'true',
            'out': file_name
        }

        payload = json.dumps({
            'jsonrpc': '2.0',
            'id': '1',
            'method': 'aria2.addUri',
            'params': ['token:' + rpc_secret, [download_link], options]
        })

        try:
            response = requests.post(aria2_rpc_url, data=payload, timeout=10)
            data = json.loads(response.text)
            if 'result' not in data:
                raise ValueError(f"Failed to start download: {data}")
            gid = data['result']
        except Exception as e:
            print(f"Failed to start download: {e}")
            gl.download_fail = True
            return

        while True:
            if gl.cancel_status:
                payload = json.dumps({
                    'jsonrpc': '2.0',
                    'id': '1',
                    'method': 'aria2.remove',
                    'params': ['token:' + rpc_secret, gid]
                })
                requests.post(aria2_rpc_url, data=payload, timeout=10)
                if progress != None:
                    progress(0, desc="Download cancelled.")
                return

            try:
                payload = json.dumps({
                    'jsonrpc': '2.0',
                    'id': '1',
                    'method': 'aria2.tellStatus',
                    'params': ['token:' + rpc_secret, gid]
                })

                response = requests.post(aria2_rpc_url, data=payload, timeout=10)
                status_info = json.loads(response.text)['result']

                total_length = int(status_info['totalLength'])
                completed_length = int(status_info['completedLength'])
                download_speed = int(status_info['downloadSpeed'])

                progress_percent = (completed_length / total_length) * 100 if total_length else 0

                remaining_size = total_length - completed_length
                if download_speed > 0:
                    eta_seconds = remaining_size / download_speed
                    eta_formatted = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                else:
                    eta_formatted = "XX:XX:XX"
                if progress != None:
                    progress(progress_percent / 100, desc=f"Downloading: {file_name} - {convert_size(completed_length)}/{convert_size(total_length)} - Speed: {convert_size(download_speed)}/s - ETA: {eta_formatted} - Queue: {current_count}/{total_count}")

                if status_info['status'] == 'complete':
                    msg = f"Model saved to: {file_path}"
                    print(msg)
                    debug_print(f"[Download] aria2 complete: {file_name}")
                    if progress != None:
                        progress(1, desc=msg)
                    gl.download_fail = False
                    return

                if status_info['status'] == 'error':
                    if progress != None:
                        progress(0, desc=f"Encountered an error during download of: '{file_name}' Please try again.")
                    gl.download_fail = True
                    return

                time.sleep(0.25)

            except Exception as e:
                print(f"Error occurred during Aria2 status update: {e}")
                max_retries -= 1
                if max_retries == 0:
                    if progress != None:
                        progress(0, desc=f"Encountered an error during download of: '{file_name}' Please try again.")
                    gl.download_fail = True
                    return
                # #5 Aria2 RPC unreachable — restart and re-add the download
                print("Aria2 RPC unreachable, attempting to reconnect...")
                start_aria2_rpc()
                time.sleep(3)
                try:
                    _reconnect_resp = requests.post(aria2_rpc_url, timeout=5, data=json.dumps({
                        'jsonrpc': '2.0', 'id': '1', 'method': 'aria2.addUri',
                        'params': ['token:' + rpc_secret, [download_link], options]
                    }))
                    _reconnect_data = json.loads(_reconnect_resp.text)
                    if 'result' in _reconnect_data:
                        gid = _reconnect_data['result']
                        print(f"Aria2 reconnected, resumed download of '{file_name}'.")
                except Exception:
                    pass
                time.sleep(2)
    except:
        if progress != None:
            progress(0, desc=f"Encountered an error during download of: '{file_name}' Please try again.")
        gl.download_fail = True
        if os.path.exists(file_path):
            os.remove(file_path)
        time.sleep(5)

def info_to_json(install_path, model_id, model_sha256, unpackList=None):
    json_file = os.path.splitext(install_path)[0] + '.json'
    data = _api.safe_json_load(json_file) or {}
    data.update({
        'modelId': model_id,
        'sha256': model_sha256
    })
    if unpackList:
        data['unpackList'] = unpackList

    _api.safe_json_save(json_file, data)

def download_file_old(url, file_path, model_id, progress=gr.Progress() if queue else None):
    try:
        gl.download_fail = False
        max_retries = 5
        tokens = re.split(re.escape(os.sep), file_path)
        file_name_display = tokens[-1]
        debug_print(f"[Download] fallback start: {file_name_display}")
        if os.path.exists(file_path):
            os.remove(file_path)

        downloaded_size = 0
        start_time = time.time()
        last_update_time = 0
        update_interval = 0.25

        download_link = _get_download_link_with_retry(url, model_id, file_name_display, progress)
        if download_link is None:
            return
        if not download_link:
            msg = f"File: '{file_name_display}' not found on CivitAI servers, it looks like the file is not available for download."
            print(msg)
            if progress != None:
                progress(0, desc=msg)
                time.sleep(5)
            gl.download_fail = True
            return

        elif download_link == 'NO_API':
            msg = f"File: '{file_name_display}' requires a personal CivitAI API key to be downloaded, you can set your own API key in the CivitAI Browser+ settings in the SD-WebUI settings tab"
            print(msg)
            gl.download_fail = 'NO_API'
            if progress != None:
                progress(0, desc=msg)
                time.sleep(5)
            return

        headers = _api.get_headers(model_id, True)
        proxies, ssl = _api.get_proxies()

        while True:
            if gl.cancel_status:
                if progress != None:
                    progress(0, desc='Download cancelled.')
                return
            if os.path.exists(file_path):
                downloaded_size = os.path.getsize(file_path)
                headers['Range'] = f"bytes={downloaded_size}-"

            with open(file_path, 'ab') as f:
                while gl.isDownloading:
                    try:
                        if gl.cancel_status:
                            if progress != None:
                                progress(0, desc='Download cancelled.')
                            return
                        try:
                            if gl.cancel_status:
                                if progress != None:
                                    progress(0, desc='Download cancelled.')
                                return
                            response = requests.get(download_link, headers=headers, stream=True, timeout=10, proxies=proxies, verify=ssl)
                            if response.status_code == 404:
                                if progress != None:
                                    progress(0, desc=f"Encountered an error during download of: {file_name_display}, file is not found on CivitAI servers.")
                                gl.download_fail = True
                                return
                            total_size = int(response.headers.get('Content-Length', 0))
                        except:
                            raise TimeOutFunction('Timed Out')

                        if total_size == 0:
                            total_size = downloaded_size

                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                if gl.cancel_status:
                                    if progress != None:
                                        progress(0, desc='Download cancelled.')
                                    return
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                elapsed_time = time.time() - start_time
                                download_speed = downloaded_size / elapsed_time
                                remaining_size = total_size - downloaded_size
                                if download_speed > 0:
                                    eta_seconds = remaining_size / download_speed
                                    eta_formatted = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                                else:
                                    eta_formatted = 'XX:XX:XX'
                                current_time = time.time()
                                if current_time - last_update_time >= update_interval:
                                    if progress != None:
                                        progress(downloaded_size / total_size, desc=f"Downloading: {file_name_display} {convert_size(downloaded_size)} / {convert_size(total_size)} - Speed: {convert_size(int(download_speed))}/s - ETA: {eta_formatted} - Queue: {current_count}/{total_count}")
                                    last_update_time = current_time
                                if gl.isDownloading == False:
                                    response.close
                                    break
                        downloaded_size = os.path.getsize(file_path)
                        break

                    except TimeOutFunction:
                        if progress != None:
                            progress(0, desc='CivitAI API did not respond, retrying...')
                        max_retries -= 1
                        if max_retries == 0:
                            if progress != None:
                                progress(0, desc=f"Encountered an error during download of: {file_name_display}, please try again.")
                            gl.download_fail = True
                            return
                        time.sleep(5)

            if (gl.isDownloading == False):
                break

            gl.isDownloading = False
            downloaded_size = os.path.getsize(file_path)
            if downloaded_size >= total_size:
                if not gl.cancel_status:
                    msg = f"Model saved to: {file_path}"
                    print(msg)
                    debug_print(f"[Download] fallback complete: {file_name_display}")
                    if progress != None:
                        progress(1, desc=msg)
                    gl.download_fail = False
                    return

            else:
                if progress != None:
                    progress(0, desc=f"Encountered an error during download of: {file_name_display}, please try again.")
                print(f"File download failed: {file_name_display}")
                gl.download_fail = True
                if os.path.exists(file_path):
                    os.remove(file_path)
    except:
        if progress != None:
            progress(0, desc=f"Encountered an error during download of: {file_name_display}, please try again.")
        gl.download_fail = True
        if os.path.exists(file_path):
            os.remove(file_path)
        time.sleep(5)

def download_create_thread(download_finish, queue_trigger, progress=gr_progress_threadable() if queue else None):
    global current_count

    # Auto-exit update mode when queue is empty on entry
    auto_exit_update = False
    if gl.update_mode and not gl.download_queue:
        gl.update_mode = False
        gl.update_items = []
        auto_exit_update = True

    if not gl.download_queue:
        return (
            gr.update(),  # Download Progress HTML
            gr.update(value=None),  # Current Model
            gr.update(value=random_number(download_finish)),  # Download Finish Trigger
            gr.update(),  # Queue Trigger — no update to prevent re-trigger loop
            gr.update(value='') if auto_exit_update else gr.update()  # Update Mode Banner
        )

    card_name = None
    # Process the entire queue in an internal loop to eliminate gaps
    # caused by Gradio event queue round-trips between items.
    while gl.download_queue:
        current_count += 1
        item = gl.download_queue[0]
        # If the user clicked Cancel between items (during time.sleep), respect it.
        if gl.cancel_status:
            _dl_log.log_cancelled(item['dl_id'])
            debug_print(f"[Download] Cancelled between items: '{item['model_name']}'")
            if len(gl.download_queue) != 0:
                gl.download_queue.pop(0)
            gl.cancel_status = False  # Reset after handling this single-item cancellation
            continue
        gl.cancel_status = False
        gl.download_fail = False
        use_aria2 = getattr(opts, 'use_aria2', True)
        unpack_zip = getattr(opts, 'unpack_zip', False)
        save_all_images = getattr(opts, 'auto_save_all_img', False)
        gl.recent_model = item['model_name']
        gl.last_version = item['version_name']
        debug_print(f"[Download] Starting item {current_count}/{total_count}: '{item['model_name']}' v'{item['version_name']}' → {item['install_path']}")

        # #2 Lazy API fetch: deferred from enqueue time for batch performance
        if not item.get('_api_ready'):
            debug_print(f"[Download] Lazy API fetch for '{item['model_name']}' (id={item['model_id']})")
            _lazy_versions = _api.update_model_versions(item['model_id'])
            if _lazy_versions:
                item['model_versions'] = _lazy_versions
                debug_print(f"[Download] Loaded model versions for '{item['model_name']}'")
            try:
                _lazy_result = _api.update_model_info(None, (item['model_versions'] or {}).get('value'), False, item['model_id'])
                item['preview_html'] = _lazy_result[0].get('value', '') if isinstance(_lazy_result[0], dict) else ''
                item['existing_path'] = (_lazy_result[11].get('value') if isinstance(_lazy_result[11], dict) else None) or item['install_path']
                debug_print(f"[Download] Loaded preview + existing_path for '{item['model_name']}'")
            except Exception as _e:
                debug_print(f"[Lazy fetch] Could not load API data for {item['model_name']}: {_e}")
            item['_api_ready'] = True

        # === Ambiguity check for SHA256: query both civitai.com and civitai.red by-hash
        try:
            if item.get('model_sha256') and not item.get('ambiguous_checked'):
                sha = item.get('model_sha256')
                candidates = []
                try:
                    # helper: ask both domains for candidates
                    def _find_sha_candidates(sha_val):
                        res = []
                        seen = set()
                        for domain in ('https://civitai.com', 'https://civitai.red'):
                            try:
                                url = f"{domain}/api/v1/model-versions/by-hash/{sha_val}"
                                data = _api.request_civit_api(url)
                                if isinstance(data, dict) and data.get('id'):
                                    vid = data.get('id')
                                    mid = data.get('modelId')
                                    vname = data.get('name')
                                    files = data.get('files', [])
                                    # use first file entry as representative
                                    for f in files:
                                        file_sha = (f.get('hashes', {}) or {}).get('SHA256', '')
                                        dl = f.get('downloadUrl') or data.get('downloadUrl') or ''
                                        fname = f.get('name')
                                        key = (mid, vid)
                                        if key in seen:
                                            continue
                                        seen.add(key)
                                        res.append({
                                            'modelId': mid,
                                            'versionId': vid,
                                            'version_name': vname,
                                            'file_name': fname,
                                            'sha256': (file_sha or '').upper(),
                                            'downloadUrl': dl,
                                            'domain': domain
                                        })
                            except Exception:
                                pass
                        return res

                    candidates = _find_sha_candidates(sha)
                except Exception:
                    candidates = []

                item['ambiguous_checked'] = True
                item['ambiguous_candidates'] = candidates

                # Log ambiguity for debugging and traceability
                try:
                    debug_print(f"[Debug] Ambiguous SHA detected for {sha}: {len(candidates)} candidates")
                    for c in candidates:
                        debug_print(f"[Debug] Candidate -> modelId={c.get('modelId')} versionId={c.get('versionId')} file={c.get('file_name')} domain={c.get('domain')}")
                except Exception:
                    pass

                # If multiple distinct candidates exist and none matches the queued model id/version, pause and ask user
                if len(candidates) > 1:
                    match = False
                    for c in candidates:
                        try:
                            if int(c.get('modelId', -1)) == int(item.get('model_id', -1)):
                                # prefer candidate that matches the queued model id
                                item['dl_url'] = c.get('downloadUrl') or item['dl_url']
                                match = True
                                break
                        except Exception:
                            continue

                    if not match:
                        try:
                            debug_print(f"[Debug] No matching queued model for SHA {sha}; prompting user to choose candidate for '{item.get('model_name')}'")
                        except Exception:
                            pass
                        # Prepare an HTML chooser and return early so the UI can show selection
                        html = '<div style="padding:12px;background:#1f1f1f;border-radius:8px;color:#fff">'
                        html += '<h3>Ambiguous SHA256 detected</h3>'
                        html += '<p>More than one model/version matches this SHA. Please choose which one to download:</p>'
                        html += '<ul style="list-style:none;padding:0;margin:0">'
                        for i, c in enumerate(candidates):
                            display = f"{c.get('modelId')} / {c.get('versionId')} — {c.get('version_name') or ''} — {c.get('file_name') or ''} ({c.get('domain')})"
                            # radio inputs; when confirmed they set hidden ambiguity_choice and click hidden confirm button
                            html += f'<li style="margin:6px 0;padding:6px;border:1px solid #333;border-radius:6px;background:#111"><label style="cursor:pointer"><input type="radio" name="ambig_choice" value="{i}"> {display}</label></li>'
                        html += '</ul>'
                        html += '<div style="margin-top:10px"><button onclick="(function(){let v=document.querySelector(\'input[name=\\\'ambig_choice\\\']:checked\'); if(v){let ta=document.getElementById(\'ambiguity_choice\'); if(ta){ta.value=v.value; ta.dispatchEvent(new Event(\'input\'));} let btn=document.getElementById(\'ambiguity_confirm\'); if(btn){btn.click();}} else {alert(\'Please select an option to continue\');}})()">Confirm selection</button></div>'
                        html += '</div>'

                        return (
                            gr.update(value=html),  # Download Progress HTML -> shows chooser
                            gr.update(value=item.get('model_name')),  # Current Model
                            gr.update(),  # Download Finish Trigger — no update
                            gr.update(),  # Queue Trigger — no update to prevent loop
                            gr.update()  # Update Mode Banner — no change
                        )
        except Exception:
            pass

        # Fix #3: do not mutate item dict — use a local effective path
        # Fallback to install_path if existing_path is None (e.g. lazy fetch returned value=None)
        effective_install_path = (item['existing_path'] or item['install_path']) if item['from_batch'] else item['install_path']

        gl.isDownloading = True
        _not_downloading.clear()  # signal: download in progress
        _dl_log.log_downloading(item['dl_id'])
        _file.make_dir(effective_install_path)

        path_to_new_file = os.path.join(effective_install_path, item['model_filename'])
        debug_print(f"[Download] File target: {path_to_new_file}")

        if use_aria2 and os_type != 'Darwin':
            debug_print(f"[Download] Using aria2 for '{item['model_filename']}'")
            thread = threading.Thread(target=download_file, args=(item['dl_url'], path_to_new_file, effective_install_path, item['model_id'], progress))
        else:
            debug_print(f"[Download] Using fallback downloader for '{item['model_filename']}'")
            thread = threading.Thread(target=download_file_old, args=(item['dl_url'], path_to_new_file, item['model_id'], progress))
        thread.start()
        try:
            if progress != None and hasattr(progress, 'join'):
                progress.join(thread)
            else:
                thread.join()
        except Exception:
            # If the progress channel throws (client disconnected / screen locked),
            # fall back to a plain join so download always completes before cleanup.
            thread.join()

        # Fix #1: SHA256 integrity check after download
        if not gl.cancel_status and not gl.download_fail and item.get('model_sha256') and os.path.exists(path_to_new_file):
            if progress is not None:
                progress(0.99, desc=f"Verifying integrity: {item['model_filename']}...")
            sha256_hash = hashlib.sha256()
            with open(path_to_new_file, 'rb') as _f:
                for _chunk in iter(lambda: _f.read(8 * 1024 * 1024), b''):
                    sha256_hash.update(_chunk)
            actual_sha256 = sha256_hash.hexdigest().upper()
            if actual_sha256 != item['model_sha256'].upper():
                sha_mismatch_resolved = False
                version_id = item.get('version_id')
                if version_id:
                    try:
                        domain = _api.get_civitai_domain()
                        api_url = f"https://{domain}/api/v1/model-versions/{version_id}"
                        proxies, ssl = _api.get_proxies()
                        response = requests.get(api_url, headers=_api.get_headers(), timeout=(60, 30), proxies=proxies, verify=ssl)
                        if response.status_code == 200:
                            data = response.json()
                            files = data.get('files', [])
                            primary_file = next((f for f in files if f.get('primary', False)), None)
                            if not primary_file and files:
                                primary_file = files[0]
                            if primary_file:
                                api_sha = primary_file.get('hashes', {}).get('SHA256', '').upper()
                                if api_sha and api_sha == actual_sha256:
                                    print(f"SHA256 updated silently for '{item['model_filename']}': old={item['model_sha256'][:12]}…, new={actual_sha256[:12]}…")
                                    item['model_sha256'] = actual_sha256
                                    sha_mismatch_resolved = True
                    except Exception as e:
                        debug_print(f"[SHA256 recheck] API error for version {version_id}: {e}")

                if not sha_mismatch_resolved:
                    file_size = os.path.getsize(path_to_new_file)
                    print(f"SHA256 mismatch for '{item['model_filename']}': expected {item['model_sha256'][:12]}…, got {actual_sha256[:12]}… (size: {file_size} bytes)")
                    gl.download_fail = True
                    if progress is not None:
                        progress(0, desc=f"Integrity check failed for '{item['model_filename']}' — file may be corrupted.")

        if not gl.cancel_status and not gl.download_fail:
            if os.path.exists(path_to_new_file):
                debug_print(f"[Download] Download succeeded for '{item['model_name']}'; starting post-processing")
                # Determine content type once — used for both zip extraction and post-download saves
                _item_content_type = ((item.get('model_json') or {}).get('items') or [{}])[0].get('type', '')
                _is_wildcard_dl = _item_content_type == 'Wildcards'
                unpackList = []
                if unpack_zip:
                    try:
                        if path_to_new_file.endswith('.zip'):
                            directory = Path(os.path.dirname(path_to_new_file))
                            if _is_wildcard_dl:
                                # Wildcards: flat extraction — files only, no internal folder structure.
                                # Prevents double-nesting (e.g. wildcards/pack/pack/file.txt).
                                import zipfile as _zipfile
                                with _zipfile.ZipFile(path_to_new_file, 'r') as zf:
                                    for member in zf.infolist():
                                        if member.is_dir():
                                            continue
                                        filename = os.path.basename(member.filename)
                                        if not filename:
                                            continue
                                        target_path = os.path.join(str(directory), filename)
                                        with zf.open(member) as src, open(target_path, 'wb') as dst:
                                            dst.write(src.read())
                                        unpackList.append(filename)
                            else:
                                zip_handler = ZipHandler(path_to_new_file)

                                for _, decoded_name in zip_handler.name_map.items():
                                    unpackList.append(decoded_name)

                                zip_handler.extract_all(directory)
                                zip_handler.zip_ref.close()

                            print(f"Successfully extracted {item['model_filename']} to {directory}")
                            os.remove(path_to_new_file)
                    except ImportError:
                        print('Python module "ZipUnicode" has not been imported correctly, cannot extract zip file. Please try to restart or install it manually.')
                    except Exception as e:
                        print(f"Failed to extract {item['model_filename']} with error: {e}")
                if not gl.cancel_status:
                    if item['create_json']:
                        _file.save_model_info(effective_install_path, item['model_filename'], item['sub_folder'], item['model_sha256'], item['preview_html'], api_response=item['model_json'])
                    info_to_json(path_to_new_file, item['model_id'], item['model_sha256'], unpackList)

                    if _item_content_type == 'Checkpoint' and os.path.exists(path_to_new_file):
                        sidecar_path = os.path.splitext(path_to_new_file)[0] + '.json'
                        sidecar_data = _api.safe_json_load(sidecar_path) if os.path.exists(sidecar_path) else {}
                        sidecar_data = sidecar_data if isinstance(sidecar_data, dict) else {}
                        _file.sync_checkpoint_sha256_on_download(
                            path_to_new_file,
                            sidecar_data.get('sha256') or item.get('model_sha256'),
                            model_id=sidecar_data.get('modelId') or item.get('model_id'),
                            model_version_id=sidecar_data.get('modelVersionId')
                        )

                    if not _is_wildcard_dl:
                        _file.save_preview(path_to_new_file, item['model_json'], True, item['model_sha256'])
                        if save_all_images:
                            _file.save_images(item['preview_html'], item['model_filename'], effective_install_path, item['sub_folder'], api_response=item['model_json'])

                    # Retention policy for updates where old and new filenames differ.
                    # (Same-filename case is already handled inside download_file via handle_existing_model_file.)
                    old_fp = item.get('old_file_path', '')
                    if old_fp and os.path.exists(old_fp) and os.path.abspath(old_fp) != os.path.abspath(path_to_new_file):
                        _file.handle_existing_model_file(old_fp)

        base_name = os.path.splitext(item['model_filename'])[0]
        base_name_preview = base_name + '.preview'

        if gl.download_fail:
            # EARLY_ACCESS and NO_API: download never started, no file was created by
            # this attempt — preserve any pre-existing file from a previous good download.
            download_never_started = gl.download_fail in ('EARLY_ACCESS', 'NO_API')
            if not download_never_started:
                for root, dirs, files in os.walk(effective_install_path, followlinks=True):
                    for file in files:
                        file_base_name = os.path.splitext(file)[0]
                        if file_base_name == base_name or file_base_name == base_name_preview:
                            path_file = os.path.join(root, file)
                            os.remove(path_file)

            if gl.cancel_status:
                print(f"Cancelled download of '{item['model_filename']}'")
            else:
                if gl.download_fail not in ('NO_API', 'EARLY_ACCESS'):
                    print(f"Error occured during download of '{item['model_filename']}'")

        if gl.cancel_status:
            card_name = None
        else:
            model_string = f"{item['model_name']} ({item['model_id']})"
            (card_name, _, _) = _file.card_update(item['model_versions'], model_string, item['version_name'], True)

        # Log final download status before removing from queue
        if gl.cancel_status:
            _dl_log.log_cancelled(item['dl_id'])
            debug_print(f"[Download] Cancelled: '{item['model_name']}'")
            if len(gl.download_queue) != 0:
                gl.download_queue.pop(0)
        elif gl.download_fail:
            # Permanent errors (Early Access, missing API key) should not be retried
            permanent_error = gl.download_fail in ('EARLY_ACCESS', 'NO_API')
            if not permanent_error:
                retry_count = item.get('retry_count', 0) + 1
                item['retry_count'] = retry_count
                if retry_count < 3:
                    # Move to end of queue for automatic retry
                    debug_print(f"[Download] Retry {retry_count}/3 for '{item['model_name']}' — moving to end of queue")
                    gl.download_queue.pop(0)
                    gl.download_queue.append(item)
                else:
                    debug_print(f"[Download] Failed permanently after 3 retries: '{item['model_name']}'")
                    _dl_log.log_failed(item['dl_id'])
                    if len(gl.download_queue) != 0:
                        gl.download_queue.pop(0)
            else:
                debug_print(f"[Download] Failed permanently (non-retryable): '{item['model_name']}'")
                _dl_log.log_failed(item['dl_id'])
                if len(gl.download_queue) != 0:
                    gl.download_queue.pop(0)
        else:
            _dl_log.log_completed(item['dl_id'])
            debug_print(f"[Download] Completed: '{item['model_name']}' → {path_to_new_file}")
            if len(gl.download_queue) != 0:
                gl.download_queue.pop(0)

        gl.isDownloading = False
        _not_downloading.set()  # signal: download finished, safe to cancel/cleanup
        time.sleep(2)

    # Queue exhausted — return final result
    # Auto-exit update mode when all downloads finish
    auto_exit_update = False
    if gl.update_mode and not gl.download_queue:
        gl.update_mode = False
        gl.update_items = []
        auto_exit_update = True

    finish_nr = random_number(download_finish)
    return (
        gr.update(),  # Download Progress HTML
        gr.update(value=card_name),  # Current Model
        gr.update(value=finish_nr),  # Download Finish Trigger
        gr.update(),  # Queue Trigger — no update to prevent re-trigger loop
        gr.update(value='') if auto_exit_update else gr.update()  # Update Mode Banner
    )

def remove_from_queue(dl_id):
    global total_count
    for item in gl.download_queue:
        if int(dl_id) == int(item['dl_id']):
            gl.download_queue.remove(item)
            total_count -= 1
            _dl_log.log_cancelled(int(dl_id))
            return

def arrange_queue(input):
    id_and_index = input.split('.')
    dl_id = int(id_and_index[0])
    index = int(id_and_index[1]) + 1
    for item in gl.download_queue:
        if int(item['dl_id']) == dl_id:
            current_item = gl.download_queue.pop(gl.download_queue.index(item))
            gl.download_queue.insert(index, current_item)
            break

def get_style(size, left_border):
    return f"flex-grow: {size};" + ("border-left: 1px solid var(--border-color-primary);" if left_border else '') + "padding: 5px 10px 5px 10px;width: 0;align-self: center;"

def download_manager_html(current_html):
    html = current_html.rsplit('</div>', 1)[0]
    pattern = r'dl_id="(\d+)"'
    matches = re.findall(pattern, html)
    existing_item_ids = [int(match) for match in matches]

    for item in gl.download_queue:
        if item['dl_id'] not in existing_item_ids:
            download_item = (
                f'<div class="civitai_dl_item" dl_id="{item["dl_id"]}" style="display: flex; font-size: var(--section-header-text-size);">'
                f'<div class="dl_name" style="{get_style(1, False)}"><span title="{item["model_name"]}">{item["model_name"]}</span></div>'
                f'<div class="dl_ver" style="{get_style(0.75, True)}"><span title="{item["version_name"]}">{item["version_name"]}</span></div>'
                f'<div class="dl_path" style="{get_style(1.5, True)}"><span title="{item["install_path"]}">{item["install_path"]}</span></div>'
                f'<div class="dl_stat" style="{get_style(1.5, True)}"><div class="dl_progress_bar" style="width:0%">In queue...</div></div>'
                f'<div class="dl_action_btn" style="{get_style(0.3, True)}text-align: center;"><span onclick="removeDlItem({item["dl_id"]}, this)" class="civitai-btn-text" style="font-size: larger;">Remove</span></div>'
                '</div>'
            )
            html += download_item
    html += '</div>'

    return html


# ─── Queue Restore  (called from civitai_gui.py) ─────────────────────────────

def get_interrupted_downloads_json():
    """Called by Gradio .load() on every page load.
    Returns a JSON string of interrupted items, or '' if none.
    Also purges stale completed/cancelled entries from the log file."""
    _dl_log.purge_old_entries(days=7)
    interrupted = _dl_log.get_interrupted()
    if not interrupted:
        return ''

    # Filter out false positives: if the file already exists on disk
    # and there is no Aria2 sidecar (.aria2), the download actually
    # completed and only the log wasn't updated (e.g. WebUI restarted
    # during post-processing).
    genuinely_interrupted = []
    for entry in interrupted:
        install_path = entry.get('install_path', '')
        filename = entry.get('model_filename', '')
        file_path = os.path.join(install_path, filename) if install_path and filename else ''
        aria2_path = file_path + '.aria2' if file_path else ''

        if file_path and os.path.exists(file_path) and not os.path.exists(aria2_path):
            # File is complete — auto-mark as completed in the log
            try:
                _dl_log.log_completed(entry.get('dl_id'))
            except Exception:
                pass
            continue  # skip from interrupted list

        genuinely_interrupted.append(entry)

    if not genuinely_interrupted:
        return ''
    import json as _json
    return _json.dumps(genuinely_interrupted)


def dismiss_interrupted_downloads():
    """Called when the user dismisses the restore banner without restoring."""
    _dl_log.dismiss_interrupted()
    return ''


def _restore_queue_item(data):
    """Rebuild a full queue item dict from a log entry, fetching fresh API data."""
    model_id = int(data['model_id'])

    # Duplicate guard
    for existing in gl.download_queue:
        if existing.get('dl_url') == data['dl_url']:
            return None

    # Fetch from API first — gl.json_data may be empty/None at restore time
    api_url = f'https://{_api.get_civitai_domain()}/api/v1/models/{model_id}'
    raw = _api.request_civit_api(api_url)
    model_json = {'items': [raw]} if raw and isinstance(raw, dict) else {'items': []}

    model_versions = _api.update_model_versions(model_id, json_input=model_json if model_json['items'] else None)
    if not model_versions:
        return None

    preview_html_val = ''
    existing_path_val = data['install_path']

    try:
        result = _api.update_model_info(None, model_versions.get('value'), False, model_id, json_input=model_json if model_json['items'] else None)
        # update_model_info returns a 13-tuple; index 0 = preview_html, index 11 = existing_path
        preview_html_val = result[0].get('value', '') if isinstance(result[0], dict) else ''
        existing_path_val = (result[11].get('value') if isinstance(result[11], dict) else None) or data['install_path']
    except Exception as e:
        debug_print(f"[Restore] Could not fetch API data for model {model_id}: {e}")

    global dl_manager_count
    dl_manager_count += 1

    return {
        'dl_id':          dl_manager_count,
        'dl_url':         data['dl_url'],
        'model_filename': data['model_filename'],
        'install_path':   data['install_path'],
        'model_name':     data['model_name'],
        'version_name':   data.get('version_name', ''),
        'model_sha256':   data.get('model_sha256'),
        'model_id':       model_id,
        'create_json':    data.get('create_json', True),
        'model_json':     model_json,
        'model_versions': model_versions,
        'preview_html':   preview_html_val,
        'existing_path':  existing_path_val,
        'from_batch':     data.get('from_batch', False),
        'sub_folder':     data.get('sub_folder', ''),
        '_api_ready':     True,  # already fetched above
    }


def restore_interrupted_to_queue(current_html):
    """Re-enqueue all interrupted downloads and kick off the download chain.
    Returns the same 6-tuple as download_start / selected_to_queue so the GUI
    wires up identically."""
    global total_count, current_count

    interrupted = _dl_log.get_interrupted()
    if not interrupted:
        return (
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(),
            gr.update(value=current_html),
        )

    number = random_number()
    total_count = 0
    current_count = 0

    for data in interrupted:
        item = _restore_queue_item(data)
        if item:
            gl.download_queue.append(item)
            total_count += 1

    _dl_log.dismiss_interrupted()  # don't show banner again unless new interruptions

    if not gl.download_queue:
        return (
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(),
            gr.update(value=current_html),
        )

    html = download_manager_html(current_html)

    return (
        gr.update(interactive=False, visible=False),   # Download Button
        gr.update(interactive=True, visible=True),     # Cancel Button
        gr.update(interactive=len(gl.download_queue) > 1, visible=True),  # Cancel All Button
        gr.update(value=number),                       # Download Start Trigger
        gr.update(value='<div style="min-height: 100px;"></div>'),  # Download Progress
        gr.update(value=html),                         # Download Manager HTML
    )