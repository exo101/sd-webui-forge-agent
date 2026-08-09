# FUNCTION_MAP.md — CivitAI Browser Neo

> Architectural summary of all major functions, classes, and modules.  
> Generated for **v0.9.0** · Last updated: 2026-05-09

---

## Module Overview

| Module | Role | Key Responsibilities |
|--------|------|---------------------|
| `scripts/civitai_api.py` | API Client & HTML Builder | CivitAI REST calls, card HTML generation, URL building, SHA256 search, image fetching |
| `scripts/civitai_download.py` | Download Engine | Aria2 RPC, queue management, batch updates, download/fallback logic, queue restore |
| `scripts/civitai_file_manage.py` | File Operations | Organization, deletion, dashboard, creator mgmt, metadata sidecars, auto-subfolders |
| `scripts/civitai_gui.py` | Gradio UI Definition | Tab layout, event wiring, settings persistence, dropdown helpers |
| `scripts/civitai_global.py` | Global State | Mutable module-level variables, colored print helpers, runtime init |
| `scripts/download_log.py` | Queue Persistence | JSONL log for download states (queued → downloading → completed / cancelled / failed / dismissed) |
| `javascript/civitai-html.js` | Frontend Logic | Card interaction, overlay, video hover, update polling, queue UI, image viewer |

---

## Conventions Used

- **Dependencies** list only the most significant cross-module or complex internal calls.
- `gr.update(...)` means the function returns one or more Gradio component update objects.
- `_api.` = `scripts/civitai_api.py`, `_dl.` = `scripts/civitai_download.py`, `_file.` = `scripts/civitai_file_manage.py`, `_gui.` = `scripts/civitai_gui.py`, `gl.` = `scripts/civitai_global.py`, `_dl_log.` = `scripts/download_log.py`.

---

## `scripts/civitai_api.py` — API Client & HTML Builder

### Display & Domain Helpers

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `get_display_type(type_name)` | Short display name for model types (e.g. `TextualInversion` → `Embedding`). | `MODEL_TYPE_DISPLAY_NAMES` | `str` |
| `get_base_model_short(base_model)` | Abbreviation for a base model (e.g. `SDXL 1.0` → `XL`). | `BASE_MODEL_SHORT` dict | `str` |
| `get_civitai_domain()` | Returns `civitai.com` or `civitai.red` based on SFW-only toggle. | `opts.civitai_sfw_only` | `str` |

### Status & Metadata

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `is_early_access(version_data)` | Checks if a version is marked as early access. | — | `bool` |
| `get_status_badge_type(item)` | Returns `'new'`, `'updated'`, or `''` based on version publish age. | `datetime`, `timedelta` | `str` |
| `is_model_nsfw(model_data, nsfw_level)` | Determines NSFW status from metadata and first-image NSFW level. | — | `bool` |

### Data Safety & I/O

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `normalize_sha256(sha256_hash)` | Upper-cases and validates SHA256 format. | — | `str \| None` |
| `safe_json_load(file_path)` | Loads JSON with error handling. | `os.path.exists`, `json.load` | `dict \| None` |
| `safe_json_save(file_path, data)` | Saves JSON with directory creation. | `os.makedirs`, `json.dump` | `bool` |

### Folder & Path Resolution

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `contenttype_folder(content_type, desc, custom_folder)` | Resolves on-disk folder for a content type (Neo vs Classic layouts). | `cmd_opts`, `models_path`, `_resolve_embeddings_folder`, `_resolve_upscaler_folder` | `Path \| None` |
| `sub_folder_value(content_type, desc)` | Returns default subfolder setting from Forge options. | `getattr(opts, ...)` | `str` |
| `cleaned_name(file_name)` | Sanitizes filename (illegal chars stripped, length capped). | `platform.system`, `re.sub` | `str` |

### Update Mode HTML

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `update_mode_page_html(content_type_filter, base_filter, tile_count, current_page)` | Renders update-mode card grid from `gl.update_items` without API call. | `gl.update_items`, `_fam_matches`, `_type_short` | `(html, total_pages, current_page, hasPrev, hasNext)` |

### Browser Card HTML

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `model_list_html(json_data)` | Builds full HTML card grid. Detects installed/outdated/cross-family status. | `filter_versions`, `collect_existing_files`, `get_model_card`, `contenttype_folder`, `_file.FavoriteCreators`, `_file.extract_version_from_ver_name`, `_file.compare_version_parts` | `str` |
| `filter_versions(item, hide_early_access, current_time)` *(nested)* | Filters out versions with no files or early-access versions. | — | `list` |
| `collect_existing_files(model_folders)` *(nested)* | Walks folders to collect existing filenames and SHA256 hashes. | `os.walk`, `json.load` | `(dict, dict)` |
| `get_model_card(item, ...)` *(nested)* | Builds HTML for a single model card (badges, preview, actions). | `_api.get_base_model_short`, `_api.is_model_nsfw` | `str` |

### SHA256 Search

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `_search_by_sha256(sha256_hash)` | Searches by SHA256 across both domains; handles ambiguity, 404, 503. | `normalize_sha256`, `get_headers`, `get_proxies`, `get_civitai_domain`, `request_civit_api` | `dict \| str` |

### API URL & Pagination

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `create_api_url(...)` | Builds CivitAI `/models` or `/model-versions` URL from filters. Also extracts IDs from pasted CivitAI URLs. | `get_civitai_domain`, `request_civit_api` | `str` |
| `initial_model_page(...)` | Entry point for loading a page. Handles update mode, SHA256 search, normal API search. | `update_mode_page_html`, `create_api_url`, `_search_by_sha256`, `request_civit_api`, `insert_metadata`, `model_list_html`, `api_error_msg` | `tuple[gr.update, ...]` (17 items) |
| `prev_model_page(...)` | Wrapper for navigating to previous page. | `next_model_page` | `tuple[gr.update, ...]` |
| `next_model_page(...)` | Wrapper for next/previous page with API fetch. | `create_api_url`, `request_civit_api`, `insert_metadata`, `model_list_html`, `initial_model_page` | `tuple[gr.update, ...]` |
| `insert_metadata(page_nr, api_url)` | Injects `prevPage`/`nextPage` into `gl.json_data['metadata']`. | `gl.json_data`, `gl.url_list` | `dict` |

### Version & File Info

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `update_model_versions(model_id, json_input, base_filter)` | Builds version dropdown, marks installed & early-access versions. | `gl.json_data`, `contenttype_folder`, `normalize_sha256`, `is_early_access` | `gr.update` |
| `update_file_info(model_string, model_version, file_metadata)` | Updates UI fields when a file is selected. Detects install status & misclassification. | `extract_model_info`, `gl.json_data`, `contenttype_folder`, `normalize_sha256`, `sub_folder_value`, `cleaned_name`, `_dl.convert_size` | `tuple[gr.update, ...]` (8 items) |
| `update_model_info(...)` | Main preview-panel builder (description, permissions, samples, trigger words, install path). | `extract_model_info`, `gl.json_data`, `contenttype_folder`, `is_model_nsfw`, `get_civitai_domain`, `request_civit_api`, `fetch_and_process_image`, `get_local_trigger_words`, `_file.getSubfolders`, `_file.get_companion_banner`, `_file.convertCustomFolder`, `_dl.convert_size` | `tuple[gr.update, ...]` (13 items) |

### Utilities

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `fetch_and_process_image(image_url)` | Fetches image and extracts generation metadata. | `get_proxies`, `requests.get`, `PIL.Image.open`, `read_info_from_image` | `str \| None` |
| `extract_model_info(input_string)` | Parses `"Model Name (12345)"` into name and numeric ID. | — | `(str, int)` |
| `get_local_trigger_words(content_type, model_filename, sha256, allow_legacy)` | Loads trigger words from local `.json` sidecar. | `contenttype_folder`, `safe_json_load` | `list \| None` |

### Network Layer

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `get_proxies()` | Builds proxy & SSL verification settings from Forge options. | `opts.custom_civitai_proxy`, `opts.disable_sll_proxy` | `(dict, bool)` |
| `get_headers(referer, no_api)` | Builds HTTP headers with optional API key. | `opts.custom_api_key`, `get_civitai_domain` | `dict` |
| `request_civit_api(api_url, skip_error_check)` | Core API request with retry logic (5xx, timeout, DNS). | `get_headers`, `get_proxies`, `requests.get`, `json.loads` | `dict \| str` |

### Error Handling

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `inject_removed_banner(html)` | Prepends "removed by owner" warning banner. | — | `str` |
| `api_error_msg(input_string)` | User-friendly HTML error message for common failures. | — | `str` |

---

## `scripts/civitai_download.py` — Download Engine

### Utilities

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `random_number(prev)` | Generates a random 5-digit string, avoiding the previous value. | `random.randint` | `str` |
| `convert_size(size)` | Human-readable byte size (B → KB → MB → GB). | — | `str` |
| `get_style(size, left_border)` | Inline CSS style for download-manager rows. | — | `str` |

### Aria2 RPC Lifecycle

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `start_aria2_rpc()` | Starts/restarts Aria2 RPC daemon on port `24000` with fixed secret. | `subprocess.Popen`, `os.path.exists` | `None` |

### Queue Item Creation

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `create_model_item(dl_url, model_filename, install_path, model_name, version_name, model_sha256, model_id, create_json, from_batch, old_file_path, version_id)` | Builds a queue item dict; guards against duplicate URLs. | `gl.json_data`, `_api.contenttype_folder`, `_dl_log.log_queued` | `dict \| None` |
| `_resolve_versions_to_download(versions_list, model_folder)` | Determines which versions to download for batch updates (one per installed family). | `_api.safe_json_load`, `_file.extract_version_from_ver_name` | `list` |

### Enqueuing & UI Entry Points

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `selected_to_queue(model_list, subfolder, download_start, create_json, current_html)` | Converts JSON model list into queued items with auto-organization. | `_api.extract_model_info`, `_api.contenttype_folder`, `_resolve_versions_to_download`, `create_model_item`, `_file.convertCustomFolder`, `_file.normalize_base_model`, `download_manager_html` | `tuple[gr.update, ...]` (6 items) |
| `_build_model_list_for_update(items)` | Formats `gl.update_items` into JSON strings for `selected_to_queue`. | `json.dumps` | `str` |
| `update_all_models(download_start, create_json, current_html)` | Enqueues every model flagged for update. | `_build_model_list_for_update`, `selected_to_queue` | `tuple[gr.update, ...]` |
| `update_selected_models(trigger_value, download_start, create_json, current_html)` | Enqueues only checked models from Update Mode. | `update_all_models`, `selected_to_queue` | `tuple[gr.update, ...]` |
| `download_single_update(trigger_value, download_start, create_json, current_html)` | Enqueues a single model update by `model_id\|family`. | `_build_model_list_for_update`, `selected_to_queue`, `update_all_models` | `tuple[gr.update, ...]` |
| `download_start(download_start, dl_url, model_filename, install_path, model_string, version_name, model_sha256, model_id, create_json, current_html)` | Enqueues a single manual download and triggers the chain. | `_api.extract_model_info`, `create_model_item`, `random_number`, `download_manager_html` | `tuple[gr.update, ...]` |

### Download Control

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `download_finish(model_filename, version, model_id)` | Resets UI buttons after download and refreshes version dropdown. | `_api.update_model_versions` | `tuple[gr.update, ...]` |
| `download_cancel()` | Sets cancel flag, waits for stop, deletes partial file. | `_not_downloading.wait`, `_file.delete_model` | `None` |
| `download_cancel_all()` | Cancels current download and empties entire queue. | `_not_downloading.wait`, `_file.delete_model`, `_dl_log.log_all_cancelled` | `None` |

### Gradio Progress

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `gr_progress_threadable()` | Thread-safe progress callable for Gradio 4+ with buffered updates. | `gr.Progress`, `gr.__version__` | custom `progress` object |

### Download Engine (Aria2 + Fallback)

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `_is_signed_civitai_download(url)` | Checks if URL is a signed CivitAI CDN link. | `urllib.parse.urlparse` | `bool` |
| `get_download_link(url, model_id)` | Resolves CivitAI download URL to actual CDN redirect. | `_api.get_headers`, `_api.get_proxies`, `requests.get` | `str \| None` (or `'NO_API'`) |
| `_get_download_link_with_retry(url, model_id, file_name, progress)` | Wraps `get_download_link` with one timeout retry. | `get_download_link`, `time.sleep` | `str \| None` (or `'NO_API'`) |
| `download_file(url, file_path, install_path, model_id, progress)` | Actual download via Aria2 RPC with progress polling and cancel support. | `_get_download_link_with_retry`, `_is_signed_civitai_download`, `_file.handle_existing_model_file`, `start_aria2_rpc`, `requests.post` (Aria2 JSON-RPC) | `None` |
| `download_file_old(url, file_path, model_id, progress)` | Fallback direct HTTP downloader (requests streaming). | `_get_download_link_with_retry`, `_api.get_headers`, `_api.get_proxies`, `requests.get` | `None` |

### Post-Download & Metadata

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `info_to_json(install_path, model_id, model_sha256, unpackList)` | Writes/updates sidecar `.json` with `modelId`, `sha256`, `unpackList`. | `_api.safe_json_load`, `_api.safe_json_save` | `None` |

### Core Queue Processor

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `download_create_thread(download_finish, queue_trigger, progress)` | Main queue-processing loop: lazy API fetch, SHA ambiguity check, download threading, SHA256 verification, ZIP extraction, metadata, retention, retries, logging. | `_api.update_model_versions`, `_api.update_model_info`, `_api.request_civit_api`, `_api.get_civitai_domain`, `_file.make_dir`, `_file.save_model_info`, `_file.save_preview`, `_file.save_images`, `_file.handle_existing_model_file`, `_file.card_update`, `_file.sync_checkpoint_sha256_on_download`, `_dl_log.log_downloading`, `_dl_log.log_completed`, `_dl_log.log_cancelled`, `_dl_log.log_failed`, `download_file`, `download_file_old`, `hashlib.sha256`, `random_number` | `tuple[gr.update, ...]` (4 items) |

### Ambiguity & Queue Management

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `resolve_ambiguity(choice_index)` | Resolves ambiguous SHA256 match by applying user's chosen candidate. | `debug_print`, `_api.cleaned_name`, `random_number` | `tuple[gr.update, ...]` (4 items) |
| `remove_from_queue(dl_id)` | Removes single item from queue by `dl_id`. | `_dl_log.log_cancelled` | `None` |
| `arrange_queue(input)` | Reorders queue item by dot-separated `dl_id.index`. | — | `None` |
| `download_manager_html(current_html)` | Generates/appends HTML rows for download manager panel. | `get_style` | `str` |

### Queue Restore / Persistence

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `get_interrupted_downloads_json()` | Scans log for interrupted downloads, filters false positives. | `_dl_log.purge_old_entries`, `_dl_log.get_interrupted`, `_dl_log.log_completed` | `str` (JSON) |
| `dismiss_interrupted_downloads()` | Marks all interrupted entries as dismissed. | `_dl_log.dismiss_interrupted` | `str` |
| `_restore_queue_item(data)` | Reconstructs queue item from log entry, fetches fresh API data. | `_api.get_civitai_domain`, `_api.request_civit_api`, `_api.update_model_versions`, `_api.update_model_info` | `dict \| None` |
| `restore_interrupted_to_queue(current_html)` | Re-enqueues genuinely interrupted downloads. | `_dl_log.get_interrupted`, `_dl_log.dismiss_interrupted`, `_restore_queue_item`, `random_number`, `download_manager_html` | `tuple[gr.update, ...]` |

### Exception

| Class | Description |
|-------|-------------|
| `TimeOutFunction(Exception)` | Raised by fallback downloader to signal request timeout. |

---

## `scripts/civitai_file_manage.py` — File Operations

### Utilities

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `_format_size(size_bytes)` / `format_size(size_bytes)` | Human-readable byte size. | — | `str` |
| `make_dir(path)` | Creates directory tree, retries with `0o777` on permission error. | `os.makedirs` | `None` |

### Creator Management

| Class/Function | Description | Dependencies | Returns |
|----------------|-------------|--------------|---------|
| `class UserInfo` | Persistent comma-separated creator list in `.txt`. Methods: `load`, `save`, `add`, `remove`, `get_as_list`, `get_as_text`. | `os.path.exists`, `open` | — |
| `class FavoriteUsers(UserInfo)` | Uses `favoriteCreators.txt`. | — | — |
| `class BanUsers(UserInfo)` | Uses `bannedCreators.txt`. | — | — |
| `_creator_button_updates(username)` | Computes favorite/ban/clear button states. | `FavoriteCreators`, `BanCreators` | `tuple[gr.update, ...]` |
| `add_favorite_creator(username)` | Adds to favorites, removes from ban. | `_creator_button_updates` | `tuple[gr.update, ...]` |
| `ban_creator(username)` | Adds to ban, removes from favorites. | `_creator_button_updates` | `tuple[gr.update, ...]` |
| `clear_creator(username)` | Removes from both lists. | `_creator_button_updates` | `tuple[gr.update, ...]` |
| `get_banned_creators_text()` | Returns comma-joined banned creators for JS init. | — | `str` |

### Companion Files Banner

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `get_companion_banner(base_model, model_filename, model_name)` | HTML banner listing missing required companion files (VAE, text encoders, etc.). | `models_path`, `re` | `str` |

### Audit Log & Retention

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `append_update_audit_log(action, details)` | Appends JSONL line to `neo_update_audit.jsonl`. | `json.dumps`, `Path` | `None` |
| `handle_existing_model_file(file_path)` | Applies retention policy (`keep`, `move to _Trash`, `replace`) before downloading. | `append_update_audit_log`, `_trash_associated_files`, `delete_associated_files`, `shutil.move` | `None` |

### Deletion

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `delete_model(delete_finish, model_filename, model_string, list_versions, sha256, selected_list, model_ver, model_json)` | Deletes model + sidecars; trash vs permanent based on settings. | `_api.extract_model_info`, `_api.update_model_versions`, `card_update`, `_dl.random_number`, `delete_associated_files`, `send2trash` | `tuple[gr.update, ...]` |
| `delete_installed_by_sha256(sha256, delete_finish)` | Searches all folders for matching SHA256 and deletes. | `_api.contenttype_folder`, `_api.safe_json_load`, `delete_associated_files`, `send2trash`, `_dl.random_number` | `gr.update` |
| `delete_associated_files(directory, base_name)` | Deletes preview, api_info, HTML, and numbered image sidecars. | `send2trash`, `os.remove` | `None` |
| `_trash_associated_files(directory, base_name, trash_dir)` | Moves sidecars into `_Trash` folder. | `shutil.move` | `None` |

### Preview & Image Handling

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `_resize_image_bytes(image_bytes, target_size)` | Resizes image preserving aspect ratio. | `Image.open`, `Image.LANCZOS` | `bytes` |
| `save_preview(file_path, api_response, overwrite_toggle, sha256)` | Downloads and saves first preview image as `<name>.preview.png`. | `_api.get_proxies`, `_api.get_civitai_domain`, `_resize_image_bytes`, `requests.get` | `None` |
| `get_image_path(install_path, api_response, sub_folder)` | Destination folder for gallery/sample images. | `_api.contenttype_folder`, `make_dir` | `str` |
| `save_images(preview_html, model_filename, install_path, sub_folder, api_response)` | Extracts image URLs from preview HTML and downloads them. | `get_image_path`, `_resize_image_bytes`, `urllib.request` | `None` |

### Model Info, HTML & Cards

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `convert_local_images(html)` | Converts `<img data-sampleimg="true">` with local paths to base64 inline. | `BeautifulSoup`, `base64.b64encode`, `PIL.Image` | `str` |
| `_get_cached_html_stripped(model_file)` | Reads local `.html` cache and strips `<head>`. | `os.path.exists`, `open` | `str \| None` |
| `_wrap_html_with_css(body)` | Prepends `style_html.css` inside `<head><style>`. | `Path.read_text` | `str` |
| `model_from_sent(model_name, content_type)` | Resolves local model and returns preview HTML (cached or API). | `_api.contenttype_folder`, `_api.api_error_msg`, `_api.request_civit_api`, `_api.update_model_info`, `get_models`, `gen_sha256`, `find_model_version_by_sha256`, `find_model_version_by_filename`, `_wrap_html_with_css`, `_get_cached_html_stripped` | `(gr.update,)` |
| `send_to_browser(model_name, content_type, click_first_item)` | Finds local model by name, resolves CivitAI ID, loads browser card view. | `_api.contenttype_folder`, `_api.api_error_msg`, `_api.request_civit_api`, `_api.model_list_html`, `get_models`, `_dl.random_number` | `tuple[gr.update, ...]` |
| `card_update(gr_components, model_name, list_versions, is_install)` | Appends `.New`/`.Old`/`.None` suffix to model name based on install state. | — | `tuple` |

### Subfolders & Custom Folders

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `convertCustomFolder(folderValue, basemodel, nsfw, author, modelName, modelId, versionName, versionId)` | Replaces placeholders (`{BASEMODEL}`, `{AUTHOR}`, etc.) in custom folder template. | `_api.cleaned_name` | `str` |
| `getSubfolders(model_folder, basemodel, nsfw, author, modelName, modelId, versionName, versionId)` | Lists existing subdirs + custom template subfolders for dropdown. | `_api.safe_json_load`, `convertCustomFolder` | `list` |
| `updateSubfolder(subfolderInput)` | Parses `index.action.value` and edits custom subfolder config JSON. | `_api.safe_json_load`, `_api.safe_json_save` | `None` |

### Description & URL Helpers

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `is_image_url(url)` | Checks if URL points to an image by extension. | `urlparse` | `bool` |
| `clean_description(desc)` | Converts CivitAI HTML to plain text with markdown-like formatting. | `BeautifulSoup`, `re` | `str` |

### JSON Sidecars & Metadata

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `save_model_info(install_path, file_name, sub_folder, sha256, preview_html, overwrite_toggle, api_response)` | Saves `.json` sidecar, `.html` preview, `.api_info.json`. | `get_save_path_and_name`, `get_image_path`, `find_and_save`, `gen_sha256`, `_api.safe_json_save`, `_api.get_proxies`, `_api.get_civitai_domain`, `_api.normalize_sha256`, `requests.get` | `None` |
| `find_model_version_by_sha256(api_response, sha256)` | Finds model version dict matching SHA256. | `_api.normalize_sha256` | `(model_version, item) \| (None, None)` |
| `find_model_version_by_filename(api_response, file_name)` | Finds model version dict matching filename. | — | `(model_version, item) \| (None, None)` |
| `extract_safetensors_metadata(file_path)` | Parses `.safetensors` header for trigger words. | `json.loads`, `re.split` | `list` |
| `consolidate_trigger_words(safetensors_tags, json_tags, api_tags)` | Deduplicates and merges trigger words from three sources. | `re.split` | `list` |
| `find_and_save(api_response, sha256, file_name, json_file, no_hash, overwrite_toggle)` | Locates version by SHA256 or filename and writes `.json` sidecar. | `find_model_version_by_sha256`, `find_model_version_by_filename`, `extract_safetensors_metadata`, `consolidate_trigger_words`, `clean_description`, `_api.safe_json_load`, `_api.safe_json_save` | `'found' \| 'not found'` |
| `get_models(file_path, gen_hash)` | Resolves CivitAI `modelId` from local file via sidecar or SHA256 lookup. | `_api.safe_json_load`, `gen_sha256`, `_api.get_civitai_domain`, `_api.get_proxies`, `_api.safe_json_save` | `str \| 'offline' \| 'Model not found' \| None` |
| `gen_sha256(file_path)` | Computes file SHA256, caches in `.json` sidecar. | `_api.safe_json_load`, `_api.safe_json_save`, `hashlib.sha256` | `str` |
| `_normalize_sha256(sha256_value)` | Validates and lowercases SHA256 string. | `re.fullmatch` | `str \| None` |

### Checkpoint Hash Cache (Forge Integration)

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `_load_checkpoint_hash_db()` | Loads `lib/models/checkpoint_hashes.json`. | `_api.safe_json_load` | `dict` |
| `_save_checkpoint_hash_db(db_data)` | Persists checkpoint hash registry. | `_api.safe_json_save` | `None` |
| `_is_checkpoint_file(file_path)` | Checks if `.safetensors` or `.ckpt`. | — | `bool` |
| `_checkpoint_cache_key(file_path)` | Builds relative path key for Forge hash cache. | `_api.contenttype_folder` | `str` |
| `_upsert_forge_hash_cache(file_path, sha256_value, add_only)` | Writes checkpoint SHA256 into Forge `hashes` cache. | `modules.hashes.cache`, `modules.hashes.dump_cache`, `_normalize_sha256` | `(bool, str)` |
| `_cleanup_deleted_checkpoints(db_data, existing_paths)` | Removes stale entries from local DB and Forge cache. | `modules.hashes.cache`, `modules.hashes.dump_cache` | `(int, int)` |
| `sync_checkpoint_sha256_on_download(file_path, sha256_value, model_id, model_version_id)` | One-shot sync for freshly downloaded checkpoint. | `_normalize_sha256`, `_upsert_forge_hash_cache`, `_load_checkpoint_hash_db`, `_save_checkpoint_hash_db` | `None` |
| `sync_checkpoint_sha256_cache(progress)` | Batch scanner syncing sidecar SHA256s into Forge cache. | `_api.contenttype_folder`, `_is_checkpoint_file`, `_load_checkpoint_hash_db`, `_cleanup_deleted_checkpoints`, `_upsert_forge_hash_cache`, `_save_checkpoint_hash_db`, `_normalize_sha256`, `_api.safe_json_load` | `gr.update` |

### Version Matching & Update Detection

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `extract_version_from_ver_name(filename)` | Extracts semantic family name and version parts via regex. | `re.search` | `(family_name \| None, list[int])` |
| `compare_version_parts(a_parts, b_parts)` | Compares two semantic version part lists. | — | `int` (`-1, 0, 1`) |
| `version_match(file_paths, api_response, log)` | Determines outdated models by API order + semantic version comparison. | `_api.safe_json_load`, `extract_version_from_ver_name`, `compare_version_parts` | `(updated_models, outdated_models)` |
| `collect_update_items(outdated_set, api_response, file_paths)` | Builds `gl.update_items` entries for Update Mode UI cards. | `_api.safe_json_load` | `list[dict]` |

### File Scanning Orchestration

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `get_content_choices(scan_choices)` | Returns supported content type strings. | — | `list[str]` |
| `get_save_path_and_name(install_path, file_name, api_response, sub_folder)` | Directory and base name for saving metadata/previews. | `get_image_path` | `(save_path, name)` |
| `list_files(folders)` | Recursively collects model files from folders. | `os.walk` | `list[str]` |
| `_detect_content_type_from_path(file_path)` | Infers content type by matching path against known folders. | `_api.contenttype_folder` | `str` |
| `_build_local_fallback_browser_item(file_path)` | Synthetic CivitAI-style item dict for local file with no API match. | `_detect_content_type_from_path`, `_api.safe_json_load`, `gen_sha256` | `dict` |
| `file_scan(folders, tag_finish, ver_finish, installed_finish, preview_finish, organize_finish, overwrite_toggle, tile_count, gen_hash, create_html, progress)` | Central multi-purpose scanner (tags, previews, version check, installed models, organization). | `list_files`, `get_models`, `version_match`, `collect_update_items`, `save_model_info`, `save_preview`, `analyze_organization_plan`, `generate_organization_preview_html`, `save_organization_backup`, `execute_organization`, `_api.request_civit_api`, `_dl.random_number`, `_build_local_fallback_browser_item` | `tuple[gr.update, ...]` |
| `set_globals(input_global)` | Sets module-level booleans to route `file_scan` behavior. | — | `None` |
| `save_tag_start(tag_start)` / `save_preview_start(preview_start)` / `ver_search_start(ver_start)` / `installed_models_start(installed_start)` / `organize_start(organize_start)` | Sets scan state and returns UI-disabled tuple. | `set_globals`, `_dl.random_number`, `start_returns` | `tuple` |
| `finish_returns()` | Standard UI-re-enabled tuple. | — | `tuple[gr.update, ...]` |
| `start_returns(number)` | Standard UI-disabled tuple with placeholder. | — | `tuple[gr.update, ...]` |
| `scan_finish()` | Resets scan globals and re-enables UI. | `set_globals` | `tuple[gr.update, ...]` |
| `cancel_scan()` | Sets cancel flag and waits for scan to finish. | `gl.cancel_status`, `gl.scan_files` | `None` |

### Auto-Organization System

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `get_model_categories()` | Returns folder-name → base-model detection patterns mapping. | `opts.civitai_neo_model_categories`, `json.loads` | `dict` |
| `_debug_log(message)` | Prints debug message if `civitai_neo_debug_organize` is enabled. | `debug_print` | `None` |
| `normalize_base_model(base_model)` | Normalizes raw `baseModel` into folder-friendly category name. | `get_model_categories`, `_debug_log` | `str \| None` |
| `_fetch_api_info_by_hash(file_path, api_info_file)` | Fetches version info from CivitAI by SHA256 and saves `.api_info.json`. | `gen_sha256`, `_api.normalize_sha256`, `_api.get_civitai_domain`, `_api.get_headers`, `_api.get_proxies`, `_api.safe_json_save` | `dict \| None` |
| `_extract_base_model_from_api_data(data, file_path)` | Extracts `baseModel` from various API response shapes. | `gen_sha256`, `_debug_log` | `str` |
| `get_model_info_for_organization(file_path)` | Returns base model type and name (prefers `.api_info.json`, then API, then `.json`). | `_extract_base_model_from_api_data`, `_fetch_api_info_by_hash`, `_api.safe_json_load`, `_debug_log` | `(base_model_type \| None, model_name)` |
| `analyze_organization_plan(folders, progress)` | Scans folders and builds move plan grouped by normalized base model. | `list_files`, `get_model_info_for_organization`, `normalize_base_model`, `_api.contenttype_folder` | `dict` |
| `generate_organization_preview_html(organization_plan)` | HTML preview of organization plan with stats and file lists. | `format_size` | `str` |
| `save_organization_backup(organization_plan)` | Saves plan to `civitai_organization_backups.json` (keeps last 5). | `json.dump`, `gl.organization_backup_file` | `str \| None` |
| `get_last_organization_backup()` | Retrieves most recent backup entry. | `json.load`, `gl.organization_backup_file` | `dict \| None` |
| `execute_rollback(progress)` | Rolls back last organization by moving files to original locations. | `get_last_organization_backup`, `shutil.move`, `os.rmdir`, `debug_print` | `dict` |
| `_move_associated_files(source_path, target_path)` | Moves all sidecars alongside a model file. | `shutil.move`, `debug_print` | `None` |
| `_make_progress_bar_html(done, total, label)` | Inline HTML progress bar string. | — | `str` |
| `execute_organization(organization_plan, progress)` | Executes planned moves, creating dirs and moving files + sidecars. | `_move_associated_files`, `shutil.move`, `os.makedirs` | `dict` |
| `validate_organization(folders, progress)` | Read-only validation scan reporting misplaced models. | `analyze_organization_plan` | `generator` yielding `tuple` |
| `fix_misplaced_files(plan_json, progress)` | Executes organization plan from `validate_organization`. | `save_organization_backup`, `shutil.move`, `_move_associated_files`, `_make_progress_bar_html` | `generator` yielding `tuple` |
| `rollback_organization(progress)` | Public entry point for undoing last organization. | `get_last_organization_backup`, `execute_rollback`, `format_size` | `gr.update` |

### Dashboard & Statistics

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `generate_dashboard_statistics(selected_types, hide_empty_categories, detect_orphans, progress)` | Scans model folders and generates HTML dashboard (charts, top files, orphans, update summary). | `_api.contenttype_folder`, `_api.safe_json_load`, `_format_size`, `os.walk`, `os.path.getsize` | `gr.update` |
| `export_dashboard_csv()` | Exports last dashboard scan as CSV string. | `csv.writer`, `last_dashboard_data` | `gr.update` |
| `export_dashboard_json()` | Exports last dashboard scan as JSON string. | `json.dumps`, `last_dashboard_data` | `gr.update` |

### Update Mode UI

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `_render_update_mode_banner(count)` | HTML banner for Update Mode with mode switcher and retention label. | — | `str` |
| `enter_update_mode()` | Activates Update Mode and renders banner if outdated models exist. | `_render_update_mode_banner` | `gr.update` |
| `exit_update_mode(content_type, sort_type, period_type, use_search_term, search_term, tile_count, base_filter, nsfw, exact_search)` | Deactivates Update Mode, clears banner, resets browser UI. | — | `tuple[gr.update, ...]` |

### Local Browser & Navigation

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `_resolve_browser_local_folders(content_type)` | Maps content types to filesystem folders for local browsing. | `_api.contenttype_folder` | `list[str]` |
| `prepare_local_browser_url_list(content_type, tile_count, use_search_term, search_term)` | Resolves local model IDs and builds `gl.url_list` pages. | `_resolve_browser_local_folders`, `list_files`, `get_models`, `_api.get_civitai_domain` | `bool` |
| `load_to_browser(content_type, sort_type, period_type, use_search_term, search_term, tile_count, base_filter, nsfw, exact_search)` | Loads initial browser page and resets scan globals. | `_api.initial_model_page` | `tuple[gr.update, ...]` |

---

## `scripts/civitai_gui.py` — Gradio UI Definition

### Settings Persistence

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `saveSettings(ust, ct, pt, st, bf, cj, ol, hi, olf, sn, es, ss, ts)` | Persists browser filter settings to Forge UI config file. | `_api.safe_json_load`, `_api.safe_json_save` | `None` |

### UI Helpers

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `all_visible(html_check)` | Determines if "Select All" button should be visible based on checkbox count. | `gr.update` | `gr.update` |
| `HTMLChange(input)` | Pass-through helper returning `gr.update` for HTML component. | `gr.update` | `gr.update` |
| `show_multi_buttons(model_list, type_list, version_value)` | Computes visibility/interactivity states for download/delete/save-info/multi-download buttons. | `_api.contenttype_folder`, `os.walk`, `json.loads`, `gr.update` | `tuple[gr.update, ...]` (6 items) |

### Browser / API Helpers

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `txt2img_output(image_url)` | Strips `url:` prefix, fetches gen info, prepends random number for txt2img payload. | `_api.fetch_and_process_image`, `_dl.random_number`, `gr.update` | `gr.update` |
| `get_base_models()` | Fetches base model list from CivitAI API, falls back to hard-coded defaults. | `_api.get_civitai_domain`, `_api.request_civit_api`, `json.loads` | `list[str]` |

### Main UI Builder

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `on_ui_tabs()` | Builds entire Gradio UI (Browser, Update Models, Download Queue, Local Models, Dashboard tabs). Wires all event callbacks. | `_file.get_content_choices`, `_api.get_base_models`, `_api.initial_model_page`, `_dl.download_start`, `_file.delete_model`, `_file.generate_dashboard_statistics`, and many more | `(gr.Blocks, tab_name, elem_id)` |

#### Nested helpers inside `on_ui_tabs`

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `format_custom_subfolders()` | Reads custom subfolder JSON and formats as delimited string. | `_api.safe_json_load`, `gl.subfolder_json` | `str` |
| `ToggleDate(toggle_date)` | Sets `gl.sortNewest` flag. | `gl.sortNewest` | `None` |
| `select_subfolder(sub_folder)` | Updates install path textbox when subfolder selected. | `gl.main_folder`, `gr.update` | `gr.update` |
| `get_creator_for_model(mid)` | Looks up creator username for model ID from API cache. | `gl.json_data`, `gr.update` | `gr.update` |
| `update_models_dropdown(input, base_filter)` | Populates model/version dropdowns and preview panel on card click. | `_api.extract_model_info`, `_api.update_model_versions`, `_api.update_model_info`, `re.sub`, `gr.update` | `tuple[gr.update, ...]` (~18 items) |
| `save_images_wrapper(preview_html, model_filename, install_path, sub_folder, model_id)` | Wraps `_file.save_images` with fresh preview HTML generation. | `_api.update_model_versions`, `_api.update_model_info`, `_api.request_civit_api`, `_file.save_images`, `gl.json_data`, `debug_print` | `None` |
| `handle_dashboard_selection(selected)` | Handles "All" checkbox logic in Dashboard content-type selector. | `gr.update` | `gr.update` |

### File & Folder Helpers

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `subfolder_list(folder, desc)` | Resolves on-disk folder and returns subfolders. | `_api.contenttype_folder`, `_file.getSubfolders` | `list[str] \| None` |
| `make_lambda(folder, desc)` | Factory returning lambda for dynamic Gradio Dropdown choices. | `subfolder_list` | `lambda` → `dict` |

### Settings Registration

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `on_ui_settings()` | Registers all CivitAI Browser Neo options into Forge settings system. | `shared.opts.add_option`, `shared.OptionInfo`, `categories.register_category`, `gr.Dropdown`, `gr.Slider`, `gr.Radio`, `gr.Textbox`, `make_lambda` | `None` |

---

## `scripts/civitai_global.py` — Global State

### Classes

| Class | Description |
|-------|-------------|
| `Colors` | ANSI escape codes for colored terminal output. |

### Functions

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `init()` | Initializes global state variables and creates runtime config files (`config_states/`, `civitai_subfolders.json`, `civitai_organization_backups.json`). Suppresses `InsecureRequestWarning`. | `warnings.simplefilter`, `os.path.exists`, `os.mkdir`, `json.dump`, `datetime.now` | `None` |
| `print(print_message)` | Prefixes every message with `[CivitAI Browser Neo]`. | `_print`, `Colors.BLUE`, `Colors.RESET` | `None` |
| `debug_print(print_message)` | Prints `[DEBUG]` message only when `do_debug_print` is enabled. | `_print`, `do_debug_print`, `Colors.MAGENTA` | `None` |

### Important Module-Level Variables

| Variable | Purpose |
|----------|---------|
| `do_debug_print` | Controls whether `debug_print()` emits output (from `opts.civitai_debug_prints`). |
| `download_queue` | In-memory download queue items. |
| `cancel_status` | Cancellation signal for active downloads. |
| `json_data` | Raw JSON from last CivitAI API browse/search call. |
| `json_info` | JSON metadata for currently selected model version. |
| `main_folder` | Resolved base folder path where models are installed. |
| `download_fail` | Boolean indicating last download attempt failed. |
| `sortNewest` | Toggle for "sort by newest" ordering. |
| `isDownloading` | Gate indicating download in progress. |
| `scan_files` | Flag set when user triggers a file scan. |
| `update_mode` | Boolean indicating UI is in batch-update mode. |
| `update_items` | List of model items queued for batch update. |
| `url_list` | Dictionary mapping model IDs/filenames to download URLs. |
| `subfolder_json` | Path to `config_states/civitai_subfolders.json`. |
| `organization_backup_file` | Path to `config_states/civitai_organization_backups.json`. |
| `local_browser_fallback_items` | Cached local model items when CivitAI API is unreachable. |

---

## `scripts/download_log.py` — Queue Persistence

### Internal Helpers

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `_get_log_path()` | Resolves path to `config_states/neo_download_queue.jsonl`, creates dir if needed. | `os.path.join`, `os.makedirs` | `str` |
| `_read_all()` | Reads every valid JSON line from log, silently skipping corrupt lines. | `_get_log_path`, `json.loads`, `os.path.exists` | `list[dict]` |
| `_write_all(entries)` | Overwrites log file with entries as JSON lines. | `_get_log_path`, `json.dumps` | `None` |
| `_now()` | Current UTC timestamp in ISO 8601 format. | `datetime.utcnow` | `str` |

### Status Logging

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `log_queued(item)` | Appends new download item with status `queued`, skipping duplicate URLs. | `_read_all`, `_write_all`, `_now` | `None` |
| `_update_status(dl_id, status)` | Finds first non-terminal entry matching `dl_id` and updates status. | `_read_all`, `_write_all`, `_now` | `None` |
| `log_downloading(dl_id)` | Marks entry as `downloading`. | `_update_status` | `None` |
| `log_completed(dl_id)` | Marks entry as `completed`. | `_update_status` | `None` |
| `log_cancelled(dl_id)` | Marks entry as `cancelled`. | `_update_status` | `None` |
| `log_failed(dl_id)` | Marks entry as `failed`. | `_update_status` | `None` |
| `log_all_cancelled()` | Bulk-cancels all `queued` or `downloading` entries. | `_read_all`, `_write_all`, `_now` | `None` |

### Query & Cleanup

| Function | Description | Dependencies | Returns |
|----------|-------------|--------------|---------|
| `get_interrupted()` | Returns entries to offer for restore (`queued`, `downloading`, `failed`). | `_read_all` | `list[dict]` |
| `dismiss_interrupted()` | Marks all interrupted entries as `dismissed`. | `_read_all`, `_write_all`, `_now` | `None` |
| `purge_old_entries(days)` | Removes terminal entries older than N days while preserving active ones. | `_read_all`, `_write_all`, `datetime.utcnow`, `datetime.fromisoformat`, `timedelta` | `None` |

---

## `javascript/civitai-html.js` — Frontend Logic

### Card Selection & Interaction

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `select_model(model_name, event, bool, content_type, sendToBrowser)` | Writes model name to hidden Gradio textarea, triggering Python callback. | `updateInput()` | `#model_select textarea`, `#model_sent textarea`, `#send_to_browser textarea`, `#type_sent textarea` |
| `clickFirstFigureInColumn()` | Auto-clicks first `<figure>` after 500 ms delay. | — | `.column.civmodellist figure` |

### Card Size & Styling

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `updateCardSize(width, height)` | Updates CSS rules for card dimensions, font scaling, badge visibility. | `addOrUpdateRule()` | `document.styleSheets[0]` |
| `addOrUpdateRule(styleSheet, selector, newRules)` | Inserts or replaces CSS rule. | — | `styleSheet.cssRules` |

### Card Status Updates (Installed / Outdated)

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `applyPendingCardUpdates()` | Drains `pendingCardUpdates` Set and calls `updateCard()`. | `updateCard()` | — |
| `initCardUpdateObserver()` (IIFE) | MutationObserver watching for `.civmodellist`/`.civmodelcards` to flush pending updates. | `applyPendingCardUpdates()` | `document.documentElement` |
| `initCardUpdatePoller()` (IIFE) | 600 ms polling fallback for Gradio 4 innerHTML injections. | `applyPendingCardUpdates()`, `pressRefresh()` | `.civmodellist`, `.civmodelcards` |
| `updateCard(modelNameWithSuffix, allowRefresh)` | Adds/removes `civmodelcardinstalled`/`civmodelcardoutdated` classes; queues if no containers visible. | `updateCard()` (retry), `pressRefresh()`, `hideInstalled()` | `.civmodellist`, `.civmodelcards`, `.civmodelcard` |

### Video Hover-to-Play

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `attachVideoHoverPlay(card)` | Attaches `mouseenter`/`mouseleave` to card `<video>` for hover play. | — | `.civmodelcard video.video-bg` |
| `initVideoHoverObserver()` (IIFE) | MutationObserver auto-attaching hover listeners to new `.civmodelcard` elements. | `attachVideoHoverPlay()` | `.civmodellist`, `document.body` |

### Keyboard & Refresh

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `keydownHandler(e)` | Global keydown: Alt/Ctrl/Meta + Enter triggers refresh inside CivitAI tab. | `get_uiCurrentTabContent()` (Forge global) | `#tab_civitai_interface`, `#refreshBtn`, `#refreshBtnL` |
| `pressRefresh()` | Triggers page refresh by writing random value to `#page_slider_trigger`. | `updateInput()` | `#pageSlider input`, `#page_slider_trigger textarea` |

### Responsive Layout & Filters

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `adjustFilterBoxAndButtons()` | Adjusts filter accordion margins, page buttons (desktop vs mobile), model-block wrapping. | — | `#filterBox`, `#filterBoxL`, `#pageBtn1`, `#pageBtn2`, `#pageBox`, `#pageBoxMobile`, `.model-block` |
| `updateSVGIcons()` | Injects style block for filter/search icons and box-shadow colors (dark mode). | — | `#filterBox`, `#filterBoxL`, `document.body` |
| `setupClickOutsideListener()` | Closes filter dropdown when clicking outside. | — | `#filterBox`, `#filterBoxL` |

### Tooltips & Links

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `createTooltip(element, hover_element, insertText)` | Hidden tooltip div showing on mouseover. | — | dynamically created `.browser_tooltip` |
| `createLink(infoElement)` | Converts static settings text into real hyperlink. | — | `#setting_custom_api_key .info` |

### Settings Accordion

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `createAccordion(containerDiv, subfolders, name, id_name)` | Builds collapsible accordion section in settings panel. | — | dynamically created nodes appended to `containerDiv` |

### txt2img / img2img Extra-Card Buttons

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `createCivitAICardButtons()` | Injects "goto CivitAI browser" SVG button into `.card` elements in extra tabs. | `createSVGIcon()`, `modelInfoPopUp()` | `.card`, `.button-row`, `.actions .name` |
| `createSVGIcon(fontSize)` | Returns SVG element with CivitAI logo paths. | — | creates SVG node |
| `addOnClickToButtons()` | Attaches click listeners to refresh buttons to re-create card buttons. | `createCivitAICardButtons()` | `#txt2img_extra_tabs`, `#img2img_extra_tabs` |

### Model Info Overlay

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `modelInfoPopUp(modelName, content_type, no_message)` | Sends model to browser tab or opens full-screen overlay. | `select_model()`, `createCivitaiOverlay()`, `hideCivitaiOverlay()`, `sendClick()` | `#setting_civitai_send_to_browser input`, `.tab-nav` buttons, `#tab_civitai_interface` |
| `createCivitaiOverlay(noMessage)` | Creates and animates in full-screen overlay; registers click/ESC handlers. | `handleOverlayClick()`, `handleOverlayKeyPress()` | dynamically created `.civitai-overlay`, `.civitai-overlay-inner`, `document.body` |
| `handleOverlayClick(event)` | Closes overlay on backdrop click. | `hideCivitaiOverlay()` | `.civitai-overlay` |
| `handleOverlayKeyPress(event)` | Closes overlay on Escape (unless image viewer active). | `hideCivitaiOverlay()` | `currentViewerOverlay` |
| `hideCivitaiOverlay()` | Fades out and removes overlay, restores body scroll. | — | `.civitai-overlay`, `document.body` |
| `inputHTMLPreviewContent(html_input)` | Parses Gradio HTML string, injects into overlay, initializes description toggle. | `initDescriptionToggle()` | `.civitai-overlay-inner`, `.civitai-overlay-text` |

### Send to txt2img

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `metaToTxt2Img(event, type, element)` | Extracts generation metadata from clicked `<dd>` and assembles prompt string. | `genInfo_to_txt2img()`, `hideCivitaiOverlay()`, `sendClick()` | `#txt2img_prompt textarea`, `#txt2img_neg_prompt textarea`, `#txt2img_cfg_scale input`, `#txt2img_extra_tabs button` |
| `genInfo_to_txt2img(genInfo, do_slice)` | Writes generation info into txt2img prompt and clicks Paste button. | `updateInput()` | `#txt2img_prompt textarea`, `#paste` |
| `sendTagsToPrompt(tags)` | Copies trigger words into txt2img prompt (appending if Shift held) and clicks Generate. | `genInfo_to_txt2img()`, `sendClick()`, `hideCivitaiOverlay()` | `#txt2img_prompt textarea`, `#txt2img_neg_prompt textarea`, `#txt2img_cfg_scale input`, `#txt2img_extra_tabs button` |

### Multi-Select & Queue Management

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `multi_model_select(modelName, modelType, isChecked)` | Maintains `selectedModels`/`selectedTypes` arrays, syncs to hidden textareas. | `updateInput()`, `syncUpdateBtn()` | `#selected_model_list textarea`, `#selected_type_list textarea` |
| `sendClick(location)` | Dispatches synthetic `MouseEvent('click')`. | — | `location` |
| `cancelCurrentDl()` | Sets `currentDlCancelled` flag for progress worker. | — | — |
| `cancelAllDl()` | Sets `allDlCancelled` flag for progress worker. | — | — |
| `setSortable()` | Initializes Sortable.js on `#queue_list` for drag-to-reorder. | `updateInput()` | `#queue_list`, `#civitai_dl_list.prose`, `#queue_html_input textarea`, `#arrange_dl_id textarea` |
| `cancelQueueDl()` | Triggers Python cancel callback. | `updateInput()` | `#html_cancel_input textarea` |
| `removeDlItem(dl_id, element)` | Removes download item from DOM, notifies Python, syncs queue HTML. | `updateInput()` | `#queue_html_input textarea`, `#remove_dl_id textarea`, `#civitai_dl_list.prose` |
| `selectAllModels()` | Clicks every `.model-checkbox` to select all. | `sendClick()` | `.model-checkbox` |
| `deselectAllModels()` | Clicks every checked `.model-checkbox` to deselect all after 1 s. | `sendClick()` | `.model-checkbox` |

### Download Progress

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `_createWorkerInterval(callback, ms)` | Web-Worker-backed interval for background-tab progress polling. | — | creates Blob URL Worker |
| `setDownloadProgressBar()` | Moves active download to non-queue list, mirrors Gradio progress bar into custom queue UI. | `_createWorkerInterval()`, `updateInput()` | `#DownloadProgress`, `#civitai_dl_list`, `.civitai_dl_item`, `.progress-bar`, `.progress-level-inner`, `#queue_html_input textarea` |

### Queue Restore Banner

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `initRestoreBanner(json)` | Builds and injects restore/dismiss banner HTML for orphaned queued downloads. | — | `#restore_banner` |
| `triggerRestoreQueue()` | Writes timestamp to `#restore_action_trigger` to tell Python to restore queue. | `updateInput()` | `#restore_action_trigger textarea`, `#restore_banner` |
| `triggerDismissRestore()` | Writes `'1'` to `#dismiss_restore_trigger` to tell Python to dismiss banner. | `updateInput()` | `#dismiss_restore_trigger textarea`, `#restore_banner` |

### Image URL & Blob Download

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `sendImgUrl(image_url)` | Sends sample image URL to Python for gen-info extraction, switches to txt2img. | `updateInput()`, `hideCivitaiOverlay()`, `sendClick()` | `#civitai_text2img_input textarea`, `#txt2img_extra_tabs button` |
| `downloadBlobFile(content, filename, mimeType)` | Triggers browser download from Blob, resets hidden textarea. | — | `#export_csv_output textarea`, `#export_json_output textarea` |

### Trigger Words

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `copyTriggerWord(text, btn)` | Copies text to clipboard with fallback, shows checkmark on button. | — | `btn` |

### Hide Installed / Banned Creators

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `hideInstalled(toggleValue)` | Toggles visibility of `.civmodelcardinstalled` cards. | — | `.column.civmodellist > .civmodelcardinstalled` |
| `initBannedCreators(listStr, checked)` | Parses banned-creator list and applies filter. | `hideBannedCreators()` | — |
| `refreshBannedCreators(listStr, checked)` | Re-applies banned-creator filter after ban/unban. | `hideBannedCreators()` | — |
| `hideBannedCreators(checked)` | Shows/hides cards matching banned `data-creator`. | — | `.civmodelcard[data-creator]` |

### Description Toggle

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `toggleDescription(prefix)` | Expands/collapses description area between 400 px and full height. | — | `#prefixdescription-content`, `#prefixdescription-overlay`, `#prefixdescription-toggle-btn` |
| `initDescriptionToggle(prefix)` | Measures height and shows/hides toggle button based on overflow. | — | same as above |

### Custom Subfolders

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `submitNewSubfolder(subfolderId, subfolderValue)` | Writes subfolder add/update command to hidden textarea. | `updateInput()` | `#create_subfolder textarea` |
| `deleteSubfolder(subfolderId)` | Writes subfolder delete command to hidden textarea. | `updateInput()` | `#create_subfolder textarea` |
| `createCustomSubfolder(subfolderDiv, subfolderId, subfolderValue)` | Builds subfolder entry row with Save/Delete buttons. | `submitNewSubfolder()`, `deleteSubfolder()` | dynamically created `.CivitDefaultSubfolder` row |
| `insertExistingSubfolders(input)` | Stub referencing undefined vars; not actively used. | `createCustomSubfolder()` | `civitai-custom-subfolder-div` |
| `createSubfolderButton()` | Renders existing subfolder rows and Create/Guide buttons. | `createCustomSubfolder()`, `modelInfoPopUp()`, `insertGuideMessage()` | `#create-sub-accordion`, `#custom_subfolders_list textarea` |
| `insertGuideMessage(html_input)` | Injects guide HTML into open overlay. | — | `.civitai-overlay-inner` |

### Page Load & Settings

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `onPageLoad()` | Master init: SVG icons, accordions, subfolder UI, tooltips, hash hints, extra-tab buttons. | `updateSVGIcons()`, `createAccordion()`, `createSubfolderButton()`, `createTooltip()`, `addOnClickToButtons()`, `createCivitAICardButtons()`, `adjustFilterBoxAndButtons()`, `setupClickOutsideListener()` | `#settings_civitai_browser_plus`, `#settings_civitai_browser_download`, `#settings_civitai_browser`, `#toggle4`, `#toggle4L`, `#skip_hash_toggle`, `#do_html_gen` |
| `checkSettingsLoad()` | Waits for API-key info element, converts it to hyperlink. | `createLink()` | `#setting_custom_api_key .info` |

### Image Viewer

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `createViewerOverlay()` | Builds reusable image/video viewer overlay DOM. | — | dynamically created `#image-viewer-overlay` |
| `openImageViewer(mediaUrl, mediaType)` | Shows overlay with image or video source. | `createViewerOverlay()`, `setupViewerEventListeners()` | `#image-viewer-overlay`, `#viewer-image`, `#viewer-video` |
| `setupViewerEventListeners(overlay)` | Attaches Escape, backdrop-click, media-click listeners. | `closeImageViewer()` | `overlay`, `#viewer-image`, `#viewer-video`, `document` |
| `cleanupViewerEventListeners()` | Removes all tracked viewer listeners. | — | `viewerEventListeners` entries |
| `closeImageViewer()` | Hides viewer with CSS animation, restores scroll, cleans up. | `cleanupViewerEventListeners()` | `#image-viewer-overlay`, `document.body`, `.civitai-overlay` |
| `handlePreviewMediaClick(e)` | Click handler for `.preview-media`; initializes viewer lazily. | `initializeImageViewer()`, `openImageViewer()` | `e.target` |
| `initializeImageViewer()` | Attaches click handlers and MutationObserver for new `.preview-media`. | `handlePreviewMediaClick()` | `.preview-media`, `document.body` |

### Delete Installed Model

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `deleteInstalledModel(event, modelString, sha256, installedCount)` | Confirms and triggers deletion via SHA256 input; updates card to `.None`. | `select_model()`, `updateInput()`, `updateCard()` | `#delete_finish textarea`, `#sha256 textarea`, `#delete_trigger_btn`, `.update-mode-card` |

### Update Mode

| Function | Description | Calls | DOM |
|----------|-------------|-------|-----|
| `updateAllModels()` | Triggers Python to enqueue all outdated models. | `updateInput()` | `#update_all_trigger textarea` |
| `updateOrSelectedModels()` | Enqueues checked models, falls back to all if none checked. | `updateAllModels()`, `updateInput()` | `#update_selected_trigger textarea`, `.model-checkbox:checked` |
| `syncUpdateBtn()` | Updates Update All/Selected button label to reflect checkbox count. | — | `#civupdate-update-btn`, `.model-checkbox` |
| `updateSingleModel(modelId, family)` | Enqueues single model update; dims card visually. | `updateInput()` | `#update_single_trigger textarea`, `.update-mode-card` |
| `exitUpdateMode()` | Clears update-mode banner, tells Python to reset state. | `updateInput()` | `#exit_update_mode_trigger textarea`, `#update_mode_banner` |

---

## Data Flow Summary

```
User (Gradio UI)
    ↓
civitai_gui.py  →  Gradio callbacks (event wiring)
    ↓
civitai_api.py  ←→  CivitAI REST API
    ↓
civitai_download.py  ←→  Aria2 RPC
    ↓
civitai_file_manage.py  →  Filesystem
    ↓
lib/models/*.json          Metadata cache
config_states/*.jsonl      Queue persistence
```

### Key Call Chains

1. **Browse → Card Click → Preview**
   `initial_model_page()` → `model_list_html()` → `update_models_dropdown()` → `update_model_info()`

2. **Download Flow**
   `download_start()` / `selected_to_queue()` → `create_model_item()` → `download_create_thread()` → `download_file()` / `download_file_old()` → `info_to_json()` + `_file.save_model_info()` + `_file.save_preview()`

3. **Update Detection → Update Mode**
   `file_scan()` → `version_match()` → `collect_update_items()` → `enter_update_mode()` → `update_mode_page_html()` → `update_all_models()` / `update_selected_models()`

4. **Organization**
   `analyze_organization_plan()` → `generate_organization_preview_html()` → `save_organization_backup()` → `execute_organization()` / `execute_rollback()`

5. **Queue Persistence**
   `create_model_item()` → `_dl_log.log_queued()` → `download_create_thread()` → `_dl_log.log_downloading()` → `_dl_log.log_completed()` / `_dl_log.log_cancelled()` / `_dl_log.log_failed()`

6. **Frontend ↔ Backend**
   `select_model()` (JS) → hidden textarea → Python callback → `update_models_dropdown()` → `update_model_info()` → HTML → `inputHTMLPreviewContent()` (JS) → overlay

---

## Change Log

| Date | Version | Notes |
|------|---------|-------|
| 2026-05-09 | v0.9.0 | Initial `FUNCTION_MAP.md` covering all major functions across 7 modules. |
