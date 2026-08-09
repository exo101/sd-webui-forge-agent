import fnmatch
import json
import os
import re
import gradio as gr

# === WebUI imports ===
from modules import script_callbacks, shared
from modules.shared import opts, cmd_opts
from modules.paths import extensions_dir
from modules.options import categories

# === Extension imports ===
import scripts.civitai_download as _download
import scripts.civitai_file_manage as _file
import scripts.civitai_global as gl
import scripts.civitai_api as _api
from scripts.civitai_global import print, debug_print


gl.init()


# Path to extension-local defaults (tile size/count and future browser settings)
_BROWSER_DEFAULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'config_states',
    'civitai_browser_defaults.json'
)


def _load_browser_defaults():
    """Load browser defaults from extension-local JSON."""
    try:
        with open(_BROWSER_DEFAULTS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_browser_defaults(data):
    """Save browser defaults to extension-local JSON."""
    os.makedirs(os.path.dirname(_BROWSER_DEFAULTS_PATH), exist_ok=True)
    with open(_BROWSER_DEFAULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def saveSettings(ust, ct, pt, st, bf, cj, ol, hi, sn, es, ss, ts):
    config = cmd_opts.ui_config_file

    # Create a dictionary to map the settings to their respective variables
    # NOTE: prefix must match the elem_id returned in on_ui_tabs() → 'civitai_interface_neo'
    settings_map = {
        'civitai_interface_neo/Search type:/value': ust,
        'civitai_interface_neo/Content type:/value': ct,
        'civitai_interface_neo/Time period:/value': pt,
        'civitai_interface_neo/Sort by:/value': st,
        'civitai_interface_neo/Base model:/value': bf,
        'civitai_interface_neo/Save info after download/value': cj,
        'civitai_interface_neo/Divide cards by date/value': False,  # This is a toggle, so its state does not matter here
        'civitai_interface_neo/Liked models only/value': ol,
        'civitai_interface_neo/Hide installed models/value': hi,
        'civitai_interface_neo/NSFW content/value': sn,
        'civitai_interface_neo/Exact search/value': es,
        'civitai_interface_neo/Tile size:/value': ss,
        'civitai_interface_neo/Tile count:/value': ts
    }

    # Load the current contents of the config file into a dictionary
    data = _api.safe_json_load(config)
    if not data:
        print(f"Cannot save settings, failed to open '{config}'")
        print('Please try to manually repair the file or remove it to reset settings.')
        return

    # Remove any keys containing the text `civitai_interface`
    keys_to_remove = [key for key in data if 'civitai_interface' in key]
    for key in keys_to_remove:
        del data[key]

    # Update the dictionary with the new settings
    data.update(settings_map)

    # Save the modified content back to the file
    if _api.safe_json_save(config, data):
        print(f"Updated settings to: {config}")

    # Persist all browser filter defaults to extension-local file
    print(f"[CivitAI Browser] Saving filter defaults: tile_size={ss}, tile_count={ts}, search_type={ust}, content_type={ct}, base_model={bf}, period={pt}, sort={st}, liked={ol}, hide={hi}, nsfw={sn}, exact={es}, save_json={cj}")
    _save_browser_defaults({
        'search_type': ust,
        'content_type': ct,
        'time_period': pt,
        'sort_by': st,
        'base_model': bf,
        'save_info_after_download': cj,
        'liked_models_only': ol,
        'hide_installed_models': hi,
        'nsfw_content': sn,
        'exact_search': es,
        'tile_size': ss,
        'tile_count': ts,
    })

# === ANXETY EDITs ===
def all_visible(html_check):
    # Count the number of model-checkbox occurrences in the HTML
    checkbox_count = html_check.count('model-checkbox')
    # Show the button only if there are 2 or more checkboxes (more than 1 model to select)
    return gr.update(visible=checkbox_count >= 2)

def HTMLChange(input):
    return gr.update(value=input)

def show_multi_buttons(model_list, type_list, version_value):
    model_list = json.loads(model_list)
    type_list = json.loads(type_list)
    otherButtons = True
    multi_file_subfolder = False
    default_subfolder = 'Only available if the selected files are of the same model type'
    sub_folders = ['None']
    BtnDwn = version_value and not version_value.endswith('[Installed]') and not model_list
    BtnDel = version_value.endswith('[Installed]')

    dot_subfolders = getattr(opts, 'dot_subfolders', True)

    multi = bool(model_list) and not len(gl.download_queue) > 0
    if model_list:
        otherButtons = False
    if type_list and all(x == type_list[0] for x in type_list):
        multi_file_subfolder = True
        model_folder = os.path.join(_api.contenttype_folder(type_list[0]))
        default_subfolder = 'None'
        try:
            for root, dirs, _ in os.walk(model_folder, followlinks=True):
                if dot_subfolders:
                    dirs = [d for d in dirs if not d.startswith('.')]
                    dirs = [d for d in dirs if not any(part.startswith('.') for part in os.path.join(root, d).split(os.sep))]
                for d in dirs:
                    sub_folder = os.path.relpath(os.path.join(root, d), model_folder)
                    if sub_folder:
                        sub_folders.append(f'{os.sep}{sub_folder}')
            sub_folders.remove('None')
            sub_folders = sorted(sub_folders, key=lambda x: (x.lower(), x))
            sub_folders.insert(0, 'None')

            list = set()
            sub_folders = [x for x in sub_folders if not (x in list or list.add(x))]
        except:
            sub_folders = ['None']

    return (gr.update(visible=multi, interactive=multi), # Download Multi Button
            gr.update(visible=BtnDwn if multi else True if not version_value.endswith('[Installed]') else False), # Download Button
            gr.update(visible=BtnDel if not model_list else False), # Delete Button
            gr.update(visible=otherButtons), # Save model info Button
            gr.update(visible=otherButtons), # Save images Button
            gr.update(visible=multi, interactive=multi_file_subfolder, choices=sub_folders, value=default_subfolder) # Selected type sub folder
            )

def txt2img_output(image_url):
    clean_url = image_url[4:]
    geninfo = _api.fetch_and_process_image(clean_url)
    if geninfo:
        nr = _download.random_number()
        geninfo = nr + geninfo
        return gr.update(value=geninfo)

## === ANXETY EDITs ===
def get_base_models():
    api_url = f'https://{_api.get_civitai_domain()}/api/v1/models?baseModels=GetModels'
    json_return = _api.request_civit_api(api_url, True)
    # The data below is taken from the API response (last synced: 2026-02-18)
    default_options = [
        'Anima',
        'AuraFlow',
        'Chroma',
        'CogVideoX',
        'Flux.1 S',
        'Flux.1 D',
        'Flux.1 Krea',
        'Flux.1 Kontext',
        'Flux.2 D',
        'Flux.2 Klein 9B',
        'Flux.2 Klein 9B-base',
        'Flux.2 Klein 4B',
        'Flux.2 Klein 4B-base',
        'HiDream',
        'Hunyuan 1',
        'Hunyuan Video',
        'Illustrious',
        'Imagen4',
        'Kling',
        'Kolors',
        'LTXV',
        'LTXV2',
        'Lumina',
        'Mochi',
        'Nano Banana',
        'NoobAI',
        'ODOR',
        'OpenAI',
        'Other',
        'PixArt a',
        'PixArt E',
        'Playground v2',
        'Pony',
        'Pony V7',
        'Qwen',
        'Stable Cascade',
        'SD 1.4',
        'SD 1.5',
        'SD 1.5 LCM',
        'SD 1.5 Hyper',
        'SD 2.0',
        'SD 2.0 768',
        'SD 2.1',
        'SD 2.1 768',
        'SD 2.1 Unclip',
        'SD 3',
        'SD 3.5',
        'SD 3.5 Large',
        'SD 3.5 Large Turbo',
        'SD 3.5 Medium',
        'Sora 2',
        'SDXL 0.9',
        'SDXL 1.0',
        'SDXL 1.0 LCM',
        'SDXL Lightning',
        'SDXL Hyper',
        'SDXL Turbo',
        'SDXL Distilled',
        'Seedance',
        'Seedance 1.5',
        'Seedance 2.0',
        'Seedream',
        'SVD',
        'SVD XT',
        'Veo 3',
        'Vidu Q1',
        'Wan Video',
        'Wan Video 1.3B t2v',
        'Wan Video 14B t2v',
        'Wan Video 14B i2v 480p',
        'Wan Video 14B i2v 720p',
        'Wan Video 2.2 TI2V-5B',
        'Wan Video 2.2 I2V-A14B',
        'Wan Video 2.2 T2V-A14B',
        'Wan Video 2.5 T2V',
        'Wan Video 2.5 I2V',
        'ZImageTurbo',
        'ZImageBase',
    ]

    if not isinstance(json_return, dict):
        print("[CivitAI Browser Neo] - Couldn't fetch latest baseModel options, using default.")
        return default_options

    if 'error' in json_return and 'message' in json_return['error']:
        try:
            parsed_message = json.loads(json_return['error']['message'])
            options = parsed_message[0]['errors'][0][0]['values']
            return options
        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as e:
            print(f"Basemodel fetch error extracting options: {e}")
            return default_options
    else:
        return default_options

## === ANXETY EDITs ===
def on_ui_tabs():
    page_header = getattr(opts, 'page_header', False)
    lobe_directory = None

    for root, dirs, files in os.walk(extensions_dir, followlinks=True):
        for dir_name in fnmatch.filter(dirs, '*lobe*'):
            lobe_directory = os.path.join(root, dir_name)
            break

    # Different ID's for Lobe Theme
    component_id = 'togglesL' if lobe_directory else 'toggles'
    toggle1 = 'toggle1L' if lobe_directory else 'toggle1'
    toggle2 = 'toggle2L' if lobe_directory else 'toggle2'
    toggle3 = 'toggle3L' if lobe_directory else 'toggle3'
    toggle5 = 'toggle5L' if lobe_directory else 'toggle5'
    toggle6 = 'toggle6L' if lobe_directory else 'toggle6'
    refreshbtn = 'refreshBtnL' if lobe_directory else 'refreshBtn'
    filterBox = 'filterBoxL' if lobe_directory else 'filterBox'

    if page_header:
        header = 'headerL' if lobe_directory else 'header'
    else:
        header = 'header_off'

    api_key = getattr(opts, 'custom_api_key', '')
    if api_key:
        toggle4 = 'toggle4L_api' if lobe_directory else 'toggle4_api'
        show_only_liked = True
    else:
        toggle4 = 'toggle4L' if lobe_directory else 'toggle4'
        show_only_liked = False

    sync_card_delete_enabled = bool(getattr(opts, 'civitai_neo_sync_card_delete', True))

    content_choices = _file.get_content_choices()
    scan_choices = _file.get_content_choices(scan_choices=True)
    
    # Local Models tab should only scan Checkpoint and LORA folders
    local_scan_choices = ['Checkpoint', 'LORA']
    
    # Dashboard can scan all model types
    dashboard_scan_choices = ['Checkpoint', 'LORA', 'TextualInversion', 'VAE', 'Controlnet', 
                              'Upscaler', 'MotionModule', 'AestheticGradient', 'Poses', 
                              'Detection', 'Wildcards', 'Workflows', 'Other']
    dashboard_scan_choices_with_all = ['All'] + dashboard_scan_choices

    with gr.Blocks() as civitai_interface:
        ## Browser Tab
        with gr.Tab(label='Browser', elem_id='browserTab'):
            gr.Markdown('## 🔍 Browse CivitAI Models', elem_id='browser_header')
            gr.Markdown('Search, discover, and download models directly from CivitAI.')
            restore_banner_html = gr.HTML(value='', elem_id='restore_banner')
            update_mode_banner = gr.HTML(value='', elem_id='update_mode_banner')
            
            _browser_defaults = _load_browser_defaults()
            with gr.Row(elem_id='searchRow'):
                with gr.Accordion(label='', open=False, elem_id=filterBox):
                    with gr.Row():
                        use_search_term = gr.Radio(label='Search type:', choices=['Model name', 'User name', 'Tag', 'SHA256'], value=_browser_defaults.get('search_type', 'Model name'), elem_id='searchType')
                    with gr.Row():
                        content_type = gr.Dropdown(label='Content type:', choices=content_choices, value=_browser_defaults.get('content_type', None), type='value', multiselect=True, elem_id='centerText')
                    with gr.Row():
                        base_filter = gr.Dropdown(label='Base model:', multiselect=True, choices=get_base_models(), value=_browser_defaults.get('base_model', None), type='value', elem_id='centerText')
                    with gr.Row():
                        period_type = gr.Dropdown(label='Time period:', choices=['All Time', 'Year', 'Month', 'Week', 'Day'], value=_browser_defaults.get('time_period', 'Month'), type='value', elem_id='centerText')
                        sort_type = gr.Dropdown(label='Sort by:', choices=['Newest','Oldest','Most Downloaded','Highest Rated','Most Liked','Most Buzz','Most Discussed','Most Collected','Most Images'], value=_browser_defaults.get('sort_by', 'Highest Rated'), type='value', elem_id='centerText')
                    with gr.Row(elem_id=component_id):
                        create_json = gr.Checkbox(label=f"Save info after download", value=_browser_defaults.get('save_info_after_download', True), elem_id=toggle1)
                        show_nsfw = gr.Checkbox(label='NSFW content', value=_browser_defaults.get('nsfw_content', False), elem_id=toggle2)
                        toggle_date = gr.Checkbox(label='Divide cards by date', value=False, elem_id=toggle3)
                        exact_search = gr.Checkbox(label='Exact search', value=_browser_defaults.get('exact_search', True), elem_id=toggle6)
                        only_liked = gr.Checkbox(label='Liked models only', value=_browser_defaults.get('liked_models_only', False), interactive=show_only_liked, elem_id=toggle4)
                        hide_installed = gr.Checkbox(label='Hide installed models', value=_browser_defaults.get('hide_installed_models', False), elem_id=toggle5)
                        hide_banned_creators = gr.Checkbox(label='Hide banned creators', value=False, elem_id='hideBannedCreators')
                    with gr.Row():
                        size_slider = gr.Slider(label='Tile size:', minimum=8, maximum=20, value=_browser_defaults.get('tile_size', 12), step=0.25)
                        tile_count_slider = gr.Slider(label='Tile count:', minimum=1, maximum=100, value=_browser_defaults.get('tile_count', 27), step=1)
                    with gr.Row(elem_id='save_set_box'):
                        save_settings = gr.Button(value='Save settings as default', elem_id='save_set_btn')
                search_term = gr.Textbox(label='', placeholder='Enter model name, or paste a CivitAI link', elem_id='searchBox')
                refresh = gr.Button(value='', elem_id=refreshbtn, icon='placeholder')
            with gr.Row(elem_id=header):
                with gr.Row(elem_id='pageBox'):
                    get_prev_page = gr.Button(value='Prev page', interactive=False, elem_id='pageBtn1')
                    page_slider = gr.Slider(label='Current page:', step=1, minimum=1, maximum=1, min_width=80, elem_id='pageSlider')
                    get_next_page = gr.Button(value='Next page', interactive=False, elem_id='pageBtn2')
                with gr.Row(elem_id='pageBoxMobile'):
                    pass # Row used for button placement on mobile
            with gr.Row(elem_id='select_all_models_container'):
                select_all = gr.Button(value='Select All', elem_id='select_all_models', visible=False)
            with gr.Row():
                gr.HTML(value=(
                    '<div class="card-legend">'
                    '<span class="legend-title">Legend:</span>'
                    '<span class="legend-item"><span class="legend-dot not-installed"></span>Not installed</span>'
                    '<span class="legend-separator"></span>'
                    '<span class="legend-item"><span class="legend-dot installed"></span>Installed &amp; up to date</span>'
                    '<span class="legend-separator"></span>'
                    '<span class="legend-item"><span class="legend-dot outdated"></span>Update available (same family)</span>'
                    '<span class="legend-separator"></span>'
                    '<span class="legend-item"><span class="legend-dot cross-family"></span>New family variant available</span>'
                    '<span class="legend-separator"></span>'
                    '<span class="legend-item"><span class="legend-dot early-access"></span>Early Access (paid)</span>'
                    '</div>'
                ))
            with gr.Accordion(label='\U0001f464 Creator Management', open=False, elem_id='creatorMgmtBox'):
                with gr.Row():
                    creator_name_txt = gr.Textbox(label='Creator:', interactive=False, max_lines=1, scale=3, elem_id='creator_name_display')
                    btn_fav = gr.Button(value='\u2b50 Favorite', interactive=False, scale=1, min_width=110)
                    btn_ban = gr.Button(value='\U0001f6ab Ban', interactive=False, scale=1, min_width=90)
                    btn_clear = gr.Button(value='\u21ba Reset', interactive=False, scale=1, min_width=90)
            with gr.Row():
                list_html = gr.HTML(value='<div style="font-size: 24px; text-align: center; margin: 50px;">Click the search icon to load models.<br>Use the filter icon to filter results.</div>')
            with gr.Row():
                download_progress = gr.HTML(value='<div style="min-height: 0px;"></div>', elem_id='DownloadProgress')
            with gr.Row():
                list_models = gr.Dropdown(label='Model:', choices=[], interactive=False, elem_id='quicksettings1', value=None)
                list_versions = gr.Dropdown(label='Version:', choices=[], interactive=False, elem_id='quicksettings0', value=None)
                file_list = gr.Dropdown(label='File:', choices=[], interactive=False, elem_id='file_list', value=None)
            with gr.Row():
                with gr.Column(scale=4):
                    install_path = gr.Textbox(label='Download folder:', interactive=False, max_lines=1)
                with gr.Column(scale=2):
                    sub_folder = gr.Dropdown(label='Sub folder:', choices=[], interactive=False, value=None)
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Row():
                        trained_tags = gr.Textbox(label='Trained tags (if any):', value=None, interactive=False, lines=1, scale=6)
                        send_tags_btn = gr.Button(value='➕ Add to prompt', scale=1, min_width=120, interactive=False, visible=False)
                with gr.Column(scale=2, elem_id='spanWidth'):
                    base_model = gr.Textbox(label='Base model: ', value=None, interactive=False, lines=1, elem_id='baseMdl')
                    model_filename = gr.Textbox(label='Model filename:', interactive=False, value=None)
            with gr.Row():
                save_info = gr.Button(value='Save model info', interactive=False)
                save_images = gr.Button(value='Save images', interactive=False)
                delete_model = gr.Button(value='Delete model', interactive=False, visible=False)
                download_model = gr.Button(value='Download model', interactive=False)
                subfolder_selected = gr.Dropdown(label='Sub folder for selected files:', choices=[], interactive=False, visible=False, value=None, allow_custom_value=True)
                download_selected = gr.Button(value='Download all selected', interactive=False, visible=False, elem_id='download_all_button')
            with gr.Row():
                cancel_all_model = gr.Button(value='Cancel all downloads', interactive=False, visible=False)
                cancel_model = gr.Button(value='Cancel current download', interactive=False, visible=False)
            with gr.Row():
                preview_html = gr.HTML(elem_id='civitai_preview_html')

        ## Update Tab
        with gr.Tab(label='Update Models', elem_id='updateTab'):
            gr.Markdown('## 🔄 Update & Maintain Models', elem_id='update_header')
            gr.Markdown('Keep your models up-to-date and manage metadata, previews, and tags.')
            
            with gr.Row():
                selected_tags = gr.CheckboxGroup(elem_id='selected_tags', label='Selected content types:', choices=scan_choices, value=['All'])
            
            with gr.Accordion(label='🔧 Scan Options', open=False):
                with gr.Row(elem_id='civitai_update_toggles'):
                    overwrite_toggle = gr.Checkbox(elem_id='overwrite_toggle', label='Overwrite any existing files (previews, HTMLs, tags, descriptions)', value=True, min_width=300)
                    skip_hash_toggle = gr.Checkbox(elem_id='skip_hash_toggle', label='One-Time Hash Generation for externally downloaded models', value=True, min_width=300)
                    do_html_gen = gr.Checkbox(elem_id='do_html_gen', label='Save HTML file for each model when updating info & tags', value=False, min_width=300)
            
            gr.Markdown('---')
            gr.Markdown('### 📝 Update Model Information')
            gr.Markdown('Fetch and update metadata, tags, and descriptions from CivitAI.')
            
            with gr.Row():
                save_all_tags = gr.Button(value='📝 Update model info & tags', interactive=True, visible=True, variant='primary')
                cancel_all_tags = gr.Button(value='Cancel updating model info & tags', interactive=False, visible=False)
            with gr.Row():
                tag_progress = gr.HTML(value='<div style="min-height: 0px;"></div>')

            with gr.Row():
                sync_sha256_cache = gr.Button(value='🔄 Sync checkpoint SHA256 cache', interactive=True, visible=True)
            with gr.Row():
                sync_sha256_progress = gr.HTML(value='<div style="min-height: 0px;"></div>')
            
            gr.Markdown('---')
            gr.Markdown('### 🖼️ Update Model Previews')
            gr.Markdown('Download and refresh preview images for your models.')
            
            with gr.Row():
                update_preview = gr.Button(value='🖼️ Update model preview', interactive=True, visible=True, variant='primary')
                cancel_update_preview = gr.Button(value='Cancel updating model previews', interactive=False, visible=False)
            with gr.Row():
                preview_progress = gr.HTML(value='<div style="min-height: 0px;"></div>')
            
            gr.Markdown('---')
            gr.Markdown('### 🔍 Check for Updates')
            gr.Markdown('Scan your collection for newer versions available on CivitAI.')
            
            with gr.Row():
                ver_search = gr.Button(value='🔍 Scan for available updates', interactive=True, visible=True, variant='primary')
                cancel_ver_search = gr.Button(value='Cancel updates scan', interactive=False, visible=False)
                load_to_browser_outdated = gr.Button(value='Load outdated models to browser', interactive=False, visible=False)
            with gr.Row():
                version_progress = gr.HTML(value='<div style="min-height: 0px;"></div>')

        ## Queue Tab
        with gr.Tab(label='Download Queue', elem_id='queueTab'):
            gr.Markdown('## 📥 Download Queue Manager', elem_id='queue_header')
            gr.Markdown('Monitor and manage your active downloads. Drag items to reorder the queue.')
            
            def get_style(size, left_border):
                return f"flex-grow: {size};" + ('border-left: 1px solid var(--border-color-primary);' if left_border else '') + 'border-bottom: 1px solid var(--border-color-primary);padding: 5px 10px 5px 10px;width: 0;'

            download_manager_html = gr.HTML(elem_id='civitai_dl_list', value=f'''
                <div style="display: flex;font-size: var(--section-header-text-size);border: 1px solid transparent;">
                <div style="{get_style(1, False)}"><span>Model:</span></div>
                <div style="{get_style(0.75, True)}"><span>Version:</span></div>
                <div style="{get_style(1.5, True)}"><span>Path:</span></div>
                <div style="{get_style(1.5, True)}"><span>Status:</span></div>
                <div style="{get_style(0.3, True)}"><span>Action:</span></div>
                </div>
                <div class="civitai_nonqueue_list">
                </div>
                <span style="padding: 10px 0px 5px 5px;font-size: larger;border-bottom: 1px solid var(--border-color-primary);">In queue: (drag items to rearrange queue order)</span>
                <div class="list" id="queue_list">
                </div>
                ''')

        ## Local Models Tab
        with gr.Tab(label='Local Models', elem_id='localTab'):
            gr.Markdown('## 📂 Manage Local Models', elem_id='local_models_header')
            gr.Markdown('Load, organize, and manage your installed models.')
            
            with gr.Row():
                selected_tags_local = gr.CheckboxGroup(elem_id='selected_tags_local', label='Selected content types:', choices=local_scan_choices, value=['Checkpoint', 'LORA'])
            
            with gr.Accordion(label='🔧 Scan Options', open=False):
                with gr.Row(elem_id='civitai_local_toggles'):
                    overwrite_toggle_local = gr.Checkbox(elem_id='overwrite_toggle_local', label='Overwrite any existing files (previews, HTMLs, tags, descriptions)', value=True, min_width=300)
                    skip_hash_toggle_local = gr.Checkbox(elem_id='skip_hash_toggle_local', label='One-Time Hash Generation for externally downloaded models', value=True, min_width=300)
                    do_html_gen_local = gr.Checkbox(elem_id='do_html_gen_local', label='Save HTML file for each model when updating info & tags', value=False, min_width=300)
            
            gr.Markdown('---')
            gr.Markdown('### 📋 Load Installed Models')
            gr.Markdown('Scan and load information about all models currently installed on your system.')
            
            with gr.Row():
                load_installed = gr.Button(value='📋 Load all installed models', interactive=True, visible=True, variant='primary')
                cancel_installed = gr.Button(value='Cancel loading models', interactive=False, visible=False)
                load_to_browser_installed = gr.Button(value='Load installed models to browser', interactive=False, visible=False)
            with gr.Row():
                installed_progress = gr.HTML(value='<div style="min-height: 0px;"></div>')
            
            gr.Markdown('---')
            gr.Markdown('### 📁 Model Organization')
            gr.Markdown('Automatically organize your models into subfolders by base model type (SDXL, Pony, FLUX, etc.)')
            
            with gr.Row():
                organize_models = gr.Button(value='📁 Organize models into subfolders by type', interactive=True, visible=True, variant='primary')
                cancel_organize = gr.Button(value='Cancel organization', interactive=False, visible=False)
            with gr.Row():
                organize_progress = gr.HTML(value='<div style="min-height: 0px;"></div>')
            
            with gr.Row():
                undo_organization = gr.Button(value='↶ Undo Last Organization', interactive=True, visible=True, variant='secondary')
            with gr.Row():
                undo_progress = gr.HTML(value='<div style="min-height: 0px;"></div>')

            gr.Markdown('---')
            gr.Markdown('### 🔍 Validate Organization')
            gr.Markdown('Check whether all models are in their correct subfolders — without moving anything. Optionally fix any misplaced files.')

            with gr.Row():
                validate_org_btn = gr.Button(value='🔍 Validate organization', interactive=True, visible=True, variant='primary')
            with gr.Row():
                validate_org_progress = gr.HTML(value='<div style="min-height: 0px;"></div>')
            with gr.Row():
                fix_misplaced_btn = gr.Button(value='✅ Fix misplaced files', interactive=True, visible=False, variant='secondary')
            with gr.Row():
                fix_misplaced_progress = gr.HTML(value='<div style="min-height: 0px;"></div>')
            with gr.Row():
                undo_fix_btn = gr.Button(value='↶ Undo Fix', interactive=True, visible=False, variant='secondary')
            with gr.Row():
                undo_fix_progress = gr.HTML(value='<div style="min-height: 0px;"></div>')

            validate_plan_state = gr.State(value='{}')

        ## Dashboard Tab
        with gr.Tab(label='Dashboard', elem_id='dashboardTab'):
            gr.Markdown('## 📊 Model Collection Statistics', elem_id='dashboard_header')
            gr.Markdown('View disk usage statistics for your model collection organized by type.')
            
            with gr.Row():
                dashboard_content_types = gr.CheckboxGroup(
                    elem_id='dashboard_content_types', 
                    label='Content types to analyze:', 
                    choices=dashboard_scan_choices_with_all
                )

            with gr.Row():
                dashboard_hide_empty = gr.Checkbox(
                    elem_id='dashboard_hide_empty',
                    label='Hide empty categories (0 files)',
                    value=True
                )
                dashboard_detect_orphans = gr.Checkbox(
                    elem_id='dashboard_detect_orphans',
                    label='Detect orphan files (no CivitAI metadata)',
                    value=False
                )
            
            with gr.Row():
                generate_dashboard = gr.Button(
                    value='📊 Generate Dashboard', 
                    interactive=True, 
                    visible=True, 
                    variant='primary'
                )
                refresh_dashboard = gr.Button(
                    value='🔄 Refresh', 
                    interactive=True, 
                    visible=True
                )
                export_csv_btn = gr.Button(
                    value='📥 Export CSV',
                    interactive=True,
                    visible=True
                )
                export_json_btn = gr.Button(
                    value='📥 Export JSON',
                    interactive=True,
                    visible=True
                )
            
            with gr.Row():
                dashboard_html = gr.HTML(value='''
                    <div style="padding: 60px 20px; text-align: center; background: var(--block-background-fill); border-radius: 8px; margin: 20px 0;">
                        <div style="font-size: 64px; margin-bottom: 20px; opacity: 0.6;">📊</div>
                        <h3 style="margin: 0 0 15px 0; color: var(--body-text-color); font-size: 20px;">Ready to analyze your collection</h3>
                        <p style="margin: 0 0 10px 0; color: var(--body-text-color-subdued); font-size: 15px;">
                            Select the content types you want to analyze and click <strong>Generate Dashboard</strong>.
                        </p>
                        <p style="margin: 0; color: var(--body-text-color-subdued); font-size: 14px;">
                            This will scan your model directories and show detailed disk usage statistics.
                        </p>
                    </div>
                ''')

        def format_custom_subfolders():
            separator = '␞␞'
            data = _api.safe_json_load(gl.subfolder_json) or {}
            # Filter out timestamp and non-string values
            filtered_data = {key: value for key, value in data.items()
                           if key != "created_at" and isinstance(value, str)}
            result = separator.join([f"{key}{separator}{value}" for key, value in filtered_data.items()])
            return result

        #Invisible triggers/variables
        #Yes, there is probably a much better way of passing variables/triggering functions between javascript and python

        gr.Textbox(elem_id='custom_subfolders_list', visible=False, value=format_custom_subfolders())
        model_id = gr.Textbox(visible=False)
        queue_trigger = gr.Textbox(visible=False)
        dl_url = gr.Textbox(visible=False)
        civitai_text2img_output = gr.Textbox(visible=False)
        civitai_text2img_input = gr.Textbox(elem_id='civitai_text2img_input', visible=False)
        page_slider_trigger = gr.Textbox(elem_id='page_slider_trigger', visible=False)
        selected_model_list = gr.Textbox(elem_id='selected_model_list', visible=False)
        selected_type_list = gr.Textbox(elem_id='selected_type_list', visible=False)
        html_cancel_input = gr.Textbox(elem_id='html_cancel_input', visible=False)
        queue_html_input = gr.Textbox(elem_id='queue_html_input', visible=False)
        list_html_input = gr.Textbox(elem_id='list_html_input', visible=False)
        preview_html_input = gr.Textbox(elem_id='preview_html_input', visible=False)
        create_subfolder = gr.Textbox(elem_id='create_subfolder', visible=False)
        send_to_browser = gr.Textbox(elem_id='send_to_browser', visible=False)
        arrange_dl_id = gr.Textbox(elem_id='arrange_dl_id', visible=False)
        remove_dl_id = gr.Textbox(elem_id='remove_dl_id', visible=False)
        model_select = gr.Textbox(elem_id='model_select', visible=False)
        model_sent = gr.Textbox(elem_id='model_sent', visible=False)
        type_sent = gr.Textbox(elem_id='type_sent', visible=False)
        click_first_item = gr.Textbox(visible=False)
        empty = gr.Textbox(value='', visible=False)
        download_start = gr.Textbox(visible=False)
        download_finish = gr.Textbox(visible=False)
        tag_start = gr.Textbox(visible=False)
        tag_finish = gr.Textbox(visible=False)
        preview_start = gr.Textbox(visible=False)
        preview_finish = gr.Textbox(visible=False)
        ver_start = gr.Textbox(visible=False)
        ver_finish = gr.Textbox(visible=False)
        installed_start = gr.Textbox(visible=None)
        installed_finish = gr.Textbox(visible=None)
        organize_start = gr.Textbox(visible=None)
        organize_finish = gr.Textbox(visible=None)
        delete_finish = gr.Textbox(elem_id='delete_finish', visible=False)
        export_csv_output  = gr.Textbox(elem_id='export_csv_output',  visible=False)
        export_json_output = gr.Textbox(elem_id='export_json_output', visible=False)
        current_model = gr.Textbox(visible=False)
        current_sha256 = gr.Textbox(visible=False)
        # Ambiguity chooser: receives HTML for choices and a hidden textarea to send selected index
        ambiguity_html = gr.HTML(value='', visible=False, elem_id='ambiguity_html')
        ambiguity_choice = gr.Textbox(visible=False, elem_id='ambiguity_choice')
        ambiguity_confirm = gr.Button('Confirm Ambiguity', visible=False, elem_id='ambiguity_confirm')
        # Bind confirm button to resolver
        ambiguity_confirm.click(fn=_download.resolve_ambiguity, inputs=[ambiguity_choice], outputs=[download_progress, current_model, download_finish, queue_trigger], show_progress='hidden')
        model_preview_html_input = gr.Textbox(visible=False)
        banned_creators_list_txt = gr.Textbox(elem_id='banned_creators_list', visible=False, value=_file.get_banned_creators_text())
        # Hidden: queue restore triggers (survive RunPod/browser disconnects)
        restore_queue_input     = gr.Textbox(elem_id='restore_queue_input',     visible=False)
        restore_action_trigger  = gr.Textbox(elem_id='restore_action_trigger',  visible=False)
        dismiss_restore_trigger = gr.Textbox(elem_id='dismiss_restore_trigger', visible=False)
        # Hidden: update mode triggers
        update_all_trigger        = gr.Textbox(elem_id='update_all_trigger',        visible=False)
        update_single_trigger     = gr.Textbox(elem_id='update_single_trigger',     visible=False)
        update_selected_trigger   = gr.Textbox(elem_id='update_selected_trigger',   visible=False)
        exit_update_mode_trigger  = gr.Textbox(elem_id='exit_update_mode_trigger',  visible=False)
        
        # Hidden elements for quick delete by SHA256 (from model cards)
        delete_trigger_sha256 = gr.Textbox(elem_id='sha256', visible=False)
        delete_trigger_btn = gr.Button(elem_id='delete_trigger_btn', visible=False)
        delete_trigger_finish = gr.Textbox(visible=False)
        # Hidden: controls whether card delete btn syncs with selected version
        sync_card_delete_cb = gr.Checkbox(value=sync_card_delete_enabled, visible=False)

        def ToggleDate(toggle_date):
            gl.sortNewest = toggle_date

        def select_subfolder(sub_folder):
            if sub_folder == 'None' or sub_folder == 'Only available if the selected files are of the same model type':
                newpath = gl.main_folder
            else:
                newpath = str(gl.main_folder) + sub_folder
            return gr.update(value=newpath)

        # Javascript Functions #

        list_html_input.change(fn=None, inputs=hide_installed, _js='(toggleValue) => hideInstalled(toggleValue)')
        hide_installed.input(fn=None, inputs=hide_installed, _js='(toggleValue) => hideInstalled(toggleValue)')

        # === Creator Management bindings ===
        list_html_input.change(fn=None, inputs=[banned_creators_list_txt, hide_banned_creators], _js='(bList, checked) => initBannedCreators(bList, checked)')
        hide_banned_creators.change(fn=None, inputs=[banned_creators_list_txt, hide_banned_creators], _js='(bList, checked) => refreshBannedCreators(bList, checked)')

        def get_creator_for_model(mid):
            if not mid or not gl.json_data:
                return gr.update(value='')
            try:
                mid_int = int(mid)
                for it in gl.json_data.get('items', []):
                    if int(it['id']) == mid_int:
                        cr = it.get('creator', {}) or {}
                        return gr.update(value=cr.get('username', '') or '')
            except Exception:
                pass
            return gr.update(value='')

        model_id.change(fn=get_creator_for_model, inputs=[model_id], outputs=[creator_name_txt])
        creator_name_txt.change(fn=_file._creator_button_updates, inputs=[creator_name_txt], outputs=[btn_fav, btn_ban, btn_clear, banned_creators_list_txt])
        btn_fav.click(fn=_file.add_favorite_creator, inputs=[creator_name_txt], outputs=[btn_fav, btn_ban, btn_clear, banned_creators_list_txt])
        btn_ban.click(fn=_file.ban_creator, inputs=[creator_name_txt], outputs=[btn_fav, btn_ban, btn_clear, banned_creators_list_txt])
        btn_clear.click(fn=_file.clear_creator, inputs=[creator_name_txt], outputs=[btn_fav, btn_ban, btn_clear, banned_creators_list_txt])
        banned_creators_list_txt.change(fn=None, inputs=[banned_creators_list_txt, hide_banned_creators], _js='(bList, checked) => refreshBannedCreators(bList, checked)')

        civitai_text2img_output.change(fn=None, inputs=civitai_text2img_output, _js='(genInfo) => genInfo_to_txt2img(genInfo)')

        download_selected.click(fn=None, _js='() => deselectAllModels()')

        select_all.click(fn=None, _js='() => selectAllModels()')

        list_models.select(fn=None, inputs=list_models, _js='(list_models) => select_model(list_models)')

        preview_html_input.change(fn=None, _js='() => adjustFilterBoxAndButtons()')
        preview_html_input.change(fn=None, _js='() => setDescriptionToggle()')

        page_slider.release(fn=None, _js='() => pressRefresh()')

        # Post-download card updates: skip pressRefresh() fallback to avoid
        # expensive full-page API re-fetch (e.g. 100 items) when a single card
        # is missing from the DOM after download/delete/queue events.
        card_updates = [queue_trigger, download_finish, delete_finish]
        for func in card_updates:
            func.change(fn=None, inputs=func, _js='(modelName) => updateCard(modelName, false)')

        # Ensure realtime card status updates also fire when current_model itself changes.
        # Skip pressRefresh() fallback — current_model changes after download/delete/ambiguity
        # resolution, and we must not trigger expensive full-page API re-fetches.
        current_model.change(fn=None, inputs=current_model, _js='(modelName) => updateCard(modelName, false)')

        list_html_input.change(fn=None, inputs=size_slider, _js='(size) => updateCardSize(size, size * 1.5)')
        size_slider.change(fn=None, inputs=size_slider, _js='(size) => updateCardSize(size, size * 1.5)')

        model_preview_html_input.change(fn=None, inputs=model_preview_html_input, _js='(html_input) => inputHTMLPreviewContent(html_input)')

        queue_html_input.change(fn=None, _js='() => setSortable()')

        click_first_item.change(fn=None, _js='() => clickFirstFigureInColumn()')

        # Filter button Functions #

        queue_html_input.change(fn=HTMLChange, inputs=[queue_html_input], outputs=download_manager_html)
        list_html_input.change(fn=HTMLChange, inputs=[list_html_input], outputs=list_html)
        preview_html_input.change(fn=HTMLChange, inputs=[preview_html_input], outputs=preview_html)

        remove_dl_id.change(
            fn=_download.remove_from_queue,
            inputs=[remove_dl_id]
        )

        arrange_dl_id.change(
            fn=_download.arrange_queue,
            inputs=[arrange_dl_id]
        )

        html_cancel_input.change(
            fn=_download.download_cancel
        )

        html_cancel_input.change(fn=None, _js='() => cancelCurrentDl()')

        save_settings.click(
            fn=saveSettings,
            inputs=[
                use_search_term,
                content_type,
                period_type,
                sort_type,
                base_filter,
                create_json,
                only_liked,
                hide_installed,
                show_nsfw,
                exact_search,
                size_slider,
                tile_count_slider
            ]
        )

        toggle_date.input(
            fn=ToggleDate,
            inputs=[toggle_date]
        )

        # Model Button Functions #

        civitai_text2img_input.change(fn=txt2img_output,inputs=civitai_text2img_input,outputs=civitai_text2img_output)

        list_html_input.change(fn=all_visible, inputs=list_html_input, outputs=select_all)

        def update_models_dropdown(input, base_filter=None):
            # If there is no loaded model data, reset all UI elements and show a message
            if not gl.json_data:
                return (
                    gr.update(value=None, choices=[], interactive=False),  # Model list dropdown
                    gr.update(value=None, choices=[], interactive=False),  # Version list dropdown
                    gr.update(value=None),                                  # Preview HTML
                    gr.update(value=None, interactive=False),               # Trained tags textbox
                    gr.update(value=None, interactive=False),               # Base model textbox
                    gr.update(value=None, interactive=False),               # Model filename textbox
                    gr.update(value=None, interactive=False),               # Install path textbox
                    gr.update(value=None, choices=[], interactive=False),  # Subfolder dropdown
                    gr.update(interactive=False),                            # Download model button
                    gr.update(interactive=False),                            # Save image button
                    gr.update(interactive=False, visible=False),             # Delete model button
                    gr.update(value=None, choices=[], interactive=False),  # File list dropdown
                    gr.update(value=None),                                  # Download URL textbox
                    gr.update(value=None),                                  # Model ID textbox
                    gr.update(value=None),                                  # Current SHA256 textbox
                    gr.update(interactive=False),                            # Save model info button
                    gr.update(
                        value='<div style="font-size: 24px; text-align: center; margin: 50px;">Click the search icon to load models.<br>Use the filter icon to filter results.</div>'
                    )  # Model list HTML message
                )

            model_string = re.sub(r'\.\d{3}$', '', input)
            model_name, model_id = _api.extract_model_info(model_string)
            model_versions = _api.update_model_versions(model_id, base_filter=base_filter)

            # Get detailed model info for the selected version
            (
                html,
                tags,
                base_model,
                download_button,
                save_images_button,
                delete_button,
                file_list,
                model_filename,
                download_url,
                model_id_value,
                current_sha256,
                install_path,
                sub_folder
            ) = _api.update_model_info(model_string, model_versions.get('value'))

            # Return all UI updates in the expected order
            return (
                gr.update(value=model_string, interactive=True), # Model dropdown
                model_versions,                                           # Version dropdown
                html,                                                     # Preview HTML
                tags,                                                     # Trained tags
                base_model,                                               # Base model
                model_filename,                                           # Model filename
                install_path,                                             # Install path
                sub_folder,                                               # Sub folder
                download_button,                                          # Download button
                save_images_button,                                       # Save images button
                delete_button,                                            # Delete button
                file_list,                                                # File list
                download_url,                                             # Download URL
                model_id_value,                                           # Model ID
                current_sha256,                                           # Current SHA256
                gr.update(interactive=True),                       # Save model info button
                gr.update()                                       # Model list HTML
            )

        model_select.change(
            fn=update_models_dropdown,
            inputs=[model_select, base_filter],
            outputs=[
                list_models,
                list_versions,
                preview_html,
                trained_tags,
                base_model,
                model_filename,
                install_path,
                sub_folder,
                download_model,
                save_images,
                delete_model,
                file_list,
                dl_url,
                model_id,
                current_sha256,
                save_info,
                list_html_input
            ]
        )

        model_sent.change(
            fn=_file.model_from_sent,
            inputs=[model_sent, type_sent],
            outputs=[model_preview_html_input]
        )

        send_to_browser.change(
            fn=_file.send_to_browser,
            inputs=[send_to_browser, type_sent, click_first_item],
            outputs=[list_html_input, get_prev_page, get_next_page, page_slider, click_first_item]
        )

        sub_folder.select(
            fn=select_subfolder,
            inputs=[sub_folder],
            outputs=[install_path]
        )

        subfolder_selected.select(
            fn=select_subfolder,
            inputs=[subfolder_selected],
            outputs=[install_path]
        )

        list_versions.select(
            fn=_api.update_model_info,
            inputs=[
                list_models,
                list_versions
            ],
            outputs=[
                preview_html_input,
                trained_tags,
                base_model,
                download_model,
                save_images,
                delete_model,
                file_list,
                model_filename,
                dl_url,
                model_id,
                current_sha256,
                install_path,
                sub_folder
            ]
        )

        trained_tags.change(
            fn=lambda v: gr.update(interactive=bool(v and v.strip())),
            inputs=[trained_tags],
            outputs=[send_tags_btn]
        )

        send_tags_btn.click(
            fn=None,
            inputs=[trained_tags],
            _js='(tags) => sendTagsToPrompt(tags)'
        )

        # Sync card delete button visibility with selected version (if setting enabled)
        list_versions.select(
            fn=None,
            inputs=[list_versions, sync_card_delete_cb],
            _js='(ver, enabled) => syncCardDeleteBtnOnCard(ver, enabled)'
        )

        file_list.input(
            fn=_api.update_file_info,
            inputs=[
                list_models,
                list_versions,
                file_list
            ],
            outputs=[
                model_filename,
                dl_url,
                model_id,
                current_sha256,
                download_model,
                delete_model,
                install_path,
                sub_folder
            ]
        )

        # Download/Save Model Button Functions #

        selected_model_list.change(
            fn=show_multi_buttons,
            inputs=[selected_model_list, selected_type_list, list_versions],
            outputs=[
                download_selected,
                download_model,
                delete_model,
                save_info,
                save_images,
                subfolder_selected
            ]
        )

        download_model.click(
            fn=_download.download_start,
            inputs=[
                download_start,
                dl_url,
                model_filename,
                install_path,
                list_models,
                list_versions,
                current_sha256,
                model_id,
                create_json,
                download_manager_html
            ],
            outputs=[
                download_model,
                cancel_model,
                cancel_all_model,
                download_start,
                download_progress,
                download_manager_html
            ],
            show_progress='hidden'
        )

        download_selected.click(
            fn=_download.selected_to_queue,
            inputs=[
                selected_model_list,
                subfolder_selected,
                download_start,
                create_json,
                download_manager_html
            ],
            outputs=[
                download_model,
                cancel_model,
                cancel_all_model,
                download_start,
                download_progress,
                download_manager_html
            ],
            show_progress='hidden'
        )

        for component in [download_start, queue_trigger]:
            component.change(fn=None, _js='() => setDownloadProgressBar()')
            component.change(
                fn=_download.download_create_thread,
                inputs=[download_finish, queue_trigger],
                outputs=[
                    download_progress,
                    current_model,
                    download_finish,
                    queue_trigger,
                    update_mode_banner
                ]
            )

        _download_finish_event = download_finish.change(
            fn=_download.download_finish,
            inputs=[
                model_filename,
                list_versions,
                model_id
            ],
            outputs=[
                download_model,
                cancel_model,
                cancel_all_model,
                delete_model,
                download_progress,
                list_versions
            ],
            show_progress='hidden'
        )

        cancel_model.click(_download.download_cancel)
        cancel_all_model.click(_download.download_cancel_all)

        cancel_model.click(fn=None, _js='() => cancelCurrentDl()')
        cancel_all_model.click(fn=None, _js='() => cancelAllDl()')

        # === Queue Restore bindings ===
        restore_queue_input.change(
            fn=None, inputs=[restore_queue_input],
            _js='(json) => initRestoreBanner(json)'
        )
        restore_action_trigger.change(
            fn=_download.restore_interrupted_to_queue,
            inputs=[download_manager_html],
            outputs=[download_model, cancel_model, cancel_all_model,
                     download_start, download_progress, download_manager_html],
            show_progress='hidden'
        )
        dismiss_restore_trigger.change(fn=_download.dismiss_interrupted_downloads)

        delete_model.click(
            fn=_file.delete_model,
            inputs=[
                delete_finish,
                model_filename,
                list_models,
                list_versions,
                current_sha256,
                selected_model_list
            ],
            outputs=[
                download_model,
                cancel_model,
                delete_model,
                delete_finish,
                current_model,
                list_versions
            ]
        )
        
        # Quick delete button for installed models (triggered from model cards)
        delete_trigger_btn.click(
            fn=_file.delete_installed_by_sha256,
            inputs=[delete_trigger_sha256, delete_trigger_finish],
            outputs=[delete_trigger_finish]
        )

        save_info.click(
            fn=_file.save_model_info,
            inputs=[
                install_path,
                model_filename,
                sub_folder,
                current_sha256,
                preview_html_input
            ],
            outputs=[]
        )

        def save_images_wrapper(preview_html, model_filename, install_path, sub_folder, model_id):
            """Wrapper function to save images with API response data"""
            # print(f"Save Images Debug: model_id={model_id}, preview_html_length={len(preview_html) if preview_html else 0}")

            # Generate proper HTML with images if preview_html is empty or doesn't contain images
            if not preview_html or 'data-sampleimg="true"' not in preview_html:
                debug_print(f"Generating HTML with images for model {model_id}")
                if model_id:
                    try:
                        # Get model versions for the current model
                        model_versions = _api.update_model_versions(model_id)
                        if model_versions and model_versions.get('value'):
                            # Generate HTML with images using the same function as automatic version
                            result = _api.update_model_info(None, model_versions.get('value'), False, model_id)
                            if isinstance(result, tuple) and len(result) >= 1:
                                preview_html = result[0]  # First element is the HTML
                            else:
                                preview_html = str(result) if result else ""
                            debug_print(f"Generated HTML length: {len(preview_html) if preview_html else 0}")
                    except Exception as e:
                        debug_print(f"Error generating HTML: {e}")

            if model_id and gl.json_data:
                # Find the current model in the API data
                for item in gl.json_data.get('items', []):
                    if int(item.get('id', 0)) == int(model_id):
                        debug_print(f"Using existing API data for model {model_id}")
                        # Ensure preview_html is a string
                        if not isinstance(preview_html, str):
                            preview_html = str(preview_html) if preview_html else ""
                        _file.save_images(preview_html, model_filename, install_path, sub_folder, api_response=gl.json_data)
                        return
                # If model not found in current data, try to fetch it
                try:
                    debug_print(f"Fetching API data for model {model_id}")
                    api_response = _api.request_civit_api(f"https://{_api.get_civitai_domain()}/api/v1/models/{model_id}")
                    if not isinstance(preview_html, str):
                        preview_html = str(preview_html) if preview_html else ""
                    _file.save_images(preview_html, model_filename, install_path, sub_folder, api_response=api_response)
                    return
                except Exception as e:
                    debug_print(f"Error fetching API data: {e}")
            # Fallback to save without API response
            debug_print(f"Using fallback save method")
            if not isinstance(preview_html, str):
                preview_html = str(preview_html) if preview_html else ""
            _file.save_images(preview_html, model_filename, install_path, sub_folder)

        save_images.click(
            fn=save_images_wrapper,
            inputs=[
                preview_html_input,
                model_filename,
                install_path,
                sub_folder,
                model_id
            ],
            outputs=[]
        )

        # Common input&output lists #

        page_inputs = [
            content_type,
            sort_type,
            period_type,
            use_search_term,
            search_term,
            page_slider,
            base_filter,
            only_liked,
            show_nsfw,
            exact_search,
            tile_count_slider
        ]

        refresh_inputs = [empty if item == page_slider else item for item in page_inputs]

        page_outputs = [
            list_models,
            list_versions,
            list_html_input,
            get_prev_page,
            get_next_page,
            page_slider,
            save_info,
            save_images,
            download_model,
            delete_model,
            install_path,
            sub_folder,
            file_list,
            preview_html_input,
            trained_tags,
            base_model,
            model_filename
        ]

        _download_finish_event.then(
            fn=lambda *args: _api.initial_model_page(*args, from_update_tab=True),
            inputs=refresh_inputs,
            outputs=page_outputs
        )

        file_scan_inputs = [
            selected_tags,
            tag_finish,
            ver_finish,
            installed_finish,
            preview_finish,
            organize_finish,
            overwrite_toggle,
            tile_count_slider,
            skip_hash_toggle,
            do_html_gen
        ]

        file_scan_inputs_local = [
            selected_tags_local,
            tag_finish,
            ver_finish,
            installed_finish,
            preview_finish,
            organize_finish,
            overwrite_toggle_local,
            tile_count_slider,
            skip_hash_toggle_local,
            do_html_gen_local
        ]

        load_to_browser_inputs = [
            content_type,
            sort_type,
            period_type,
            use_search_term,
            search_term,
            tile_count_slider,
            base_filter,
            show_nsfw,
            exact_search
        ]

        cancel_btn_list = [cancel_all_tags, cancel_ver_search, cancel_installed, cancel_update_preview, cancel_organize]

        browser = [save_all_tags, ver_search, load_installed, update_preview]

        browser_installed_load = [cancel_installed, load_to_browser_installed, installed_progress]
        browser_outdated_load = [cancel_ver_search, load_to_browser_outdated, version_progress]

        browser_installed_list = page_outputs + browser + browser_installed_load
        browser_outdated_list = page_outputs + browser + browser_outdated_load

        # Page Button Functions #

        page_btn_list = {
            refresh.click: (_api.initial_model_page, True),
            search_term.submit: (_api.initial_model_page, True),
            page_slider_trigger.change: (_api.initial_model_page, False),
            get_next_page.click: (_api.next_model_page, False),
            get_prev_page.click: (_api.prev_model_page, False)
        }

        for trigger, (function, use_refresh_inputs) in page_btn_list.items():
            inputs_to_use = refresh_inputs if use_refresh_inputs else page_inputs
            trigger(fn=function, inputs=inputs_to_use, outputs=page_outputs)
            trigger(fn=None, _js='() => multi_model_select()')

        for button in cancel_btn_list:
            button.click(fn=_file.cancel_scan)

        # Update model Functions #

        ver_search.click(
            fn=_file.ver_search_start,
            inputs=[ver_start],
            outputs=[
                ver_start,
                ver_search,
                cancel_ver_search,
                load_installed,
                save_all_tags,
                update_preview,
                organize_models,
                version_progress
            ]
        )

        ver_start.change(
            fn=_file.file_scan,
            inputs=file_scan_inputs,
            outputs=[
                version_progress,
                ver_finish
            ]
        )

        ver_finish.change(
            fn=_file.scan_finish,
            outputs=[
                ver_search,
                save_all_tags,
                load_installed,
                update_preview,
                organize_models,
                cancel_ver_search,
                load_to_browser_outdated
            ]
        )

        load_installed.click(
            fn=_file.installed_models_start,
            inputs=[installed_start],
            outputs=[
                installed_start,
                load_installed,
                cancel_installed,
                ver_search,
                save_all_tags,
                update_preview,
                organize_models,
                installed_progress
            ]
        )

        installed_start.change(
            fn=_file.file_scan,
            inputs=file_scan_inputs_local,
            outputs=[
                installed_progress,
                installed_finish
            ]
        )

        installed_finish.change(
            fn=_file.scan_finish,
            outputs=[
                ver_search,
                save_all_tags,
                load_installed,
                update_preview,
                organize_models,
                cancel_installed,
                load_to_browser_installed
            ]
        )

        save_all_tags.click(
            fn=_file.save_tag_start,
            inputs=[tag_start],
            outputs=[
                tag_start,
                save_all_tags,
                cancel_all_tags,
                load_installed,
                ver_search,
                update_preview,
                organize_models,
                tag_progress
            ]
        )

        sync_sha256_cache.click(
            fn=_file.sync_checkpoint_sha256_cache,
            outputs=[sync_sha256_progress]
        )

        tag_start.change(
            fn=_file.file_scan,
            inputs=file_scan_inputs,
            outputs=[
                tag_progress,
                tag_finish
            ]
        )

        tag_finish.change(
            fn=_file.save_tag_finish,
            outputs=[
                ver_search,
                save_all_tags,
                load_installed,
                update_preview,
                organize_models,
                cancel_all_tags
            ]
        )

        update_preview.click(
            fn=_file.save_preview_start,
            inputs=[preview_start],
            outputs=[
                preview_start,
                update_preview,
                cancel_update_preview,
                load_installed,
                ver_search,
                save_all_tags,
                organize_models,
                preview_progress
            ]
        )

        preview_start.change(
            fn=_file.file_scan,
            inputs=file_scan_inputs,
            outputs=[
                preview_progress,
                preview_finish
            ]
        )

        preview_finish.change(
            fn=_file.save_preview_finish,
            outputs=[
                ver_search,
                save_all_tags,
                load_installed,
                update_preview,
                organize_models,
                cancel_update_preview
            ]
        )

        organize_models.click(
            fn=_file.organize_start,
            inputs=[organize_start],
            outputs=[
                organize_start,
                organize_models,
                cancel_organize,
                load_installed,
                ver_search,
                save_all_tags,
                update_preview,
                organize_progress
            ]
        )

        organize_start.change(
            fn=_file.file_scan,
            inputs=file_scan_inputs_local,
            outputs=[
                organize_progress,
                organize_finish
            ]
        )

        organize_finish.change(
            fn=_file.scan_finish,
            outputs=[
                ver_search,
                save_all_tags,
                load_installed,
                update_preview,
                organize_models,
                cancel_organize,
                load_to_browser_installed
            ]
        )

        undo_organization.click(
            fn=_file.rollback_organization,
            outputs=[
                undo_progress
            ]
        )

        validate_org_btn.click(
            fn=_file.validate_organization,
            inputs=[selected_tags_local],
            outputs=[
                validate_org_progress,
                fix_misplaced_btn,
                validate_plan_state
            ],
            show_progress="full"
        )

        fix_misplaced_btn.click(
            fn=_file.fix_misplaced_files,
            inputs=[validate_plan_state],
            outputs=[
                fix_misplaced_progress,
                fix_misplaced_btn,
                undo_fix_btn,
                validate_plan_state
            ],
            show_progress="full"
        )

        undo_fix_btn.click(
            fn=_file.rollback_organization,
            outputs=[
                undo_fix_progress
            ]
        )

        def handle_dashboard_selection(selected):
            """Handle 'All' checkbox logic"""
            if 'All' in selected:
                # If 'All' is selected, select everything except 'All' itself
                return gr.update(value=dashboard_scan_choices)
            else:
                # Return current selection
                return gr.update(value=selected)
        
        dashboard_content_types.change(
            fn=handle_dashboard_selection,
            inputs=[dashboard_content_types],
            outputs=[dashboard_content_types]
        )

        generate_dashboard.click(
            fn=_file.generate_dashboard_statistics,
            inputs=[dashboard_content_types, dashboard_hide_empty, dashboard_detect_orphans],
            outputs=[dashboard_html]
        )

        refresh_dashboard.click(
            fn=_file.generate_dashboard_statistics,
            inputs=[dashboard_content_types, dashboard_hide_empty, dashboard_detect_orphans],
            outputs=[dashboard_html]
        )

        export_csv_btn.click(
            fn=_file.export_dashboard_csv,
            inputs=[],
            outputs=[export_csv_output]
        )
        export_csv_output.change(
            fn=None,
            inputs=[export_csv_output],
            _js='(csv) => downloadBlobFile(csv, "dashboard_stats.csv", "text/csv")'
        )

        export_json_btn.click(
            fn=_file.export_dashboard_json,
            inputs=[],
            outputs=[export_json_output]
        )
        export_json_output.change(
            fn=None,
            inputs=[export_json_output],
            _js='(json) => downloadBlobFile(json, "dashboard_stats.json", "application/json")'
        )

        load_to_browser_outdated.click(
            fn=_file.load_to_browser,
            inputs=load_to_browser_inputs,
            outputs=browser_outdated_list
        ).then(
            fn=_file.enter_update_mode,
            outputs=[update_mode_banner]
        )

        update_all_trigger.change(
            fn=_download.update_all_models,
            inputs=[download_start, create_json, download_manager_html],
            outputs=[
                download_model,
                cancel_model,
                cancel_all_model,
                download_start,
                download_progress,
                download_manager_html
            ],
            show_progress='hidden'
        )

        update_single_trigger.change(
            fn=_download.download_single_update,
            inputs=[update_single_trigger, download_start, create_json, download_manager_html],
            outputs=[
                download_model,
                cancel_model,
                cancel_all_model,
                download_start,
                download_progress,
                download_manager_html
            ],
            show_progress='hidden'
        )

        update_selected_trigger.change(
            fn=_download.update_selected_models,
            inputs=[update_selected_trigger, download_start, create_json, download_manager_html],
            outputs=[
                download_model,
                cancel_model,
                cancel_all_model,
                download_start,
                download_progress,
                download_manager_html
            ],
            show_progress='hidden'
        )

        exit_update_mode_trigger.change(
            fn=_file.exit_update_mode,
            inputs=load_to_browser_inputs,
            outputs=[update_mode_banner, list_html_input, get_prev_page, get_next_page, page_slider]
        )

        load_to_browser_installed.click(
            fn=_file.load_to_browser,
            inputs=load_to_browser_inputs,
            outputs=browser_installed_list
        )

        # Settings function
        create_subfolder.change(
            fn=_file.updateSubfolder,
            inputs=create_subfolder,
            outputs=[]
        )

        # On page load, check for interrupted downloads and populate the restore banner
        civitai_interface.load(
            fn=_download.get_interrupted_downloads_json,
            outputs=[restore_queue_input]
        )

    tab_name = 'CivitAI Browser Neo'
    return (civitai_interface, tab_name, 'civitai_interface_neo'),

def subfolder_list(folder, desc=None):
    if folder == None:
        return
    model_folder = _api.contenttype_folder(folder, desc)
    sub_folders = _file.getSubfolders(model_folder)
    return sub_folders

def make_lambda(folder, desc):
    return lambda: {'choices': subfolder_list(folder, desc)}

## === ANXETY EDITs ===
def on_ui_settings():
    browser = ('civitai_browser', 'Browser')
    download = ('civitai_browser_download', 'Downloads')

    categories.register_category('civitai_browser_neo', 'CivitAI Browser Neo')
    cat_id = 'civitai_browser_neo'   # Settings category for Neo version

    if not (hasattr(shared.OptionInfo, 'info') and callable(getattr(shared.OptionInfo, 'info'))):
        def info(self, info):
            self.label += f" ({info})"
            return self
        shared.OptionInfo.info = info

    ## Browser Options
    shared.opts.add_option(
        'custom_api_key',
        shared.OptionInfo(
            default=r'',
            label='Personal CivitAI API key',
            section=browser,
            category_id=cat_id
        ).info('You can create your own API key in your CivitAI account settings, this required for some downloads. Requires UI reload')
    )

    shared.opts.add_option(
        'civitai_sfw_only',
        shared.OptionInfo(
            default=False,
            label='SFW only (use civitai.com)',
            section=browser,
            category_id=cat_id
        ).info('When enabled, all CivitAI links and API calls use civitai.com. Disable for full access via civitai.red. Requires UI reload')
    )

    shared.opts.add_option(
        'hide_early_access',
        shared.OptionInfo(
            default=False,
            label='Hide early access models',
            section=browser,
            category_id=cat_id
        ).info('Early access models are only downloadable for supporter tier members')
    )

    shared.opts.add_option(
        'dot_subfolders',
        shared.OptionInfo(
            default=True,
            label='Hide sub-folders that start with a `.`',
            section=browser,
            category_id=cat_id
        )
    )

    shared.opts.add_option(
        'use_local_html',
        shared.OptionInfo(
            default=False,
            label='Use local HTML file for model info',
            section=browser,
            category_id=cat_id
        ).info('Uses the matching local HTML file when pressing CivitAI button on model cards in txt2img and img2img')
    )

    shared.opts.add_option(
        'local_path_in_html',
        shared.OptionInfo(
            default=False,
            label='Use local images in the HTML',
            section=browser,
            category_id=cat_id
        ).info('Only works if all images of the corresponding model are downloaded')
    )

    shared.opts.add_option(
        'page_header',
        shared.OptionInfo(
            default=False,
            label='Page navigation as header',
            section=browser,
            category_id=cat_id
        ).info('Keeps the page navigation always visible at the top. Requires UI reload')
    )

    shared.opts.add_option(
        'video_playback',
        shared.OptionInfo(
            default=True,
            label='Gif/video playback in the browser',
            section=browser,
            category_id=cat_id
        ).info("Disable this option if you're experiencing high CPU usage during video/gif playback")
    )

    shared.opts.add_option(
        'individual_meta_btn',
        shared.OptionInfo(
            default=True,
            label='Individual prompt buttons',
            section=browser,
            category_id=cat_id
        ).info('Turns individual prompts from an example image into a button to send it to txt2img')
    )

    shared.opts.add_option(
        'resize_preview_cards',
        shared.OptionInfo(
            default=True,
            label='Resize preview images/videos in model cards',
            section=browser,
            category_id=cat_id
        ).info('Resize preview images/videos in model cards to improve loading speeds')
    )

    shared.opts.add_option(
        'resize_preview_size',
        shared.OptionInfo(
            default=512,
            label='Resize preview size',
            component=gr.Slider,
            component_args=lambda: {'maximum': '1024', 'minimum': '128', 'step': '32'},
            section=browser,
            category_id=cat_id
        ).info('Size in pixels the width of preview images/videos. Images retain the aspect ratio')
    )

    shared.opts.add_option(
        'resize_preview_on_save',
        shared.OptionInfo(
            default=True,
            label='Resize saved preview images',
            section=browser,
            category_id=cat_id
        ).info('Apply resizing to preview images when saving them locally?')
    )

    shared.opts.add_option(
        'model_desc_to_json',
        shared.OptionInfo(
            default=False,
            label='Save model description to json',
            section=browser,
            category_id=cat_id
        ).info('This saves the models description to the description field on model cards')
    )

    shared.opts.add_option(
        'civitai_send_to_browser',
        shared.OptionInfo(
            default=False,
            label='Send model from the cards CivitAI button to the browser, instead of showing a popup',
            section=browser,
            category_id=cat_id
        )
    )

    shared.opts.add_option(
        'image_location',
        shared.OptionInfo(
            default=r'',
            label='Custom save images location',
            section=browser,
            category_id=cat_id
        ).info('Overrides the download folder location when saving images.')
    )

    shared.opts.add_option(
        'sub_image_location',
        shared.OptionInfo(
            default=True,
            label='Use sub folders inside custom images location',
            section=browser,
            category_id=cat_id
        ).info('Will append any content type and sub folders to the custom path.')
    )

    shared.opts.add_option(
        'save_to_custom',
        shared.OptionInfo(
            default=False,
            label='Store the HTML and api_info in the custom images location',
            section=browser,
            category_id=cat_id
        )
    )

    shared.opts.add_option(
        'show_nsfw_badge',
        shared.OptionInfo(
            default=True,
            label='Show NSFW badge on model cards',
            section=browser,
            category_id=cat_id
        ).info('Display NSFW badge on model cards that are marked as NSFW content')
    )

    shared.opts.add_option(
        'show_civitai_status_badges',
        shared.OptionInfo(
            default=True,
            label='Show status badges on model cards (New / Updated + base model)',
            section=browser,
            category_id=cat_id
        ).info('Displays green "New" or "Updated" badge and base model abbreviation (e.g. IL, Pony, FLUX) on the type badge — similar to the CivitAI website. Disable if you prefer a cleaner / smaller card layout.')
    )

    shared.opts.add_option(
        'precise_version_check',
        shared.OptionInfo(
            default=True,
            label='Precise model family version comparison',
            section=browser,
            category_id=cat_id
        ).info('Check for updates using family and version (if disabled, compares only num version patterns)')
    )

    shared.opts.add_option(
        'custom_civitai_proxy',
        shared.OptionInfo(
            default=r'',
            label='Proxy address',
            component=gr.Textbox,
            component_args={'placeholder': 'socks4://0.0.0.0:00000 | socks5://0.0.0.0:00000'},
            section=browser,
            category_id=cat_id
        ).info('Only works with proxies that support HTTPS, turn Aria2 off for proxy downloads')
    )

    shared.opts.add_option(
        'cabundle_path_proxy',
        shared.OptionInfo(
            default=r'',
            label='Path to custom CA Bundle',
            component=gr.Textbox,
            component_args={'placeholder': '/path/to/custom/cabundle.pem'},
            section=browser,
            category_id=cat_id
        ).info('Specify custom CA bundle for SSL certificate checks if required')
    )

    shared.opts.add_option(
        'disable_sll_proxy',
        shared.OptionInfo(
            default=False,
            label='Disable SSL certificate checks',
            section=browser,
            category_id=cat_id
        ).info('Not recommended for security, may be required if you do not have the correct CA Bundle available')
    )

    shared.opts.add_option(
        'civitai_debug_prints',
        shared.OptionInfo(
            default=False,
            label='Enable debug prints',
            section=browser,
            category_id=cat_id
        ).info('Shows debug information in console for API calls and file operations. Requires UI reload')
    )

    shared.opts.add_option(
        'civitai_neo_delete_to_trash',
        shared.OptionInfo(
            default=True,
            label='Move deleted models to system Trash instead of permanently deleting',
            section=browser,
            category_id=cat_id
        ).info('When enabled, deleted models are sent to the OS recycle bin/trash. Disable for permanent deletion.')
    )

    ## Download Options
    shared.opts.add_option(
        'use_aria2',
        shared.OptionInfo(
            default=True,
            label='Download models using Aria2',
            section=download,
            category_id=cat_id
        ).info("Disable this option if you're experiencing any issues with downloads or if you want to use a proxy.")
    )

    shared.opts.add_option(
        'disable_dns',
        shared.OptionInfo(
            default=False,
            label='Disable Async DNS for Aria2',
            section=download,
            category_id=cat_id
        ).info('Useful for users who use PortMaster or other software that controls the DNS')
    )

    shared.opts.add_option(
        'show_log',
        shared.OptionInfo(
            default=False,
            label='Show Aria2 logs in console',
            section=download,
            category_id=cat_id
        ).info('Requires UI reload')
    )

    shared.opts.add_option(
        'split_aria2',
        shared.OptionInfo(
            default=64,
            label='Number of connections to use for downloading a model',
            component=gr.Slider,
            component_args=lambda: {'maximum': '64', 'minimum': '1', 'step': '1'},
            section=download,
            category_id=cat_id
        ).info('Only applies to Aria2')
    )

    shared.opts.add_option(
        'aria2_flags',
        shared.OptionInfo(
            default=r'',
            label='Custom Aria2 command line flags',
            section=download,
            category_id=cat_id
        ).info('Requires UI reload')
    )

    shared.opts.add_option(
        'unpack_zip',
        shared.OptionInfo(
            default=True,
            label='Automatically unpack .zip files after downloading',
            section=download,
            category_id=cat_id
        )
    )

    shared.opts.add_option(
        'save_api_info',
        shared.OptionInfo(
            default=False,
            label='Save API info of model when saving model info',
            section=download,
            category_id=cat_id
        ).info('creates an api_info.json file when saving any model info with all the API data of the model')
    )

    shared.opts.add_option(
        'auto_save_all_img',
        shared.OptionInfo(
            default=False,
            label='Automatically save all images',
            section=download,
            category_id=cat_id
        ).info('Automatically saves all the images of a model after downloading')
    )

    shared.opts.add_option(
        'save_img_count',
        shared.OptionInfo(
            default=16,
            label='Number of images to save',
            component=gr.Slider,
            component_args=lambda: {'maximum': '64', 'minimum': '4', 'step': '2'},
            section=download,
            category_id=cat_id
        ).info('Number of images to save when using the Save Images button or when Automatically save all images is enabled')
    )

    shared.opts.add_option(
        'save_html_on_save',
        shared.OptionInfo(
            default=False,
            label='Save HTML file when saving model',
            section=download,
            category_id=cat_id
        ).info('If enabled, saves the HTML file locally when saving a model')
    )

    # Default sub folders
    folders = [
        'Checkpoint',
        'LORA',
        'TextualInversion',
        'Poses',
        'Controlnet',
        'Detection',
        'MotionModule',
        ('Upscaler', 'SWINIR'),
        ('Upscaler', 'REALESRGAN'),
        ('Upscaler', 'GFPGAN'),
        ('Upscaler', 'BSRGAN'),
        ('Upscaler', 'ESRGAN'),
        'VAE',
        'AestheticGradient',
        'Wildcards',
        'Workflows',
        'Other'
    ]

    for folder in folders:
        if folder is None:
            continue

        # Handle tuple folders (e.g., ('Upscaler', 'SWINIR'))
        if isinstance(folder, tuple):
            folder_name = f"{folder[0]} - {folder[1]}"
            setting_name = f"{folder[1]}_upscale"
            folder_key = folder[0]
            desc = folder[1]
        else:
            folder_name = folder
            setting_name = folder
            folder_key = folder
            desc = None


        shared.opts.add_option(
            f"{setting_name}_default_subfolder",
            shared.OptionInfo(
                default='None',
                label=folder_name,
                component=gr.Dropdown,
                component_args=make_lambda(folder_key, desc),
                section=download,
                category_id=cat_id
            )
        )

    ## Organization Options
    organization = ('civitai_browser_organization', 'Model Organization')
    
    shared.opts.add_option(
        'civitai_neo_auto_organize',
        shared.OptionInfo(
            default=False,
            label='Auto-organize downloads into subfolders by model type',
            section=organization,
            category_id=cat_id
        ).info('When enabled, new downloads will automatically go into subfolders based on baseModel (e.g., SDXL/, Pony/, FLUX/)')
    )
    
    shared.opts.add_option(
        'civitai_neo_create_other_folder',
        shared.OptionInfo(
            default=True,
            label='Create "Other" folder for unrecognized models',
            section=organization,
            category_id=cat_id
        ).info('When disabled, unrecognized models will be placed in the root folder')
    )

    shared.opts.add_option(
        'civitai_neo_wildcard_own_folder',
        shared.OptionInfo(
            default=True,
            label='Create a subfolder per wildcard download',
            section=organization,
            category_id=cat_id
        ).info('Places each downloaded wildcard in its own subfolder (e.g. wildcards/emotion-pack/emotion-pack.txt). Compatible with sd-dynamic-prompts __subfolder/name__ syntax.')
    )

    shared.opts.add_option(
        'civitai_neo_wildcard_organize_by_base',
        shared.OptionInfo(
            default=False,
            label='Organize wildcards by base model (SDXL/, Pony/, etc.)',
            section=organization,
            category_id=cat_id
        ).info('When enabled, wildcards are also sorted into base model subfolders like Checkpoints and LORAs. Most wildcards are architecture-agnostic, so this is OFF by default.')
    )
    
    shared.opts.add_option(
        'civitai_neo_wan_subfolder_by_type',
        shared.OptionInfo(
            default=False,
            label='Organize Wan models into subfolders by type (I2V / T2V / TI2V)',
            section=organization,
            category_id=cat_id
        ).info('When enabled, Wan models are split into Wan/I2V/, Wan/T2V/, Wan/TI2V/ subfolders instead of a single Wan/ folder. Useful because I2V and T2V use different resources.')
    )

    shared.opts.add_option(
        'civitai_neo_debug_organize',
        shared.OptionInfo(
            default=False,
            label='Enable debug logs for organization system',
            section=organization,
            category_id=cat_id
        ).info('Show detailed debug logs when organizing models. Useful for troubleshooting baseModel detection issues.')
    )

    shared.opts.add_option(
        'civitai_neo_update_retention',
        shared.OptionInfo(
            default='replace',
            label='Old version retention policy after model update',
            component=gr.Radio,
            component_args=lambda: {'choices': ['keep', 'move to _Trash', 'replace']},
            section=organization,
            category_id=cat_id
        ).info("'keep' both files, 'move to _Trash' (subfolder next to old file), or 'replace' (delete old before downloading new)")
    )

    shared.opts.add_option(
        'civitai_neo_sync_card_delete',
        shared.OptionInfo(
            default=True,
            label='Sync card delete button visibility with selected version',
            section=organization,
            category_id=cat_id
        ).info('When enabled, the delete shortcut on outdated cards hides automatically if the selected version in the panel is not the installed one. Requires UI reload.')
    )

    shared.opts.add_option(
        'civitai_neo_model_categories',
        shared.OptionInfo(
            default='',
            label='Custom model categories (JSON format)',
            component=gr.Textbox,
            component_args=lambda: {'lines': 8, 'placeholder': 'Leave empty to use default categories\n\nExample:\n{\n  "SD": ["SD 1", "SD1", "SD 2", "SD2"],\n  "SDXL": ["SDXL"],\n  "Pony": ["PONY"],\n  "FLUX": ["FLUX"]\n}'},
            section=organization,
            category_id=cat_id
        ).info('Advanced: Customize folder names and detection patterns. Leave empty for defaults (SD, SDXL, Pony, Illustrious, FLUX, Wan, Qwen, Z-Image, Lumina, Anima, Cascade, PixArt, Playground, SVD, Hunyuan, Kolors, AuraFlow, Chroma)')
    )

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)