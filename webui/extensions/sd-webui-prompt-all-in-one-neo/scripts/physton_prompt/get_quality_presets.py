import os
import json
import struct
import requests

from scripts.physton_prompt.storage import Storage

# ---------------------------------------------------------------------------
# Storage key
# ---------------------------------------------------------------------------
STORAGE_KEY = 'qualityPresets'

def _default_builtin_enabled() -> dict:
    """Return a dict with all BUILTIN_TEMPLATES keys set to True."""
    # Evaluated lazily (after BUILTIN_TEMPLATES is defined) via load_presets().
    return {k: True for k in BUILTIN_TEMPLATES}

DEFAULT_PRESETS = {
    # NOTE: CivitAI API key is stored in WebUI settings (shared.opts.paio_neo_civitai_api_key),
    # not here. Use _get_civitai_api_key() to retrieve it at runtime.
    # builtin_enabled: per-family toggle (True = use this family's template by default)
    # Populated at runtime from BUILTIN_TEMPLATES keys; missing keys default to True.
    'builtin_enabled': {},
    # builtin_overrides: user-edited tags per family (values follow same schema as BUILTIN_TEMPLATES)
    'builtin_overrides': {},
    # checkpoint_cache: SHA-256 → {base_model, filename, scanned_at}
    # Persisted so repeated opens never call the API twice for the same file.
    'checkpoint_cache': {},
    'presets': [],
}


def _get_civitai_api_key() -> str:
    """Read the CivitAI API key from WebUI settings (Settings → Prompt All-in-One Neo)."""
    try:
        from modules import shared
        return getattr(shared.opts, 'paio_neo_civitai_api_key', '') or ''
    except Exception:
        return ''

# ---------------------------------------------------------------------------
# Built-in template mapping  CivitAI baseModel  →  quality tag blocks
#
# Sources (research summary):
#
#  SD 1.5 / SDXL generic
#    - "masterpiece, best quality" are Danbooru aesthetic-tier tags; useful but
#      less impactful in SDXL. Negative: classic "worst quality, low quality, lowres"
#    https://civitai.com/articles/19069
#
#  NoobAI-XL
#    - Team explicitly mapped percentile tiers in the dataset:
#        masterpiece (>95th), best quality (85-95th), good quality (60-85th),
#        normal quality (30-60th), worst quality (<30th)
#    - Official recommended prefix: "masterpiece, best quality, newest, absurdres, highres"
#    - Official negative: "worst quality, old, early, low quality, lowres"
#    https://civitai.com/models/833294/noobai-xl-nai-xl
#
#  Pony Diffusion V6 XL
#    - Uses score_* tags from the Derpibooru rating system (not Danbooru aesthetic tiers)
#    - Full chain: score_9 > score_8_up > score_7_up > score_6_up > score_5_up > score_4_up
#    - Best practice: score_9, score_8_up, score_7_up (top 3 is sufficient)
#    https://civitai.com/models/257749/pony-diffusion-v6-xl
#
#  Illustrious XL (v0.1 / 1.0 / 2.0)
#    - "Raw" base model without strong aesthetic curation; masterpiece still works
#      in tag-style prompts. Most visual quality comes from LoRA and prompt content.
#    https://civitai.com/models/1232765/illustrious-xl-10
#
#  Flux.1 (dev / schnell / pro)
#    - Trained on natural-language captions, not Danbooru tag-style data.
#    - Quality tags have minimal effect; quality comes from descriptive prose prompts.
#    - Use concrete defect terms in negative (e.g. "extra fingers") rather than
#      generic quality tags.
#    https://education.civitai.com/quickstart-guide-to-flux-1/
# ---------------------------------------------------------------------------
BUILTIN_TEMPLATES = {
    # --- Pony ---
    'Pony': {
        'positive_prefix': ['score_9', 'score_8_up', 'score_7_up'],
        'negative_prefix': ['score_1', 'score_2', 'score_3'],
    },

    # --- NoobAI (NoobAI-XL) ---
    'NoobAI': {
        'positive_prefix': ['masterpiece', 'best quality', 'newest', 'absurdres', 'highres'],
        'negative_prefix': ['worst quality', 'old', 'early', 'low quality', 'lowres'],
    },

    # --- Illustrious XL ---
    'Illustrious': {
        'positive_prefix': ['masterpiece', 'best quality'],
        'negative_prefix': ['worst quality', 'low quality', 'lowres'],
    },

    # --- SDXL variants (generic, no dataset-specific quality tags) ---
    'SDXL 1.0': {
        'positive_prefix': ['masterpiece', 'best quality'],
        'negative_prefix': ['worst quality', 'low quality', 'lowres'],
    },
    'SDXL Turbo': {
        'positive_prefix': ['masterpiece', 'best quality'],
        'negative_prefix': ['worst quality', 'low quality', 'lowres'],
    },
    'SDXL Lightning': {
        'positive_prefix': ['masterpiece', 'best quality'],
        'negative_prefix': ['worst quality', 'low quality', 'lowres'],
    },

    # --- SD 1.5 / 2.x ---
    'SD 1.5': {
        'positive_prefix': ['masterpiece', 'best quality'],
        'negative_prefix': ['worst quality', 'low quality', 'lowres'],
    },
    'SD 2.1': {
        'positive_prefix': ['masterpiece', 'best quality'],
        'negative_prefix': ['worst quality', 'low quality', 'lowres'],
    },

    # --- Flux.1 — no quality tags; natural language captions preferred ---
    'Flux.1 D': {
        'positive_prefix': [],
        'negative_prefix': [],
    },
    'Flux.1 S': {
        'positive_prefix': [],
        'negative_prefix': [],
    },
}

# ---------------------------------------------------------------------------
# Helpers: read SHA-256 from a safetensors file
# ---------------------------------------------------------------------------

def _read_safetensors_header(filepath: str) -> dict:
    """Return the parsed JSON header of a safetensors file, or {}."""
    try:
        with open(filepath, 'rb') as f:
            length_bytes = f.read(8)
            if len(length_bytes) < 8:
                return {}
            header_length = struct.unpack('<Q', length_bytes)[0]
            if header_length > 100 * 1024 * 1024:   # sanity: > 100 MB is wrong
                return {}
            header_bytes = f.read(header_length)
            return json.loads(header_bytes.decode('utf-8'))
    except Exception:
        return {}


def get_sha256_from_file(filepath: str) -> str:
    """
    Try to obtain the SHA-256 hash for a checkpoint file.
    Priority:
      1. modelspec.hash_sha256 inside safetensors __metadata__
      2. .sha256 sidecar file (written by Forge / A1111 after first load)
    Returns uppercase hex string or '' if unavailable.
    """
    # 1. safetensors metadata
    if filepath.lower().endswith('.safetensors'):
        header = _read_safetensors_header(filepath)
        meta = header.get('__metadata__', {})
        sha = meta.get('modelspec.hash_sha256', '')
        if sha:
            return sha.upper().strip()

    # 2. sidecar .sha256 file
    sidecar = os.path.splitext(filepath)[0] + '.sha256'
    if os.path.exists(sidecar):
        try:
            return open(sidecar).read().strip().upper()
        except Exception:
            pass

    return ''


# ---------------------------------------------------------------------------
# CivitAI API lookup
# ---------------------------------------------------------------------------

def _civitai_headers(api_key: str = '') -> dict:
    """
    Build request headers for CivitAI API calls.
    Uses a browser-style User-Agent to bypass Cloudflare (error 1010).
    Based on the pattern from sd-civitai-browser-neo/scripts/civitai_api.py.
    """
    headers = {
        'Connection': 'keep-alive',
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/121.0.0.0 Safari/537.36'
        ),
        'Content-Type': 'application/json',
    }
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    return headers


def fetch_base_model_from_civitai(sha256: str, api_key: str = '') -> str:
    """
    Query CivitAI /api/v1/model-versions/by-hash/{sha256}.
    Returns the raw baseModel string (e.g. 'NoobAI', 'Pony') or '' on failure.
    """
    if not sha256 or len(sha256) != 64:
        return ''
    url = f'https://civitai.com/api/v1/model-versions/by-hash/{sha256}'
    try:
        resp = requests.get(url, headers=_civitai_headers(api_key), timeout=(15, 30))
        if resp.status_code == 200:
            data = resp.json()
            if 'error' not in data:
                return data.get('baseModel', '')
    except Exception:
        pass
    return ''


# ---------------------------------------------------------------------------
# Preset storage helpers
# ---------------------------------------------------------------------------

def load_presets() -> dict:
    data = Storage.get(STORAGE_KEY)
    if not isinstance(data, dict):
        data = {}
    data.setdefault('presets', [])
    data.setdefault('builtin_overrides', {})
    data.setdefault('checkpoint_cache', {})
    # Ensure every known family is present in builtin_enabled (default True)
    enabled = data.get('builtin_enabled', {})
    for key in BUILTIN_TEMPLATES:
        enabled.setdefault(key, True)
    data['builtin_enabled'] = enabled
    return data


def save_presets(data: dict) -> None:
    Storage.set(STORAGE_KEY, data)


# ---------------------------------------------------------------------------
# Core detection: given a checkpoint filepath, resolve the matching preset
# or fall back to a built-in template.
# ---------------------------------------------------------------------------

def detect_preset_for_checkpoint(filepath: str) -> dict:
    """
    Examine the checkpoint at `filepath` and return the best-matching quality
    preset as a dict:

        {
            'source':           'civitai' | 'preset' | 'builtin' | 'none',
            'base_model':       str,        # raw CivitAI baseModel value or ''
            'preset_name':      str,        # name of matched user preset or ''
            'positive_prefix':  [str, ...],
            'negative_prefix':  [str, ...],
            'positive_embeddings': [str, ...],
            'negative_embeddings': [str, ...],
            'auto_insert':      bool,
        }

    Detection order:
      1. SHA-256  →  CivitAI by-hash  →  built-in template
      2. Filename substring  →  user preset match_substr
      3. No match → source='none', empty tag lists
    """
    result = {
        'source': 'none',
        'base_model': '',
        'preset_name': '',
        'positive_prefix': [],
        'negative_prefix': [],
        'positive_embeddings': [],
        'negative_embeddings': [],
        'auto_insert': False,
    }

    if not filepath or not os.path.exists(filepath):
        return result

    filename_stem = os.path.splitext(os.path.basename(filepath))[0].lower()

    storage = load_presets()
    user_presets = storage.get('presets', [])
    api_key = _get_civitai_api_key()

    # -- Step 1a: exact filename match in user presets ----------------------
    for preset in user_presets:
        exact_list = [e.lower() for e in preset.get('match_exact', [])]
        if filename_stem in exact_list:
            result.update({
                'source': 'preset',
                'preset_name': preset.get('name', ''),
                'positive_prefix': preset.get('positive_prefix', []),
                'negative_prefix': preset.get('negative_prefix', []),
                'positive_embeddings': preset.get('positive_embeddings', []),
                'negative_embeddings': preset.get('negative_embeddings', []),
                'auto_insert': preset.get('auto_insert', False),
            })
            return result

    # -- Step 1b: CivitAI SHA-256 lookup ------------------------------------
    sha256 = get_sha256_from_file(filepath)
    base_model = ''
    if sha256:
        base_model = fetch_base_model_from_civitai(sha256, api_key)

    if base_model:
        result['base_model'] = base_model
        # Check if any user preset overrides this base model family
        for preset in user_presets:
            substr_list = [s.lower() for s in preset.get('match_substr', [])]
            if any(s in filename_stem for s in substr_list):
                result.update({
                    'source': 'preset',
                    'preset_name': preset.get('name', ''),
                    'positive_prefix': preset.get('positive_prefix', []),
                    'negative_prefix': preset.get('negative_prefix', []),
                    'positive_embeddings': preset.get('positive_embeddings', []),
                    'negative_embeddings': preset.get('negative_embeddings', []),
                    'auto_insert': preset.get('auto_insert', False),
                })
                return result

        # Fall back to built-in template for this baseModel (if enabled)
        enabled = storage.get('builtin_enabled', {})
        if enabled.get(base_model, True):
            overrides = storage.get('builtin_overrides', {})
            template = overrides.get(base_model) or BUILTIN_TEMPLATES.get(base_model)
            if template:
                result.update({
                    'source': 'civitai',
                    'positive_prefix': list(template.get('positive_prefix', [])),
                    'negative_prefix': list(template.get('negative_prefix', [])),
                    'auto_insert': False,   # built-in templates are suggested, not auto-inserted
                })
                return result

    # -- Step 2: filename substring match against user presets --------------
    for preset in user_presets:
        substr_list = [s.lower() for s in preset.get('match_substr', [])]
        if any(s in filename_stem for s in substr_list):
            result.update({
                'source': 'preset',
                'preset_name': preset.get('name', ''),
                'positive_prefix': preset.get('positive_prefix', []),
                'negative_prefix': preset.get('negative_prefix', []),
                'positive_embeddings': preset.get('positive_embeddings', []),
                'negative_embeddings': preset.get('negative_embeddings', []),
                'auto_insert': preset.get('auto_insert', False),
            })
            return result

    # -- No match -----------------------------------------------------------
    return result


# ---------------------------------------------------------------------------
# WebUI integration: installed checkpoints list
# ---------------------------------------------------------------------------

def get_installed_checkpoints() -> list:
    """
    Return a list of dicts for every checkpoint known to the WebUI:
        [
            {
                'filename':   str,   # basename with extension
                'filepath':   str,   # absolute path
                'title':      str,   # display name (model title or filename stem)
                'sha256':     str,   # from sidecar/.safetensors meta, or ''
                'base_model': str,   # from checkpoint_cache (already scanned), or ''
            },
            ...
        ]
    """
    storage = load_presets()
    cache = storage.get('checkpoint_cache', {})

    checkpoints = []
    try:
        from modules import sd_models
        for info in sd_models.checkpoints_list.values():
            filepath = getattr(info, 'filename', '')
            filename = os.path.basename(filepath)
            title    = getattr(info, 'title', '') or os.path.splitext(filename)[0]
            sha256   = get_sha256_from_file(filepath) if filepath else ''
            cached   = cache.get(sha256, {}) if sha256 else {}
            checkpoints.append({
                'filename':   filename,
                'filepath':   filepath,
                'title':      title,
                'sha256':     sha256,
                'base_model': cached.get('base_model', ''),
            })
    except Exception:
        pass
    return checkpoints


def scan_checkpoint(filepath: str) -> dict:
    """
    Query CivitAI for a single checkpoint and update checkpoint_cache.
    Returns {'filename', 'sha256', 'base_model'} — empty strings on failure.
    """
    storage = load_presets()
    api_key = _get_civitai_api_key()

    sha256 = get_sha256_from_file(filepath)
    if not sha256:
        return {'filename': os.path.basename(filepath), 'sha256': '', 'base_model': ''}

    base_model = fetch_base_model_from_civitai(sha256, api_key)

    import time
    cache = storage.get('checkpoint_cache', {})
    cache[sha256] = {
        'base_model':  base_model,
        'filename':    os.path.basename(filepath),
        'scanned_at':  int(time.time()),
    }
    storage['checkpoint_cache'] = cache
    save_presets(storage)

    return {
        'filename':   os.path.basename(filepath),
        'sha256':     sha256,
        'base_model': base_model,
    }


# ---------------------------------------------------------------------------
# WebUI integration: resolve the currently loaded checkpoint path
# ---------------------------------------------------------------------------

def get_current_checkpoint_path() -> str:
    """
    Return the absolute path to the currently loaded checkpoint, or ''.
    Works with Forge Neo / A1111 / SD.Next.
    """
    try:
        from modules import shared, sd_models
        name = getattr(shared.opts, 'sd_model_checkpoint', '')
        if not name:
            return ''
        info = sd_models.get_closet_checkpoint_match(name)
        if info and hasattr(info, 'filename'):
            return info.filename
    except Exception:
        pass
    return ''
