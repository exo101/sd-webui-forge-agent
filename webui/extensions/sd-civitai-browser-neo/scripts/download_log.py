"""
Download Queue Log  —  neo_download_queue.jsonl

Persists the download queue state so that interrupted sessions
(e.g. RunPod disconnects) can be detected and restored on the next page load.

Entry lifecycle:
  queued  →  downloading  →  completed
                           ↘  cancelled   (user cancelled)
                           ↘  failed      (network / API error)
                           ↘  dismissed   (user dismissed the restore banner)
"""

import json
import os
from datetime import datetime, timedelta

_LOG_FILE = None  # Resolved lazily to survive module-level import order


def _get_log_path():
    global _LOG_FILE
    if _LOG_FILE is None:
        config_folder = os.path.join(os.getcwd(), 'config_states')
        os.makedirs(config_folder, exist_ok=True)
        _LOG_FILE = os.path.join(config_folder, 'neo_download_queue.jsonl')
    return _LOG_FILE


def _read_all():
    path = _get_log_path()
    if not os.path.exists(path):
        return []
    entries = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        return []
    return entries


def _write_all(entries):
    path = _get_log_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except OSError:
        pass


def _now():
    return datetime.utcnow().isoformat()


# ─── Write helpers ────────────────────────────────────────────────────────────

def log_queued(item):
    """Record a newly enqueued download item."""
    entries = _read_all()
    # Skip duplicates for the same URL that are still active
    for e in entries:
        if e.get('dl_url') == item.get('dl_url') and e.get('status') in ('queued', 'downloading'):
            return
    entry = {
        'dl_id':          item['dl_id'],
        'dl_url':         item['dl_url'],
        'model_filename': item['model_filename'],
        'install_path':   item['install_path'],
        'model_name':     item['model_name'],
        'version_name':   item.get('version_name', ''),
        'model_sha256':   item.get('model_sha256'),
        'model_id':       item['model_id'],
        'create_json':    item.get('create_json', True),
        'from_batch':     item.get('from_batch', False),
        'sub_folder':     item.get('sub_folder', ''),
        'status':         'queued',
        'queued_at':      _now(),
        'updated_at':     _now(),
    }
    entries.append(entry)
    _write_all(entries)


def _update_status(dl_id, status):
    entries = _read_all()
    changed = False
    for e in entries:
        if e.get('dl_id') == dl_id and e.get('status') not in ('completed', 'cancelled', 'dismissed'):
            e['status'] = status
            e['updated_at'] = _now()
            changed = True
            break
    if changed:
        _write_all(entries)


def log_downloading(dl_id):
    _update_status(dl_id, 'downloading')


def log_completed(dl_id):
    _update_status(dl_id, 'completed')


def log_cancelled(dl_id):
    _update_status(dl_id, 'cancelled')


def log_failed(dl_id):
    _update_status(dl_id, 'failed')


def log_all_cancelled():
    """Mark every queued/downloading entry as cancelled (Cancel All was pressed)."""
    entries = _read_all()
    changed = False
    for e in entries:
        if e.get('status') in ('queued', 'downloading'):
            e['status'] = 'cancelled'
            e['updated_at'] = _now()
            changed = True
    if changed:
        _write_all(entries)


# ─── Read helpers ─────────────────────────────────────────────────────────────

def get_interrupted():
    """Return entries whose status is still queued, downloading, or failed.
    'failed' is included so that items which exhausted their automatic retries
    can be re-attempted manually via the restore banner."""
    return [e for e in _read_all() if e.get('status') in ('queued', 'downloading', 'failed')]


def dismiss_interrupted():
    """Mark all interrupted (and failed) entries as dismissed so they are ignored on next reload."""
    entries = _read_all()
    changed = False
    for e in entries:
        if e.get('status') in ('queued', 'downloading', 'failed'):
            e['status'] = 'dismissed'
            e['updated_at'] = _now()
            changed = True
    if changed:
        _write_all(entries)


def purge_old_entries(days=7):
    """Remove resolved entries (completed / cancelled / dismissed / failed)
    older than *days* days to keep the log file small."""
    entries = _read_all()
    cutoff = datetime.utcnow() - timedelta(days=days)
    keep = []
    for e in entries:
        if e.get('status') in ('queued', 'downloading'):
            keep.append(e)  # never purge active entries
        else:
            try:
                updated = datetime.fromisoformat(e.get('updated_at', ''))
                if updated >= cutoff:
                    keep.append(e)
            except (ValueError, TypeError):
                keep.append(e)
    if len(keep) != len(entries):
        _write_all(keep)
