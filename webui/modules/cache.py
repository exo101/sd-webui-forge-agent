import threading
import diskcache
import os

from modules.paths import data_path

cache_lock = threading.Lock()
cache_dir = os.environ.get("SD_WEBUI_CACHE_DIR", os.path.join(data_path, "cache"))
caches = {}


dump_cache = lambda: None
"""does nothing since diskcache"""


def cache(subsection):
    """
    Retrieves or initializes a cache for a specific subsection.

    Parameters:
        subsection (str): The subsection identifier for the cache.

    Returns:
        diskcache.Cache: The cache data for the specified subsection.
    """

    cache_obj = caches.get(subsection)
    if not cache_obj:
        with cache_lock:
            cache_obj = caches.get(subsection)
            if not cache_obj:
                cache_obj = diskcache.Cache(os.path.join(cache_dir, subsection))
                caches[subsection] = cache_obj

    return cache_obj


def cached_data_for_file(subsection, title, filename, func):
    """
    Retrieves or generates data for a specific file, using a caching mechanism.

    Parameters:
        subsection (str): The subsection of the cache to use.
        title (str): The title of the data entry in the subsection of the cache.
        filename (str): The path to the file to be checked for modifications.
        func (callable): A function that generates the data if it is not available in the cache.

    Returns:
        dict or None: The cached or generated data, or None if data generation fails.

    The function implements a caching mechanism for data stored in files.
    It checks if the data associated with the given `title` is present in the cache and compares the
    modification time of the file with the cached modification time. If the file has been modified,
    the cache is considered invalid and the data is regenerated using the provided `func`.
    Otherwise, the cached data is returned.

    If the data generation fails, None is returned to indicate the failure. Otherwise, the generated
    or cached data is returned as a dictionary.
    """

    existing_cache = cache(subsection)
    ondisk_mtime = os.path.getmtime(filename)

    entry = existing_cache.get(title)
    if entry:
        cached_mtime = entry.get("mtime", 0)
        if ondisk_mtime > cached_mtime:
            entry = None

    if not entry or "value" not in entry:
        value = func()
        if value is None:
            return None

        entry = {"mtime": ondisk_mtime, "value": value}
        existing_cache[title] = entry

        dump_cache()

    return entry["value"]
