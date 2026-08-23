"""Tier A — Import validation for UI sub-codebase.

Tests all third-party packages and UI modules import without error.
Must run with QT_QPA_PLATFORM=offscreen for Qt-dependent imports.
"""

import sys
import os
import importlib
import traceback

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_API'] = 'pyqt6'

# Ensure repo root is on the path for common/ imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
# Ensure ui/ is on the path for ui.ui imports
ui_dir = os.path.dirname(os.path.abspath(__file__))
if ui_dir not in sys.path:
    sys.path.insert(0, ui_dir)

results = {'pass': [], 'fail': []}

def test_import(module_name, label=None):
    label = label or module_name
    try:
        importlib.import_module(module_name)
        results['pass'].append(label)
        print(f"  [PASS] {label}")
    except Exception as e:
        results['fail'].append((label, str(e)))
        print(f"  [FAIL] {label}: {e}")
        traceback.print_exc(limit=2)

# ── Section 1: Third-party packages ──

print("\n=== Third-party Package Imports ===\n")

packages = [
    ('PyQt6.QtWidgets', 'PyQt6'),
    ('qtpy', 'qtpy'),
    ('cv2', 'opencv-python'),
    ('PIL', 'pillow'),
    ('PIL.Image', 'pillow (Image)'),
    ('pillow_jxl', 'pillow-jxl-plugin'),
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('matplotlib', 'matplotlib'),
    ('matplotlib.pyplot', 'matplotlib.pyplot'),
    ('huggingface_hub', 'huggingface-hub'),
    ('simple_parsing', 'simple-parsing'),
    ('termcolor', 'termcolor'),
    ('colorama', 'colorama'),
    ('yaml', 'pyyaml'),
    ('natsort', 'natsort'),
    ('py7zr', 'py7zr'),
    ('einops', 'einops'),
    ('pycocotools', 'pycocotools'),
]

for mod, label in packages:
    test_import(mod, label)

# ── Section 2: Common (shared) modules ──

print("\n=== Common Module Imports ===\n")

common_modules = [
    'utils.io_utils',
    'utils.cv',
    'utils.package',
]

for mod in common_modules:
    test_import(mod)

# utils.visualize may need torch — test separately
try:
    importlib.import_module('utils.visualize')
    results['pass'].append('utils.visualize')
    print(f"  [PASS] utils.visualize")
except ImportError as e:
    if 'torch' in str(e):
        results['pass'].append('utils.visualize (skipped - torch not in Tier 0)')
        print(f"  [SKIP] utils.visualize (requires torch, not in Tier 0)")
    else:
        results['fail'].append(('utils.visualize', str(e)))
        print(f"  [FAIL] utils.visualize: {e}")

# ── Section 3: UI modules ──

print("\n=== UI Module Imports ===\n")

# Core modules (no heavy deps beyond Qt + common)
ui_modules = [
    'ui.shared',
    'ui.structures',
    'ui.logger',
    'ui.ui_config',
    'ui.cursor',
    'ui.misc',
    'ui.label',
    'ui.slider',
    'ui.combobox',
    'ui.checkbox',
    'ui.lineedit',
    'ui.scrollbar',
    'ui.message',
    'ui.search_widget',
    'ui.shared_widget',
    'ui.widget',
    'ui.top_area',
    'ui.commands',
    'ui.drawable_item',
    'ui.instance_preview',
    'ui.tag_tree',
    'ui.proj',
    'ui.canvas',
    'ui.io_thread',
    'ui.mainwindowbars',
    'ui.mainwindow',
    'ui.launch',
]

for mod in ui_modules:
    test_import(mod)

# Modules with annotator deps (expected to fail gracefully or at import time)
print("\n=== UI Modules with Annotator Dependencies ===\n")
test_import('ui.run_thread')

# Frameless window modules
print("\n=== Frameless Window Modules ===\n")
fw_modules = [
    'ui.framelesswindow',
    'ui.framelesswindow.fw_qt6',
    'ui.framelesswindow.fw_qt6.linux',
    'ui.framelesswindow.fw_qt6.linux.window_effect',
]
for mod in fw_modules:
    test_import(mod)

# ── Summary ──

print(f"\n{'='*50}")
print(f"TOTAL: {len(results['pass'])} passed, {len(results['fail'])} failed")
print(f"{'='*50}")

if results['fail']:
    print("\nFailed imports:")
    for name, err in results['fail']:
        print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("\nAll imports passed!")
    sys.exit(0)
