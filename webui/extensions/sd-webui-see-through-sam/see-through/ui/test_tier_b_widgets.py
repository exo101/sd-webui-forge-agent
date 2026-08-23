"""Tier B — Offscreen widget construction test.

Creates QApplication with offscreen platform, instantiates MainWindow,
and loads a real project to validate the data model works.
"""

import sys
import os

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_API'] = 'pyqt6'

# Set up paths (same as launch.py)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
ui_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ui_dir)

import traceback

results = {'pass': [], 'fail': []}

def record(label, success, error=None):
    if success:
        results['pass'].append(label)
        print(f"  [PASS] {label}")
    else:
        results['fail'].append((label, str(error)))
        print(f"  [FAIL] {label}: {error}")

# ── Step 1: Create QApplication ──

print("\n=== Step 1: QApplication Creation ===\n")

try:
    from qtpy.QtWidgets import QApplication
    from qtpy.QtGui import QGuiApplication
    from ui import shared
    from ui.logger import setup_logging
    from ui import ui_config as program_config

    setup_logging(shared.LOGGING_PATH)
    program_config.load_config()
    config = program_config.pcfg

    app = QApplication(sys.argv)
    app.setApplicationName('Live2D Parsing Test')

    ps = QGuiApplication.primaryScreen()
    if ps is not None:
        shared.LDPI = ps.logicalDotsPerInch()
        shared.SCREEN_W = ps.geometry().width()
        shared.SCREEN_H = ps.geometry().height()
    else:
        # offscreen may not have a primary screen
        shared.LDPI = 96.0
        shared.SCREEN_W = 1920
        shared.SCREEN_H = 1080

    record("QApplication created", True)
    record(f"Screen: {shared.SCREEN_W}x{shared.SCREEN_H}, DPI={shared.LDPI}", True)
except Exception as e:
    record("QApplication created", False, e)
    traceback.print_exc()
    sys.exit(1)

# ── Step 2: Instantiate MainWindow ──

print("\n=== Step 2: MainWindow Instantiation ===\n")

try:
    from ui.mainwindow import MainWindow
    mainwindow = MainWindow(app, config)
    record("MainWindow instantiated", True)
except Exception as e:
    record("MainWindow instantiated", False, e)
    traceback.print_exc()
    sys.exit(1)

# ── Step 3: Load real project ──

print("\n=== Step 3: Project Loading ===\n")

# Use the annotated dataset as test data
test_proj_dir = os.path.join(
    repo_root,
    'workspace', 'datasets', 'model_pack2_jxl_crop_annotated'
)
test_proj_dir = os.path.abspath(test_proj_dir)
exec_list = os.path.join(test_proj_dir, 'exec_list.txt')

if os.path.exists(exec_list):
    record(f"Test project found: {os.path.basename(test_proj_dir)}", True)
else:
    record(f"Test project found at {exec_list}", False, "exec_list.txt not found")
    sys.exit(1)

try:
    mainwindow.openProj(exec_list)
    record("Project loaded via openProj()", True)
except Exception as e:
    record("Project loaded via openProj()", False, e)
    traceback.print_exc()

# ── Step 4: Validate project data model ──

print("\n=== Step 4: Data Model Validation ===\n")

proj = mainwindow.proj

try:
    page_count = len(proj.pages)
    record(f"Pages loaded: {page_count}", page_count > 0,
           f"Expected >0 pages, got {page_count}" if page_count == 0 else None)
except Exception as e:
    record("Pages dict accessible", False, e)

try:
    page_names = list(proj.pages.keys())
    record(f"First page: {page_names[0]}", True)
    record(f"Last page: {page_names[-1]}", True)
except Exception as e:
    record("Page names accessible", False, e)

try:
    current = proj.current_model
    record(f"Current model set: {current}", current is not None,
           "current_model is None" if current is None else None)
except Exception as e:
    record("Current model accessible", False, e)

try:
    l2d = proj.l2dmodel
    record(f"Live2DScrapModel loaded: {l2d is not None}", l2d is not None,
           "l2dmodel is None" if l2d is None else None)
    if l2d is not None:
        drawable_count = len(l2d.drawable_list) if hasattr(l2d, 'drawable_list') else -1
        if drawable_count >= 0:
            record(f"Drawable count: {drawable_count}", drawable_count > 0)
except Exception as e:
    record("Live2DScrapModel accessible", False, e)

# ── Step 5: Clean exit ──

print("\n=== Step 5: Clean Exit ===\n")

try:
    mainwindow.close()
    app.quit()
    record("Clean shutdown", True)
except Exception as e:
    record("Clean shutdown", False, e)

# ── Summary ──

print(f"\n{'='*50}")
print(f"TOTAL: {len(results['pass'])} passed, {len(results['fail'])} failed")
print(f"{'='*50}")

if results['fail']:
    print("\nFailed checks:")
    for name, err in results['fail']:
        print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("\nAll widget construction checks passed!")
    sys.exit(0)
