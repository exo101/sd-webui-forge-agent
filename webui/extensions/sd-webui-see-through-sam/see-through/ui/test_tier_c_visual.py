"""Tier C — Xvfb visual validation.

Launches the full UI under xvfb-run, takes screenshots of:
1. Empty state (MainWindow shown, no project)
2. Project loaded (model_pack2_jxl_crop_annotated via exec_list)

Screenshots saved to ui/test_screenshots/
Must be run via: xvfb-run -a -s "-screen 0 1920x1080x24" python ui/test_tier_c_visual.py
"""

import sys
import os
import time

os.environ['QT_API'] = 'pyqt6'

# Set up paths
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
ui_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ui_dir)

# Screenshot output directory
screenshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_screenshots')
os.makedirs(screenshot_dir, exist_ok=True)

results = {'pass': [], 'fail': []}

def record(label, success, error=None):
    if success:
        results['pass'].append(label)
        print(f"  [PASS] {label}")
    else:
        results['fail'].append((label, str(error)))
        print(f"  [FAIL] {label}: {error}")

def take_screenshot(widget, filename, label):
    """Capture a screenshot of the widget using Qt's grab()."""
    filepath = os.path.join(screenshot_dir, filename)
    try:
        # Process pending events to ensure rendering is complete
        app.processEvents()
        time.sleep(0.5)
        app.processEvents()

        # Use widget.grab() - captures the widget as a QPixmap
        pixmap = widget.grab()
        if pixmap.isNull():
            record(label, False, "grab() returned null pixmap")
            return False

        saved = pixmap.save(filepath, 'PNG')
        if saved:
            size = os.path.getsize(filepath)
            record(f"{label} ({pixmap.width()}x{pixmap.height()}, {size//1024}KB)", True)
            return True
        else:
            record(label, False, "pixmap.save() returned False")
            return False
    except Exception as e:
        record(label, False, e)
        return False


# ── Step 1: Create QApplication with real display ──

print("\n=== Step 1: QApplication (Xvfb display) ===\n")

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
    app.setApplicationName('Live2D Parsing Visual Test')

    ps = QGuiApplication.primaryScreen()
    if ps is not None:
        shared.LDPI = ps.logicalDotsPerInch()
        shared.SCREEN_W = ps.geometry().width()
        shared.SCREEN_H = ps.geometry().height()
        record(f"Xvfb display: {shared.SCREEN_W}x{shared.SCREEN_H}, DPI={shared.LDPI}", True)
    else:
        record("Primary screen detected", False, "No primary screen - is Xvfb running?")
        sys.exit(1)
except Exception as e:
    record("QApplication (Xvfb)", False, e)
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Step 2: Create and show MainWindow ──

print("\n=== Step 2: MainWindow (empty state) ===\n")

try:
    from ui.mainwindow import MainWindow
    mainwindow = MainWindow(app, config)
    mainwindow.show()
    mainwindow.resize(1600, 900)

    # Let Qt render the window
    app.processEvents()
    time.sleep(1)
    app.processEvents()

    record("MainWindow shown", True)
except Exception as e:
    record("MainWindow shown", False, e)
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Screenshot 1: Empty state ──

print("\n=== Screenshot 1: Empty State ===\n")
take_screenshot(mainwindow, 'screenshot_01_empty.png', 'Screenshot 1 (empty state)')

# ── Step 3: Load project ──

print("\n=== Step 3: Load Project ===\n")

test_proj_dir = os.path.join(
    repo_root,
    'workspace', 'datasets', 'model_pack2_jxl_crop_annotated'
)
test_proj_dir = os.path.abspath(test_proj_dir)
exec_list = os.path.join(test_proj_dir, 'exec_list.txt')

try:
    mainwindow.openProj(exec_list)
    # Let the canvas render
    app.processEvents()
    time.sleep(2)
    app.processEvents()
    record(f"Project loaded: {len(mainwindow.proj.pages)} pages", True)
except Exception as e:
    record("Project loaded", False, e)
    import traceback; traceback.print_exc()

# ── Screenshot 2: Project loaded ──

print("\n=== Screenshot 2: Project Loaded ===\n")
take_screenshot(mainwindow, 'screenshot_02_project.png', 'Screenshot 2 (project loaded)')

# ── Step 4: Navigate to a different page for variety ──

print("\n=== Step 4: Page Navigation ===\n")

try:
    pages = list(mainwindow.proj.pages.keys())
    if len(pages) > 5:
        target_page = pages[5]
        mainwindow.proj.set_current_page(target_page)
        mainwindow.updatePageList()
        app.processEvents()
        time.sleep(1)
        app.processEvents()
        record(f"Navigated to page: {target_page}", True)
        take_screenshot(mainwindow, 'screenshot_03_page_nav.png', 'Screenshot 3 (page navigation)')
    else:
        record("Page navigation (skipped - not enough pages)", True)
except Exception as e:
    record("Page navigation", False, e)
    import traceback; traceback.print_exc()

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

# List screenshots
print(f"\nScreenshots saved to: {screenshot_dir}")
for f in sorted(os.listdir(screenshot_dir)):
    if f.endswith('.png'):
        size = os.path.getsize(os.path.join(screenshot_dir, f))
        print(f"  {f} ({size//1024}KB)")

sys.exit(1 if results['fail'] else 0)
