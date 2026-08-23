import sys
import os
import os.path as osp
import argparse

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

os.environ['QT_API'] = 'pyqt6'

parser = argparse.ArgumentParser()
parser.add_argument("--proj", default='', type=str, help='Open project directory on startup')
args, _ = parser.parse_known_args()


MAIN_WINDOW = None
APP = None

def restart():
    global MAIN_WINDOW
    print('restarting...\n')
    if MAIN_WINDOW:
        MAIN_WINDOW.close()
    os.execv(sys.executable, ['python'] + sys.argv)


def main():

    from qtpy.QtWidgets import QApplication
    from qtpy.QtGui import QGuiApplication

    from ui import shared
    from ui.logger import setup_logging, logger as LOGGER
    from ui import ui_config as program_config

    # os.chdir(shared.PROGRAM_PATH)
    setup_logging(shared.LOGGING_PATH)

    program_config.load_config()
    config = program_config.pcfg

    app_args = sys.argv
    app = QApplication(app_args)
    app.setApplicationName('Live2D Parsing')

    ps = QGuiApplication.primaryScreen()
    shared.LDPI = ps.logicalDotsPerInch()
    shared.SCREEN_W = ps.geometry().width()
    shared.SCREEN_H = ps.geometry().height()

    from ui.mainwindow import MainWindow
    mainwindow = MainWindow(app, config)
    global MAIN_WINDOW
    MAIN_WINDOW = mainwindow
    mainwindow.restart_signal.connect(restart)

    mainwindow.show()

    if args.proj is not None and osp.exists(args.proj):
        mainwindow.openProj(args.proj)

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
