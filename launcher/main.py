import sys
import os
import traceback
from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog, QVBoxLayout, QTextEdit, QPushButton, QLabel
from PyQt6.QtCore import Qt
from gui.main_window import MainWindow


class ErrorDialog(QDialog):
    """详细错误对话框"""
    def __init__(self, title, message, details):
        super().__init__()
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)
        self._build_ui(message, details)
    
    def _build_ui(self, message, details):
        layout = QVBoxLayout(self)
        
        lbl_msg = QLabel(message)
        lbl_msg.setStyleSheet("font-size: 14px; color: #dc2626; font-weight: bold;")
        lbl_msg.setWordWrap(True)
        layout.addWidget(lbl_msg)
        
        lbl_details = QLabel("详细信息：")
        lbl_details.setStyleSheet("font-size: 12px; color: #374151; margin-top: 10px;")
        layout.addWidget(lbl_details)
        
        txt_details = QTextEdit()
        txt_details.setReadOnly(True)
        txt_details.setPlainText(details)
        txt_details.setStyleSheet("""
            QTextEdit {
                background-color: #f3f4f6;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 8px;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(txt_details)
        
        btn_ok = QPushButton("确定")
        btn_ok.clicked.connect(self.accept)
        btn_ok.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2563eb; }
        """)
        layout.addWidget(btn_ok, alignment=Qt.AlignmentFlag.AlignCenter)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        tb_str = traceback.format_exc()
        
        print("=" * 80)
        print("启动器崩溃！")
        print("=" * 80)
        print(tb_str)
        print("=" * 80)
        
        # 尝试显示错误对话框
        try:
            msg = f"启动器发生错误，无法正常启动：\n\n{str(e)}\n\n请将详细信息发送给开发者排查问题。"
            dlg = ErrorDialog("启动器错误", msg, tb_str)
            dlg.exec()
        except Exception:
            # 如果连对话框都显示不了，至少显示一个简单的MessageBox
            try:
                QMessageBox.critical(
                    None, 
                    "启动器错误", 
                    f"启动器崩溃！\n\n{str(e)}\n\n{tb_str}"
                )
            except Exception:
                pass
        
        sys.exit(1)


if __name__ == '__main__':
    main()
