from qtpy.QtWidgets import QHBoxLayout, QComboBox, QTextEdit, QLabel, QPlainTextEdit, QCheckBox, QVBoxLayout,  QGraphicsDropShadowEffect, QWidget
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QKeyEvent, QTextCursor, QHideEvent, QInputMethodEvent, QFontMetrics, QColor, QShowEvent, QSyntaxHighlighter, QTextCharFormat

from typing import List, Union, Tuple, Dict
import re

from .ui_config import pcfg
from .widget import Widget, ClickableLabel
from .proj import ProjSeg
from live2d.scrap_model import Live2DScrapModel
# from .textitem import TextBlkItem
# from .textedit_area import TransPairWidget, SourceTextEdit, TransTextEdit

SEARCHRST_HIGHLIGHT_COLOR = QColor(30, 147, 229, 60)
CURRENT_TEXT_COLOR = QColor(244, 249, 28)


class Matched:
    def __init__(self, local_no: int, start: int, end: int) -> None:
        self.local_no = local_no
        self.start = start
        self.end = end


def match_text(pattern: re.Pattern, text: str) -> Tuple[int, Dict]:
    found_counter = 0
    match_map = {}
    rst_iter = pattern.finditer(text)
    for rst in rst_iter:
        span = rst.span()
        match_map[span[1]] = Matched(found_counter, span[0], span[1])
        found_counter += 1
    return found_counter, match_map


class SearchEditor(QPlainTextEdit):
    height_changed = Signal()
    commit = Signal()
    enter_pressed = Signal()
    shift_enter_pressed = Signal()
    def __init__(self, parent: QWidget = None, original_height: int = 32, commit_latency: int = -1, shift_enter_prev: bool = True, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.original_height = original_height
        self.commit_latency = commit_latency
        self.shift_enter_prev = shift_enter_prev
        if commit_latency > 0:
            self.commit_timer = QTimer(self)
            self.commit_timer.timeout.connect(self.on_commit_timer_timeout)
        else:
            self.commit_timer = None
        self.pre_editing = False
        self.setFixedHeight(original_height)
        self.document().documentLayout().documentSizeChanged.connect(self.adjustSize)
        self.textChanged.connect(self.on_text_changed)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)

    def adjustSize(self):
        fm = QFontMetrics(self.font())
        h = fm.height() * self.document().size().height() * 1.05
        h += self.document().documentMargin() * 2
        h = int(h)
        if self.geometry().height() != h:
            self.setFixedHeight(max(h, self.original_height))
            self.height_changed.emit()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Return:
            if self.commit_timer is not None:
                self.commit_timer.stop()
            if e.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                if self.shift_enter_prev:
                    e.setAccepted(True)
                    self.shift_enter_pressed.emit()
                    return
            else:
                e.setAccepted(True)
                self.enter_pressed.emit()
                return
        return super().keyPressEvent(e)

    def on_text_changed(self):
        if self.commit_timer is not None:
            if not self.pre_editing:
                self.commit_timer.stop()
                self.commit_timer.start(self.commit_latency)
        elif not self.pre_editing:
            self.commit.emit()

    def on_commit_timer_timeout(self):
        self.commit_timer.stop()
        self.commit.emit()

    def hideEvent(self, e: QHideEvent) -> None:
        if self.commit_timer is not None:
            self.commit_timer.stop()
        return super().hideEvent(e)

    def inputMethodEvent(self, e: QInputMethodEvent) -> None:
        if e.preeditString() == '':
            self.pre_editing = False
            if self.commit_timer is not None:
                self.commit_timer.start(self.commit_latency)
        else:
            if self.commit_timer is not None:
                self.commit_timer.stop()
            self.pre_editing = True
        return super().inputMethodEvent(e)


class PageSearchWidget(Widget):

    search = Signal()

    def __init__(self, parent: QWidget = None, *args, **kwargs) -> None:
        super().__init__(parent)

        self.search_counter_list: List[int] = []
        self.counter_sum = 0

        self.current_cursor: QTextCursor = None
        self.result_pos = 0
        self.update_cursor_on_insert = True

        self.search_editor = SearchEditor(self, commit_latency=-1)
        self.search_editor.setPlaceholderText(self.tr('Find'))
        self.search_editor.height_changed.connect(self.on_editor_height_changed)
        
        self.no_result_str = self.tr('No result')
        # self.result_counter_label = QLabel(self.no_result_str)
        # self.result_counter_label.setMaximumHeight(32)
        # self.prev_match_btn = ClickableLabel(None, self)
        # self.prev_match_btn.setObjectName('PrevMatchBtn')
        # self.prev_match_btn.clicked.connect(self.on_prev_search_result)
        # self.prev_match_btn.setToolTip(self.tr('Previous Match (Shift+Enter)'))

        # self.next_match_btn = ClickableLabel(None, self)
        # self.next_match_btn.setObjectName('NextMatchBtn')
        # self.next_match_btn.clicked.connect(self.on_next_search_result)
        # self.next_match_btn.setToolTip(self.tr('Next Match (Enter)'))

        self.case_sensitive_toggle = QCheckBox(self)
        self.case_sensitive_toggle.setObjectName('CaseSensitiveToggle')
        self.case_sensitive_toggle.setToolTip(self.tr('Match Case'))
        self.case_sensitive_toggle.clicked.connect(self.on_case_clicked)

        self.whole_word_toggle = QCheckBox(self)
        self.whole_word_toggle.setObjectName('WholeWordToggle')
        self.whole_word_toggle.setToolTip(self.tr('Match Whole Word'))
        self.whole_word_toggle.clicked.connect(self.on_whole_word_clicked)

        self.regex_toggle = QCheckBox(self)
        self.regex_toggle.setObjectName('RegexToggle')
        self.regex_toggle.setToolTip(self.tr('Use Regular Expression'))
        self.regex_toggle.clicked.connect(self.on_regex_clicked)

        hlayout_bar1_0 = QHBoxLayout()
        hlayout_bar1_0.addWidget(self.search_editor)
        # hlayout_bar1_0.addWidget(self.result_counter_label)
        hlayout_bar1_0.setAlignment(Qt.AlignmentFlag.AlignTop)
        hlayout_bar1_0.setSpacing(10)

        hlayout_bar1_1 = QHBoxLayout()
        hlayout_bar1_1.addWidget(self.case_sensitive_toggle)
        hlayout_bar1_1.addWidget(self.whole_word_toggle)
        hlayout_bar1_1.addWidget(self.regex_toggle)
        # hlayout_bar1_1.addWidget(self.prev_match_btn)
        # hlayout_bar1_1.addWidget(self.next_match_btn)
        hlayout_bar1_1.setAlignment(hlayout_bar1_1.alignment() | Qt.AlignmentFlag.AlignTop)
        hlayout_bar1_1.setSpacing(5)

        hlayout_bar1 = QHBoxLayout()
        hlayout_bar1.addLayout(hlayout_bar1_0)
        hlayout_bar1.addLayout(hlayout_bar1_1)

        vlayout = QVBoxLayout(self)
        vlayout.addLayout(hlayout_bar1)
        # vlayout.addLayout(hlayout_bar2)

        self.search_editor.commit.connect(self.search)
        self.close_btn = ClickableLabel(None, self)
        self.close_btn.setObjectName('SearchCloseBtn')
        self.close_btn.setToolTip(self.tr('Close (Escape)'))
        self.close_btn.clicked.connect(self.on_close_button_clicked)
        hlayout_bar1_1.addWidget(self.close_btn)
        e = QGraphicsDropShadowEffect(self)
        e.setOffset(0, 0)
        e.setBlurRadius(35)
        self.setGraphicsEffect(e)
        self.setFixedWidth(360)
        self.search_editor.setFixedWidth(200)
        self.search_editor.enter_pressed.connect(self.search)

        self.adjustSize()
        

    def on_close_button_clicked(self):
        self.hide()

    def showEvent(self, e: QShowEvent) -> None:
        self.focus_to_editor()

        # text = self.search_editor.toPlainText()
        # if text != '':
        #     self.on_commit_search()
        return super().showEvent(e)
    
    def focus_to_editor(self):
        self.search_editor.setFocus()
        cursor = self.search_editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        self.search_editor.setTextCursor(cursor)

    def on_editor_height_changed(self):
        self.adjustSize()

    def adjustSize(self) -> None:
        tgt_size = self.search_editor.height() + 15
        self.setFixedHeight(tgt_size)

    def clearSearchResult(self):
        self.search_counter_list.clear()
        self.updateCounterText()

    def reInitialize(self):
        self.clearSearchResult()

    def get_regex_pattern(self) -> re.Pattern:
        target_text = self.search_editor.toPlainText()
        regexr = target_text
        if target_text == '':
            return None

        flag = re.DOTALL
        if not self.case_sensitive_toggle.isChecked():
            flag |= re.IGNORECASE
        if not self.regex_toggle.isChecked():
            regexr = re.escape(regexr)
        if self.whole_word_toggle.isChecked():
            regexr = r'\b' + target_text + r'\b'

        return re.compile(regexr, flag)

    def _match_text(self, text: str) -> Tuple[int, Dict]:
        try:
            return match_text(self.get_regex_pattern(), text)
        except re.error:
            return 0, {}

    def updateCounterText(self):
        pass

    def clean_current_selection(self):
        cursor = self.current_edit.textCursor()
        if cursor.hasSelection():
            cursor.clearSelection()
            self.current_edit.setTextCursor(cursor)

    def on_next_search_result(self):
        if self.current_cursor is None:
            return
        move = self.move_cursor(1)
        if move == 0:
            self.result_pos = min(self.result_pos + 1, self.counter_sum - 1)
        else:
            self.result_pos = 0
        self.updateCounterText()

    def on_prev_search_result(self):
        if self.current_cursor is None:
            return
        move = self.move_cursor(-1)
        if move == 0:
            self.result_pos = max(self.result_pos - 1, 0)
        else:
            self.result_pos = self.counter_sum - 1
        self.updateCounterText()

    def on_whole_word_clicked(self):
        pcfg.fsearch_whole_word = self.whole_word_toggle.isChecked()
        self.search.emit()

    def on_regex_clicked(self):
        pcfg.fsearch_regex = self.regex_toggle.isChecked()
        self.search.emit()

    def on_case_clicked(self):
        pcfg.fsearch_case = self.case_sensitive_toggle.isChecked()
        self.search.emit()

    def on_commit_search(self):
        self.search.emit()

