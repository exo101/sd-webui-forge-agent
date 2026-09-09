# coding: utf-8

from ... import shared

class LinuxMoveResize:
    """ Tool class for moving and resizing window """

    @classmethod
    def startSystemMove(cls, window, globalPos):
        """ move window """
        window.windowHandle().startSystemMove()

    @classmethod
    def starSystemResize(cls, window, globalPos, edges):
        """ resize window

        Parameters
        ----------
        window: QWidget
            window

        globalPos: QPoint
            the global point of mouse release event

        edges: `Qt.Edges`
            window edges
        """
        window.windowHandle().startSystemResize(edges)

    @classmethod
    def toggleMaxState(cls, window):
        if shared.HEADLESS:
            return
        if window.isMaximized():
            window.showNormal()
        else:
            window.showMaximized()