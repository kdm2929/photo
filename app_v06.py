from __future__ import annotations

import os
import sys

os.environ.setdefault('QT_ENABLE_HIGHDPI_SCALING', '1')

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from main import Main as BaseMain, self_test as base_self_test
from storage import APP, VERSION
from ui_pro import install_pro_ui


class Main(BaseMain):
    def __init__(self):
        super().__init__()
        install_pro_ui(self)
        self.setWindowTitle(f'PhotoRef Sorter {VERSION} Pro')

    def refresh_home(self):
        super().refresh_home()
        if hasattr(self, '_pro_ui'):
            self._pro_ui.refresh_status()

    def refresh_gallery(self):
        super().refresh_gallery()
        if hasattr(self, '_pro_ui'):
            self._pro_ui.refresh_status()

    def scan_done(self, stats):
        super().scan_done(stats)
        if hasattr(self, '_pro_ui'):
            self._pro_ui.refresh_status()


def ui_smoke_test() -> int:
    try:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        app = QApplication.instance() or QApplication([])
        app.setApplicationName(APP)
        app.setStyle('Fusion')
        w = Main()
        for i in range(w.stack.count()):
            w.change_page(i)
            app.processEvents()
        assert w.stack.count() >= 10
        assert w.gallery.model() is w.gallery_model
        assert hasattr(w, '_pro_ui')
        assert w._pro_ui.zoom.minimum() <= w._pro_ui.zoom.value() <= w._pro_ui.zoom.maximum()
        w.close()
        app.processEvents()
        return 0
    except Exception as ex:
        print('UI SMOKE TEST FAILED', repr(ex))
        return 1


def main() -> int:
    if '--self-test' in sys.argv:
        return base_self_test()
    if '--ui-smoke-test' in sys.argv:
        return ui_smoke_test()
    app = QApplication(sys.argv)
    app.setApplicationName(APP)
    app.setStyle('Fusion')
    w = Main()
    w.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
