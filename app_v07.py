from __future__ import annotations

import os
import sys

os.environ.setdefault('QT_ENABLE_HIGHDPI_SCALING', '1')

from portable_config import APP_NAME, bootstrap_portable_environment, settings_store

BOOT_STATE = bootstrap_portable_environment()
SETTINGS = settings_store(BOOT_STATE)

# Storage must be imported only after the portable/profile/custom data root is selected.
import storage
storage.VERSION = '0.7'

from PySide6.QtWidgets import QApplication
from app_v06 import Main as V06Main, base_self_test
from ui_folderfirst import install_folder_first_ui


class Main(V06Main):
    def __init__(self):
        super().__init__()
        install_folder_first_ui(self, BOOT_STATE, SETTINGS)
        self.setWindowTitle('PhotoRef Sorter 0.7 · Folder First')

    def refresh_home(self):
        if hasattr(self, '_folder_ui'):
            self._folder_ui.refresh_people_count()
            if hasattr(self, '_pro_ui'):
                self._pro_ui.refresh_status()
            return
        super().refresh_home()

    def scan_progress(self, i, n, name, speed, cached, new):
        super().scan_progress(i, n, name, speed, cached, new)
        if hasattr(self, '_folder_ui'):
            self._folder_ui.on_progress(i, n, name, speed, cached, new)

    def scan_done(self, stats):
        super().scan_done(stats)
        if hasattr(self, '_folder_ui'):
            self._folder_ui.on_done(stats)

    def scan_fail(self, text):
        super().scan_fail(text)
        if hasattr(self, '_folder_ui'):
            self._folder_ui.on_fail(text)


def ui_smoke_test() -> int:
    try:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        app = QApplication.instance() or QApplication([])
        app.setApplicationName(APP_NAME)
        app.setStyle('Fusion')
        w = Main()
        assert hasattr(w, '_folder_ui')
        assert hasattr(w, '_pro_ui')
        assert w.stack.count() >= 10
        assert w._folder_ui.home_start.text()
        assert w._folder_ui.storage_mode.count() == 3
        assert BOOT_STATE.data_dir.exists()
        for i in (0, 1, 2, 3, 9):
            w.change_page(i)
            app.processEvents()
        w.close(); app.processEvents()
        return 0
    except Exception as ex:
        print('V0.7 UI SMOKE TEST FAILED', repr(ex))
        return 1


def main() -> int:
    if '--self-test' in sys.argv:
        return base_self_test()
    if '--ui-smoke-test' in sys.argv:
        return ui_smoke_test()
    if '--portable-info' in sys.argv:
        print('mode=', BOOT_STATE.mode)
        print('data_dir=', BOOT_STATE.data_dir)
        print('settings=', SETTINGS.path)
        return 0
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle('Fusion')
    w = Main(); w.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
