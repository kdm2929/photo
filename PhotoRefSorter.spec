# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, collect_submodules

binaries = collect_dynamic_libs("cv2")
datas = [("models", "models")]
hiddenimports = [
    "cv2", "numpy", "core", "storage", "recognition", "scanner", "gallery_model",
    "ui_base", "ui_pro", "ui_folderfirst", "portable_config", "main", "app_v06", "app_v07",
    "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
    "PIL", "PIL.Image", "pillow_heif", "pillow_avif", "rawpy", "onnxruntime"
]
for pkg in ("onnxruntime", "rawpy", "pillow_heif"):
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass
for pkg in ("pillow_heif", "pillow_avif"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass
try:
    hiddenimports += collect_submodules("onnxruntime")
except Exception:
    pass

a = Analysis(["app_v07.py"], pathex=[], binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=[], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="PhotoRefSorter", debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=False,
          disable_windowed_traceback=False, argv_emulation=False, target_arch=None,
          codesign_identity=None, entitlements_file=None)
