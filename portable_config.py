from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

APP_NAME = "PhotoRefSorter"
BOOTSTRAP_NAME = "PhotoRefSorter.bootstrap.json"
SETTINGS_NAME = "settings.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    "remember_last_folder": True,
    "last_source": "",
    "last_output": "",
    "output_subdir": "PhotoRef_Result",
    "recursive": True,
    "include_images": True,
    "include_videos": True,
    "include_raw": True,
    "multi_person": True,
    "recognition_threshold": 0.43,
    "auto_workers": True,
    "workers": max(2, min(6, (os.cpu_count() or 4) // 2)),
    "gpu": True,
    "hardlink": False,
    "reuse_cache": True,
    "theme": "dark",
    "show_advanced_progress": False,
}


@dataclass
class BootstrapState:
    app_dir: Path
    mode: str
    data_root: Path
    data_dir: Path
    bootstrap_path: Path
    original_localappdata: Path | None
    custom_root: str = ""
    portable_fallback: bool = False
    migrated_from: str = ""


def executable_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def read_bootstrap(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    return {}


def resolve_data_root(mode: str, app_dir: Path, original_localappdata: Path | None, custom_root: str = "") -> Path:
    mode = (mode or "portable").lower()
    if mode == "profile" and original_localappdata:
        return original_localappdata
    if mode == "custom" and custom_root.strip():
        return Path(custom_root).expanduser().resolve()
    return app_dir / "data"


def _copy_tree_contents(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_dir():
        return False
    dst.mkdir(parents=True, exist_ok=True)
    copied = False
    for item in src.iterdir():
        target = dst / item.name
        try:
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                if not target.exists():
                    shutil.copy2(item, target)
            copied = True
        except Exception:
            continue
    return copied


def bootstrap_portable_environment() -> BootstrapState:
    app_dir = executable_dir()
    bootstrap_path = app_dir / BOOTSTRAP_NAME
    original = Path(os.environ["LOCALAPPDATA"]).resolve() if os.environ.get("LOCALAPPDATA") else None
    cfg = read_bootstrap(bootstrap_path)
    mode = str(cfg.get("mode") or "portable").lower()
    custom = str(cfg.get("custom_root") or "")
    root = resolve_data_root(mode, app_dir, original, custom)
    fallback = False

    try:
        (root / APP_NAME).mkdir(parents=True, exist_ok=True)
        probe = root / APP_NAME / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except Exception:
        if original is None:
            raise
        root = original
        mode = "profile"
        fallback = True
        (root / APP_NAME).mkdir(parents=True, exist_ok=True)

    data_dir = root / APP_NAME
    migrated = ""
    if mode in {"portable", "custom"} and original:
        old_dir = original / APP_NAME
        if old_dir.resolve() != data_dir.resolve() and old_dir.exists():
            target_db = data_dir / "library.sqlite3"
            target_legacy = data_dir / "people.sqlite3"
            if not target_db.exists() and not target_legacy.exists():
                if _copy_tree_contents(old_dir, data_dir):
                    migrated = str(old_dir)

    os.environ["LOCALAPPDATA"] = str(root)
    state = BootstrapState(app_dir, mode, root, data_dir, bootstrap_path, original, custom, fallback, migrated)
    global ACTIVE_STATE
    ACTIVE_STATE = state
    return state


def write_bootstrap(mode: str, custom_root: str = "", app_dir: Path | None = None) -> Path:
    base = app_dir or executable_dir()
    path = base / BOOTSTRAP_NAME
    payload = {"mode": mode, "custom_root": custom_root}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class SettingsStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = dict(DEFAULT_SETTINGS)
        self.load()

    def load(self) -> dict[str, Any]:
        self.data = dict(DEFAULT_SETTINGS)
        try:
            if self.path.exists():
                obj = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    self.data.update(obj)
        except Exception:
            pass
        return dict(self.data)

    def save(self, updates: dict[str, Any] | None = None) -> dict[str, Any]:
        if updates:
            self.data.update(updates)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        return dict(self.data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


def settings_store(state: BootstrapState | None = None) -> SettingsStore:
    st = state or ACTIVE_STATE
    return SettingsStore(st.data_dir / SETTINGS_NAME)


ACTIVE_STATE: BootstrapState | None = None
