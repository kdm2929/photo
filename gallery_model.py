from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QIcon

from storage import LibraryDB, generate_thumbnail, thumbnail_path


class ThumbSignals(QObject):
    ready = Signal(str, str)


class ThumbTask(QRunnable):
    def __init__(self, path: str, size: int):
        super().__init__(); self.path = path; self.size = size; self.signals = ThumbSignals()
    def run(self):
        out = generate_thumbnail(self.path, self.size)
        self.signals.ready.emit(self.path, str(out) if out else "")


class MediaRoles:
    PathRole = Qt.UserRole + 1
    KindRole = Qt.UserRole + 2
    FaceCountRole = Qt.UserRole + 3
    CapturedRole = Qt.UserRole + 4
    FavoriteRole = Qt.UserRole + 5


class MediaListModel(QAbstractListModel):
    """Paged QListView model: only rows requested by the viewport are rendered."""
    def __init__(self, db: LibraryDB, thumb_size: int = 220, page_size: int = 300, parent=None):
        super().__init__(parent)
        self.db = db; self.thumb_size = thumb_size; self.page_size = page_size
        self.rows: list[tuple] = []; self.total = 0; self.search = ""; self.person_id = None; self.album_id = None; self.month = None
        self.pool = QThreadPool.globalInstance(); self.pending: dict[str, ThumbTask] = {}; self.icons: dict[str, QIcon] = {}
        self.placeholder = QIcon()

    def roleNames(self):
        return {MediaRoles.PathRole:b"path", MediaRoles.KindRole:b"kind", MediaRoles.FaceCountRole:b"faceCount", MediaRoles.CapturedRole:b"captured", MediaRoles.FavoriteRole:b"favorite"}

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self.rows)):
            return None
        path, kind, faces, captured, width, height, duration, favorite = self.rows[index.row()]
        if role == Qt.DisplayRole:
            date = (captured or "")[:10]; badge = f"👤{faces}" if faces else ""; return f"{Path(path).name}\n{date}  {badge}".strip()
        if role == Qt.DecorationRole:
            if path in self.icons: return self.icons[path]
            tp = thumbnail_path(path, self.thumb_size)
            if tp.exists():
                icon = QIcon(str(tp)); self.icons[path] = icon; return icon
            self._schedule(path); return self.placeholder
        if role == Qt.ToolTipRole:
            dim = f"{width}×{height}" if width and height else ""; dur = f" · {duration:.1f}s" if duration else ""; return f"{path}\n{kind} · 얼굴 {faces} · {dim}{dur}"
        if role == MediaRoles.PathRole: return path
        if role == MediaRoles.KindRole: return kind
        if role == MediaRoles.FaceCountRole: return int(faces or 0)
        if role == MediaRoles.CapturedRole: return captured
        if role == MediaRoles.FavoriteRole: return bool(favorite)
        if role == Qt.SizeHintRole: return QSize(self.thumb_size + 24, self.thumb_size + 54)
        return None

    def _schedule(self, path: str):
        if path in self.pending: return
        task = ThumbTask(path, self.thumb_size); self.pending[path] = task; task.signals.ready.connect(self._thumb_ready); self.pool.start(task)

    def _thumb_ready(self, path: str, out: str):
        self.pending.pop(path, None)
        if out: self.icons[path] = QIcon(out)
        for row, r in enumerate(self.rows):
            if r[0] == path:
                ix = self.index(row, 0); self.dataChanged.emit(ix, ix, [Qt.DecorationRole]); break

    def refresh(self, search: str = "", person_id=None, album_id=None, month: str | None = None):
        self.beginResetModel(); self.search = search; self.person_id = person_id; self.album_id = album_id; self.month = month; self.rows = []; self.icons.clear(); self.total = self.db.media_count(search, person_id, album_id, month); self.endResetModel(); self.fetchMore(QModelIndex())

    def canFetchMore(self, parent: QModelIndex):
        return not parent.isValid() and len(self.rows) < self.total

    def fetchMore(self, parent: QModelIndex):
        if parent.isValid(): return
        start = len(self.rows); remain = self.total - start; count = min(self.page_size, remain)
        if count <= 0: return
        batch = self.db.media_rows(count, start, self.search, self.person_id, self.album_id, self.month)
        if not batch: return
        self.beginInsertRows(QModelIndex(), start, start + len(batch) - 1); self.rows.extend(batch); self.endInsertRows()

    def prefetch_around(self, row: int, radius: int = 24):
        if not self.rows: return
        lo=max(0,row-radius); hi=min(len(self.rows),row+radius+1)
        for i in range(lo,hi):
            path=self.rows[i][0]
            if path not in self.icons and not thumbnail_path(path,self.thumb_size).exists(): self._schedule(path)

    def path_at(self, row: int) -> str | None:
        return self.rows[row][0] if 0 <= row < len(self.rows) else None

    def selected_paths(self, indexes) -> list[str]:
        return [self.rows[i.row()][0] for i in indexes if i.isValid() and 0 <= i.row() < len(self.rows)]
