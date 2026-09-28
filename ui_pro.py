from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEasingCurve, QModelIndex, QRect, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import QLabel, QSlider, QStyle, QStyledItemDelegate, QToolBar, QWidget

from gallery_model import MediaRoles


class ProGalleryDelegate(QStyledItemDelegate):
    """Modern card renderer for large virtualized galleries.

    The view/model stay virtualized; this only paints the visible cells, so the
    richer UI does not create thousands of child widgets.
    """
    def __init__(self, parent=None, thumb_size: int = 220):
        super().__init__(parent)
        self.thumb_size = thumb_size

    def set_thumb_size(self, value: int):
        self.thumb_size = max(120, min(340, int(value)))

    def sizeHint(self, option, index):
        return QSize(self.thumb_size + 24, self.thumb_size + 76)

    def paint(self, painter: QPainter, option, index: QModelIndex):
        painter.save()
        r = option.rect.adjusted(5, 5, -5, -5)
        selected = bool(option.state & QStyle.State_Selected)
        hovered = bool(option.state & QStyle.State_MouseOver)
        bg = QColor('#242C3A') if selected else QColor('#181D25')
        if hovered and not selected:
            bg = QColor('#202733')
        border = QColor('#7180FF') if selected else QColor('#303846')

        path = QPainterPath()
        path.addRoundedRect(QRectF(r), 13, 13)
        painter.fillPath(path, bg)
        painter.setPen(QPen(border, 1.5 if selected else 1.0))
        painter.drawPath(path)

        img_rect = QRect(r.left()+8, r.top()+8, r.width()-16, max(60, r.width()-16))
        deco = index.data(Qt.DecorationRole)
        pix = deco.pixmap(img_rect.size()) if deco is not None and hasattr(deco, 'pixmap') else QPixmap()
        if not pix.isNull():
            scaled = pix.scaled(img_rect.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = max(0, (scaled.width()-img_rect.width())//2)
            y = max(0, (scaled.height()-img_rect.height())//2)
            crop = scaled.copy(x, y, img_rect.width(), img_rect.height())
            clip = QPainterPath(); clip.addRoundedRect(QRectF(img_rect), 9, 9)
            painter.setClipPath(clip)
            painter.drawPixmap(img_rect, crop)
            painter.setClipping(False)
        else:
            painter.fillRect(img_rect, QColor('#11151B'))
            painter.setPen(QColor('#6E7888'))
            painter.drawText(img_rect, Qt.AlignCenter, '썸네일 생성 중')

        faces = int(index.data(MediaRoles.FaceCountRole) or 0)
        kind = str(index.data(MediaRoles.KindRole) or 'image')
        fav = bool(index.data(MediaRoles.FavoriteRole))
        badges = []
        if faces:
            badges.append(f'👤 {faces}')
        if kind == 'video':
            badges.append('▶ VIDEO')
        if fav:
            badges.append('★')
        bx = img_rect.left()+7
        for text in badges:
            fm = painter.fontMetrics(); bw = fm.horizontalAdvance(text)+14
            br = QRect(bx, img_rect.top()+7, bw, 23)
            painter.fillRect(br, QColor(10,12,16,205))
            painter.setPen(QColor('#F4F7FB')); painter.drawText(br, Qt.AlignCenter, text)
            bx += bw+5

        display = str(index.data(Qt.DisplayRole) or '')
        lines = display.split('\n', 1)
        name = lines[0]
        sub = lines[1] if len(lines) > 1 else ''
        text_top = img_rect.bottom()+10
        painter.setPen(QColor('#F0F3F8'))
        f = painter.font(); f.setBold(True); painter.setFont(f)
        name_rect = QRect(r.left()+10, text_top, r.width()-20, 22)
        elided = painter.fontMetrics().elidedText(name, Qt.ElideMiddle, name_rect.width())
        painter.drawText(name_rect, Qt.AlignLeft|Qt.AlignVCenter, elided)
        f.setBold(False); f.setPointSize(max(8, f.pointSize()-1)); painter.setFont(f)
        painter.setPen(QColor('#8D98A8'))
        painter.drawText(QRect(r.left()+10, text_top+24, r.width()-20, 19), Qt.AlignLeft|Qt.AlignVCenter, sub)
        painter.restore()


class ProUIController:
    def __init__(self, window):
        self.w = window
        self.delegate = ProGalleryDelegate(window.gallery, getattr(window.gallery_model, 'thumb_size', 220))
        window.gallery.setItemDelegate(self.delegate)
        window.gallery.setMouseTracking(True)
        window.gallery.setSpacing(2)
        self._search_timer = QTimer(window)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(260)
        self._search_timer.timeout.connect(window.refresh_gallery)
        try:
            window.photo_search.textChanged.disconnect()
        except Exception:
            pass
        window.photo_search.textChanged.connect(lambda *_: self._search_timer.start())
        self._install_toolbar()
        self._install_shortcuts()
        self._sync_status()

    def _install_toolbar(self):
        tb = QToolBar('Library Controls', self.w)
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setObjectName('proToolbar')
        self.w.addToolBar(Qt.TopToolBarArea, tb)
        self.status = QLabel('준비됨')
        self.status.setObjectName('statusChip')
        tb.addWidget(self.status)
        spacer = QWidget(); spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy(), spacer.sizePolicy().verticalPolicy())
        spacer.setMinimumWidth(18); tb.addWidget(spacer)
        zlabel = QLabel('사진 크기')
        zlabel.setObjectName('toolbarMuted'); tb.addWidget(zlabel)
        self.zoom = QSlider(Qt.Horizontal); self.zoom.setRange(140, 320); self.zoom.setValue(getattr(self.w.gallery_model, 'thumb_size', 220)); self.zoom.setFixedWidth(150)
        self.zoom.valueChanged.connect(self._zoom_changed); tb.addWidget(self.zoom)
        self.w.setStyleSheet(self.w.styleSheet() + '''
            QToolBar#proToolbar{background:#12161C;border:none;border-bottom:1px solid #252C36;padding:6px 12px;spacing:8px}
            #statusChip{background:#202735;border:1px solid #303949;border-radius:10px;padding:5px 10px;color:#BFC8D6;font-weight:650}
            #toolbarMuted{color:#7F8998;font-size:11px}
        ''')

    def _install_shortcuts(self):
        shortcuts = [
            ('Ctrl+L', lambda: self._focus_search()),
            ('Ctrl+1', lambda: self.w.change_page(0)),
            ('Ctrl+2', lambda: self.w.change_page(1)),
            ('Ctrl+3', lambda: self.w.change_page(2)),
            ('Ctrl+4', lambda: self.w.change_page(3)),
            ('Ctrl+5', lambda: self.w.change_page(4)),
            ('Ctrl++', lambda: self.zoom.setValue(min(self.zoom.maximum(), self.zoom.value()+20))),
            ('Ctrl+-', lambda: self.zoom.setValue(max(self.zoom.minimum(), self.zoom.value()-20))),
        ]
        self.shortcuts=[]
        for key, cb in shortcuts:
            sc=QShortcut(QKeySequence(key), self.w); sc.activated.connect(cb); self.shortcuts.append(sc)

    def _focus_search(self):
        self.w.change_page(1)
        self.w.photo_search.setFocus()
        self.w.photo_search.selectAll()

    def _zoom_changed(self, value: int):
        self.delegate.set_thumb_size(value)
        self.w.gallery_model.thumb_size = int(value)
        self.w.gallery.setIconSize(QSize(value, value))
        self.w.gallery.setGridSize(QSize(value+28, value+82))
        self.w.gallery.viewport().update()

    def _sync_status(self):
        try:
            d = self.w.db.dashboard()
            self.status.setText(f"미디어 {d['media']:,} · 인물 {d['people']:,} · 검토 {d['review']+d['unknown']:,}")
        except Exception:
            self.status.setText('라이브러리 준비됨')

    def refresh_status(self):
        self._sync_status()


def install_pro_ui(window):
    ctl = ProUIController(window)
    window._pro_ui = ctl
    return ctl
