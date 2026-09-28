from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QSize, Qt
from PySide6.QtGui import QAction, QFont, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QFrame, QGraphicsOpacityEffect,
    QHBoxLayout, QLabel, QLineEdit, QListView, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter,
    QStackedWidget, QVBoxLayout, QWidget
)

from gallery_model import MediaListModel, MediaRoles
from recognition import FaceEngine
from scanner import ScanEngine, auto_worker_count
from storage import (
    APP, VERSION, LibraryDB, ScanConfig, THUMBS, generate_thumbnail, pillow_heif, rawpy,
    read_image, unpack_faces
)


def cv_to_pixmap(img: np.ndarray | None) -> QPixmap:
    if img is None or not img.size:
        return QPixmap()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    q = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(q)


def preview_pixmap(path: str, size: int = 720) -> QPixmap:
    out = generate_thumbnail(path, size)
    return QPixmap(str(out)) if out and out.exists() else QPixmap()


def face_crop_pixmap(db: LibraryDB, path: str, face_idx: int, size: int = 130) -> QPixmap:
    m = db.cache_get(Path(path))
    if not m or not (0 <= face_idx < len(m.faces)):
        return QPixmap()
    if Path(path).suffix.lower() in {'.mp4','.mov','.m4v','.avi','.mkv','.webm'}:
        return preview_pixmap(path, size)
    img = read_image(Path(path), 2200)
    if img is None:
        return QPixmap()
    x,y,w,h = m.faces[face_idx].bbox
    ih, iw = img.shape[:2]
    pad = .20
    x1=max(0,int(x-w*pad)); y1=max(0,int(y-h*pad)); x2=min(iw,int(x+w*(1+pad))); y2=min(ih,int(y+h*(1+pad)))
    if x2<=x1 or y2<=y1:
        return QPixmap()
    crop=img[y1:y2,x1:x2]
    return cv_to_pixmap(crop).scaled(size,size,Qt.KeepAspectRatioByExpanding,Qt.SmoothTransformation)


class StatCard(QFrame):
    def __init__(self, title: str, value: str = '0', sub: str = ''):
        super().__init__(); self.setObjectName('statCard')
        l=QVBoxLayout(self); l.setContentsMargins(16,14,16,14)
        t=QLabel(title); t.setObjectName('statTitle'); self.value=QLabel(value); self.value.setObjectName('statValue'); self.sub=QLabel(sub); self.sub.setObjectName('tiny')
        l.addWidget(t); l.addWidget(self.value); l.addWidget(self.sub)


class PersonCard(QFrame):
    def __init__(self, pid: int, name: str, pos: int, neg: int, threshold: float, quality: float, hero: str | None = None):
        super().__init__(); self.pid=pid; self.setObjectName('personCard')
        l=QHBoxLayout(self); l.setContentsMargins(12,10,12,10)
        av=QLabel(); av.setFixedSize(52,52); av.setObjectName('avatar')
        if hero:
            px=preview_pixmap(hero,96)
            if not px.isNull(): av.setPixmap(px.scaled(52,52,Qt.KeepAspectRatioByExpanding,Qt.SmoothTransformation))
        l.addWidget(av)
        vv=QVBoxLayout(); n=QLabel(name); n.setObjectName('personName'); d=QLabel(f'{pos} refs · {neg} negative · 기준 {threshold:.3f}'); d.setObjectName('tiny'); q=QLabel(f'품질 {quality:.0%}'); q.setObjectName('tiny'); vv.addWidget(n); vv.addWidget(d); vv.addWidget(q); l.addLayout(vv,1)


class UIBase(QMainWindow):
    def __init__(self):
        super().__init__(); self.db=LibraryDB(); self.pid=None; self.scan_worker=None; self.scan_thread=None; self.clusters=[]; self.review_rows=[]; self.review_pos=0; self.dark=True
        self.setWindowTitle(f'PhotoRef Sorter {VERSION}'); self.resize(1450,900); self.setMinimumSize(1120,720)
        self._build(); self.apply_theme(True); self._shortcuts()

    def _shortcuts(self):
        for key, cb in [('1',lambda:self.review_choose(0)),('2',lambda:self.review_choose(1)),('3',lambda:self.review_choose(2)),('X',self.review_reject),('Right',self.review_next),('Left',self.review_prev)]:
            sc=QShortcut(QKeySequence(key),self); sc.activated.connect(cb)

    def _build(self):
        root=QWidget(); self.setCentralWidget(root); h=QHBoxLayout(root); h.setContentsMargins(0,0,0,0); h.setSpacing(0)
        side=QFrame(); side.setObjectName('side'); side.setFixedWidth(218); sl=QVBoxLayout(side); sl.setContentsMargins(16,22,16,18)
        logo=QLabel('PhotoRef'); logo.setObjectName('logo'); sl.addWidget(logo); ver=QLabel(f'v{VERSION} · Smart Library'); ver.setObjectName('sideMuted'); sl.addWidget(ver); sl.addSpacing(22)
        nav_labels=['⌂   홈','▦   모든 사진','👤   인물','✓   빠른 검토','◎   미확인 얼굴','▱   앨범','◫   중복 사진','⚡   대량 분류','⏱   작업 기록','⚙   설정']
        self.nav=[]; group=QButtonGroup(self); group.setExclusive(True)
        for i,text in enumerate(nav_labels):
            b=QPushButton(text); b.setObjectName('nav'); b.setCheckable(True); b.setMinimumHeight(42); sl.addWidget(b); self.nav.append(b); group.addButton(b,i)
        group.idClicked.connect(self.change_page); self.nav[0].setChecked(True); sl.addStretch()
        privacy=QLabel('LOCAL AI\n얼굴 특징값과 사진은\n이 PC 밖으로 전송하지 않음'); privacy.setObjectName('sideMuted'); sl.addWidget(privacy); h.addWidget(side)
        self.stack=QStackedWidget(); h.addWidget(self.stack,1)
        pages=[self.home_page(),self.photos_page(),self.people_page(),self.review_page(),self.cluster_page(),self.albums_page(),self.duplicates_page(),self.scan_page(),self.history_page(),self.settings_page()]
        for p in pages:self.stack.addWidget(p)

    def header(self,title,sub=''):
        v=QVBoxLayout(); a=QLabel(title); a.setObjectName('h1'); v.addWidget(a)
        if sub:
            b=QLabel(sub); b.setObjectName('subtitle'); b.setWordWrap(True); v.addWidget(b)
        return v

    def panel(self):
        p=QFrame(); p.setObjectName('panel'); return p

    def home_page(self):
        w=QWidget(); v=QVBoxLayout(w); v.setContentsMargins(30,26,30,26); v.addLayout(self.header('라이브러리','사진 라이브러리 상태와 최근 작업을 한눈에 확인합니다.'))
        cards=QHBoxLayout(); self.home_cards={}
        for k,title in [('media','전체 미디어'),('people','등록 인물'),('matched','인식 얼굴'),('review','검토 대기'),('unknown','미확인')]:
            c=StatCard(title); cards.addWidget(c); self.home_cards[k]=c
        v.addLayout(cards); v.addSpacing(14)
        split=QSplitter(Qt.Horizontal); recent=self.panel(); rl=QVBoxLayout(recent); t=QLabel('최근 작업'); t.setObjectName('h2'); rl.addWidget(t); self.home_recent=QListWidget(); rl.addWidget(self.home_recent); split.addWidget(recent)
        tips=self.panel(); tl=QVBoxLayout(tips); tt=QLabel('스마트 정리'); tt.setObjectName('h2'); tl.addWidget(tt); self.home_health=QLabel(); self.home_health.setWordWrap(True); self.home_health.setObjectName('muted'); tl.addWidget(self.home_health); tl.addStretch(); go=QPushButton('미확인 인물 그룹 정리 →'); go.setObjectName('primary'); go.clicked.connect(lambda:self.change_page(4)); tl.addWidget(go); split.addWidget(tips); split.setSizes([700,450]); v.addWidget(split,1); return w

    def photos_page(self):
        w=QWidget(); outer=QHBoxLayout(w); outer.setContentsMargins(28,24,24,24); outer.setSpacing(14)
        left=QWidget(); v=QVBoxLayout(left); v.setContentsMargins(0,0,0,0); v.addLayout(self.header('모든 사진','가상화 썸네일 + 비동기 로딩. 수만 장에서도 화면에 필요한 항목만 렌더링합니다.'))
        filters=QHBoxLayout(); self.photo_search=QLineEdit(); self.photo_search.setPlaceholderText('파일명/경로 검색'); self.photo_search.returnPressed.connect(self.refresh_gallery); filters.addWidget(self.photo_search,1)
        self.photo_month=QComboBox(); self.photo_month.currentIndexChanged.connect(self.refresh_gallery); filters.addWidget(self.photo_month)
        self.photo_person=QComboBox(); self.photo_person.currentIndexChanged.connect(self.refresh_gallery); filters.addWidget(self.photo_person)
        self.photo_album=QComboBox(); self.photo_album.currentIndexChanged.connect(self.refresh_gallery); filters.addWidget(self.photo_album)
        refresh=QPushButton('새로고침'); refresh.clicked.connect(self.refresh_gallery); filters.addWidget(refresh); v.addLayout(filters)
        self.gallery_model=MediaListModel(self.db,220,300,self); self.gallery=QListView(); self.gallery.setModel(self.gallery_model); self.gallery.setViewMode(QListView.IconMode); self.gallery.setResizeMode(QListView.Adjust); self.gallery.setUniformItemSizes(True); self.gallery.setIconSize(QSize(220,220)); self.gallery.setGridSize(QSize(244,280)); self.gallery.setSelectionMode(QListView.ExtendedSelection); self.gallery.setSpacing(4); self.gallery.clicked.connect(self.gallery_selected); self.gallery.doubleClicked.connect(self.open_selected_file); self.gallery.setContextMenuPolicy(Qt.ActionsContextMenu)
        fav=QAction('★ 즐겨찾기',self.gallery); fav.triggered.connect(lambda:self.gallery_favorite(True)); self.gallery.addAction(fav); addref=QAction('선택 사진을 레퍼런스로',self.gallery); addref.triggered.connect(self.gallery_add_reference); self.gallery.addAction(addref)
        v.addWidget(self.gallery,1); self.gallery_count=QLabel(); self.gallery_count.setObjectName('tiny'); v.addWidget(self.gallery_count); outer.addWidget(left,1)
        ins=self.panel(); ins.setFixedWidth(340); il=QVBoxLayout(ins); il.setContentsMargins(16,16,16,16); self.ins_preview=QLabel('사진을 선택하세요'); self.ins_preview.setAlignment(Qt.AlignCenter); self.ins_preview.setFixedHeight(275); self.ins_preview.setObjectName('preview'); il.addWidget(self.ins_preview); self.ins_name=QLabel(); self.ins_name.setObjectName('h2'); self.ins_name.setWordWrap(True); il.addWidget(self.ins_name); self.ins_meta=QLabel(); self.ins_meta.setObjectName('muted'); self.ins_meta.setWordWrap(True); il.addWidget(self.ins_meta); il.addWidget(QLabel('검출 얼굴 / 후보')); self.face_strip=QListWidget(); self.face_strip.setViewMode(QListWidget.IconMode); self.face_strip.setIconSize(QSize(82,82)); self.face_strip.setGridSize(QSize(102,126)); self.face_strip.setFlow(QListWidget.LeftToRight); self.face_strip.setWrapping(True); self.face_strip.setMaximumHeight(270); self.face_strip.itemDoubleClicked.connect(self.face_strip_assign); il.addWidget(self.face_strip); self.ins_albums=QLabel(); self.ins_albums.setObjectName('tiny'); self.ins_albums.setWordWrap(True); il.addWidget(self.ins_albums); il.addStretch(); batch=QLabel('선택 항목 일괄 작업'); batch.setObjectName('h3'); il.addWidget(batch); self.batch_person=QComboBox(); il.addWidget(self.batch_person); br=QHBoxLayout(); bp=QPushButton('인물 학습'); bp.clicked.connect(self.gallery_add_reference); bf=QPushButton('★'); bf.clicked.connect(lambda:self.gallery_favorite(True)); br.addWidget(bp); br.addWidget(bf); il.addLayout(br); self.batch_album=QComboBox(); il.addWidget(self.batch_album); ba=QPushButton('앨범에 추가'); ba.clicked.connect(self.gallery_add_album); il.addWidget(ba); outer.addWidget(ins); return w

    def people_page(self):
        w=QWidget(); v=QVBoxLayout(w); v.setContentsMargins(28,24,28,24); v.addLayout(self.header('인물','프로필, 대표사진, 자동 임계값, 혼동 인물까지 한 화면에서 관리합니다.'))
        addrow=QHBoxLayout(); self.person_new=QLineEdit(); self.person_new.setPlaceholderText('새 인물 이름'); addrow.addWidget(self.person_new,1); add=QPushButton('인물 추가'); add.setObjectName('primary'); add.clicked.connect(self.add_person); addrow.addWidget(add); v.addLayout(addrow)
        split=QSplitter(Qt.Horizontal); left=self.panel(); ll=QVBoxLayout(left); self.people_list=QListWidget(); self.people_list.currentItemChanged.connect(self.person_selected); ll.addWidget(self.people_list); split.addWidget(left)
        right=self.panel(); rl=QVBoxLayout(right); top=QHBoxLayout(); self.person_hero=QLabel(); self.person_hero.setFixedSize(190,190); self.person_hero.setAlignment(Qt.AlignCenter); self.person_hero.setObjectName('preview'); top.addWidget(self.person_hero); info=QVBoxLayout(); self.person_title=QLabel('인물을 선택하세요'); self.person_title.setObjectName('h1'); info.addWidget(self.person_title); self.person_stats=QLabel(); self.person_stats.setObjectName('muted'); self.person_stats.setWordWrap(True); info.addWidget(self.person_stats); self.person_warning=QLabel(); self.person_warning.setObjectName('warning'); self.person_warning.setWordWrap(True); info.addWidget(self.person_warning); top.addLayout(info,1); rl.addLayout(top)
        sub=QSplitter(Qt.Horizontal); refs=self.panel(); rfl=QVBoxLayout(refs); rfl.addWidget(QLabel('레퍼런스')); self.ref_list=QListWidget(); rfl.addWidget(self.ref_list); rr=QHBoxLayout(); ar=QPushButton('추가'); ar.clicked.connect(self.add_refs); dr=QPushButton('선택 삭제'); dr.clicked.connect(self.delete_ref); rr.addWidget(ar); rr.addWidget(dr); rfl.addLayout(rr); sub.addWidget(refs)
        conf=self.panel(); cl=QVBoxLayout(conf); cl.addWidget(QLabel('가장 헷갈리는 인물')); self.confusion_list=QListWidget(); cl.addWidget(self.confusion_list); cl.addWidget(QLabel('최근 인식 사진')); self.person_recent=QListWidget(); self.person_recent.setViewMode(QListWidget.IconMode); self.person_recent.setIconSize(QSize(90,90)); self.person_recent.setGridSize(QSize(110,118)); self.person_recent.setMaximumHeight(250); self.person_recent.itemDoubleClicked.connect(self.set_hero_from_recent); cl.addWidget(self.person_recent); sub.addWidget(conf); rl.addWidget(sub,1); actions=QHBoxLayout(); dele=QPushButton('인물 삭제'); dele.setObjectName('danger'); dele.clicked.connect(self.delete_person); actions.addStretch(); actions.addWidget(dele); rl.addLayout(actions); split.addWidget(right); split.setSizes([330,900]); v.addWidget(split,1); return w

    def review_page(self):
        w=QWidget(); v=QVBoxLayout(w); v.setContentsMargins(28,24,28,24); v.addLayout(self.header('빠른 검토','키보드 1/2/3으로 후보 확정 · X로 1위 후보 제외 학습 · ←/→로 이동'))
        top=QHBoxLayout(); self.review_progress=QLabel(); self.review_progress.setObjectName('muted'); top.addWidget(self.review_progress); top.addStretch(); rb=QPushButton('검토 목록 새로고침'); rb.clicked.connect(self.refresh_review); top.addWidget(rb); v.addLayout(top)
        split=QSplitter(Qt.Horizontal); photo=self.panel(); pl=QVBoxLayout(photo); self.review_image=QLabel('검토할 얼굴 없음'); self.review_image.setAlignment(Qt.AlignCenter); self.review_image.setObjectName('reviewImage'); pl.addWidget(self.review_image,1); self.review_path=QLabel(); self.review_path.setObjectName('tiny'); self.review_path.setWordWrap(True); pl.addWidget(self.review_path); split.addWidget(photo)
        choices=self.panel(); choices.setFixedWidth(390); ch=QVBoxLayout(choices); self.review_face=QLabel(); self.review_face.setFixedSize(180,180); self.review_face.setAlignment(Qt.AlignCenter); self.review_face.setObjectName('preview'); ch.addWidget(self.review_face,0,Qt.AlignHCenter); self.review_state=QLabel(); self.review_state.setObjectName('h2'); ch.addWidget(self.review_state); self.review_candidate_buttons=[]
        for i in range(3):
            b=QPushButton(f'{i+1}. 후보 없음'); b.setObjectName('candidate'); b.setMinimumHeight(58); b.clicked.connect(lambda checked=False,n=i:self.review_choose(n)); ch.addWidget(b); self.review_candidate_buttons.append(b)
        x=QPushButton('X · 1위 후보가 아님 (negative 학습)'); x.clicked.connect(self.review_reject); ch.addWidget(x); skip=QPushButton('→ 건너뛰기'); skip.clicked.connect(self.review_next); ch.addWidget(skip); ch.addStretch(); hint=QLabel('수정 결과는 즉시 positive/negative 레퍼런스에 반영되어 다음 스캔의 판정에 사용됩니다.'); hint.setWordWrap(True); hint.setObjectName('tiny'); ch.addWidget(hint); split.addWidget(choices); split.setSizes([900,390]); v.addWidget(split,1); return w

    def cluster_page(self):
        w=QWidget(); v=QVBoxLayout(w); v.setContentsMargins(28,24,28,24); v.addLayout(self.header('미확인 얼굴','미확인 얼굴을 사람별로 묶고 한 번의 이름 지정으로 대표 얼굴을 자동 학습합니다.'))
        top=QHBoxLayout(); make=QPushButton('◎ 스마트 그룹 생성'); make.setObjectName('primary'); make.clicked.connect(self.make_clusters); top.addWidget(make); self.cluster_status=QLabel('아직 그룹화하지 않음'); self.cluster_status.setObjectName('muted'); top.addWidget(self.cluster_status); top.addStretch(); v.addLayout(top)
        split=QSplitter(Qt.Horizontal); self.cluster_list=QListWidget(); self.cluster_list.currentRowChanged.connect(self.show_cluster); split.addWidget(self.cluster_list); detail=self.panel(); dl=QVBoxLayout(detail); self.cluster_preview=QListWidget(); self.cluster_preview.setViewMode(QListWidget.IconMode); self.cluster_preview.setIconSize(QSize(105,105)); self.cluster_preview.setGridSize(QSize(125,132)); dl.addWidget(self.cluster_preview,1); self.cluster_detail=QLabel(); self.cluster_detail.setObjectName('muted'); dl.addWidget(self.cluster_detail); row=QHBoxLayout(); self.cluster_person=QComboBox(); self.cluster_name=QLineEdit(); self.cluster_name.setPlaceholderText('새 인물 이름'); use=QPushButton('이 그룹 확정 + 학습'); use.setObjectName('primary'); use.clicked.connect(self.assign_cluster); row.addWidget(self.cluster_person); row.addWidget(self.cluster_name,1); row.addWidget(use); dl.addLayout(row); split.addWidget(detail); split.setSizes([330,900]); v.addWidget(split,1); return w

    def albums_page(self):
        w=QWidget(); v=QVBoxLayout(w); v.setContentsMargins(28,24,28,24); v.addLayout(self.header('앨범 / 이벤트','인물 분류와 별개로 콘서트·공항·팬사인회 같은 사용자 앨범을 만들 수 있습니다.'))
        row=QHBoxLayout(); self.album_new=QLineEdit(); self.album_new.setPlaceholderText('새 앨범 이름'); row.addWidget(self.album_new,1); add=QPushButton('앨범 만들기'); add.setObjectName('primary'); add.clicked.connect(self.add_album); row.addWidget(add); v.addLayout(row); self.album_list=QListWidget(); self.album_list.itemDoubleClicked.connect(self.open_album); v.addWidget(self.album_list,1); rr=QHBoxLayout(); op=QPushButton('선택 앨범 열기'); op.clicked.connect(self.open_album); de=QPushButton('앨범 삭제'); de.setObjectName('danger'); de.clicked.connect(self.delete_album); rr.addWidget(op); rr.addStretch(); rr.addWidget(de); v.addLayout(rr); return w

    def duplicates_page(self):
        w=QWidget(); v=QVBoxLayout(w); v.setContentsMargins(28,24,28,24); v.addLayout(self.header('중복 / 연사','pHash 기반으로 완전 중복과 유사 연사 후보를 확인합니다. 원본은 자동 삭제하지 않습니다.'))
        top=QHBoxLayout(); run=QPushButton('중복 분석 새로고침'); run.setObjectName('primary'); run.clicked.connect(self.refresh_duplicates); top.addWidget(run); self.dup_status=QLabel(); self.dup_status.setObjectName('muted'); top.addWidget(self.dup_status); top.addStretch(); v.addLayout(top); split=QSplitter(Qt.Horizontal); self.dup_groups=QListWidget(); self.dup_groups.currentRowChanged.connect(self.dup_group_selected); split.addWidget(self.dup_groups); self.dup_files=QListWidget(); split.addWidget(self.dup_files); split.setSizes([350,850]); v.addWidget(split,1); return w

    def scan_page(self):
        w=QWidget(); v=QVBoxLayout(w); v.setContentsMargins(30,26,30,26); v.addLayout(self.header('대량 분류','첫 스캔은 병렬 AI 분석, 이후에는 얼굴 임베딩 캐시를 재사용합니다.'))
        p=self.panel(); l=QVBoxLayout(p); l.setContentsMargins(20,20,20,20); self.src=QLineEdit(); self.dst=QLineEdit()
        for title,edit,cb in [('분류할 폴더',self.src,self.pick_src),('결과 폴더',self.dst,self.pick_dst)]:
            l.addWidget(QLabel(title)); rr=QHBoxLayout(); rr.addWidget(edit,1); b=QPushButton('선택'); b.clicked.connect(cb); rr.addWidget(b); l.addLayout(rr)
        opts=QHBoxLayout(); self.recursive=QCheckBox('하위 폴더'); self.recursive.setChecked(True); self.multi=QCheckBox('여러 인물 동시 분류'); self.multi.setChecked(True); self.hard=QCheckBox('같은 드라이브 하드링크'); opts.addWidget(self.recursive); opts.addWidget(self.multi); opts.addWidget(self.hard); opts.addStretch(); l.addLayout(opts)
        perf=QHBoxLayout(); self.auto_workers=QCheckBox('저장장치/CPU 자동 튜닝'); self.auto_workers.setChecked(True); perf.addWidget(self.auto_workers); perf.addWidget(QLabel('수동 워커')); self.workers=QSpinBox(); self.workers.setRange(1,8); self.workers.setValue(min(6,max(2,(os.cpu_count() or 4)//2))); perf.addWidget(self.workers); self.gpu=QCheckBox('DirectML GPU'); ok,desc=FaceEngine.gpu_available(); self.gpu.setEnabled(ok); self.gpu.setChecked(ok); self.gpu.setToolTip(desc); perf.addWidget(self.gpu); perf.addStretch(); l.addLayout(perf)
        tr=QHBoxLayout(); tr.addWidget(QLabel('기본 인식 기준')); self.th=QSlider(Qt.Horizontal); self.th.setRange(36,58); self.th.setValue(43); self.tv=QLabel('0.43'); self.th.valueChanged.connect(lambda x:self.tv.setText(f'{x/100:.2f}')); tr.addWidget(self.th,1); tr.addWidget(self.tv); l.addLayout(tr)
        actions=QHBoxLayout(); self.go=QPushButton('⚡ 분류 시작'); self.go.setObjectName('primaryBig'); self.go.clicked.connect(self.start_scan); self.stop=QPushButton('중단'); self.stop.setEnabled(False); self.stop.clicked.connect(self.stop_scan); actions.addWidget(self.go,1); actions.addWidget(self.stop); l.addLayout(actions); self.pb=QProgressBar(); l.addWidget(self.pb); self.scan_status=QLabel('준비됨'); self.scan_status.setObjectName('muted'); l.addWidget(self.scan_status); self.speed=QLabel(); self.speed.setObjectName('tiny'); l.addWidget(self.speed); v.addWidget(p); v.addStretch(); return w

    def history_page(self):
        w=QWidget(); v=QVBoxLayout(w); v.setContentsMargins(28,24,28,24); v.addLayout(self.header('작업 기록','분류 실행 이력과 마지막 작업 되돌리기를 관리합니다.'))
        self.history_list=QListWidget(); v.addWidget(self.history_list,1); rr=QHBoxLayout(); reload=QPushButton('새로고침'); reload.clicked.connect(self.refresh_history); undo=QPushButton('마지막 분류 되돌리기'); undo.clicked.connect(self.undo_last); rr.addWidget(reload); rr.addStretch(); rr.addWidget(undo); v.addLayout(rr); return w

    def settings_page(self):
        w=QWidget(); v=QVBoxLayout(w); v.setContentsMargins(30,26,30,26); v.addLayout(self.header('설정','캐시, 지원 포맷, UI 테마와 가속 상태를 관리합니다.'))
        p=self.panel(); l=QVBoxLayout(p); self.sys_info=QLabel(); self.sys_info.setWordWrap(True); l.addWidget(self.sys_info); self.theme_toggle=QCheckBox('다크 모드'); self.theme_toggle.setChecked(True); self.theme_toggle.toggled.connect(self.apply_theme); l.addWidget(self.theme_toggle); rr=QHBoxLayout(); clean=QPushButton('없는 파일 캐시 정리'); clean.clicked.connect(self.cleanup_cache); dups=QPushButton('중복 통계'); dups.clicked.connect(self.refresh_duplicates); clear=QPushButton('전체 분석 캐시 비우기'); clear.setObjectName('danger'); clear.clicked.connect(self.clear_cache); rr.addWidget(clean); rr.addWidget(dups); rr.addWidget(clear); l.addLayout(rr); fm=QLabel('JPG · PNG · WebP · TIFF · HEIC/HEIF · AVIF · GIF · RAW · MP4/MOV/MKV/AVI/WebM'); fm.setObjectName('muted'); fm.setWordWrap(True); l.addWidget(fm); v.addWidget(p); v.addStretch(); return w

    def apply_theme(self, dark: bool):
        self.dark=bool(dark)
        if dark:
            self.setStyleSheet('''*{font-family:"Segoe UI","Malgun Gothic";font-size:13px} QMainWindow,QWidget{background:#0F1115;color:#E9EDF3} #side{background:#11141A;border-right:1px solid #242A33} #logo{font-size:26px;font-weight:850;color:white} #sideMuted,#tiny{color:#788292;font-size:11px} #nav{border:none;background:transparent;color:#AEB7C4;text-align:left;border-radius:10px;padding:9px 12px;font-weight:650} #nav:hover{background:#1B2028;color:white} #nav:checked{background:#242A36;color:#FFFFFF;border-left:3px solid #6C7BFF} #h1{font-size:28px;font-weight:850} #h2{font-size:18px;font-weight:800} #h3{font-size:14px;font-weight:750} #subtitle,#muted{color:#8C96A5} #panel,#statCard,#personCard{background:#161A21;border:1px solid #272D37;border-radius:14px} #personCard:hover{border:1px solid #4D5870} #statTitle{color:#929DAC;font-size:12px} #statValue{font-size:27px;font-weight:850} #personName{font-size:15px;font-weight:800} QLineEdit,QListWidget,QListView,QComboBox,QSpinBox{background:#13171D;border:1px solid #2D3440;border-radius:9px;padding:7px;color:#EEF2F7;selection-background-color:#4858D7} QPushButton{border:none;border-radius:9px;background:#252B35;color:#E9EDF3;padding:9px 13px;font-weight:650} QPushButton:hover{background:#303846} #primary,#primaryBig{background:#596AF5;color:white} #primary:hover,#primaryBig:hover{background:#6878FF} #primaryBig{font-size:15px;padding:12px} #danger{background:#3A2026;color:#FF9CAA} #candidate{background:#1D2330;text-align:left;font-size:15px;padding:14px} #candidate:hover{background:#293247} #preview,#reviewImage{background:#0B0D11;border:1px solid #272D37;border-radius:12px} #warning{color:#F0B86E} QProgressBar{border:none;background:#222833;border-radius:7px;height:15px;text-align:center} QProgressBar::chunk{background:#6171F6;border-radius:7px} QScrollBar:vertical{background:#11151B;width:11px} QScrollBar::handle:vertical{background:#343C49;border-radius:5px;min-height:30px}''')
        else:
            self.setStyleSheet('''*{font-family:"Segoe UI","Malgun Gothic";font-size:13px} QMainWindow,QWidget{background:#F5F7FA;color:#20252C} #side{background:#171A20} #logo{font-size:26px;font-weight:850;color:white} #sideMuted,#tiny{color:#858E9A;font-size:11px} #nav{border:none;background:transparent;color:#BBC3CE;text-align:left;border-radius:10px;padding:9px 12px;font-weight:650} #nav:hover{background:#252A33;color:white} #nav:checked{background:#2D3440;color:white} #h1{font-size:28px;font-weight:850} #h2{font-size:18px;font-weight:800} #subtitle,#muted{color:#6F7884} #panel,#statCard,#personCard{background:white;border:1px solid #DFE4EA;border-radius:14px} #statValue{font-size:27px;font-weight:850} QLineEdit,QListWidget,QListView,QComboBox,QSpinBox{background:white;border:1px solid #D8DEE6;border-radius:9px;padding:7px} QPushButton{border:none;border-radius:9px;background:#E9EDF2;padding:9px 13px;font-weight:650} #primary,#primaryBig{background:#5263F5;color:white} #danger{background:#FFF0F1;color:#B93D4D} #preview,#reviewImage{background:#EEF1F5;border-radius:12px}''')

    def animate_page(self, widget):
        eff=QGraphicsOpacityEffect(widget); widget.setGraphicsEffect(eff); anim=QPropertyAnimation(eff,b'opacity',widget); anim.setDuration(170); anim.setStartValue(.25); anim.setEndValue(1.0); anim.setEasingCurve(QEasingCurve.OutCubic); anim.finished.connect(lambda:widget.setGraphicsEffect(None)); anim.start(); widget._fade_anim=anim

    def change_page(self, i: int):
        if not (0 <= i < self.stack.count()): return
        self.stack.setCurrentIndex(i); self.nav[i].setChecked(True); self.animate_page(self.stack.currentWidget())
        if i==0:self.refresh_home()
        elif i==1:self.refresh_gallery_filters();self.refresh_gallery()
        elif i==2:self.refresh_people()
        elif i==3:self.refresh_review()
        elif i==4:self.refresh_cluster_people()
        elif i==5:self.refresh_albums()
        elif i==6:self.refresh_duplicates()
        elif i==8:self.refresh_history()
        elif i==9:self.refresh_settings()
