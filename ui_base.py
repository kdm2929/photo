from __future__ import annotations

import os, sqlite3, sys, time
from pathlib import Path
import numpy as np

from PySide6.QtCore import QObject,QThread,Qt,Signal,QSize
from PySide6.QtGui import QFont,QIcon,QPixmap
from PySide6.QtWidgets import QApplication,QButtonGroup,QCheckBox,QComboBox,QFileDialog,QFrame,QHBoxLayout,QLabel,QLineEdit,QListWidget,QListWidgetItem,QMainWindow,QMessageBox,QProgressBar,QPushButton,QScrollArea,QSlider,QSpinBox,QStackedWidget,QVBoxLayout,QWidget

from core import APP,VERSION,DB,THUMBS,pillow_heif,rawpy,FaceEngine,LibraryDB,ScanConfig,ScanEngine,Clusterer,build_models,read_image,unpack_faces


def thumb(path,size=140):
    p=Path(path)
    if not p.exists():return QPixmap()
    px=QPixmap(str(p))
    if px.isNull():
        img=read_image(p,600)
        if img is not None:
            import cv2
            t=THUMBS/f'ui_{abs(hash((str(p),p.stat().st_mtime_ns)))}.jpg';cv2.imwrite(str(t),img);px=QPixmap(str(t))
    return px.scaled(size,size,Qt.KeepAspectRatioByExpanding,Qt.SmoothTransformation) if not px.isNull() else px


class ScanWorker(QObject):
    progress=Signal(int,int,str,float,int,int);status=Signal(str);done=Signal(dict);fail=Signal(str)
    def __init__(self,cfg):super().__init__();self.cfg=cfg;self.stop=False
    def request_stop(self):self.stop=True
    def run(self):
        try:
            engine=ScanEngine(self.cfg,lambda *a:self.progress.emit(*a),lambda s:self.status.emit(s),lambda:self.stop)
            self.done.emit(engine.run())
        except Exception as e:self.fail.emit(str(e))


class PersonCard(QFrame):
    clicked=Signal(int)
    def __init__(self,pid,name,pos,neg,threshold,quality):
        super().__init__();self.pid=pid;self.setObjectName('personCard');self.setCursor(Qt.PointingHandCursor)
        l=QVBoxLayout(self);l.setContentsMargins(14,12,14,12);n=QLabel(name);n.setObjectName('personName');l.addWidget(n);t=QLabel(f'레퍼런스 {pos or 0} · 제외학습 {neg or 0} · 자동기준 {threshold:.3f}');t.setObjectName('muted');l.addWidget(t);q=QLabel(f'레퍼런스 품질 {quality:.0%}');q.setObjectName('tiny');l.addWidget(q)
    def mousePressEvent(self,e):self.clicked.emit(self.pid);super().mousePressEvent(e)


class UIBase(QMainWindow):
    def __init__(self):
        super().__init__();self.db=LibraryDB();self.pid=None;self.scan_thread=None;self.scan_worker=None;self.clusters=[]
        self.setWindowTitle(f'PhotoRef Sorter {VERSION}');self.resize(1180,780);self.setMinimumSize(980,680);self._build();self._style();self.refresh_people();self.refresh_review();self.refresh_settings()
    def _build(self):
        root=QWidget();self.setCentralWidget(root);h=QHBoxLayout(root);h.setContentsMargins(0,0,0,0);h.setSpacing(0)
        side=QFrame();side.setObjectName('side');side.setFixedWidth(220);sl=QVBoxLayout(side);sl.setContentsMargins(18,24,18,24);logo=QLabel('PhotoRef\nSorter');logo.setObjectName('logo');sl.addWidget(logo);ver=QLabel(f'v{VERSION} · Smart Library');ver.setObjectName('sideMuted');sl.addWidget(ver);sl.addSpacing(22)
        self.nav=[]
        for text in ['👤   인물','⚡   대량 분류','✅   검토 / 학습','🧩   미확인 그룹','⚙️   관리']:
            b=QPushButton(text);b.setCheckable(True);b.setObjectName('nav');b.setMinimumHeight(45);sl.addWidget(b);self.nav.append(b)
        sl.addStretch();priv=QLabel('얼굴 특징값은 이 PC에만 저장\n원본 사진은 외부 전송 안 함');priv.setObjectName('sideMuted');sl.addWidget(priv);h.addWidget(side)
        self.stack=QStackedWidget();h.addWidget(self.stack,1)
        for page in [self.people_page(),self.scan_page(),self.review_page(),self.cluster_page(),self.settings_page()]:self.stack.addWidget(page)
        g=QButtonGroup(self);g.setExclusive(True)
        for i,b in enumerate(self.nav):g.addButton(b,i)
        g.idClicked.connect(self.change_page);self.nav[0].setChecked(True)
    def _style(self):
        self.setStyleSheet('''*{font-family:"Segoe UI","Malgun Gothic";font-size:14px}QMainWindow,QWidget{background:#F4F6F8;color:#1F242C}#side{background:#15181E}#logo{color:white;font-size:25px;font-weight:850}#sideMuted{color:#88919E;font-size:11px}#nav{border:none;background:transparent;color:#B8C0CB;text-align:left;border-radius:10px;padding:9px 12px;font-weight:650}#nav:hover{background:#222731;color:white}#nav:checked{background:#2A3040;color:white}#h1{font-size:29px;font-weight:850}#h2{font-size:20px;font-weight:800}#subtitle,#muted{color:#707986}#tiny{color:#8B94A1;font-size:11px}#panel,#personCard{background:white;border:1px solid #DFE4EA;border-radius:14px}#personCard:hover{border:1px solid #9EA9B7;background:#FBFCFE}#personName{font-size:16px;font-weight:800}QLineEdit,QListWidget,QComboBox,QSpinBox{background:white;border:1px solid #D8DEE6;border-radius:9px;padding:7px}QPushButton{border:none;border-radius:9px;background:#E9EDF2;padding:9px 13px;font-weight:650}QPushButton:hover{background:#DDE3EA}#primary,#primaryBig{background:#5263F5;color:white}#primary:hover,#primaryBig:hover{background:#4555E4}#primaryBig{font-size:15px;padding:12px}#danger{background:#FFF0F1;color:#B93D4D}QProgressBar{border:none;background:#E5E9EF;border-radius:7px;height:15px;text-align:center}QProgressBar::chunk{background:#5A69F5;border-radius:7px}''')
    def header(self,title,sub):
        v=QVBoxLayout();a=QLabel(title);a.setObjectName('h1');b=QLabel(sub);b.setObjectName('subtitle');b.setWordWrap(True);v.addWidget(a);v.addWidget(b);return v
    def change_page(self,i):
        self.stack.setCurrentIndex(i)
        if i==0:self.refresh_people()
        elif i==2:self.refresh_review()
        elif i==4:self.refresh_settings()
    def people_page(self):
        w=QWidget();v=QVBoxLayout(w);v.setContentsMargins(34,28,34,28);v.addLayout(self.header('인물 라이브러리','레퍼런스 품질과 개인별 얼굴 변화 폭을 분석해 인물마다 다른 인식 기준을 자동 적용합니다.'))
        row=QHBoxLayout();self.name=QLineEdit();self.name.setPlaceholderText('새 인물 이름');add=QPushButton('인물 추가');add.setObjectName('primary');add.clicked.connect(self.add_person);row.addWidget(self.name,1);row.addWidget(add);v.addLayout(row);v.addSpacing(12)
        split=QHBoxLayout();split.setSpacing(16);left=QScrollArea();left.setWidgetResizable(True);left.setFrameShape(QFrame.NoFrame);inn=QWidget();self.cards=QVBoxLayout(inn);self.cards.setAlignment(Qt.AlignTop);left.setWidget(inn);split.addWidget(left,1)
        right=QFrame();right.setObjectName('panel');r=QVBoxLayout(right);r.setContentsMargins(18,18,18,18);self.person_title=QLabel('인물을 선택하세요');self.person_title.setObjectName('h2');r.addWidget(self.person_title);self.person_detail=QLabel('좋은 레퍼런스 5~20장을 권장합니다.');self.person_detail.setObjectName('muted');r.addWidget(self.person_detail);self.ref_list=QListWidget();r.addWidget(self.ref_list,1)
        br=QHBoxLayout();ar=QPushButton('레퍼런스 추가');ar.setObjectName('primary');ar.clicked.connect(self.add_refs);rr=QPushButton('선택 레퍼런스 삭제');rr.clicked.connect(self.delete_ref);dp=QPushButton('인물 삭제');dp.setObjectName('danger');dp.clicked.connect(self.delete_person);br.addWidget(ar);br.addWidget(rr);br.addWidget(dp);r.addLayout(br);split.addWidget(right,1);v.addLayout(split,1);return w
    def scan_page(self):
        w=QWidget();v=QVBoxLayout(w);v.setContentsMargins(34,28,34,28);v.addLayout(self.header('대량 자동 분류','수만 장도 빠르게: 병렬 AI 분석 + 임베딩 캐시 + 변경 파일만 재분석. 영상은 대표 프레임을 샘플링합니다.'))
        p=QFrame();p.setObjectName('panel');l=QVBoxLayout(p);l.setContentsMargins(20,20,20,20);self.src=QLineEdit();self.dst=QLineEdit()
        for title,edit,cb in [('분류할 폴더',self.src,self.pick_src),('결과 폴더',self.dst,self.pick_dst)]:
            l.addWidget(QLabel(title));rr=QHBoxLayout();rr.addWidget(edit,1);b=QPushButton('선택');b.clicked.connect(cb);rr.addWidget(b);l.addLayout(rr)
        opts=QHBoxLayout();self.recursive=QCheckBox('하위 폴더');self.recursive.setChecked(True);self.multi=QCheckBox('여러 인물 동시 분류');self.multi.setChecked(True);self.hard=QCheckBox('같은 드라이브는 하드링크(초고속/무용량)');opts.addWidget(self.recursive);opts.addWidget(self.multi);opts.addWidget(self.hard);opts.addStretch();l.addLayout(opts)
        perf=QHBoxLayout();perf.addWidget(QLabel('AI 워커'));self.workers=QSpinBox();self.workers.setRange(1,8);self.workers.setValue(min(6,max(2,(os.cpu_count() or 4)//2)));perf.addWidget(self.workers);self.gpu=QCheckBox('DirectML GPU 가속');ok,desc=FaceEngine.gpu_available();self.gpu.setEnabled(ok);self.gpu.setChecked(ok);self.gpu.setToolTip(desc);perf.addWidget(self.gpu);perf.addStretch();l.addLayout(perf)
        tr=QHBoxLayout();tr.addWidget(QLabel('기본 인식 기준'));self.th=QSlider(Qt.Horizontal);self.th.setRange(36,58);self.th.setValue(43);self.tv=QLabel('0.43');self.th.valueChanged.connect(lambda x:self.tv.setText(f'{x/100:.2f}'));tr.addWidget(self.th,1);tr.addWidget(self.tv);l.addLayout(tr)
        actions=QHBoxLayout();self.go=QPushButton('⚡ 고속 분류 시작');self.go.setObjectName('primaryBig');self.go.clicked.connect(self.start_scan);self.stop=QPushButton('중단');self.stop.setEnabled(False);self.stop.clicked.connect(self.stop_scan);actions.addWidget(self.go,1);actions.addWidget(self.stop);l.addLayout(actions);self.pb=QProgressBar();l.addWidget(self.pb);self.scan_status=QLabel('준비됨');self.scan_status.setObjectName('muted');l.addWidget(self.scan_status);self.speed=QLabel('캐시 0 · 새 분석 0 · 0.0장/s');self.speed.setObjectName('tiny');l.addWidget(self.speed);v.addWidget(p);v.addStretch();return w
    def review_page(self):
        w=QWidget();v=QVBoxLayout(w);v.setContentsMargins(34,28,34,28);v.addLayout(self.header('검토 & 자동 학습','애매하거나 미확인인 얼굴을 바로 수정하세요. 확정은 positive 학습, “아님”은 negative 학습으로 다음 분류에 반영됩니다.'))
        bar=QHBoxLayout();reloadb=QPushButton('새로고침');reloadb.clicked.connect(self.refresh_review);bar.addWidget(reloadb);bar.addStretch();v.addLayout(bar);self.review_list=QListWidget();self.review_list.setViewMode(QListWidget.IconMode);self.review_list.setIconSize(QSize(140,140));self.review_list.setResizeMode(QListWidget.Adjust);self.review_list.setSelectionMode(QListWidget.ExtendedSelection);self.review_list.setGridSize(QSize(180,195));v.addWidget(self.review_list,1)
        act=QHBoxLayout();self.assign_combo=QComboBox();act.addWidget(self.assign_combo,1);ap=QPushButton('✓ 이 인물로 확정 + 학습');ap.setObjectName('primary');ap.clicked.connect(self.assign_review);neg=QPushButton('✕ 현재 후보 아님(제외학습)');neg.clicked.connect(self.reject_review_candidate);act.addWidget(ap);act.addWidget(neg);v.addLayout(act);return w
    def cluster_page(self):
        w=QWidget();v=QVBoxLayout(w);v.setContentsMargins(34,28,34,28);v.addLayout(self.header('미확인 인물 자동 그룹','등록되지 않은 얼굴들을 유사도 기반으로 묶습니다. 그룹 하나에 이름을 지정하면 대표 얼굴들을 자동 레퍼런스로 학습합니다.'))
        top=QHBoxLayout();b=QPushButton('🧩 미확인 얼굴 그룹 만들기');b.setObjectName('primary');b.clicked.connect(self.make_clusters);self.cluster_status=QLabel('아직 분석하지 않음');self.cluster_status.setObjectName('muted');top.addWidget(b);top.addWidget(self.cluster_status);top.addStretch();v.addLayout(top);self.cluster_list=QListWidget();self.cluster_list.currentRowChanged.connect(self.show_cluster);v.addWidget(self.cluster_list,1);self.cluster_detail=QLabel('그룹을 선택하세요.');self.cluster_detail.setWordWrap(True);self.cluster_detail.setObjectName('muted');v.addWidget(self.cluster_detail)
        rr=QHBoxLayout();self.cluster_name=QLineEdit();self.cluster_name.setPlaceholderText('새 인물 이름 또는 기존 인물 선택');self.cluster_person=QComboBox();use=QPushButton('그룹을 인물로 등록/학습');use.setObjectName('primary');use.clicked.connect(self.assign_cluster);rr.addWidget(self.cluster_name,1);rr.addWidget(self.cluster_person);rr.addWidget(use);v.addLayout(rr);return w
    def settings_page(self):
        w=QWidget();v=QVBoxLayout(w);v.setContentsMargins(34,28,34,28);v.addLayout(self.header('라이브러리 관리','캐시·지원 포맷·되돌리기·가속 상태를 관리합니다.'))
        p=QFrame();p.setObjectName('panel');l=QVBoxLayout(p);l.setContentsMargins(20,20,20,20);self.sys_info=QLabel();self.sys_info.setWordWrap(True);l.addWidget(self.sys_info);rr=QHBoxLayout();clean=QPushButton('없는 파일 캐시 정리');clean.clicked.connect(self.cleanup_cache);dups=QPushButton('중복/연사 통계');dups.clicked.connect(self.duplicate_stats);clear=QPushButton('전체 분석 캐시 비우기');clear.setObjectName('danger');clear.clicked.connect(self.clear_cache);undo=QPushButton('마지막 분류 되돌리기');undo.clicked.connect(self.undo_last);rr.addWidget(clean);rr.addWidget(dups);rr.addWidget(clear);rr.addWidget(undo);l.addLayout(rr);formats=QLabel('지원: JPG/PNG/WebP/TIFF/BMP · HEIC/HEIF · AVIF · GIF · RAW(DNG/CR2/CR3/NEF/ARW 등) · MP4/MOV/MKV/AVI/WebM');formats.setWordWrap(True);formats.setObjectName('muted');l.addWidget(formats);v.addWidget(p);v.addStretch();return w
