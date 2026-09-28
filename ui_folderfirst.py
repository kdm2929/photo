from __future__ import annotations

import os
import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QProgressBar, QPushButton, QSlider, QSpinBox, QTabWidget,
    QVBoxLayout, QWidget
)

import storage
from portable_config import BootstrapState, SettingsStore, write_bootstrap


class FolderFirstController:
    def __init__(self, window, state: BootstrapState, store: SettingsStore):
        self.w = window
        self.state = state
        self.store = store
        self.more_open = False
        self._simplify_navigation()
        self._replace_home()
        self._replace_settings()
        self.apply_settings_to_ui()

    def _simplify_navigation(self):
        labels = ['⌂   분석 시작', '▦   사진', '👤   인물', '✓   검토']
        for i, text in enumerate(labels):
            self.w.nav[i].setText(text)
        self.w.nav[9].setText('⚙   설정')
        self.extra_indexes = [4, 5, 6, 7, 8]
        for i in self.extra_indexes:
            self.w.nav[i].setVisible(False)
        side = self.w.nav[0].parentWidget()
        layout = side.layout()
        self.more_btn = QPushButton('⋯   더보기')
        self.more_btn.setObjectName('nav')
        self.more_btn.setMinimumHeight(42)
        self.more_btn.clicked.connect(self.toggle_more)
        pos = layout.indexOf(self.w.nav[3]) + 1
        layout.insertWidget(pos, self.more_btn)
        try:
            layout.removeWidget(self.w.nav[9])
            layout.insertWidget(pos + 1, self.w.nav[9])
        except Exception:
            pass

    def toggle_more(self):
        self.more_open = not self.more_open
        for i in self.extra_indexes:
            self.w.nav[i].setVisible(self.more_open)
        self.more_btn.setText('⌃   간단히 보기' if self.more_open else '⋯   더보기')

    def _replace_stack_widget(self, index: int, widget: QWidget):
        old = self.w.stack.widget(index)
        self.w.stack.removeWidget(old)
        old.setParent(None)
        self.w.stack.insertWidget(index, widget)

    def _replace_home(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(42, 34, 42, 34)
        outer.setSpacing(16)

        title = QLabel('사진과 영상을 자동으로 인물별 정리')
        title.setObjectName('h1')
        sub = QLabel('폴더 하나만 지정하면 PhotoRef가 파일을 읽고 얼굴을 찾아 등록된 인물과 비교한 뒤 결과 폴더로 분류합니다.')
        sub.setObjectName('subtitle'); sub.setWordWrap(True)
        outer.addWidget(title); outer.addWidget(sub)

        card = QFrame(); card.setObjectName('panel')
        c = QVBoxLayout(card); c.setContentsMargins(24, 22, 24, 22); c.setSpacing(12)
        step = QLabel('1  분석할 폴더 선택')
        step.setObjectName('h2'); c.addWidget(step)
        r = QHBoxLayout()
        self.home_source = QLineEdit(); self.home_source.setPlaceholderText('사진/영상이 들어 있는 폴더')
        b = QPushButton('폴더 선택'); b.setObjectName('primary'); b.clicked.connect(self.pick_home_source)
        r.addWidget(self.home_source, 1); r.addWidget(b); c.addLayout(r)

        c.addWidget(QLabel('결과 저장 위치'))
        r2 = QHBoxLayout()
        self.home_output = QLineEdit(); self.home_output.setPlaceholderText('비워두면 원본 폴더 안에 자동 생성')
        b2 = QPushButton('변경'); b2.clicked.connect(self.pick_home_output)
        r2.addWidget(self.home_output, 1); r2.addWidget(b2); c.addLayout(r2)

        opts = QHBoxLayout()
        self.home_recursive = QCheckBox('하위 폴더 포함')
        self.home_images = QCheckBox('사진')
        self.home_videos = QCheckBox('영상')
        self.home_multi = QCheckBox('여러 인물 모두 분류')
        for x in (self.home_recursive, self.home_images, self.home_videos, self.home_multi): opts.addWidget(x)
        opts.addStretch(); c.addLayout(opts)

        self.home_people = QLabel('등록된 인물 확인 중…')
        self.home_people.setObjectName('muted'); c.addWidget(self.home_people)
        self.home_start = QPushButton('⚡  분석 및 자동 분류 시작')
        self.home_start.setObjectName('primaryBig'); self.home_start.setMinimumHeight(52)
        self.home_start.clicked.connect(self.start_from_home); c.addWidget(self.home_start)
        outer.addWidget(card)

        progress = QFrame(); progress.setObjectName('panel')
        p = QVBoxLayout(progress); p.setContentsMargins(20, 16, 20, 16)
        self.home_progress_title = QLabel('준비됨'); self.home_progress_title.setObjectName('h2'); p.addWidget(self.home_progress_title)
        self.home_progress = QProgressBar(); p.addWidget(self.home_progress)
        self.home_progress_detail = QLabel('폴더를 선택하고 분석을 시작하세요.'); self.home_progress_detail.setObjectName('muted'); p.addWidget(self.home_progress_detail)
        self.home_progress_advanced = QLabel(''); self.home_progress_advanced.setObjectName('tiny'); p.addWidget(self.home_progress_advanced)
        outer.addWidget(progress)

        hint = QLabel('흐름:  ① 인물 등록  →  ② 폴더 선택  →  ③ 자동 분석/분류  →  ④ 애매한 얼굴만 검토')
        hint.setObjectName('muted'); outer.addWidget(hint); outer.addStretch()
        self._replace_stack_widget(0, page)

    def _replace_settings(self):
        page = QWidget(); root = QVBoxLayout(page); root.setContentsMargins(30, 26, 30, 26)
        t = QLabel('설정'); t.setObjectName('h1'); root.addWidget(t)
        s = QLabel('평소에는 기본값 그대로 사용해도 됩니다. 고급 옵션은 여기에서만 관리합니다.')
        s.setObjectName('subtitle'); root.addWidget(s)
        tabs = QTabWidget(); root.addWidget(tabs, 1)

        general = QWidget(); f = QFormLayout(general)
        self.set_theme = QComboBox(); self.set_theme.addItem('다크', 'dark'); self.set_theme.addItem('라이트', 'light')
        self.set_remember = QCheckBox('마지막 폴더 기억')
        self.set_output_subdir = QLineEdit()
        f.addRow('테마', self.set_theme); f.addRow('', self.set_remember); f.addRow('기본 결과 폴더 이름', self.set_output_subdir)
        tabs.addTab(general, '일반')

        analysis = QWidget(); af = QFormLayout(analysis)
        self.set_images = QCheckBox('사진 분석'); self.set_videos = QCheckBox('영상 분석'); self.set_raw = QCheckBox('RAW 포함')
        media_row = QWidget(); mh = QHBoxLayout(media_row); mh.setContentsMargins(0,0,0,0)
        mh.addWidget(self.set_images); mh.addWidget(self.set_videos); mh.addWidget(self.set_raw); mh.addStretch()
        self.set_recursive = QCheckBox('하위 폴더 포함'); self.set_multi = QCheckBox('한 파일에 여러 인물이 있으면 각 인물 폴더에 모두 분류')
        self.set_threshold = QSlider(Qt.Horizontal); self.set_threshold.setRange(36, 58)
        self.set_threshold_label = QLabel(); self.set_threshold.valueChanged.connect(lambda x:self.set_threshold_label.setText(f'{x/100:.2f}'))
        thw = QWidget(); thl=QHBoxLayout(thw); thl.setContentsMargins(0,0,0,0); thl.addWidget(self.set_threshold,1); thl.addWidget(self.set_threshold_label)
        af.addRow('미디어', media_row); af.addRow('', self.set_recursive); af.addRow('', self.set_multi); af.addRow('기본 인식 기준', thw)
        tabs.addTab(analysis, '분석/인식')

        perf = QWidget(); pf = QFormLayout(perf)
        self.set_auto_workers = QCheckBox('CPU/저장장치 자동 튜닝')
        self.set_workers = QSpinBox(); self.set_workers.setRange(1,8)
        self.set_gpu = QCheckBox('GPU 가속 사용 (가능한 경우 DirectML)')
        self.set_hardlink = QCheckBox('같은 드라이브에서는 하드링크로 빠르게 분류')
        self.set_cache = QCheckBox('기존 분석 캐시 재사용')
        pf.addRow('', self.set_auto_workers); pf.addRow('수동 워커 수', self.set_workers); pf.addRow('', self.set_gpu); pf.addRow('', self.set_hardlink); pf.addRow('', self.set_cache)
        tabs.addTab(perf, '성능')

        storage_tab = QWidget(); sf = QFormLayout(storage_tab)
        self.storage_mode = QComboBox(); self.storage_mode.addItem('포터블 — EXE 옆 data 폴더', 'portable'); self.storage_mode.addItem('사용자 프로필 — LOCALAPPDATA', 'profile'); self.storage_mode.addItem('사용자 지정', 'custom')
        self.custom_root = QLineEdit(); cb = QPushButton('찾기'); cb.clicked.connect(self.pick_custom_root)
        cr = QWidget(); crh=QHBoxLayout(cr); crh.setContentsMargins(0,0,0,0); crh.addWidget(self.custom_root,1); crh.addWidget(cb)
        self.storage_info = QLabel(); self.storage_info.setWordWrap(True); self.storage_info.setObjectName('muted')
        apply_storage = QPushButton('저장 위치 적용 (재시작 필요)'); apply_storage.clicked.connect(self.save_storage_mode)
        open_data = QPushButton('현재 데이터 폴더 열기'); open_data.clicked.connect(self.open_data_folder)
        sf.addRow('데이터 저장 방식', self.storage_mode); sf.addRow('사용자 지정 경로', cr); sf.addRow('', self.storage_info); sf.addRow('', apply_storage); sf.addRow('', open_data)
        tabs.addTab(storage_tab, '저장 위치')

        maintain = QWidget(); mf = QVBoxLayout(maintain)
        cache_label = QLabel('DB, 얼굴 분석 캐시, 썸네일, 설정과 작업 기록을 한곳에서 관리합니다.'); cache_label.setWordWrap(True); cache_label.setObjectName('muted'); mf.addWidget(cache_label)
        for text, cbk in [
            ('없는 원본 파일 캐시 정리', self.w.cleanup_cache),
            ('전체 분석/썸네일 캐시 비우기', self.w.clear_cache),
            ('마지막 분류 작업 되돌리기', self.w.undo_last),
        ]:
            bb=QPushButton(text); bb.clicked.connect(cbk); mf.addWidget(bb)
        ex = QPushButton('설정 내보내기'); ex.clicked.connect(self.export_settings); im = QPushButton('설정 가져오기'); im.clicked.connect(self.import_settings)
        rr=QHBoxLayout(); rr.addWidget(ex); rr.addWidget(im); mf.addLayout(rr); mf.addStretch()
        tabs.addTab(maintain, '유지보수')

        save = QPushButton('설정 저장'); save.setObjectName('primaryBig'); save.clicked.connect(self.save_settings); root.addWidget(save)
        self._replace_stack_widget(9, page)

    def pick_home_source(self):
        p = QFileDialog.getExistingDirectory(self.w, '분석할 사진/영상 폴더')
        if p:
            self.home_source.setText(p)
            if not self.home_output.text().strip():
                self.home_output.setText(str(Path(p) / self.store.get('output_subdir', 'PhotoRef_Result')))

    def pick_home_output(self):
        p = QFileDialog.getExistingDirectory(self.w, '결과 저장 폴더')
        if p: self.home_output.setText(p)

    def pick_custom_root(self):
        p = QFileDialog.getExistingDirectory(self.w, '데이터 저장 루트 폴더')
        if p: self.custom_root.setText(p)

    def apply_settings_to_ui(self):
        d = self.store.data
        self.home_source.setText(d.get('last_source',''))
        self.home_output.setText(d.get('last_output',''))
        self.home_recursive.setChecked(bool(d.get('recursive',True)))
        self.home_images.setChecked(bool(d.get('include_images',True)))
        self.home_videos.setChecked(bool(d.get('include_videos',True)))
        self.home_multi.setChecked(bool(d.get('multi_person',True)))
        self.set_theme.setCurrentIndex(max(0,self.set_theme.findData(d.get('theme','dark'))))
        self.set_remember.setChecked(bool(d.get('remember_last_folder',True))); self.set_output_subdir.setText(str(d.get('output_subdir','PhotoRef_Result')))
        self.set_images.setChecked(bool(d.get('include_images',True))); self.set_videos.setChecked(bool(d.get('include_videos',True))); self.set_raw.setChecked(bool(d.get('include_raw',True)))
        self.set_recursive.setChecked(bool(d.get('recursive',True))); self.set_multi.setChecked(bool(d.get('multi_person',True)))
        self.set_threshold.setValue(int(round(float(d.get('recognition_threshold',.43))*100)))
        self.set_auto_workers.setChecked(bool(d.get('auto_workers',True))); self.set_workers.setValue(int(d.get('workers',4)))
        self.set_gpu.setChecked(bool(d.get('gpu',True))); self.set_hardlink.setChecked(bool(d.get('hardlink',False))); self.set_cache.setChecked(bool(d.get('reuse_cache',True)))
        self.storage_mode.setCurrentIndex(max(0,self.storage_mode.findData(self.state.mode))); self.custom_root.setText(self.state.custom_root)
        extra = f"\n기존 데이터 자동 이전: {self.state.migrated_from}" if self.state.migrated_from else ''
        fallback = '\n※ EXE 폴더에 쓸 수 없어 사용자 프로필로 자동 전환되었습니다.' if self.state.portable_fallback else ''
        self.storage_info.setText(f"현재 데이터: {self.state.data_dir}\n설정 파일: {self.store.path}{extra}{fallback}")
        self.w.apply_theme(d.get('theme','dark') == 'dark')
        self.refresh_people_count()

    def collect_settings(self):
        return {
            'theme': self.set_theme.currentData(), 'remember_last_folder': self.set_remember.isChecked(), 'output_subdir': self.set_output_subdir.text().strip() or 'PhotoRef_Result',
            'include_images': self.set_images.isChecked(), 'include_videos': self.set_videos.isChecked(), 'include_raw': self.set_raw.isChecked(),
            'recursive': self.set_recursive.isChecked(), 'multi_person': self.set_multi.isChecked(), 'recognition_threshold': self.set_threshold.value()/100,
            'auto_workers': self.set_auto_workers.isChecked(), 'workers': self.set_workers.value(), 'gpu': self.set_gpu.isChecked(), 'hardlink': self.set_hardlink.isChecked(), 'reuse_cache': self.set_cache.isChecked(),
        }

    def save_settings(self):
        d = self.collect_settings(); self.store.save(d); self.apply_settings_to_ui(); QMessageBox.information(self.w,'설정','설정을 저장했습니다.')

    def save_storage_mode(self):
        mode = self.storage_mode.currentData(); custom = self.custom_root.text().strip()
        if mode == 'custom' and not custom:
            QMessageBox.warning(self.w,'경로','사용자 지정 폴더를 선택하세요.'); return
        try:
            write_bootstrap(mode, custom, self.state.app_dir)
            QMessageBox.information(self.w,'저장 위치','저장 위치 설정을 기록했습니다. 프로그램을 다시 실행하면 적용됩니다. 기존 데이터는 자동 이전 대상이 됩니다.')
        except Exception as e:
            QMessageBox.critical(self.w,'저장 실패',str(e))

    def open_data_folder(self):
        self.state.data_dir.mkdir(parents=True, exist_ok=True)
        try: os.startfile(str(self.state.data_dir))
        except Exception: pass

    def export_settings(self):
        p,_=QFileDialog.getSaveFileName(self.w,'설정 내보내기','PhotoRefSorter-settings.json','JSON (*.json)')
        if p:
            self.store.save(self.collect_settings()); shutil.copy2(self.store.path,p)

    def import_settings(self):
        p,_=QFileDialog.getOpenFileName(self.w,'설정 가져오기','','JSON (*.json)')
        if p:
            shutil.copy2(p,self.store.path); self.store.load(); self.apply_settings_to_ui(); QMessageBox.information(self.w,'설정','설정을 가져왔습니다.')

    def start_from_home(self):
        src = Path(self.home_source.text().strip())
        if not src.is_dir(): QMessageBox.warning(self.w,'폴더','분석할 폴더를 선택하세요.'); return
        if not (self.home_images.isChecked() or self.home_videos.isChecked()): QMessageBox.warning(self.w,'미디어','사진 또는 영상 중 하나 이상을 선택하세요.'); return
        out = Path(self.home_output.text().strip()) if self.home_output.text().strip() else src / (self.store.get('output_subdir','PhotoRef_Result'))
        self.home_output.setText(str(out))
        d = self.collect_settings(); d.update({'recursive':self.home_recursive.isChecked(),'include_images':self.home_images.isChecked(),'include_videos':self.home_videos.isChecked(),'multi_person':self.home_multi.isChecked()})
        if d.get('remember_last_folder',True): d.update({'last_source':str(src),'last_output':str(out)})
        self.store.save(d)

        allowed=set()
        if d['include_images']: allowed |= set(storage.IMAGE_EXTS)
        if d.get('include_raw',True): allowed |= set(storage.RAW_EXTS)
        if d['include_videos']: allowed |= set(storage.VIDEO_EXTS)
        storage.SUPPORTED = allowed
        if not d.get('reuse_cache',True): self.w.db.clear_cache()

        self.w.src.setText(str(src)); self.w.dst.setText(str(out)); self.w.recursive.setChecked(d['recursive']); self.w.multi.setChecked(d['multi_person'])
        self.w.th.setValue(int(round(float(d['recognition_threshold'])*100))); self.w.auto_workers.setChecked(d['auto_workers']); self.w.workers.setValue(int(d['workers']))
        self.w.gpu.setChecked(bool(d['gpu']) and self.w.gpu.isEnabled()); self.w.hard.setChecked(bool(d['hardlink']))
        self.home_start.setEnabled(False); self.home_progress_title.setText('분석 준비 중…'); self.home_progress.setValue(0)
        self.w.start_scan()

    def refresh_people_count(self):
        try:
            n=len(self.w.db.people()); self.home_people.setText(f'등록된 인물 {n:,}명 · 인물이 없으면 먼저 왼쪽 “인물”에서 레퍼런스를 등록하세요.')
        except Exception: self.home_people.setText('등록 인물 정보를 읽을 수 없습니다.')

    def on_progress(self, i, n, name, speed, cached, new):
        self.home_progress.setValue(int(i*100/max(1,n))); eta=(n-i)/max(.01,speed)
        self.home_progress_title.setText(f'분석 중 · {i:,} / {n:,}')
        self.home_progress_detail.setText(f'{name} · 약 {eta/60:.1f}분 남음')
        if self.store.get('show_advanced_progress',False): self.home_progress_advanced.setText(f'캐시 {cached:,} · 새 AI {new:,} · {speed:.1f}개/s')
        else: self.home_progress_advanced.setText('')

    def on_status(self, text: str):
        if self.home_progress.value()==0: self.home_progress_detail.setText(text)

    def on_done(self, stats: dict):
        self.home_start.setEnabled(True); self.home_progress.setValue(100); self.home_progress_title.setText('분류 완료')
        self.home_progress_detail.setText(f"전체 {stats.get('전체',0):,} · 자동 분류 {stats.get('분류',0):,} · 확인 필요 {stats.get('확인필요',0):,} · 미확인 {stats.get('미확인',0):,}")
        self.refresh_people_count()

    def on_fail(self, text: str):
        self.home_start.setEnabled(True); self.home_progress_title.setText('분석 오류'); self.home_progress_detail.setText(text)


def install_folder_first_ui(window, state: BootstrapState, store: SettingsStore):
    ctl=FolderFirstController(window,state,store); window._folder_ui=ctl; return ctl
