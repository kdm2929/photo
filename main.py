from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QFileDialog, QListWidgetItem, QMessageBox

from recognition import Clusterer, FaceEngine, RecognitionIndex, build_models
from scanner import ScanEngine
from storage import APP, VERSION, LibraryDB, ScanConfig, DB, pillow_heif, rawpy
from ui_base import UIBase, face_crop_pixmap, preview_pixmap


class ScanWorker(QObject):
    progress=Signal(int,int,str,float,int,int); status=Signal(str); done=Signal(dict); fail=Signal(str)
    def __init__(self,cfg): super().__init__(); self.cfg=cfg; self.stop=False
    def request_stop(self): self.stop=True
    def run(self):
        try:
            e=ScanEngine(self.cfg,lambda *a:self.progress.emit(*a),lambda s:self.status.emit(s),lambda:self.stop)
            self.done.emit(e.run())
        except Exception as ex:self.fail.emit(str(ex))


class Main(UIBase):
    def __init__(self):
        super().__init__(); self.refresh_home(); self.refresh_gallery_filters(); self.refresh_people(); self.refresh_albums(); self.refresh_settings()

    def refresh_home(self):
        d=self.db.dashboard()
        for k in ('media','people','matched','review','unknown'): self.home_cards[k].value.setText(f"{d[k]:,}")
        self.home_cards['media'].sub.setText(f"검출 얼굴 {d['faces']:,}"); self.home_cards['people'].sub.setText(f"앨범 {d['albums']:,}"); self.home_cards['review'].sub.setText('키보드 검토 가능')
        self.home_recent.clear()
        for when,summary in d['recent']: self.home_recent.addItem(f"{when}  ·  {summary or '분류 작업'}")
        pending=d['review']+d['unknown']
        self.home_health.setText(f"현재 {pending:,}개의 얼굴이 사람 확인을 기다리고 있습니다.\n\n미확인 그룹 기능으로 반복 등장하는 얼굴을 자동으로 묶은 뒤 이름을 한 번만 붙이면 대표 얼굴들이 자동 학습됩니다." if pending else "현재 검토 대기 얼굴이 없습니다. 라이브러리가 깔끔하게 정리된 상태입니다.")

    def refresh_gallery_filters(self):
        person=self.photo_person.currentData(); album=self.photo_album.currentData(); month=self.photo_month.currentData()
        for box in (self.photo_person,self.photo_album,self.photo_month): box.blockSignals(True); box.clear()
        self.photo_person.addItem('모든 인물',None); self.batch_person.clear()
        for r in self.db.people(): self.photo_person.addItem(r[1],r[0]); self.batch_person.addItem(r[1],r[0])
        self.photo_album.addItem('모든 앨범',None); self.batch_album.clear()
        for aid,name,count in self.db.albums(): self.photo_album.addItem(f'{name} ({count})',aid); self.batch_album.addItem(name,aid)
        self.photo_month.addItem('전체 타임라인',None)
        for m,count in self.db.timeline_months(): self.photo_month.addItem(f'{m} ({count:,})',m)
        def restore(box,data):
            i=box.findData(data); box.setCurrentIndex(max(0,i))
        restore(self.photo_person,person); restore(self.photo_album,album); restore(self.photo_month,month)
        for box in (self.photo_person,self.photo_album,self.photo_month): box.blockSignals(False)

    def refresh_gallery(self):
        self.gallery_model.refresh(self.photo_search.text().strip(),self.photo_person.currentData(),self.photo_album.currentData(),self.photo_month.currentData())
        self.gallery_count.setText(f"{self.gallery_model.total:,}개 미디어 · 현재 {len(self.gallery_model.rows):,}개 로드됨 (스크롤 시 추가 로드)")
        self.ins_preview.clear(); self.ins_name.clear(); self.ins_meta.clear(); self.face_strip.clear(); self.ins_albums.clear()

    def selected_gallery_paths(self):
        return self.gallery_model.selected_paths(self.gallery.selectionModel().selectedIndexes())

    def gallery_selected(self,index):
        path=self.gallery_model.path_at(index.row())
        if not path:return
        self.gallery_model.prefetch_around(index.row()); px=preview_pixmap(path,640)
        self.ins_preview.setPixmap(px.scaled(self.ins_preview.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)) if not px.isNull() else self.ins_preview.setText('미리보기 없음')
        self.ins_name.setText(Path(path).name); detail=self.db.media_detail(path); self.face_strip.clear()
        if not detail:return
        mc,faces,albums=detail; cap=mc.get('captured_at') or '-'; dim=f"{mc.get('width',0)}×{mc.get('height',0)}"; dur=f" · {float(mc.get('duration') or 0):.1f}s" if mc.get('duration') else ''
        self.ins_meta.setText(f"{cap}\n{dim}{dur} · {mc.get('kind','image')} · 얼굴 {mc.get('face_count',0)}\n{path}")
        for fs in faces:
            idx=int(fs['face_idx']); cand=self.db.top_candidates(path,idx); label='\n'.join([f"{i+1}. {n} {s:.0%}" for i,(_,n,s) in enumerate(cand)]) or '미확인'
            it=QListWidgetItem(QIcon(face_crop_pixmap(self.db,path,idx,96)),label); it.setData(Qt.UserRole,(path,idx)); self.face_strip.addItem(it)
        self.ins_albums.setText('앨범: '+(', '.join(n for _,n in albums) if albums else '없음'))

    def face_strip_assign(self,item):
        pid=self.batch_person.currentData(); data=item.data(Qt.UserRole)
        if pid is None or not data:return
        path,idx=data; m=self.db.cache_get(Path(path))
        if m and 0<=idx<len(m.faces):
            f=m.faces[idx]; self.db.add_ref(pid,path,f.embedding,f.quality,False,'face-strip'); self.db.mark_state(path,idx,'learned'); self.refresh_people(); QMessageBox.information(self,'학습','선택한 얼굴을 현재 인물에 학습했습니다.')

    def open_selected_file(self,index=None):
        path=self.gallery_model.path_at(index.row()) if index is not None else (self.selected_gallery_paths()[0] if self.selected_gallery_paths() else None)
        if path and Path(path).exists():
            try: os.startfile(path)
            except Exception: pass

    def gallery_favorite(self,value=True):
        paths=self.selected_gallery_paths()
        if paths:self.db.set_favorite(paths,value); self.refresh_gallery()

    def gallery_add_reference(self):
        pid=self.batch_person.currentData(); paths=self.selected_gallery_paths()
        if pid is None or not paths: QMessageBox.information(self,'선택','사진과 인물을 선택하세요.');return
        eng=FaceEngine(False); ok=0;fail=0; QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for p in paths:
                try:v,q,_=eng.reference(Path(p));self.db.add_ref(pid,p,v,q,False,'gallery-batch');ok+=1
                except Exception:fail+=1
        finally:QApplication.restoreOverrideCursor()
        self.refresh_people(); QMessageBox.information(self,'레퍼런스',f'{ok}개 학습 완료 · 실패 {fail}개')

    def gallery_add_album(self):
        aid=self.batch_album.currentData();paths=self.selected_gallery_paths()
        if aid is not None and paths:self.db.add_to_album(aid,paths);self.refresh_gallery_filters();QMessageBox.information(self,'앨범',f'{len(paths)}개 항목을 앨범에 추가했습니다.')

    def refresh_people(self):
        self.people_list.clear(); models=build_models(self.db); current=self.pid; self.cluster_person.clear();self.cluster_person.addItem('기존 인물 선택',None); self.batch_person.clear()
        for r in self.db.people():
            pid,name,pos,neg,_,hero=r; refs=self.db.refs(pid,False); q=float(np.mean([x[3] for x in refs])) if refs else 0; th=models[pid].threshold if pid in models else .43
            it=QListWidgetItem(f"{name}\n{int(pos or 0)} refs · {int(neg or 0)} negative · 기준 {th:.3f}");it.setData(Qt.UserRole,pid);it.setIcon(QIcon(preview_pixmap(hero,90)) if hero else QIcon());self.people_list.addItem(it);self.cluster_person.addItem(name,pid);self.batch_person.addItem(name,pid)
            if pid==current:self.people_list.setCurrentItem(it)
        self.refresh_gallery_filters()

    def add_person(self):
        n=self.person_new.text().strip()
        if not n:return
        try:self.pid=self.db.add_person(n);self.person_new.clear();self.refresh_people()
        except sqlite3.IntegrityError:QMessageBox.warning(self,'중복','이미 등록된 이름입니다.')

    def person_selected(self,current,previous=None):
        if not current:return
        self.pid=current.data(Qt.UserRole); p=self.db.person_profile(self.pid)
        if not p:return
        models=build_models(self.db);th=models[self.pid].threshold if self.pid in models else .43
        self.person_title.setText(p['name']);self.person_stats.setText(f"인식 사진 {p['photos']:,} · positive {p['positive']} · negative {p['negative']}\n평균 레퍼런스 품질 {p['avg_quality']:.0%} · 자동 threshold {th:.3f}\n최근 사진을 더블클릭하면 대표사진으로 고정합니다.")
        hero=preview_pixmap(p['hero'],320) if p['hero'] else None;self.person_hero.setPixmap(hero.scaled(190,190,Qt.KeepAspectRatioByExpanding,Qt.SmoothTransformation)) if hero and not hero.isNull() else self.person_hero.setText('대표사진 없음')
        self.ref_list.clear()
        for rid,path,emb,q,neg,source in self.db.refs(self.pid,None):
            it=QListWidgetItem(f"{'🚫' if neg else '✓'} {Path(path).name if path else source} · 품질 {q:.0%} · {source}");it.setData(Qt.UserRole,rid);self.ref_list.addItem(it)
        self.confusion_list.clear(); conf=self.db.confusion(self.pid)
        for opid,name,n,margin in conf:self.confusion_list.addItem(f'{name} · {n}회  · 평균 점수차 {float(margin):.3f}')
        risky=[x for x in conf if x[3] is not None and float(x[3])<.06]
        self.person_warning.setText('⚠ 닮은 인물 경고: '+', '.join(x[1] for x in risky[:3])+' — 서로 다른 각도의 레퍼런스를 추가하면 구분력이 좋아집니다.' if risky else '✓ 현재 크게 혼동되는 인물이 없습니다.')
        self.person_recent.clear()
        for path,cap,score in p['recent']:
            it=QListWidgetItem(QIcon(preview_pixmap(path,100)),f"{float(score):.0%}\n{(cap or '')[:10]}");it.setData(Qt.UserRole,path);self.person_recent.addItem(it)

    def add_refs(self):
        if not self.pid:QMessageBox.information(self,'선택','인물을 먼저 선택하세요.');return
        ps,_=QFileDialog.getOpenFileNames(self,'레퍼런스 선택','','사진/영상 (*.*)')
        if not ps:return
        eng=FaceEngine(False);ok=0;fail=[];QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for p in ps:
                try:v,q,_=eng.reference(Path(p));self.db.add_ref(self.pid,p,v,q,False,'manual');ok+=1
                except Exception as ex:fail.append(Path(p).name)
        finally:QApplication.restoreOverrideCursor()
        self.refresh_people();self.person_selected(self.people_list.currentItem());QMessageBox.information(self,'레퍼런스',f'{ok}개 등록 · 실패 {len(fail)}개')

    def delete_ref(self):
        it=self.ref_list.currentItem()
        if it:self.db.delete_ref(int(it.data(Qt.UserRole)));self.refresh_people();self.person_selected(self.people_list.currentItem())

    def delete_person(self):
        if self.pid and QMessageBox.question(self,'삭제','이 인물과 학습 데이터를 삭제할까요?')==QMessageBox.Yes:self.db.delete_person(self.pid);self.pid=None;self.refresh_people();self.person_title.setText('인물을 선택하세요')

    def set_hero_from_recent(self,item):
        if self.pid:self.db.set_person_hero(self.pid,item.data(Qt.UserRole));self.person_selected(self.people_list.currentItem());self.refresh_people()

    def refresh_review(self):
        self.review_rows=list(self.db.review_items(5000)); self.review_pos=min(self.review_pos,max(0,len(self.review_rows)-1)); self.show_review()

    def show_review(self):
        if not self.review_rows:
            self.review_progress.setText('검토 대기 0');self.review_image.setText('검토할 얼굴이 없습니다.');self.review_face.clear();self.review_state.clear();self.review_path.clear()
            for b in self.review_candidate_buttons:b.setText('후보 없음');b.setEnabled(False)
            return
        r=self.review_rows[self.review_pos];path=r[0];idx=int(r[1]);state=r[2]
        self.review_progress.setText(f'{self.review_pos+1:,} / {len(self.review_rows):,}');px=preview_pixmap(path,900);self.review_image.setPixmap(px.scaled(self.review_image.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)) if not px.isNull() else self.review_image.setText('미리보기 없음');fp=face_crop_pixmap(self.db,path,idx,180);self.review_face.setPixmap(fp);self.review_path.setText(path);self.review_state.setText('확인 필요' if state=='review' else '미확인 얼굴')
        cand=self.db.top_candidates(path,idx)
        for i,b in enumerate(self.review_candidate_buttons):
            if i<len(cand):pid,name,s=cand[i];b.setText(f'{i+1} · {name}   {s:.1%}');b.setProperty('pid',pid);b.setEnabled(True)
            else:b.setText(f'{i+1} · 후보 없음');b.setProperty('pid',None);b.setEnabled(False)

    def review_choose(self,n):
        if not self.review_rows or n<0 or n>=len(self.review_candidate_buttons):return
        pid=self.review_candidate_buttons[n].property('pid')
        if pid is None:return
        r=self.review_rows[self.review_pos];path=r[0];idx=int(r[1]);m=self.db.cache_get(Path(path))
        if m and 0<=idx<len(m.faces):
            f=m.faces[idx];self.db.add_ref(int(pid),path,f.embedding,f.quality,False,'review-hotkey');self.db.mark_state(path,idx,'learned');self.review_rows.pop(self.review_pos);self.review_pos=min(self.review_pos,max(0,len(self.review_rows)-1));self.show_review();self.refresh_people()

    def review_reject(self):
        if not self.review_rows:return
        r=self.review_rows[self.review_pos];path=r[0];idx=int(r[1]);pid=r[3];m=self.db.cache_get(Path(path))
        if pid and m and 0<=idx<len(m.faces):
            f=m.faces[idx];self.db.add_ref(int(pid),path,f.embedding,f.quality,True,'review-negative');self.db.mark_state(path,idx,'unknown')
        self.review_rows.pop(self.review_pos);self.review_pos=min(self.review_pos,max(0,len(self.review_rows)-1));self.show_review();self.refresh_people()

    def review_next(self):
        if self.review_rows:self.review_pos=(self.review_pos+1)%len(self.review_rows);self.show_review()
    def review_prev(self):
        if self.review_rows:self.review_pos=(self.review_pos-1)%len(self.review_rows);self.show_review()

    def refresh_cluster_people(self):
        cur=self.cluster_person.currentData();self.cluster_person.clear();self.cluster_person.addItem('기존 인물 선택',None)
        for r in self.db.people():self.cluster_person.addItem(r[1],r[0])
        i=self.cluster_person.findData(cur);self.cluster_person.setCurrentIndex(max(0,i))

    def make_clusters(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:items=self.db.unknown_faces(40000);self.clusters=Clusterer(.475).cluster(items)
        finally:QApplication.restoreOverrideCursor()
        self.cluster_list.clear()
        for i,c in enumerate(self.clusters):
            q=np.mean([x[2].quality for x in c['items']]);self.cluster_list.addItem(f"인물 그룹 {i+1:03d} · {len(c['items']):,}개 · 품질 {q:.0%}")
        self.cluster_status.setText(f'{len(items):,}개 미확인 얼굴 → {len(self.clusters):,}개 반복 인물 그룹')
        if self.clusters:self.cluster_list.setCurrentRow(0)

    def show_cluster(self,row):
        self.cluster_preview.clear()
        if row<0 or row>=len(self.clusters):return
        c=self.clusters[row]
        for path,idx,f,*_ in c['items'][:32]:
            it=QListWidgetItem(QIcon(face_crop_pixmap(self.db,path,idx,110)),Path(path).name);it.setToolTip(path);self.cluster_preview.addItem(it)
        self.cluster_detail.setText(f"이 그룹에는 {len(c['items']):,}개의 얼굴이 있습니다. 확정하면 품질과 다양성을 기준으로 최대 24개 얼굴만 레퍼런스로 채택합니다.")

    def assign_cluster(self):
        row=self.cluster_list.currentRow()
        if row<0 or row>=len(self.clusters):return
        pid=self.cluster_person.currentData();name=self.cluster_name.text().strip()
        if pid is None:
            if not name:QMessageBox.information(self,'이름','새 인물 이름을 입력하거나 기존 인물을 선택하세요.');return
            try:pid=self.db.add_person(name)
            except sqlite3.IntegrityError:QMessageBox.warning(self,'중복','같은 이름이 이미 있습니다.');return
        c=self.clusters[row];chosen=[]
        for item in sorted(c['items'],key=lambda x:x[2].quality,reverse=True):
            f=item[2]
            if f.quality<.25:continue
            if not chosen or max(float(np.dot(f.embedding,z[2].embedding)) for z in chosen)<.93:chosen.append(item)
            if len(chosen)>=24:break
        if not chosen:chosen=c['items'][:min(12,len(c['items']))]
        for path,idx,f,*_ in chosen:self.db.add_ref(pid,path,f.embedding,f.quality,False,'cluster-learning')
        for path,idx,*_ in c['items']:self.db.mark_state(path,idx,'clustered')
        self.cluster_name.clear();self.refresh_people();self.refresh_review();self.make_clusters();QMessageBox.information(self,'그룹 학습',f'{len(chosen)}개의 다양한 대표 얼굴을 학습했습니다.')

    def refresh_albums(self):
        self.album_list.clear()
        for aid,name,count in self.db.albums():
            it=QListWidgetItem(f'{name}   ·   {count:,}개');it.setData(Qt.UserRole,aid);self.album_list.addItem(it)
        self.refresh_gallery_filters()
    def add_album(self):
        n=self.album_new.text().strip()
        if not n:return
        try:self.db.add_album(n);self.album_new.clear();self.refresh_albums()
        except sqlite3.IntegrityError:QMessageBox.warning(self,'중복','같은 이름의 앨범이 있습니다.')
    def delete_album(self):
        it=self.album_list.currentItem()
        if it and QMessageBox.question(self,'앨범 삭제','앨범만 삭제하고 원본 사진은 유지할까요?')==QMessageBox.Yes:self.db.delete_album(int(it.data(Qt.UserRole)));self.refresh_albums()
    def open_album(self,item=None):
        it=item or self.album_list.currentItem()
        if not it:return
        aid=int(it.data(Qt.UserRole));self.change_page(1);self.refresh_gallery_filters();i=self.photo_album.findData(aid);self.photo_album.setCurrentIndex(max(0,i));self.refresh_gallery()

    def refresh_duplicates(self):
        total,exact,near=self.db.duplicate_summary();self.dup_status.setText(f'캐시 {total:,} · 완전 중복 그룹 {exact:,} · 유사 연사 후보 쌍 {near:,}');self.dup_groups.clear();self._dup_data=self.db.duplicate_groups(300)
        for i,(_,size,n,paths) in enumerate(self._dup_data):self.dup_groups.addItem(f'중복 그룹 {i+1:03d} · {n}개 · {size/1024/1024:.1f}MB')
        if self._dup_data:self.dup_groups.setCurrentRow(0)
    def dup_group_selected(self,row):
        self.dup_files.clear()
        if row<0 or row>=len(getattr(self,'_dup_data',[])):return
        for p in self._dup_data[row][3]:self.dup_files.addItem(p)

    def pick_src(self):
        p=QFileDialog.getExistingDirectory(self,'분류할 폴더')
        if p:self.src.setText(p);self.dst.setText(str(Path(p)/'분류결과'))
    def pick_dst(self):
        p=QFileDialog.getExistingDirectory(self,'결과 폴더')
        if p:self.dst.setText(p)
    def start_scan(self):
        s=Path(self.src.text().strip())
        if not s.is_dir():QMessageBox.warning(self,'폴더','분류할 폴더를 선택하세요.');return
        if not build_models(self.db):QMessageBox.warning(self,'인물','인물과 레퍼런스를 먼저 등록하세요.');return
        d=Path(self.dst.text().strip()) if self.dst.text().strip() else s/'분류결과';cfg=ScanConfig(s,d,self.recursive.isChecked(),self.multi.isChecked(),self.th.value()/100,self.workers.value(),self.gpu.isChecked(),self.hard.isChecked(),self.auto_workers.isChecked())
        self.go.setEnabled(False);self.stop.setEnabled(True);self.pb.setValue(0);self.scan_thread=QThread(self);self.scan_worker=ScanWorker(cfg);self.scan_worker.moveToThread(self.scan_thread);self.scan_thread.started.connect(self.scan_worker.run);self.scan_worker.progress.connect(self.scan_progress);self.scan_worker.status.connect(self.scan_status.setText);self.scan_worker.done.connect(self.scan_done);self.scan_worker.fail.connect(self.scan_fail);self.scan_worker.done.connect(self.scan_thread.quit);self.scan_worker.fail.connect(self.scan_thread.quit);self.scan_thread.start()
    def stop_scan(self):
        if self.scan_worker:self.scan_worker.request_stop();self.scan_status.setText('중단 요청됨 — 현재 작업 묶음 정리 중')
    def scan_progress(self,i,n,name,speed,cached,new):
        self.pb.setValue(int(i*100/max(1,n)));eta=(n-i)/max(.01,speed);self.scan_status.setText(f'{i:,}/{n:,} · {name} · ETA {eta/60:.1f}분');self.speed.setText(f'캐시 {cached:,} · 새 AI {new:,} · {speed:.1f}장/s')
    def scan_done(self,s):
        self.go.setEnabled(True);self.stop.setEnabled(False);self.pb.setValue(100);self.scan_status.setText('완료' if not s.get('중단') else '중단됨');self.refresh_home();self.refresh_gallery_filters();self.refresh_gallery();self.refresh_review();self.refresh_history();QMessageBox.information(self,'분류 결과',f"전체 {s['전체']:,}\n자동 {s['분류']:,}\n검토 {s['확인필요']:,}\n미확인 {s['미확인']:,}\n얼굴 없음 {s['얼굴없음']:,}\n캐시 {s['캐시']:,}\n새 분석 {s['새분석']:,}\n중복 재사용 {s['중복재사용']:,}\n워커 {s.get('워커',0)} · {s['초']/60:.1f}분")
    def scan_fail(self,e):self.go.setEnabled(True);self.stop.setEnabled(False);self.scan_status.setText('오류');QMessageBox.critical(self,'오류',e)

    def refresh_history(self):
        self.history_list.clear()
        for oid,created,src,out,summary in self.db.operation_history():self.history_list.addItem(f'{created}  ·  {summary}\n{src}\n→ {out}')
    def undo_last(self):
        r,m=self.db.undo_last();self.refresh_history();QMessageBox.information(self,'되돌리기',f'생성 파일/링크 {r:,}개 제거 · 이미 없던 항목 {m:,}개')
    def refresh_settings(self):
        n,faces,reviews,size=self.db.cache_stats();gpu,desc=FaceEngine.gpu_available();self.sys_info.setText(f"분석 캐시 {n:,}개 미디어 / {faces:,}개 얼굴 / 검토대기 {reviews:,} / DB {size/1024/1024:.1f}MB\nGPU: {'DirectML 가능' if gpu else 'CPU 모드'} ({desc})\nHEIC: {'지원' if pillow_heif else '미지원'} · RAW: {'지원' if rawpy else '미지원'}\n썸네일 캐시: {sum(1 for _ in Path(Path(DB).parent/'thumbs').glob('*.jpg')):,}개")
    def cleanup_cache(self):
        t,d=self.db.cleanup_cache();self.refresh_settings();QMessageBox.information(self,'캐시 정리',f'{t:,}개 중 없는 원본 {d:,}개 제거')
    def clear_cache(self):
        if QMessageBox.question(self,'캐시 삭제','분석/썸네일 캐시를 모두 비울까요? 인물 레퍼런스는 유지됩니다.')==QMessageBox.Yes:self.db.clear_cache();self.refresh_settings();self.refresh_gallery();self.refresh_review()


def self_test():
    try:
        e=FaceEngine(False);assert e.det is not None and e.rec is not None;db=LibraryDB();db.cache_stats();models=build_models(db)
        if models: RecognitionIndex(models)
        return 0
    except Exception as ex:
        print('SELF TEST FAILED',repr(ex));return 1


def main():
    if '--self-test' in sys.argv:return self_test()
    app=QApplication(sys.argv);app.setApplicationName(APP);app.setStyle('Fusion');w=Main();w.show();return app.exec()
if __name__=='__main__':sys.exit(main())
