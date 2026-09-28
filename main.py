from ui_base import *
class Main(UIBase):
    def refresh_people(self):
        while self.cards.count():
            it=self.cards.takeAt(0);wd=it.widget();wd.deleteLater() if wd else None
        self.assign_combo.clear();self.cluster_person.clear();self.cluster_person.addItem('기존 인물 선택',None);models=build_models(self.db)
        for pid,name,pos,neg,_ in self.db.people():
            refs=self.db.refs(pid,False);q=float(np.mean([r[3] for r in refs])) if refs else 0;th=models[pid].threshold if pid in models else .43;card=PersonCard(pid,name,pos or 0,neg or 0,th,q);card.clicked.connect(self.select_person);self.cards.addWidget(card);self.assign_combo.addItem(name,pid);self.cluster_person.addItem(name,pid)
        self.cards.addStretch()
    def add_person(self):
        n=self.name.text().strip()
        if not n:return
        try:self.db.add_person(n);self.name.clear();self.refresh_people()
        except sqlite3.IntegrityError:QMessageBox.warning(self,'중복','이미 등록된 이름입니다.')
    def select_person(self,pid):
        self.pid=pid;row=next((r for r in self.db.people() if r[0]==pid),None)
        if not row:return
        refs=self.db.refs(pid,None);models=build_models(self.db);self.person_title.setText(row[1]);self.ref_list.clear();pos=[r for r in refs if not r[4]];q=float(np.mean([r[3] for r in pos])) if pos else 0;th=models[pid].threshold if pid in models else .43;self.person_detail.setText(f'positive {len(pos)} · negative {len(refs)-len(pos)} · 평균 품질 {q:.0%} · 자동 threshold {th:.3f}')
        for rid,path,emb,qual,neg,source in refs:
            prefix='🚫 ' if neg else ('✨ ' if source!='manual' else '✓ ');it=QListWidgetItem(f'{prefix}{Path(path).name if path else source} · 품질 {qual:.0%} · {source}');it.setData(Qt.UserRole,rid);self.ref_list.addItem(it)
    def add_refs(self):
        if not self.pid:QMessageBox.information(self,'선택','먼저 인물을 선택하세요.');return
        ps,_=QFileDialog.getOpenFileNames(self,'레퍼런스 선택','','사진/영상 (*.*)')
        if not ps:return
        QApplication.setOverrideCursor(Qt.WaitCursor);ok=0;fail=[]
        try:
            e=FaceEngine(False)
            for p in ps:
                try:emb,q,count=e.reference(Path(p));self.db.add_ref(self.pid,p,emb,q,False,'manual');ok+=1
                except Exception as ex:fail.append(f'{Path(p).name}: {ex}')
        finally:QApplication.restoreOverrideCursor()
        self.refresh_people();self.select_person(self.pid);QMessageBox.information(self,'레퍼런스',f'{ok}장 등록 완료'+(f'\n실패 {len(fail)}장' if fail else ''))
    def delete_ref(self):
        it=self.ref_list.currentItem()
        if not it:return
        self.db.delete_ref(int(it.data(Qt.UserRole)));self.refresh_people();self.select_person(self.pid) if self.pid else None
    def delete_person(self):
        if self.pid and QMessageBox.question(self,'삭제','이 인물과 학습 정보를 삭제할까요?')==QMessageBox.Yes:self.db.delete_person(self.pid);self.pid=None;self.ref_list.clear();self.refresh_people()
    def pick_src(self):
        p=QFileDialog.getExistingDirectory(self,'분류할 폴더')
        if p:self.src.setText(p);self.dst.setText(str(Path(p)/'분류결과'))
    def pick_dst(self):
        p=QFileDialog.getExistingDirectory(self,'결과 폴더')
        if p:self.dst.setText(p)
    def start_scan(self):
        s=Path(self.src.text().strip())
        if not s.is_dir():QMessageBox.warning(self,'폴더','분류할 폴더를 선택하세요.');return
        d=Path(self.dst.text().strip()) if self.dst.text().strip() else s/'분류결과';cfg=ScanConfig(s,d,self.recursive.isChecked(),self.multi.isChecked(),self.th.value()/100,self.workers.value(),self.gpu.isChecked(),self.hard.isChecked());self.go.setEnabled(False);self.stop.setEnabled(True);self.pb.setValue(0);self.scan_thread=QThread(self);self.scan_worker=ScanWorker(cfg);self.scan_worker.moveToThread(self.scan_thread);self.scan_thread.started.connect(self.scan_worker.run);self.scan_worker.progress.connect(self.scan_progress);self.scan_worker.status.connect(self.scan_status.setText);self.scan_worker.done.connect(self.scan_done);self.scan_worker.fail.connect(self.scan_fail);self.scan_worker.done.connect(self.scan_thread.quit);self.scan_worker.fail.connect(self.scan_thread.quit);self.scan_thread.start()
    def stop_scan(self):
        if self.scan_worker:self.scan_worker.request_stop();self.scan_status.setText('중단 요청됨 — 현재 처리 묶음까지만 마무리합니다.')
    def scan_progress(self,i,n,name,speed,cached,new):
        self.pb.setValue(int(i*100/max(1,n)));eta=(n-i)/max(.01,speed);self.scan_status.setText(f'{i:,}/{n:,} · {name} · 남은시간 약 {eta/60:.1f}분');self.speed.setText(f'캐시 {cached:,} · 새 분석 {new:,} · {speed:.1f}장/s')
    def scan_done(self,s):
        self.go.setEnabled(True);self.stop.setEnabled(False);self.pb.setValue(100);self.scan_status.setText('완료' if not s.get('중단') else '중단됨');self.refresh_review();self.refresh_settings();QMessageBox.information(self,'분류 결과',f"전체 {s['전체']:,}장\n자동 분류 {s['분류']:,}\n확인 필요 {s['확인필요']:,}\n미확인 {s['미확인']:,}\n얼굴 없음 {s['얼굴없음']:,}\n캐시 재사용 {s['캐시']:,}\n새 AI 분석 {s['새분석']:,}\n중복 재사용 {s['중복재사용']:,}\n처리시간 {s['초']/60:.1f}분")
    def scan_fail(self,e):self.go.setEnabled(True);self.stop.setEnabled(False);self.scan_status.setText('오류');QMessageBox.critical(self,'오류',e)
    def refresh_review(self):
        if not hasattr(self,'review_list'):return
        self.review_list.clear()
        for path,idx,state,pid,score,name,packed in self.db.review_items(600):
            fs=unpack_faces(packed or b'')
            if not(0<=idx<len(fs)):continue
            label=(f"{'확인' if state=='review' else '미확인'}\n{name or '후보없음'} {score:.3f}" if score is not None else '미확인');it=QListWidgetItem(QIcon(thumb(path,140)),label);it.setData(Qt.UserRole,(path,idx,pid));it.setToolTip(path);self.review_list.addItem(it)
    def selected_reviews(self):return [it.data(Qt.UserRole) for it in self.review_list.selectedItems()]
    def face_from_cache(self,path,idx):
        m=self.db.cache_get(Path(path));return m.faces[idx] if m and 0<=idx<len(m.faces) else None
    def assign_review(self):
        pid=self.assign_combo.currentData();sel=self.selected_reviews()
        if pid is None or not sel:QMessageBox.information(self,'선택','얼굴과 인물을 선택하세요.');return
        n=0
        for path,idx,_ in sel:
            f=self.face_from_cache(path,idx)
            if f:self.db.add_ref(pid,path,f.embedding,f.quality,False,'review-correction');self.db.mark_state(path,idx,'learned');n+=1
        self.refresh_people();self.refresh_review();QMessageBox.information(self,'학습',f'{n}개 얼굴을 positive 레퍼런스로 학습했습니다.')
    def reject_review_candidate(self):
        n=0
        for path,idx,candidate_pid in self.selected_reviews():
            if not candidate_pid:continue
            f=self.face_from_cache(path,idx)
            if f:self.db.add_ref(candidate_pid,path,f.embedding,f.quality,True,'negative-correction');self.db.mark_state(path,idx,'unknown');n+=1
        self.refresh_people();self.refresh_review();QMessageBox.information(self,'제외 학습',f'{n}개 얼굴을 negative 학습에 반영했습니다.')
    def make_clusters(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:items=self.db.unknown_faces(20000);self.clusters=Clusterer(.475).cluster(items)
        finally:QApplication.restoreOverrideCursor()
        self.cluster_list.clear()
        for i,c in enumerate(self.clusters):q=np.mean([x[2].quality for x in c['items']]);self.cluster_list.addItem(f"미확인 그룹 {i+1:03d} · {len(c['items'])}개 · 품질 {q:.0%}")
        self.cluster_status.setText(f'{len(items):,}개 미확인 얼굴 → {len(self.clusters):,}개 반복 인물 그룹')
    def show_cluster(self,row):
        if row<0 or row>=len(self.clusters):return
        c=self.clusters[row];self.cluster_detail.setText(f"멤버 {len(c['items'])}개\n대표 파일: "+', '.join(Path(x[0]).name for x in c['items'][:8]))
    def assign_cluster(self):
        row=self.cluster_list.currentRow()
        if row<0 or row>=len(self.clusters):return
        pid=self.cluster_person.currentData();name=self.cluster_name.text().strip()
        if pid is None:
            if not name:QMessageBox.information(self,'이름','새 인물 이름을 입력하거나 기존 인물을 선택하세요.');return
            try:pid=self.db.add_person(name)
            except sqlite3.IntegrityError:QMessageBox.warning(self,'중복','같은 이름이 이미 있습니다. 기존 인물을 선택하세요.');return
        c=self.clusters[row];chosen=[]
        for item in sorted(c['items'],key=lambda x:x[2].quality,reverse=True):
            f=item[2]
            if f.quality<.25:continue
            if not chosen or max(float(np.dot(f.embedding,z[2].embedding)) for z in chosen)<.93:chosen.append(item)
            if len(chosen)>=24:break
        if not chosen:chosen=c['items'][:min(12,len(c['items']))]
        for path,idx,f,*_ in chosen:self.db.add_ref(pid,path,f.embedding,f.quality,False,'cluster-learning')
        for path,idx,*_ in c['items']:self.db.mark_state(path,idx,'clustered')
        self.cluster_name.clear();self.refresh_people();self.refresh_review();self.make_clusters();QMessageBox.information(self,'그룹 학습',f'{len(chosen)}개 대표 얼굴을 레퍼런스로 학습했습니다.')
    def refresh_settings(self):
        if not hasattr(self,'sys_info'):return
        n,faces,reviews,size=self.db.cache_stats();gpu,desc=FaceEngine.gpu_available();self.sys_info.setText(f"분석 캐시: {n:,}개 미디어 / {faces:,}개 얼굴 / 검토대기 {reviews:,}개 / DB {size/1024/1024:.1f}MB\nGPU: {'DirectML 사용 가능' if gpu else 'CPU 모드'} ({desc})\nHEIC: {'지원' if pillow_heif else '미지원'} · RAW: {'지원' if rawpy else '미지원'}")
    def cleanup_cache(self):
        total,dead=self.db.cleanup_cache();self.refresh_settings();QMessageBox.information(self,'캐시 정리',f'{total:,}개 중 존재하지 않는 파일 {dead:,}개 캐시 제거')
    def duplicate_stats(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:total,exact,near=self.db.duplicate_summary()
        finally:QApplication.restoreOverrideCursor()
        QMessageBox.information(self,'중복/연사 분석',f'분석 캐시 {total:,}개 기준\n완전 중복 그룹 {exact:,}개\n유사 연사 후보 쌍 {near:,}개\n\n완전 중복은 다음 스캔에서 얼굴 AI 결과를 재사용합니다.')
    def clear_cache(self):
        if QMessageBox.question(self,'캐시 삭제','분석 캐시를 전부 비울까요? 인물 레퍼런스는 유지됩니다.')==QMessageBox.Yes:self.db.clear_cache();self.refresh_settings();self.refresh_review()
    def undo_last(self):r,m=self.db.undo_last();QMessageBox.information(self,'되돌리기',f'마지막 분류에서 만든 파일/링크 {r:,}개 제거\n이미 없던 항목 {m:,}개')


def self_test():
    try:e=FaceEngine(False);assert e.det is not None and e.rec is not None;LibraryDB().cache_stats();return 0
    except Exception:return 1

def main():
    if '--self-test' in sys.argv:return self_test()
    app=QApplication(sys.argv);app.setApplicationName(APP);w=Main();w.show();return app.exec()
if __name__=='__main__':sys.exit(main())
