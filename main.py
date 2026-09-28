from __future__ import annotations
import csv, os, shutil, sqlite3, sys, urllib.request
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QLineEdit,QListWidget,QFileDialog,QMessageBox,QProgressBar,QSlider,QCheckBox,QStackedWidget,QFrame

APP='PhotoRefSorter'
SUPPORTED={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
YUNET='https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
SFACE='https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx'
DATA=Path(os.getenv('LOCALAPPDATA') or Path.home())/APP
DATA.mkdir(parents=True,exist_ok=True)
MODELS=DATA/'models'; MODELS.mkdir(exist_ok=True)
DB=DATA/'people.sqlite3'

def bundled_models():
    if getattr(sys,'frozen',False) and hasattr(sys,'_MEIPASS'):
        p=Path(sys._MEIPASS)/'models'
        if p.exists(): return p
    p=Path(__file__).resolve().parent/'models'
    return p if p.exists() else None

def read_img(p:Path):
    try:
        b=np.fromfile(str(p),dtype=np.uint8)
        return cv2.imdecode(b,cv2.IMREAD_COLOR) if b.size else None
    except: return None

def safe_name(s:str):
    bad='<>:"/\\|?*'
    return ''.join('_' if c in bad else c for c in s).strip().rstrip('.') or '이름없음'

def unique_copy(src:Path,dst:Path):
    dst.mkdir(parents=True,exist_ok=True)
    out=dst/src.name
    if not out.exists(): shutil.copy2(src,out); return
    if out.stat().st_size==src.stat().st_size: return
    i=2
    while True:
        c=dst/f'{src.stem}_{i}{src.suffix}'
        if not c.exists(): shutil.copy2(src,c); return
        i+=1

class PeopleDB:
    def __init__(self):
        with sqlite3.connect(DB) as c:
            c.executescript('''
            CREATE TABLE IF NOT EXISTS people(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT UNIQUE COLLATE NOCASE);
            CREATE TABLE IF NOT EXISTS refs(id INTEGER PRIMARY KEY AUTOINCREMENT,person_id INTEGER,image_path TEXT,embedding BLOB,dim INTEGER,FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE);
            ''')
    def con(self):
        c=sqlite3.connect(DB); c.execute('PRAGMA foreign_keys=ON'); return c
    def add_person(self,n):
        with self.con() as c: return c.execute('INSERT INTO people(name) VALUES(?)',(n.strip(),)).lastrowid
    def delete_person(self,pid):
        with self.con() as c: c.execute('DELETE FROM people WHERE id=?',(pid,))
    def add_ref(self,pid,path,e):
        v=np.asarray(e,dtype=np.float32).reshape(-1)
        with self.con() as c: c.execute('INSERT INTO refs(person_id,image_path,embedding,dim) VALUES(?,?,?,?)',(pid,path,v.tobytes(),v.size))
    def people(self):
        with self.con() as c:
            return c.execute('''SELECT p.id,p.name,COUNT(r.id),(SELECT image_path FROM refs r2 WHERE r2.person_id=p.id ORDER BY r2.id LIMIT 1) FROM people p LEFT JOIN refs r ON r.person_id=p.id GROUP BY p.id,p.name ORDER BY p.name''').fetchall()
    def refs(self,pid):
        with self.con() as c: rows=c.execute('SELECT image_path,embedding,dim FROM refs WHERE person_id=?',(pid,)).fetchall()
        return [(p,np.frombuffer(b,dtype=np.float32,count=d).copy()) for p,b,d in rows]
    def all(self):
        out={}
        for pid,name,_,_ in self.people():
            rs=[e for _,e in self.refs(pid)]
            if rs: out[pid]=(name,rs)
        return out

class FaceEngine:
    def __init__(self):
        b=bundled_models()
        det=(b/'face_detection_yunet_2023mar.onnx') if b else MODELS/'face_detection_yunet_2023mar.onnx'
        rec=(b/'face_recognition_sface_2021dec.onnx') if b else MODELS/'face_recognition_sface_2021dec.onnx'
        for url,dst in ((YUNET,det),(SFACE,rec)):
            if not dst.exists():
                dst=MODELS/dst.name
                if not dst.exists(): urllib.request.urlretrieve(url,dst)
                if 'yunet' in dst.name: det=dst
                else: rec=dst
        self.det=cv2.FaceDetectorYN.create(str(det),'',(320,320),0.86,0.3,5000)
        self.rec=cv2.FaceRecognizerSF.create(str(rec),'')
    def detect(self,img):
        h,w=img.shape[:2]; scale=min(1.0,2200/max(h,w))
        if scale<1: img=cv2.resize(img,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA)
        h,w=img.shape[:2]; self.det.setInputSize((w,h)); _,faces=self.det.detect(img)
        return img,([] if faces is None else [f for f in faces if f[-1]>=0.86])
    def emb(self,img,f):
        x=self.rec.feature(self.rec.alignCrop(img,f)).astype(np.float32).reshape(-1)
        n=np.linalg.norm(x); return x/n if n>1e-8 else x
    def ref(self,p):
        img=read_img(p)
        if img is None: raise ValueError('이미지를 읽을 수 없음')
        img,fs=self.detect(img)
        if not fs: raise ValueError('얼굴을 찾지 못함')
        f=max(fs,key=lambda x:float(x[2]*x[3]))
        return self.emb(img,f)
    def image(self,p):
        img=read_img(p)
        if img is None: return []
        img,fs=self.detect(img); out=[]
        for f in fs:
            try: out.append(self.emb(img,f))
            except: pass
        return out

@dataclass
class Config:
    source:Path; output:Path; auto:float; review:float; recursive:bool; multi:bool

class Worker(QObject):
    progress=Signal(int,int,str); done=Signal(dict); fail=Signal(str)
    def __init__(self,cfg): super().__init__(); self.cfg=cfg
    def run(self):
        try:
            db=PeopleDB(); people=db.all()
            if not people: raise RuntimeError('등록된 인물과 레퍼런스가 없습니다.')
            eng=FaceEngine(); refs={}; cent={}
            for pid,(name,vecs) in people.items():
                m=np.vstack(vecs).astype(np.float32); refs[pid]=m; c=m.mean(0); n=np.linalg.norm(c); cent[pid]=(name,c/n if n else c)
            it=self.cfg.source.rglob('*') if self.cfg.recursive else self.cfg.source.glob('*')
            outres=self.cfg.output.resolve(); files=[]
            for p in it:
                if p.is_file() and p.suffix.lower() in SUPPORTED:
                    try:
                        if p.resolve().is_relative_to(outres): continue
                    except: pass
                    files.append(p)
            self.cfg.output.mkdir(parents=True,exist_ok=True)
            stats={'전체':len(files),'분류':0,'확인필요':0,'미확인':0,'얼굴없음':0,'오류':0}
            with open(self.cfg.output/'분류결과.csv','w',newline='',encoding='utf-8-sig') as f:
                wr=csv.writer(f); wr.writerow(['원본파일','상태','인물','유사도'])
                for i,p in enumerate(files,1):
                    self.progress.emit(i,len(files),p.name)
                    try:
                        vs=eng.image(p)
                        if not vs:
                            unique_copy(p,self.cfg.output/'_얼굴없음'); stats['얼굴없음']+=1; wr.writerow([p,'얼굴없음','','']); continue
                        found={}; review=None
                        for v in vs:
                            best=None
                            for pid,(name,c) in cent.items():
                                s=max(float(np.dot(v,c)),float((refs[pid]@v).max())*.985)
                                if best is None or s>best[0]: best=(s,pid,name)
                            if best:
                                s,pid,name=best
                                if s>=self.cfg.auto: found[pid]=max(found.get(pid,-1),s)
                                elif s>=self.cfg.review and (review is None or s>review[0]): review=best
                        if found:
                            arr=sorted(found.items(),key=lambda x:x[1],reverse=True)
                            if not self.cfg.multi: arr=arr[:1]
                            labels=[]
                            for pid,s in arr:
                                name=people[pid][0]; unique_copy(p,self.cfg.output/safe_name(name)); labels.append(f'{name}:{s:.3f}')
                            stats['분류']+=1; wr.writerow([p,'분류',' / '.join(labels),max(found.values())])
                        elif review:
                            s,_,name=review; unique_copy(p,self.cfg.output/'_확인필요'); stats['확인필요']+=1; wr.writerow([p,'확인필요',name,f'{s:.3f}'])
                        else:
                            unique_copy(p,self.cfg.output/'_미확인'); stats['미확인']+=1; wr.writerow([p,'미확인','',''])
                    except Exception as e:
                        stats['오류']+=1; wr.writerow([p,'오류','',repr(e)])
            self.done.emit(stats)
        except Exception as e: self.fail.emit(str(e))

class Main(QMainWindow):
    def __init__(self):
        super().__init__(); self.db=PeopleDB(); self.pid=None; self.thread=None; self.worker=None
        self.setWindowTitle('PhotoRef Sorter'); self.resize(980,680)
        root=QWidget(); self.setCentralWidget(root); hl=QHBoxLayout(root); hl.setContentsMargins(0,0,0,0)
        side=QFrame(); side.setFixedWidth(210); side.setObjectName('side'); sl=QVBoxLayout(side); sl.setContentsMargins(18,24,18,24)
        logo=QLabel('PhotoRef\nSorter'); logo.setObjectName('logo'); sl.addWidget(logo); sl.addSpacing(25)
        self.b1=QPushButton('👤   인물 등록'); self.b2=QPushButton('✨   자동 분류')
        for b in (self.b1,self.b2): b.setObjectName('nav'); b.setMinimumHeight(44); sl.addWidget(b)
        sl.addStretch(); hl.addWidget(side)
        self.stack=QStackedWidget(); hl.addWidget(self.stack,1); self.stack.addWidget(self.people_page()); self.stack.addWidget(self.scan_page())
        self.b1.clicked.connect(lambda:self.stack.setCurrentIndex(0)); self.b2.clicked.connect(lambda:self.stack.setCurrentIndex(1))
        self.setStyleSheet('''*{font-family:"Segoe UI","Malgun Gothic";font-size:14px}QMainWindow,QWidget{background:#F6F7F9;color:#20242A}#side{background:#171A20}#logo{color:white;font-size:24px;font-weight:800}#nav{background:transparent;color:#C7CDD6;text-align:left;border:none;border-radius:9px;padding:10px}#nav:hover{background:#272C35;color:white}#h1{font-size:28px;font-weight:800}#muted{color:#737B86}#panel{background:white;border:1px solid #E1E5EA;border-radius:14px}QLineEdit,QListWidget{background:white;border:1px solid #D8DEE6;border-radius:9px;padding:8px}QPushButton{background:#ECEFF3;border:none;border-radius:9px;padding:9px 14px;font-weight:600}#primary{background:#4F5FF5;color:white}QProgressBar{border:none;border-radius:6px;background:#E6EAF0}QProgressBar::chunk{background:#5968F3;border-radius:6px}''')
        self.refresh()
    def title(self,text,sub):
        v=QVBoxLayout(); a=QLabel(text); a.setObjectName('h1'); b=QLabel(sub); b.setObjectName('muted'); b.setWordWrap(True); v.addWidget(a); v.addWidget(b); return v
    def people_page(self):
        w=QWidget(); v=QVBoxLayout(w); v.setContentsMargins(32,28,32,28); v.addLayout(self.title('인물 등록','이름을 만들고 레퍼런스 사진을 여러 장 등록하세요.'))
        r=QHBoxLayout(); self.name=QLineEdit(); self.name.setPlaceholderText('예: 리즈'); add=QPushButton('인물 추가'); add.setObjectName('primary'); add.clicked.connect(self.add_person); r.addWidget(self.name,1); r.addWidget(add); v.addLayout(r)
        panel=QFrame(); panel.setObjectName('panel'); p=QHBoxLayout(panel); self.people=QListWidget(); self.people.currentRowChanged.connect(self.select_person); self.refs=QListWidget(); p.addWidget(self.people,1); p.addWidget(self.refs,1); v.addWidget(panel,1)
        br=QHBoxLayout(); ar=QPushButton('레퍼런스 사진 추가'); ar.setObjectName('primary'); ar.clicked.connect(self.add_refs); dr=QPushButton('선택 인물 삭제'); dr.clicked.connect(self.del_person); br.addWidget(ar); br.addWidget(dr); v.addLayout(br); return w
    def scan_page(self):
        w=QWidget(); v=QVBoxLayout(w); v.setContentsMargins(32,28,32,28); v.addLayout(self.title('자동 분류','등록된 인물을 찾아 인물별 폴더로 복사합니다. 원본은 변경하지 않습니다.'))
        panel=QFrame(); panel.setObjectName('panel'); p=QVBoxLayout(panel)
        self.src=QLineEdit(); self.dst=QLineEdit(); bs=QPushButton('분류할 폴더 선택'); bo=QPushButton('결과 폴더 선택'); bs.clicked.connect(self.pick_src); bo.clicked.connect(self.pick_dst)
        for label,edit,btn in [('분류할 폴더',self.src,bs),('결과 폴더',self.dst,bo)]:
            p.addWidget(QLabel(label)); rr=QHBoxLayout(); rr.addWidget(edit,1); rr.addWidget(btn); p.addLayout(rr)
        self.rec=QCheckBox('하위 폴더까지'); self.rec.setChecked(True); self.multi=QCheckBox('여러 등록 인물이 있으면 각 폴더에 복사'); self.multi.setChecked(True); p.addWidget(self.rec); p.addWidget(self.multi)
        rr=QHBoxLayout(); rr.addWidget(QLabel('자동 분류 기준')); self.th=QSlider(Qt.Horizontal); self.th.setRange(36,60); self.th.setValue(43); self.tv=QLabel('0.43'); self.th.valueChanged.connect(lambda x:self.tv.setText(f'{x/100:.2f}')); rr.addWidget(self.th,1); rr.addWidget(self.tv); p.addLayout(rr)
        self.go=QPushButton('분류 시작'); self.go.setObjectName('primary'); self.go.clicked.connect(self.start); p.addWidget(self.go); self.pb=QProgressBar(); p.addWidget(self.pb); self.st=QLabel('준비됨'); self.st.setObjectName('muted'); p.addWidget(self.st); v.addWidget(panel); v.addStretch(); return w
    def refresh(self):
        rows=self.db.people(); self.people.clear(); self._rows=rows
        for _,name,c,_ in rows: self.people.addItem(f'{name}   ·   레퍼런스 {c}장')
        self.refs.clear(); self.pid=None
    def add_person(self):
        n=self.name.text().strip()
        if not n:return
        try:self.db.add_person(n);self.name.clear();self.refresh()
        except sqlite3.IntegrityError: QMessageBox.warning(self,'중복','이미 있는 이름입니다.')
    def select_person(self,row):
        if row<0 or row>=len(self._rows): return
        self.pid=self._rows[row][0]; self.refs.clear()
        for p,_ in self.db.refs(self.pid): self.refs.addItem(Path(p).name)
    def add_refs(self):
        if not self.pid: QMessageBox.information(self,'선택','먼저 인물을 선택하세요.'); return
        ps,_=QFileDialog.getOpenFileNames(self,'레퍼런스 선택','','사진 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)')
        if not ps:return
        try: eng=FaceEngine()
        except Exception as e: QMessageBox.critical(self,'AI 오류',str(e)); return
        ok=0; fail=[]
        for p in ps:
            try:self.db.add_ref(self.pid,p,eng.ref(Path(p)));ok+=1
            except Exception as e: fail.append(f'{Path(p).name}: {e}')
        self.refresh(); QMessageBox.information(self,'완료',f'{ok}장 등록 완료'+(('\n\n'+ '\n'.join(fail[:8])) if fail else ''))
    def del_person(self):
        if self.pid and QMessageBox.question(self,'삭제','선택한 인물을 삭제할까요?')==QMessageBox.Yes:self.db.delete_person(self.pid);self.refresh()
    def pick_src(self):
        p=QFileDialog.getExistingDirectory(self,'분류할 폴더')
        if p:self.src.setText(p); self.dst.setText(str(Path(p)/'분류결과'))
    def pick_dst(self):
        p=QFileDialog.getExistingDirectory(self,'결과 폴더')
        if p:self.dst.setText(p)
    def start(self):
        s=Path(self.src.text().strip())
        if not s.is_dir(): QMessageBox.warning(self,'폴더','분류할 폴더를 선택하세요.'); return
        if not self.db.all(): QMessageBox.warning(self,'레퍼런스','인물과 레퍼런스를 먼저 등록하세요.'); return
        d=Path(self.dst.text().strip()) if self.dst.text().strip() else s/'분류결과'; t=self.th.value()/100
        cfg=Config(s,d,t,max(.30,t-.09),self.rec.isChecked(),self.multi.isChecked()); self.go.setEnabled(False); self.thread=QThread(); self.worker=Worker(cfg); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.progress.connect(self.prog); self.worker.done.connect(self.done); self.worker.fail.connect(self.fail); self.worker.done.connect(self.thread.quit); self.worker.fail.connect(self.thread.quit); self.thread.start()
    def prog(self,i,n,name): self.pb.setValue(int(i*100/max(1,n))); self.st.setText(f'{i}/{n} · {name}')
    def done(self,s):
        self.go.setEnabled(True); self.pb.setValue(100); self.st.setText('완료'); QMessageBox.information(self,'완료',f"전체 {s['전체']}장\n자동 분류 {s['분류']}장\n확인 필요 {s['확인필요']}장\n미확인 {s['미확인']}장\n얼굴 없음 {s['얼굴없음']}장\n오류 {s['오류']}장")
    def fail(self,e): self.go.setEnabled(True); self.st.setText('오류'); QMessageBox.critical(self,'오류',e)

def main():
    if '--self-test' in sys.argv:
        try:
            e=FaceEngine(); return 0 if e.det is not None and e.rec is not None else 1
        except: return 1
    a=QApplication(sys.argv); a.setApplicationName(APP); w=Main(); w.show(); return a.exec()
if __name__=='__main__': sys.exit(main())
