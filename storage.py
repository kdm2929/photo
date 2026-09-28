from __future__ import annotations

import csv, json, os, shutil, sqlite3, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    from PIL import Image, ImageOps
except Exception:
    Image = ImageOps = None
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None
try:
    import pillow_avif  # noqa: F401
except Exception:
    try:
        import pillow_avif_plugin  # noqa: F401
    except Exception:
        pass
try:
    import rawpy
except Exception:
    rawpy = None
try:
    import onnxruntime as ort
except Exception:
    ort = None

APP = 'PhotoRefSorter'
VERSION = '0.5'
DATA = Path(os.getenv('LOCALAPPDATA') or Path.home()) / APP
DATA.mkdir(parents=True, exist_ok=True)
MODELS = DATA / 'models'; MODELS.mkdir(exist_ok=True)
THUMBS = DATA / 'thumbs'; THUMBS.mkdir(exist_ok=True)
DB = DATA / 'library.sqlite3'
LEGACY_DB = DATA / 'people.sqlite3'
YUNET = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
SFACE = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx'
IMAGE_EXTS={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff','.heic','.heif','.avif','.gif'}
RAW_EXTS={'.dng','.cr2','.cr3','.nef','.arw','.orf','.rw2','.raf','.pef','.srw'}
VIDEO_EXTS={'.mp4','.mov','.m4v','.avi','.mkv','.webm'}
SUPPORTED=IMAGE_EXTS|RAW_EXTS|VIDEO_EXTS


def bundled_models():
    if getattr(sys,'frozen',False) and hasattr(sys,'_MEIPASS'):
        p=Path(sys._MEIPASS)/'models'
        if p.exists(): return p
    p=Path(__file__).resolve().parent/'models'
    return p if p.exists() else None


def model_paths():
    b=bundled_models()
    if b:
        d=b/'face_detection_yunet_2023mar.onnx'; r=b/'face_recognition_sface_2021dec.onnx'
        if d.exists() and r.exists(): return d,r
    d=MODELS/'face_detection_yunet_2023mar.onnx'; r=MODELS/'face_recognition_sface_2021dec.onnx'
    if not d.exists(): urllib.request.urlretrieve(YUNET,d)
    if not r.exists(): urllib.request.urlretrieve(SFACE,r)
    return d,r


def safe_name(s):
    bad='<>:"/\\|?*'
    return ''.join('_' if c in bad else c for c in s).strip().rstrip('.') or '이름없음'


def vec_blob(v):
    a=np.asarray(v,dtype=np.float32).reshape(-1); return a.tobytes(),int(a.size)


def blob_vec(blob,dim):
    return np.frombuffer(blob,dtype=np.float32,count=dim).copy()

@dataclass
class FaceData:
    embedding: np.ndarray
    quality: float
    bbox: tuple[float,float,float,float]

@dataclass
class MediaAnalysis:
    path: str
    size: int
    mtime_ns: int
    phash: int
    faces: list[FaceData]
    kind: str

@dataclass
class PersonModel:
    pid: int
    name: str
    positives: np.ndarray
    pos_quality: np.ndarray
    negatives: np.ndarray|None
    centroid: np.ndarray
    threshold: float

@dataclass
class ScanConfig:
    source: Path
    output: Path
    recursive: bool
    multi: bool
    threshold: float
    workers: int
    use_gpu: bool
    hardlink: bool


def pack_faces(faces):
    if not faces:return b''
    arr=np.vstack([f.embedding for f in faces]).astype(np.float32)
    meta=np.array([[f.quality,*f.bbox] for f in faces],dtype=np.float32)
    head=np.array([len(faces),arr.shape[1]],dtype=np.int32).tobytes()
    return head+arr.tobytes()+meta.tobytes()


def unpack_faces(blob):
    if not blob or len(blob)<8:return []
    head=np.frombuffer(blob[:8],dtype=np.int32); n,dim=int(head[0]),int(head[1])
    if n<=0 or dim<=0:return []
    eb=n*dim*4
    arr=np.frombuffer(blob[8:8+eb],dtype=np.float32).reshape(n,dim).copy()
    meta=np.frombuffer(blob[8+eb:],dtype=np.float32).reshape(n,5).copy()
    return [FaceData(arr[i],float(meta[i,0]),tuple(float(x) for x in meta[i,1:5])) for i in range(n)]


def phash64(img):
    if img is None or not img.size:return 0
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); g=cv2.resize(g,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
    low=cv2.dct(g)[:8,:8].flatten(); med=float(np.median(low[1:])); bits=low>med; value=0
    for i,b in enumerate(bits):
        if b:value|=(1<<i)
    return value&((1<<64)-1)


def load_pil(path,max_side=2200):
    if Image is None:return None
    ext=path.suffix.lower()
    if ext in RAW_EXTS and rawpy is not None:
        with rawpy.imread(str(path)) as raw:
            rgb=raw.postprocess(use_camera_wb=True,half_size=True,no_auto_bright=True)
        im=Image.fromarray(rgb)
    else:
        im=Image.open(path)
        try: im.seek(0)
        except Exception: pass
        if ImageOps is not None: im=ImageOps.exif_transpose(im)
    im.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
    return im


def read_image(path,max_side=2200):
    ext=path.suffix.lower()
    try:
        if ext in {'.jpg','.jpeg'}:
            data=np.fromfile(str(path),dtype=np.uint8)
            if not data.size:return None
            flag=cv2.IMREAD_REDUCED_COLOR_2 if path.stat().st_size>3_000_000 else cv2.IMREAD_COLOR
            img=cv2.imdecode(data,flag)
        elif ext in RAW_EXTS or ext in {'.heic','.heif','.avif','.gif'}:
            im=load_pil(path,max_side)
            if im is None:return None
            img=cv2.cvtColor(np.asarray(im.convert('RGB')),cv2.COLOR_RGB2BGR)
        else:
            data=np.fromfile(str(path),dtype=np.uint8); img=cv2.imdecode(data,cv2.IMREAD_COLOR) if data.size else None
        if img is None:return None
        h,w=img.shape[:2]; scale=min(1.0,max_side/max(h,w))
        if scale<1:img=cv2.resize(img,(max(1,int(w*scale)),max(1,int(h*scale))),interpolation=cv2.INTER_AREA)
        return img
    except Exception:return None


def media_frames(path,video_samples=10,max_side=2200):
    if path.suffix.lower() not in VIDEO_EXTS:
        img=read_image(path,max_side); return [] if img is None else [img]
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():return []
    out=[]
    try:
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); fps=float(cap.get(cv2.CAP_PROP_FPS) or 0)
        duration=frames/fps if frames>0 and fps>0 else 0
        n=min(video_samples,max(3,int(duration/8)+2)) if duration else video_samples
        for pos in np.linspace(.05,.95,n):
            if frames>0:cap.set(cv2.CAP_PROP_POS_FRAMES,int(frames*float(pos)))
            ok,frame=cap.read()
            if not ok or frame is None:continue
            h,w=frame.shape[:2]; scale=min(1.0,max_side/max(h,w))
            if scale<1:frame=cv2.resize(frame,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA)
            out.append(frame)
    finally:cap.release()
    return out


def iter_media(root,recursive=True,exclude=None):
    ex=exclude.resolve() if exclude else None
    def walk(d):
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        p=Path(e.path)
                        if ex:
                            try:
                                if p.resolve().is_relative_to(ex):continue
                            except Exception:pass
                        if e.is_dir(follow_symlinks=False):
                            if recursive:yield from walk(p)
                        elif e.is_file(follow_symlinks=False) and p.suffix.lower() in SUPPORTED:yield p
                    except OSError:continue
        except OSError:return
    yield from walk(root)


class LibraryDB:
    def __init__(self):
        first = not DB.exists()
        self._init()
        if first and LEGACY_DB.exists():
            self._migrate_legacy()
    def connect(self):
        c=sqlite3.connect(DB,timeout=30); c.execute('PRAGMA foreign_keys=ON'); c.execute('PRAGMA journal_mode=WAL'); c.execute('PRAGMA synchronous=NORMAL'); return c
    def _init(self):
        with self.connect() as c:
            c.executescript('''
            CREATE TABLE IF NOT EXISTS people(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL UNIQUE COLLATE NOCASE,manual_threshold REAL,created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS refs(id INTEGER PRIMARY KEY AUTOINCREMENT,person_id INTEGER NOT NULL,image_path TEXT,embedding BLOB NOT NULL,dim INTEGER NOT NULL,quality REAL DEFAULT .5,is_negative INTEGER DEFAULT 0,source TEXT DEFAULT 'manual',created_at TEXT DEFAULT CURRENT_TIMESTAMP,FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS media_cache(path TEXT PRIMARY KEY,size INTEGER NOT NULL,mtime_ns INTEGER NOT NULL,phash TEXT DEFAULT '0',kind TEXT DEFAULT 'image',faces BLOB,face_count INTEGER DEFAULT 0,analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS face_state(path TEXT NOT NULL,face_idx INTEGER NOT NULL,state TEXT NOT NULL,best_person INTEGER,best_score REAL,second_person INTEGER,second_score REAL,PRIMARY KEY(path,face_idx));
            CREATE TABLE IF NOT EXISTS operations(id INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT DEFAULT CURRENT_TIMESTAMP,source_root TEXT,output_root TEXT,payload TEXT NOT NULL);
            ''')
    def _migrate_legacy(self):
        try:
            old=sqlite3.connect(LEGACY_DB)
            people=old.execute('SELECT id,name FROM people ORDER BY id').fetchall()
            refs=old.execute('SELECT person_id,image_path,embedding,dim FROM refs ORDER BY id').fetchall()
            with self.connect() as c:
                idmap={}
                for old_id,name in people:
                    c.execute('INSERT OR IGNORE INTO people(name) VALUES(?)',(name,))
                    row=c.execute('SELECT id FROM people WHERE name=? COLLATE NOCASE',(name,)).fetchone()
                    if row:idmap[old_id]=row[0]
                for opid,path,blob,dim in refs:
                    if opid in idmap:
                        c.execute('INSERT INTO refs(person_id,image_path,embedding,dim,quality,is_negative,source) VALUES(?,?,?,?,.55,0,?)',(idmap[opid],path,blob,dim,'legacy-v0.4'))
            old.close()
        except Exception:
            pass
    def add_person(self,name):
        with self.connect() as c:return c.execute('INSERT INTO people(name) VALUES(?)',(name.strip(),)).lastrowid
    def delete_person(self,pid):
        with self.connect() as c:c.execute('DELETE FROM people WHERE id=?',(pid,))
    def people(self):
        with self.connect() as c:return c.execute('''SELECT p.id,p.name,SUM(CASE WHEN r.is_negative=0 THEN 1 ELSE 0 END),SUM(CASE WHEN r.is_negative=1 THEN 1 ELSE 0 END),p.manual_threshold FROM people p LEFT JOIN refs r ON r.person_id=p.id GROUP BY p.id,p.name ORDER BY p.name COLLATE NOCASE''').fetchall()
    def add_ref(self,pid,image_path,emb,quality,negative=False,source='manual'):
        blob,dim=vec_blob(emb)
        with self.connect() as c:c.execute('INSERT INTO refs(person_id,image_path,embedding,dim,quality,is_negative,source) VALUES(?,?,?,?,?,?,?)',(pid,image_path,blob,dim,float(quality),int(bool(negative)),source))
    def refs(self,pid,negative=None):
        q='SELECT id,image_path,embedding,dim,quality,is_negative,source FROM refs WHERE person_id=?'; args=[pid]
        if negative is not None:q+=' AND is_negative=?';args.append(int(negative))
        q+=' ORDER BY id'
        with self.connect() as c:rows=c.execute(q,args).fetchall()
        return [(rid,p,blob_vec(b,d),float(qv or .5),bool(n),s) for rid,p,b,d,qv,n,s in rows]
    def delete_ref(self,rid):
        with self.connect() as c:c.execute('DELETE FROM refs WHERE id=?',(rid,))
    def cache_get(self,path):
        try:st=path.stat()
        except OSError:return None
        with self.connect() as c:row=c.execute('SELECT size,mtime_ns,phash,kind,faces FROM media_cache WHERE path=?',(str(path),)).fetchone()
        if not row or int(row[0])!=st.st_size or int(row[1])!=st.st_mtime_ns:return None
        return MediaAnalysis(str(path),int(row[0]),int(row[1]),int(row[2] or 0),unpack_faces(row[4] or b''),row[3] or 'image')
    def cache_put(self,m):
        with self.connect() as c:c.execute('''INSERT INTO media_cache(path,size,mtime_ns,phash,kind,faces,face_count,analyzed_at) VALUES(?,?,?,?,?,?,?,CURRENT_TIMESTAMP) ON CONFLICT(path) DO UPDATE SET size=excluded.size,mtime_ns=excluded.mtime_ns,phash=excluded.phash,kind=excluded.kind,faces=excluded.faces,face_count=excluded.face_count,analyzed_at=CURRENT_TIMESTAMP''',(m.path,m.size,m.mtime_ns,str(int(m.phash)),m.kind,pack_faces(m.faces),len(m.faces)))
    def find_exact_duplicate(self,phash,size,exclude_path):
        if not phash:return None
        with self.connect() as c:row=c.execute('SELECT path,size,mtime_ns,phash,kind,faces FROM media_cache WHERE phash=? AND size=? AND path<>? LIMIT 1',(str(int(phash)),int(size),exclude_path)).fetchone()
        if not row:return None
        return MediaAnalysis(row[0],int(row[1]),int(row[2]),int(row[3]),unpack_faces(row[5] or b''),row[4])
    def set_face_state(self,path,idx,state,bpid,bscore,spid=None,sscore=None):
        with self.connect() as c:c.execute('''INSERT INTO face_state(path,face_idx,state,best_person,best_score,second_person,second_score) VALUES(?,?,?,?,?,?,?) ON CONFLICT(path,face_idx) DO UPDATE SET state=excluded.state,best_person=excluded.best_person,best_score=excluded.best_score,second_person=excluded.second_person,second_score=excluded.second_score''',(path,idx,state,bpid,bscore,spid,sscore))
    def review_items(self,limit=1000):
        with self.connect() as c:return c.execute('''SELECT fs.path,fs.face_idx,fs.state,fs.best_person,fs.best_score,p.name,mc.faces FROM face_state fs LEFT JOIN people p ON p.id=fs.best_person LEFT JOIN media_cache mc ON mc.path=fs.path WHERE fs.state IN ('review','unknown') ORDER BY CASE fs.state WHEN 'review' THEN 0 ELSE 1 END,fs.best_score DESC LIMIT ?''',(limit,)).fetchall()
    def unknown_faces(self,limit=20000):
        out=[]
        for row in self.review_items(limit):
            path,idx,state,pid,score,name,packed=row; fs=unpack_faces(packed or b'')
            if 0<=idx<len(fs):out.append((path,idx,fs[idx],state,pid,score,name))
        return out
    def mark_state(self,path,idx,state):
        with self.connect() as c:c.execute('UPDATE face_state SET state=? WHERE path=? AND face_idx=?',(state,path,idx))
    def clear_cache(self):
        with self.connect() as c:c.execute('DELETE FROM media_cache');c.execute('DELETE FROM face_state')
    def cleanup_cache(self):
        with self.connect() as c:paths=[r[0] for r in c.execute('SELECT path FROM media_cache').fetchall()]
        dead=[p for p in paths if not Path(p).exists()]
        with self.connect() as c:
            for p in dead:c.execute('DELETE FROM media_cache WHERE path=?',(p,));c.execute('DELETE FROM face_state WHERE path=?',(p,))
        return len(paths),len(dead)
    def cache_stats(self):
        with self.connect() as c:
            n,faces=c.execute('SELECT COUNT(*),COALESCE(SUM(face_count),0) FROM media_cache').fetchone(); reviews=c.execute("SELECT COUNT(*) FROM face_state WHERE state IN ('review','unknown')").fetchone()[0]
        return int(n),int(faces),int(reviews),DB.stat().st_size if DB.exists() else 0
    def duplicate_summary(self,limit=50000):
        with self.connect() as c:
            rows=c.execute("SELECT path,phash,size FROM media_cache WHERE phash<>'0' LIMIT ?",(limit,)).fetchall()
        exact={};buckets={}
        for path,ph,size in rows:
            try:h=int(ph)
            except Exception:continue
            exact.setdefault((h,int(size)),[]).append(path)
            key=(h>>52)&0xFFF
            buckets.setdefault(key,[]).append((h,path))
        exact_groups=sum(1 for v in exact.values() if len(v)>1);near_pairs=0;seen=set()
        for key,vals in buckets.items():
            candidates=list(vals)
            for bit in range(12):candidates.extend(buckets.get(key^(1<<bit),[]))
            for h,p in vals:
                for h2,p2 in candidates:
                    if p>=p2 or (p,p2) in seen:continue
                    seen.add((p,p2))
                    if (h^h2).bit_count()<=5:near_pairs+=1
        return len(rows),exact_groups,near_pairs

    def log_operation(self,source,output,created):
        with self.connect() as c:c.execute('INSERT INTO operations(source_root,output_root,payload) VALUES(?,?,?)',(str(source),str(output),json.dumps(created,ensure_ascii=False)))
    def undo_last(self):
        with self.connect() as c:
            row=c.execute('SELECT id,payload FROM operations ORDER BY id DESC LIMIT 1').fetchone()
            if not row:return 0,0
            c.execute('DELETE FROM operations WHERE id=?',(row[0],))
        removed=missed=0
        for s in reversed(json.loads(row[1])):
            p=Path(s)
            try:
                if p.exists() and p.is_file():p.unlink();removed+=1
                else:missed+=1
            except Exception:missed+=1
        return removed,missed
