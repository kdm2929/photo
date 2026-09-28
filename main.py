from __future__ import annotations

import csv
import os
import shutil
import sqlite3
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QSlider, QStackedWidget, QVBoxLayout, QWidget,
)

APP = "PhotoRefSorter"
APP_VERSION = "0.4"
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
YUNET = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

DATA = Path(os.getenv("LOCALAPPDATA") or Path.home()) / APP
DATA.mkdir(parents=True, exist_ok=True)
MODELS = DATA / "models"
MODELS.mkdir(exist_ok=True)
DB = DATA / "people.sqlite3"

try:
    cv2.setNumThreads(1)
except Exception:
    pass


def bundled_models():
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        p = Path(sys._MEIPASS) / "models"
        if p.exists():
            return p
    p = Path(__file__).resolve().parent / "models"
    return p if p.exists() else None


def read_img_fast(p: Path):
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        if not data.size:
            return None
        suffix = p.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            img = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_2)
            if img is not None and max(img.shape[:2]) < 1050:
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def safe_name(s: str):
    bad = '<>:"/\\|?*'
    return "".join("_" if c in bad else c for c in s).strip().rstrip(".") or "이름없음"


def is_inside(child: str, parent: str):
    try:
        return os.path.commonpath([os.path.abspath(child), os.path.abspath(parent)]) == os.path.abspath(parent)
    except Exception:
        return False


def iter_images(root: Path, output: Path, recursive: bool):
    root_s = os.path.abspath(str(root))
    out_s = os.path.abspath(str(output))
    stack = [root_s]
    while stack:
        folder = stack.pop()
        if folder == out_s or is_inside(folder, out_s):
            continue
        try:
            with os.scandir(folder) as it:
                for ent in it:
                    try:
                        if ent.is_dir(follow_symlinks=False):
                            if recursive:
                                ep = os.path.abspath(ent.path)
                                if ep != out_s and not is_inside(ep, out_s):
                                    stack.append(ep)
                        elif ent.is_file(follow_symlinks=False):
                            ext = os.path.splitext(ent.name)[1].lower()
                            if ext in SUPPORTED:
                                yield Path(ent.path)
                    except OSError:
                        continue
        except OSError:
            continue


def place_file(src: Path, dst: Path, mode: str):
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / src.name
    if out.exists():
        try:
            if out.stat().st_size == src.stat().st_size:
                return out
        except OSError:
            pass

    def create(target: Path):
        if mode == "hardlink":
            try:
                os.link(src, target)
                return
            except OSError:
                pass
        shutil.copy2(src, target)

    if not out.exists():
        create(out)
        return out
    i = 2
    while True:
        candidate = dst / f"{src.stem}_{i}{src.suffix}"
        if not candidate.exists():
            create(candidate)
            return candidate
        i += 1


class PeopleDB:
    def __init__(self):
        with self.con() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS people(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE COLLATE NOCASE
                );
                CREATE TABLE IF NOT EXISTS refs(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    image_path TEXT,
                    embedding BLOB,
                    dim INTEGER,
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS scan_cache(
                    path TEXT PRIMARY KEY,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    embeddings BLOB,
                    rows INTEGER NOT NULL,
                    dim INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_scan_cache_stamp
                    ON scan_cache(size, mtime_ns);
                """
            )

    def con(self):
        c = sqlite3.connect(DB, timeout=30)
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA temp_store=MEMORY")
        return c

    def add_person(self, name):
        with self.con() as c:
            return c.execute("INSERT INTO people(name) VALUES(?)", (name.strip(),)).lastrowid

    def delete_person(self, pid):
        with self.con() as c:
            c.execute("DELETE FROM people WHERE id=?", (pid,))

    def add_ref(self, pid, path, emb):
        v = np.asarray(emb, dtype=np.float32).reshape(-1)
        with self.con() as c:
            c.execute(
                "INSERT INTO refs(person_id,image_path,embedding,dim) VALUES(?,?,?,?)",
                (pid, path, v.tobytes(), v.size),
            )

    def people(self):
        with self.con() as c:
            return c.execute(
                """
                SELECT p.id,p.name,COUNT(r.id),
                    (SELECT image_path FROM refs r2
                     WHERE r2.person_id=p.id ORDER BY r2.id LIMIT 1)
                FROM people p
                LEFT JOIN refs r ON r.person_id=p.id
                GROUP BY p.id,p.name
                ORDER BY p.name COLLATE NOCASE
                """
            ).fetchall()

    def refs(self, pid):
        with self.con() as c:
            rows = c.execute(
                "SELECT image_path,embedding,dim FROM refs WHERE person_id=? ORDER BY id",
                (pid,),
            ).fetchall()
        return [(p, np.frombuffer(b, dtype=np.float32, count=d).copy()) for p, b, d in rows]

    def all(self):
        out = {}
        for pid, name, _, _ in self.people():
            rs = [e for _, e in self.refs(pid)]
            if rs:
                out[pid] = (name, rs)
        return out

    def clear_cache(self):
        with self.con() as c:
            c.execute("DELETE FROM scan_cache")

    def cache_count(self):
        with self.con() as c:
            return c.execute("SELECT COUNT(*) FROM scan_cache").fetchone()[0]


class FaceEngine:
    def __init__(self):
        b = bundled_models()
        det = (b / "face_detection_yunet_2023mar.onnx") if b else MODELS / "face_detection_yunet_2023mar.onnx"
        rec = (b / "face_recognition_sface_2021dec.onnx") if b else MODELS / "face_recognition_sface_2021dec.onnx"
        for url, dst in ((YUNET, det), (SFACE, rec)):
            if not dst.exists():
                dst = MODELS / dst.name
                if not dst.exists():
                    urllib.request.urlretrieve(url, dst)
                if "yunet" in dst.name:
                    det = dst
                else:
                    rec = dst
        self.det = cv2.FaceDetectorYN.create(str(det), "", (320, 320), 0.84, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(rec), "")

    def detect(self, img):
        h, w = img.shape[:2]
        scale = min(1.0, 1650 / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        return img, ([] if faces is None else [f for f in faces if f[-1] >= 0.84])

    def emb(self, img, face):
        x = self.rec.feature(self.rec.alignCrop(img, face)).astype(np.float32).reshape(-1)
        n = np.linalg.norm(x)
        return x / n if n > 1e-8 else x

    def ref_info(self, p: Path):
        img = read_img_fast(p)
        if img is None:
            raise ValueError("이미지를 읽을 수 없음")
        img, faces = self.detect(img)
        if not faces:
            raise ValueError("얼굴을 찾지 못함")
        face = max(faces, key=lambda x: float(x[2] * x[3]))
        return self.emb(img, face), len(faces), float(face[-1])

    def image(self, p: Path):
        img = read_img_fast(p)
        if img is None:
            raise ValueError("이미지 읽기 실패")
        img, faces = self.detect(img)
        out = []
        for f in faces:
            try:
                out.append(self.emb(img, f))
            except Exception:
                continue
        return out


@dataclass
class Config:
    source: Path
    output: Path
    auto: float
    review: float
    recursive: bool
    multi: bool
    workers: int
    mode: str
    use_cache: bool


class Matcher:
    def __init__(self, people):
        self.people = people
        self.pids = list(people.keys())
        self.names = [people[pid][0] for pid in self.pids]
        centroids = []
        ref_chunks = []
        self.ref_slices = []
        offset = 0
        for pid in self.pids:
            vecs = np.vstack(people[pid][1]).astype(np.float32)
            c = vecs.mean(axis=0)
            n = np.linalg.norm(c)
            if n > 1e-8:
                c /= n
            centroids.append(c)
            ref_chunks.append(vecs)
            self.ref_slices.append((offset, offset + len(vecs)))
            offset += len(vecs)
        self.centroids = np.ascontiguousarray(np.vstack(centroids), dtype=np.float32)
        self.refs = np.ascontiguousarray(np.vstack(ref_chunks), dtype=np.float32)

    def best_person(self, v):
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        centroid_scores = self.centroids @ v
        ref_scores = self.refs @ v
        person_scores = centroid_scores.copy()
        for i, (a, b) in enumerate(self.ref_slices):
            person_scores[i] = max(float(person_scores[i]), float(np.max(ref_scores[a:b])) * 0.985)
        idx = int(np.argmax(person_scores))
        return float(person_scores[idx]), self.pids[idx], self.names[idx]


class Worker(QObject):
    progress = Signal(int, int, str, float, int, int)
    phase = Signal(str)
    done = Signal(dict)
    fail = Signal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stop_event = threading.Event()
        self.local = threading.local()

    def stop(self):
        self.stop_event.set()

    def _engine(self):
        eng = getattr(self.local, "engine", None)
        if eng is None:
            eng = FaceEngine()
            self.local.engine = eng
        return eng

    def _analyze(self, p: Path):
        if self.stop_event.is_set():
            return p, None, "STOP"
        try:
            return p, self._engine().image(p), None
        except Exception as e:
            return p, None, str(e)

    @staticmethod
    def _pack(vectors):
        if not vectors:
            return b"", 0, 0
        arr = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
        return arr.tobytes(), arr.shape[0], arr.shape[1]

    @staticmethod
    def _unpack(blob, rows, dim):
        if not rows or not dim or not blob:
            return []
        arr = np.frombuffer(blob, dtype=np.float32, count=rows * dim).reshape(rows, dim)
        return [row.copy() for row in arr]

    def _classify_and_place(self, p, vectors, matcher, stats, writer):
        if not vectors:
            place_file(p, self.cfg.output / "_얼굴없음", self.cfg.mode)
            stats["얼굴없음"] += 1
            writer.writerow([p, "얼굴없음", "", ""])
            return
        found = {}
        review = None
        for v in vectors:
            s, pid, name = matcher.best_person(v)
            if s >= self.cfg.auto:
                found[pid] = max(found.get(pid, -1.0), s)
            elif s >= self.cfg.review and (review is None or s > review[0]):
                review = (s, pid, name)
        if found:
            arr = sorted(found.items(), key=lambda x: x[1], reverse=True)
            if not self.cfg.multi:
                arr = arr[:1]
            labels = []
            for pid, score in arr:
                name = matcher.people[pid][0]
                place_file(p, self.cfg.output / safe_name(name), self.cfg.mode)
                labels.append(f"{name}:{score:.3f}")
            stats["분류"] += 1
            writer.writerow([p, "분류", " / ".join(labels), max(found.values())])
        elif review:
            s, _, name = review
            place_file(p, self.cfg.output / "_확인필요", self.cfg.mode)
            stats["확인필요"] += 1
            writer.writerow([p, "확인필요", name, f"{s:.3f}"])
        else:
            place_file(p, self.cfg.output / "_미확인", self.cfg.mode)
            stats["미확인"] += 1
            writer.writerow([p, "미확인", "", ""])

    def run(self):
        started = time.perf_counter()
        try:
            db = PeopleDB()
            people = db.all()
            if not people:
                raise RuntimeError("등록된 인물과 레퍼런스가 없습니다.")
            matcher = Matcher(people)
            self.phase.emit("사진 목록을 빠르게 읽는 중…")
            files = list(iter_images(self.cfg.source, self.cfg.output, self.cfg.recursive))
            total = len(files)
            if total == 0:
                raise RuntimeError("분석할 사진을 찾지 못했습니다.")
            self.cfg.output.mkdir(parents=True, exist_ok=True)
            stats = {"전체": total, "분류": 0, "확인필요": 0, "미확인": 0, "얼굴없음": 0, "오류": 0, "캐시": 0, "새분석": 0, "중단": False}
            cache_con = db.con()
            get_cache = cache_con.cursor()
            put_cache = cache_con.cursor()
            report = self.cfg.output / "분류결과.csv"
            with open(report, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["원본파일", "상태", "인물", "유사도"])
                misses = []
                processed = 0
                self.phase.emit("캐시 확인 중…")
                for p in files:
                    if self.stop_event.is_set():
                        stats["중단"] = True
                        break
                    try:
                        st = p.stat()
                    except OSError:
                        stats["오류"] += 1
                        continue
                    cached = None
                    if self.cfg.use_cache:
                        cached = get_cache.execute("SELECT embeddings,rows,dim FROM scan_cache WHERE path=? AND size=? AND mtime_ns=?", (str(p), st.st_size, st.st_mtime_ns)).fetchone()
                    if cached is not None:
                        vectors = self._unpack(*cached)
                        self._classify_and_place(p, vectors, matcher, stats, writer)
                        stats["캐시"] += 1
                        processed += 1
                        if processed % 25 == 0 or processed == total:
                            elapsed = max(0.001, time.perf_counter() - started)
                            self.progress.emit(processed, total, p.name, processed / elapsed, stats["캐시"], stats["새분석"])
                    else:
                        misses.append((p, st.st_size, st.st_mtime_ns))
                if not stats["중단"] and misses:
                    self.phase.emit(f"AI 얼굴 분석 중 · {self.cfg.workers}개 워커 · {len(misses):,}장 새 분석")
                    chunk_size = max(64, self.cfg.workers * 24)
                    with ThreadPoolExecutor(max_workers=self.cfg.workers, thread_name_prefix="face") as ex:
                        for start in range(0, len(misses), chunk_size):
                            if self.stop_event.is_set():
                                stats["중단"] = True
                                break
                            chunk = misses[start:start + chunk_size]
                            paths = [x[0] for x in chunk]
                            meta = {str(x[0]): (x[1], x[2]) for x in chunk}
                            for p, vectors, err in ex.map(self._analyze, paths):
                                if self.stop_event.is_set():
                                    stats["중단"] = True
                                    break
                                if err == "STOP":
                                    stats["중단"] = True
                                    break
                                if err is not None:
                                    stats["오류"] += 1
                                    writer.writerow([p, "오류", "", err])
                                else:
                                    self._classify_and_place(p, vectors, matcher, stats, writer)
                                    if self.cfg.use_cache:
                                        try:
                                            size, mtime_ns = meta[str(p)]
                                            blob, rows, dim = self._pack(vectors)
                                            put_cache.execute("""INSERT INTO scan_cache(path,size,mtime_ns,embeddings,rows,dim,updated_at) VALUES(?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET size=excluded.size,mtime_ns=excluded.mtime_ns,embeddings=excluded.embeddings,rows=excluded.rows,dim=excluded.dim,updated_at=excluded.updated_at""", (str(p), size, mtime_ns, blob, rows, dim, int(time.time())))
                                        except Exception:
                                            pass
                                stats["새분석"] += 1
                                processed += 1
                                if processed % 5 == 0 or processed == total:
                                    elapsed = max(0.001, time.perf_counter() - started)
                                    self.progress.emit(processed, total, p.name, processed / elapsed, stats["캐시"], stats["새분석"])
                            cache_con.commit()
            cache_con.close()
            stats["초"] = round(time.perf_counter() - started, 2)
            self.done.emit(stats)
        except Exception as e:
            self.fail.emit(str(e))


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db = PeopleDB()
        self.pid = None
        self.thread = None
        self.worker = None
        self._scan_started = None
        self.setWindowTitle(f"PhotoRef Sorter {APP_VERSION}")
        self.resize(1120, 760)
        self.setMinimumSize(940, 640)
        root = QWidget()
        self.setCentralWidget(root)
        hl = QHBoxLayout(root)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)
        side = QFrame()
        side.setFixedWidth(226)
        side.setObjectName("side")
        sl = QVBoxLayout(side)
        sl.setContentsMargins(18, 26, 18, 22)
        logo = QLabel("PhotoRef\nSorter")
        logo.setObjectName("logo")
        sl.addWidget(logo)
        ver = QLabel(f"v{APP_VERSION} · Fast Scan")
        ver.setObjectName("sideMuted")
        sl.addWidget(ver)
        sl.addSpacing(28)
        self.b1 = QPushButton("👤   인물 & 레퍼런스")
        self.b2 = QPushButton("⚡   빠른 자동 분류")
        for b in (self.b1, self.b2):
            b.setObjectName("nav")
            b.setMinimumHeight(46)
            sl.addWidget(b)
        sl.addStretch()
        privacy = QLabel("로컬 얼굴 분석\n원본 파일 보존")
        privacy.setObjectName("sideMuted")
        sl.addWidget(privacy)
        hl.addWidget(side)
        self.stack = QStackedWidget()
        hl.addWidget(self.stack, 1)
        self.stack.addWidget(self.people_page())
        self.stack.addWidget(self.scan_page())
        self.b1.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.b2.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.setStyleSheet("""
            * { font-family: "Segoe UI", "Malgun Gothic"; font-size: 14px; }
            QMainWindow, QWidget { background:#F5F7FA; color:#20242A; }
            #side { background:#151922; }
            #logo { color:white; font-size:26px; font-weight:800; }
            #sideMuted { color:#818A98; font-size:12px; }
            QPushButton#nav { background:transparent; color:#C8CFD9; text-align:left; border:none; border-radius:11px; padding:10px 12px; font-weight:700; }
            QPushButton#nav:hover { background:#232936; color:white; }
            #h1 { font-size:30px; font-weight:800; }
            #h2 { font-size:18px; font-weight:800; }
            #muted { color:#6E7784; font-size:12px; }
            #small { color:#89919D; font-size:11px; }
            #metric { font-size:22px; font-weight:800; }
            #panel, #card { background:white; border:1px solid #E1E6EC; border-radius:16px; }
            QLineEdit, QListWidget, QComboBox { background:white; border:1px solid #D8DFE8; border-radius:10px; padding:9px 10px; }
            QLineEdit:focus, QListWidget:focus, QComboBox:focus { border:1px solid #6372F4; }
            QPushButton { background:#EDF1F5; border:none; border-radius:10px; padding:9px 14px; font-weight:700; color:#303640; }
            QPushButton:hover { background:#E4E9EF; }
            #primary { background:#5364F5; color:white; }
            #primary:hover { background:#4556E7; }
            #danger { background:#FFF0F0; color:#B63A3A; }
            #stop { background:#FFF3E9; color:#AA5C1B; }
            QProgressBar { border:none; border-radius:7px; background:#E7EBF0; height:14px; text-align:center; }
            QProgressBar::chunk { background:#5C6CF4; border-radius:7px; }
        """)
        self.refresh()

    def title(self, text, sub):
        v = QVBoxLayout()
        a = QLabel(text)
        a.setObjectName("h1")
        b = QLabel(sub)
        b.setObjectName("muted")
        b.setWordWrap(True)
        v.addWidget(a)
        v.addWidget(b)
        return v

    def metric_card(self, title, value="—"):
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(15, 12, 15, 12)
        t = QLabel(title)
        t.setObjectName("small")
        val = QLabel(value)
        val.setObjectName("metric")
        lay.addWidget(t)
        lay.addWidget(val)
        return card, val

    def people_page(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(34, 30, 34, 30)
        v.setSpacing(14)
        v.addLayout(self.title("인물 & 레퍼런스", "인물별 기준 사진을 등록합니다. 각 레퍼런스의 일관성 점수도 함께 보여줍니다."))
        r = QHBoxLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("예: 리즈")
        self.name.setMinimumHeight(42)
        add = QPushButton("인물 추가")
        add.setObjectName("primary")
        add.setMinimumHeight(42)
        add.clicked.connect(self.add_person)
        self.name.returnPressed.connect(self.add_person)
        r.addWidget(self.name, 1)
        r.addWidget(add)
        v.addLayout(r)
        panel = QFrame()
        panel.setObjectName("panel")
        p = QHBoxLayout(panel)
        p.setContentsMargins(16, 16, 16, 16)
        p.setSpacing(16)
        left = QVBoxLayout()
        left.addWidget(QLabel("등록된 인물"))
        self.people = QListWidget()
        self.people.currentRowChanged.connect(self.select_person)
        left.addWidget(self.people, 1)
        right = QVBoxLayout()
        self.person_summary = QLabel("인물을 선택하세요")
        self.person_summary.setObjectName("h2")
        self.person_detail = QLabel("레퍼런스 5~20장 권장")
        self.person_detail.setObjectName("muted")
        self.refs = QListWidget()
        right.addWidget(self.person_summary)
        right.addWidget(self.person_detail)
        right.addWidget(self.refs, 1)
        p.addLayout(left, 1)
        p.addLayout(right, 1)
        v.addWidget(panel, 1)
        br = QHBoxLayout()
        ar = QPushButton("레퍼런스 사진 추가")
        ar.setObjectName("primary")
        ar.clicked.connect(self.add_refs)
        dr = QPushButton("선택 인물 삭제")
        dr.setObjectName("danger")
        dr.clicked.connect(self.del_person)
        br.addWidget(ar)
        br.addWidget(dr)
        br.addStretch()
        v.addLayout(br)
        return w

    def scan_page(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(34, 30, 34, 30)
        v.setSpacing(14)
        v.addLayout(self.title("빠른 자동 분류", "수만 장 폴더를 위한 병렬 분석 + 변경 감지 캐시. 두 번째 스캔부터는 변경 없는 사진을 다시 읽지 않습니다."))
        metrics = QHBoxLayout()
        c1, self.metric_speed = self.metric_card("현재 처리 속도", "—")
        c2, self.metric_cache = self.metric_card("캐시 사용", "0")
        c3, self.metric_new = self.metric_card("새 AI 분석", "0")
        c4, self.metric_eta = self.metric_card("예상 남은 시간", "—")
        for c in (c1, c2, c3, c4):
            metrics.addWidget(c, 1)
        v.addLayout(metrics)
        panel = QFrame()
        panel.setObjectName("panel")
        p = QVBoxLayout(panel)
        p.setContentsMargins(20, 18, 20, 18)
        p.setSpacing(11)
        self.src = QLineEdit()
        self.dst = QLineEdit()
        bs = QPushButton("폴더 선택")
        bo = QPushButton("결과 선택")
        bs.clicked.connect(self.pick_src)
        bo.clicked.connect(self.pick_dst)
        for label, edit, btn in (("분류할 폴더", self.src, bs), ("결과 폴더", self.dst, bo)):
            p.addWidget(QLabel(label))
            rr = QHBoxLayout()
            rr.addWidget(edit, 1)
            rr.addWidget(btn)
            p.addLayout(rr)
        opts = QGridLayout()
        self.rec = QCheckBox("하위 폴더까지")
        self.rec.setChecked(True)
        self.multi = QCheckBox("여러 등록 인물이 있으면 각 폴더에 분류")
        self.multi.setChecked(True)
        self.cache = QCheckBox("분석 캐시 사용 (강력 권장)")
        self.cache.setChecked(True)
        opts.addWidget(self.rec, 0, 0)
        opts.addWidget(self.multi, 0, 1)
        opts.addWidget(self.cache, 1, 0)
        self.workers = QComboBox()
        cpu = max(2, os.cpu_count() or 4)
        auto_workers = min(6, max(2, cpu // 2))
        self.workers.addItem(f"자동 ({auto_workers})", auto_workers)
        for n in (2, 3, 4, 6, 8):
            if n <= cpu + 2:
                self.workers.addItem(f"{n}개 워커", n)
        opts.addWidget(QLabel("병렬 분석"), 2, 0)
        opts.addWidget(self.workers, 2, 1)
        self.mode = QComboBox()
        self.mode.addItem("안전한 파일 복사", "copy")
        self.mode.addItem("빠른 NTFS 하드링크 (같은 드라이브)", "hardlink")
        opts.addWidget(QLabel("결과 생성 방식"), 3, 0)
        opts.addWidget(self.mode, 3, 1)
        p.addLayout(opts)
        rr = QHBoxLayout()
        rr.addWidget(QLabel("자동 분류 기준"))
        self.th = QSlider(Qt.Horizontal)
        self.th.setRange(36, 60)
        self.th.setValue(43)
        self.tv = QLabel("0.43")
        self.th.valueChanged.connect(lambda x: self.tv.setText(f"{x/100:.2f}"))
        rr.addWidget(self.th, 1)
        rr.addWidget(self.tv)
        p.addLayout(rr)
        btns = QHBoxLayout()
        self.go = QPushButton("⚡ 빠른 분류 시작")
        self.go.setObjectName("primary")
        self.go.setMinimumHeight(46)
        self.go.clicked.connect(self.start)
        self.stop_btn = QPushButton("중단")
        self.stop_btn.setObjectName("stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_scan)
        self.clear_cache_btn = QPushButton("분석 캐시 비우기")
        self.clear_cache_btn.clicked.connect(self.clear_cache)
        btns.addWidget(self.go, 1)
        btns.addWidget(self.stop_btn)
        btns.addWidget(self.clear_cache_btn)
        p.addLayout(btns)
        self.pb = QProgressBar()
        self.pb.setRange(0, 100)
        self.pb.setValue(0)
        p.addWidget(self.pb)
        self.phase_label = QLabel("준비됨")
        self.phase_label.setObjectName("h2")
        self.st = QLabel(f"캐시 {self.db.cache_count():,}장 · 첫 스캔은 AI 분석, 이후 변경 없는 사진은 캐시 재사용")
        self.st.setObjectName("muted")
        self.st.setWordWrap(True)
        p.addWidget(self.phase_label)
        p.addWidget(self.st)
        v.addWidget(panel)
        v.addStretch()
        return w

    def refresh(self):
        rows = self.db.people()
        self.people.clear()
        self._rows = rows
        for _, name, count, _ in rows:
            self.people.addItem(f"{name}   ·   레퍼런스 {count}장")
        self.refs.clear()
        self.pid = None
        self.person_summary.setText("인물을 선택하세요")
        self.person_detail.setText("레퍼런스 5~20장 권장")

    def add_person(self):
        n = self.name.text().strip()
        if not n:
            return
        try:
            self.db.add_person(n)
            self.name.clear()
            self.refresh()
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "중복", "이미 있는 이름입니다.")

    def select_person(self, row):
        if row < 0 or row >= len(self._rows):
            return
        self.pid = self._rows[row][0]
        name = self._rows[row][1]
        refs = self.db.refs(self.pid)
        self.refs.clear()
        self.person_summary.setText(name)
        if not refs:
            self.person_detail.setText("레퍼런스 없음")
            return
        mat = np.vstack([e for _, e in refs]).astype(np.float32)
        centroid = mat.mean(axis=0)
        n = np.linalg.norm(centroid)
        if n > 1e-8:
            centroid /= n
        sims = mat @ centroid
        avg = float(np.mean(sims))
        self.person_detail.setText(f"레퍼런스 {len(refs)}장 · 평균 일관성 {avg:.3f} · {'좋음' if avg >= .72 else '다양함/검토 권장'}")
        for (path, _), sim in zip(refs, sims):
            self.refs.addItem(f"{Path(path).name}   ·   기준유사도 {float(sim):.3f}")

    def add_refs(self):
        if not self.pid:
            QMessageBox.information(self, "선택", "먼저 인물을 선택하세요.")
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "레퍼런스 선택", "", "사진 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        if not paths:
            return
        try:
            eng = FaceEngine()
        except Exception as e:
            QMessageBox.critical(self, "AI 오류", str(e))
            return
        ok = 0
        warnings = []
        failed = []
        self.setCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            for i, p in enumerate(paths, 1):
                self.statusBar().showMessage(f"레퍼런스 분석 {i}/{len(paths)} · {Path(p).name}")
                QApplication.processEvents()
                try:
                    emb, face_count, conf = eng.ref_info(Path(p))
                    self.db.add_ref(self.pid, p, emb)
                    ok += 1
                    if face_count > 1:
                        warnings.append(f"{Path(p).name}: 얼굴 {face_count}명 → 가장 큰 얼굴 사용")
                    if conf < 0.90:
                        warnings.append(f"{Path(p).name}: 얼굴 검출 신뢰도 {conf:.2f}")
                except Exception as e:
                    failed.append(f"{Path(p).name}: {e}")
        finally:
            self.unsetCursor()
            self.statusBar().clearMessage()
        selected_pid = self.pid
        self.refresh()
        for i, row in enumerate(self._rows):
            if row[0] == selected_pid:
                self.people.setCurrentRow(i)
                break
        msg = f"{ok}장 등록 완료"
        if warnings:
            msg += "\n\n주의:\n" + "\n".join(warnings[:8])
        if failed:
            msg += "\n\n실패:\n" + "\n".join(failed[:8])
        QMessageBox.information(self, "레퍼런스 분석 결과", msg)

    def del_person(self):
        if self.pid and QMessageBox.question(self, "삭제", "선택한 인물을 삭제할까요?") == QMessageBox.Yes:
            self.db.delete_person(self.pid)
            self.refresh()

    def pick_src(self):
        p = QFileDialog.getExistingDirectory(self, "분류할 폴더")
        if p:
            self.src.setText(p)
            self.dst.setText(str(Path(p) / "분류결과"))

    def pick_dst(self):
        p = QFileDialog.getExistingDirectory(self, "결과 폴더")
        if p:
            self.dst.setText(p)

    def clear_cache(self):
        count = self.db.cache_count()
        if count == 0:
            QMessageBox.information(self, "캐시", "비울 분석 캐시가 없습니다.")
            return
        if QMessageBox.question(self, "분석 캐시 비우기", f"캐시 {count:,}장의 얼굴 분석 결과를 삭제할까요?\n원본 사진은 삭제되지 않습니다.") == QMessageBox.Yes:
            self.db.clear_cache()
            self.st.setText("분석 캐시를 비웠습니다. 다음 스캔은 전체 AI 분석을 수행합니다.")
            self.metric_cache.setText("0")

    def start(self):
        s = Path(self.src.text().strip())
        if not s.is_dir():
            QMessageBox.warning(self, "폴더", "분류할 폴더를 선택하세요.")
            return
        if not self.db.all():
            QMessageBox.warning(self, "레퍼런스", "인물과 레퍼런스를 먼저 등록하세요.")
            return
        d = Path(self.dst.text().strip()) if self.dst.text().strip() else s / "분류결과"
        t = self.th.value() / 100
        workers = int(self.workers.currentData())
        mode = str(self.mode.currentData())
        cfg = Config(s, d, t, max(0.30, t - 0.09), self.rec.isChecked(), self.multi.isChecked(), workers, mode, self.cache.isChecked())
        self.go.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.clear_cache_btn.setEnabled(False)
        self.pb.setValue(0)
        self.metric_speed.setText("—")
        self.metric_cache.setText("0")
        self.metric_new.setText("0")
        self.metric_eta.setText("—")
        self.phase_label.setText("시작 중…")
        self.st.setText("사진 목록과 기존 캐시를 확인합니다.")
        self._scan_started = time.perf_counter()
        self.thread = QThread()
        self.worker = Worker(cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.prog)
        self.worker.phase.connect(self.phase_label.setText)
        self.worker.done.connect(self.done)
        self.worker.fail.connect(self.fail)
        self.worker.done.connect(self.thread.quit)
        self.worker.fail.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def stop_scan(self):
        if self.worker:
            self.worker.stop()
            self.phase_label.setText("중단 요청됨…")
            self.stop_btn.setEnabled(False)

    def prog(self, i, n, name, speed, cache_hits, new_count):
        self.pb.setValue(int(i * 100 / max(1, n)))
        self.metric_speed.setText(f"{speed:.1f} 장/s")
        self.metric_cache.setText(f"{cache_hits:,}")
        self.metric_new.setText(f"{new_count:,}")
        remain = max(0, n - i)
        eta = remain / max(speed, 0.01)
        if eta < 60:
            eta_text = f"{eta:.0f}초"
        elif eta < 3600:
            eta_text = f"{eta/60:.1f}분"
        else:
            eta_text = f"{eta/3600:.1f}시간"
        self.metric_eta.setText(eta_text)
        self.st.setText(f"{i:,}/{n:,} · {name} · 캐시 {cache_hits:,} · 새 분석 {new_count:,}")

    def _reset_after_scan(self):
        self.go.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.clear_cache_btn.setEnabled(True)

    def done(self, s):
        self._reset_after_scan()
        if not s.get("중단"):
            self.pb.setValue(100)
            self.phase_label.setText("완료")
        else:
            self.phase_label.setText("중단됨")
        elapsed = s.get("초", 0)
        self.st.setText(f"처리 {s['전체']:,}장 · 캐시 {s['캐시']:,} · 새 AI 분석 {s['새분석']:,} · {elapsed:.1f}초")
        QMessageBox.information(self, "분류 결과", f"전체 대상 {s['전체']:,}장\n자동 분류 {s['분류']:,}장\n확인 필요 {s['확인필요']:,}장\n미확인 {s['미확인']:,}장\n얼굴 없음 {s['얼굴없음']:,}장\n오류 {s['오류']:,}장\n\n캐시 재사용 {s['캐시']:,}장\n새 AI 분석 {s['새분석']:,}장\n총 소요 {elapsed:.1f}초" + ("\n\n사용자 요청으로 중단됨" if s.get("중단") else ""))

    def fail(self, e):
        self._reset_after_scan()
        self.phase_label.setText("오류")
        self.st.setText(str(e))
        QMessageBox.critical(self, "오류", e)


def main():
    if "--self-test" in sys.argv:
        try:
            e = FaceEngine()
            return 0 if e.det is not None and e.rec is not None else 1
        except Exception:
            return 1
    a = QApplication(sys.argv)
    a.setApplicationName(APP)
    w = Main()
    w.show()
    return a.exec()


if __name__ == "__main__":
    sys.exit(main())
