from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageOps, ExifTags
except Exception:
    Image = ImageOps = ExifTags = None
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None
try:
    import pillow_avif  # type: ignore  # noqa: F401
except Exception:
    try:
        import pillow_avif_plugin  # type: ignore  # noqa: F401
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

APP = "PhotoRefSorter"
VERSION = "0.6"
DATA = Path(os.getenv("LOCALAPPDATA") or Path.home()) / APP
DATA.mkdir(parents=True, exist_ok=True)
MODELS = DATA / "models"; MODELS.mkdir(exist_ok=True)
THUMBS = DATA / "thumbs"; THUMBS.mkdir(exist_ok=True)
DB = DATA / "library.sqlite3"
LEGACY_DB = DATA / "people.sqlite3"
YUNET = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif", ".avif", ".gif"}
RAW_EXTS = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf", ".pef", ".srw"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
SUPPORTED = IMAGE_EXTS | RAW_EXTS | VIDEO_EXTS


def bundled_models() -> Path | None:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        p = Path(sys._MEIPASS) / "models"
        if p.exists():
            return p
    p = Path(__file__).resolve().parent / "models"
    return p if p.exists() else None


def model_paths() -> tuple[Path, Path]:
    b = bundled_models()
    if b:
        d, r = b / "face_detection_yunet_2023mar.onnx", b / "face_recognition_sface_2021dec.onnx"
        if d.exists() and r.exists():
            return d, r
    d, r = MODELS / "face_detection_yunet_2023mar.onnx", MODELS / "face_recognition_sface_2021dec.onnx"
    if not d.exists():
        urllib.request.urlretrieve(YUNET, d)
    if not r.exists():
        urllib.request.urlretrieve(SFACE, r)
    return d, r


def safe_name(s: str) -> str:
    bad = '<>:"/\\|?*'
    return "".join("_" if c in bad else c for c in s).strip().rstrip(".") or "이름없음"


def vec_blob(v: np.ndarray) -> tuple[bytes, int]:
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    return a.tobytes(), int(a.size)


def blob_vec(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim).copy()


@dataclass
class FaceData:
    embedding: np.ndarray
    quality: float
    bbox: tuple[float, float, float, float]


@dataclass
class MediaAnalysis:
    path: str
    size: int
    mtime_ns: int
    phash: int
    faces: list[FaceData]
    kind: str
    captured_at: str | None = None
    width: int = 0
    height: int = 0
    duration: float = 0.0


@dataclass
class PersonModel:
    pid: int
    name: str
    positives: np.ndarray
    pos_quality: np.ndarray
    negatives: np.ndarray | None
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
    auto_workers: bool = True


def pack_faces(faces: list[FaceData]) -> bytes:
    if not faces:
        return b""
    arr = np.vstack([f.embedding for f in faces]).astype(np.float32)
    meta = np.asarray([[f.quality, *f.bbox] for f in faces], dtype=np.float32)
    head = np.asarray([len(faces), arr.shape[1]], dtype=np.int32).tobytes()
    return head + arr.tobytes() + meta.tobytes()


def unpack_faces(blob: bytes) -> list[FaceData]:
    if not blob or len(blob) < 8:
        return []
    n, dim = [int(x) for x in np.frombuffer(blob[:8], dtype=np.int32)]
    if n <= 0 or dim <= 0:
        return []
    emb_bytes = n * dim * 4
    arr = np.frombuffer(blob[8:8 + emb_bytes], dtype=np.float32).reshape(n, dim).copy()
    meta = np.frombuffer(blob[8 + emb_bytes:], dtype=np.float32).reshape(n, 5).copy()
    return [FaceData(arr[i], float(meta[i, 0]), tuple(float(x) for x in meta[i, 1:5])) for i in range(n)]


def phash64(img: np.ndarray | None) -> int:
    if img is None or not img.size:
        return 0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(g)[:8, :8].flatten()
    med = float(np.median(low[1:]))
    value = 0
    for i, bit in enumerate(low > med):
        if bit:
            value |= 1 << i
    return value & ((1 << 64) - 1)


def _extract_capture_time(im) -> str | None:
    if im is None:
        return None
    try:
        exif = im.getexif()
        for key in (36867, 36868, 306):
            raw = exif.get(key)
            if raw:
                s = str(raw).strip()
                for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                    try:
                        return datetime.strptime(s[:19], fmt).isoformat(sep=" ")
                    except Exception:
                        pass
    except Exception:
        pass
    return None


def load_pil(path: Path, max_side: int = 2200):
    if Image is None:
        return None
    ext = path.suffix.lower()
    if ext in RAW_EXTS and rawpy is not None:
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True)
        im = Image.fromarray(rgb)
    else:
        im = Image.open(path)
        try:
            im.seek(0)
        except Exception:
            pass
        if ImageOps is not None:
            im = ImageOps.exif_transpose(im)
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return im


def read_image(path: Path, max_side: int = 2200) -> np.ndarray | None:
    ext = path.suffix.lower()
    try:
        if ext in {".jpg", ".jpeg"}:
            data = np.fromfile(str(path), dtype=np.uint8)
            if not data.size:
                return None
            flag = cv2.IMREAD_REDUCED_COLOR_2 if path.stat().st_size > 3_000_000 else cv2.IMREAD_COLOR
            img = cv2.imdecode(data, flag)
        elif ext in RAW_EXTS or ext in {".heic", ".heif", ".avif", ".gif"}:
            im = load_pil(path, max_side)
            if im is None:
                return None
            img = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1:
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        return img
    except Exception:
        return None


def media_frames(path: Path, video_samples: int = 10, max_side: int = 2200) -> list[np.ndarray]:
    if path.suffix.lower() not in VIDEO_EXTS:
        img = read_image(path, max_side)
        return [] if img is None else [img]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    out: list[np.ndarray] = []
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        duration = frames / fps if frames > 0 and fps > 0 else 0
        n = min(video_samples, max(3, int(duration / 8) + 2)) if duration else video_samples
        for pos in np.linspace(.05, .95, n):
            if frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames * float(pos)))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            scale = min(1.0, max_side / max(h, w))
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            out.append(frame)
    finally:
        cap.release()
    return out


def media_basic_info(path: Path) -> tuple[str | None, int, int, float]:
    ext = path.suffix.lower()
    captured = None
    w = h = 0
    duration = 0.0
    try:
        if ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                duration = frames / fps if fps > 0 else 0.0
            cap.release()
        elif Image is not None and ext not in RAW_EXTS:
            im = Image.open(path)
            w, h = im.size
            captured = _extract_capture_time(im)
            im.close()
        if not captured:
            captured = datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ")
    except Exception:
        try:
            captured = datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ")
        except Exception:
            pass
    return captured, int(w), int(h), float(duration)


def iter_media(root: Path, recursive: bool = True, exclude: Path | None = None):
    ex = exclude.resolve() if exclude else None
    def walk(d: Path):
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        p = Path(e.path)
                        if ex:
                            try:
                                if p.resolve().is_relative_to(ex):
                                    continue
                            except Exception:
                                pass
                        if e.is_dir(follow_symlinks=False):
                            if recursive:
                                yield from walk(p)
                        elif e.is_file(follow_symlinks=False) and p.suffix.lower() in SUPPORTED:
                            yield p
                    except OSError:
                        continue
        except OSError:
            return
    yield from walk(root)


def thumb_key(path: str, size: int = 320) -> str:
    p = Path(path)
    try:
        st = p.stat(); stamp = f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        stamp = "missing"
    h = hashlib.sha1(f"{path}|{stamp}|{size}".encode("utf-8", "ignore")).hexdigest()
    return h


def thumbnail_path(path: str, size: int = 320) -> Path:
    return THUMBS / f"{thumb_key(path, size)}.jpg"


def generate_thumbnail(path: str, size: int = 320) -> Path | None:
    out = thumbnail_path(path, size)
    if out.exists():
        return out
    pp=Path(path)
    if pp.suffix.lower() in VIDEO_EXTS:
        frames=media_frames(pp,1,max_side=max(640,size*2)); img=frames[0] if frames else None
    else:
        img=read_image(pp, max_side=max(640, size * 2))
    if img is None:
        return None
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return None
    side = min(h, w); y = (h - side) // 2; x = (w - side) // 2
    crop = img[y:y + side, x:x + side]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    try:
        cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(out))
        return out
    except Exception:
        return None


class LibraryDB:
    def __init__(self):
        first = not DB.exists()
        self._init()
        if first and LEGACY_DB.exists():
            self._migrate_legacy()

    def connect(self):
        c = sqlite3.connect(DB, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        return c

    def _ensure_column(self, c, table: str, name: str, decl: str):
        cols = {r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
        if name not in cols:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    def _init(self):
        with self.connect() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS people(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                manual_threshold REAL,
                hero_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS refs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                image_path TEXT,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                quality REAL DEFAULT .5,
                is_negative INTEGER DEFAULT 0,
                source TEXT DEFAULT 'manual',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS media_cache(
                path TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                phash TEXT DEFAULT '0',
                kind TEXT DEFAULT 'image',
                faces BLOB,
                face_count INTEGER DEFAULT 0,
                captured_at TEXT,
                width INTEGER DEFAULT 0,
                height INTEGER DEFAULT 0,
                duration REAL DEFAULT 0,
                analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS face_state(
                path TEXT NOT NULL,
                face_idx INTEGER NOT NULL,
                state TEXT NOT NULL,
                best_person INTEGER,
                best_score REAL,
                second_person INTEGER,
                second_score REAL,
                third_person INTEGER,
                third_score REAL,
                PRIMARY KEY(path,face_idx)
            );
            CREATE TABLE IF NOT EXISTS albums(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS album_media(
                album_id INTEGER NOT NULL,
                path TEXT NOT NULL,
                PRIMARY KEY(album_id,path),
                FOREIGN KEY(album_id) REFERENCES albums(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS media_flags(
                path TEXT PRIMARY KEY,
                favorite INTEGER DEFAULT 0,
                hidden INTEGER DEFAULT 0,
                note TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS operations(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                source_root TEXT,
                output_root TEXT,
                payload TEXT NOT NULL,
                summary TEXT DEFAULT ''
            );
            """)
            self._ensure_column(c, "people", "hero_path", "TEXT")
            for name, decl in [("captured_at", "TEXT"), ("width", "INTEGER DEFAULT 0"), ("height", "INTEGER DEFAULT 0"), ("duration", "REAL DEFAULT 0")]:
                self._ensure_column(c, "media_cache", name, decl)
            self._ensure_column(c, "face_state", "third_person", "INTEGER")
            self._ensure_column(c, "face_state", "third_score", "REAL")
            self._ensure_column(c, "operations", "summary", "TEXT DEFAULT ''")
            c.executescript("""
            CREATE INDEX IF NOT EXISTS idx_media_captured ON media_cache(captured_at DESC);
            CREATE INDEX IF NOT EXISTS idx_media_phash ON media_cache(phash,size);
            CREATE INDEX IF NOT EXISTS idx_face_state_state ON face_state(state,best_score DESC);
            CREATE INDEX IF NOT EXISTS idx_face_state_person ON face_state(best_person,state);
            CREATE INDEX IF NOT EXISTS idx_refs_person ON refs(person_id,is_negative);
            CREATE INDEX IF NOT EXISTS idx_album_media_path ON album_media(path);
            """)

    def _migrate_legacy(self):
        try:
            old = sqlite3.connect(LEGACY_DB)
            people = old.execute("SELECT id,name FROM people ORDER BY id").fetchall()
            refs = old.execute("SELECT person_id,image_path,embedding,dim FROM refs ORDER BY id").fetchall()
            with self.connect() as c:
                idmap = {}
                for old_id, name in people:
                    c.execute("INSERT OR IGNORE INTO people(name) VALUES(?)", (name,))
                    row = c.execute("SELECT id FROM people WHERE name=? COLLATE NOCASE", (name,)).fetchone()
                    if row: idmap[old_id] = row[0]
                for opid, path, blob, dim in refs:
                    if opid in idmap:
                        c.execute("INSERT INTO refs(person_id,image_path,embedding,dim,quality,is_negative,source) VALUES(?,?,?,?,.55,0,?)", (idmap[opid], path, blob, dim, "legacy-v0.4"))
            old.close()
        except Exception:
            pass

    def add_person(self, name: str) -> int:
        with self.connect() as c:
            return c.execute("INSERT INTO people(name) VALUES(?)", (name.strip(),)).lastrowid

    def delete_person(self, pid: int):
        with self.connect() as c:
            c.execute("DELETE FROM people WHERE id=?", (pid,))

    def rename_person(self, pid: int, name: str):
        with self.connect() as c:
            c.execute("UPDATE people SET name=? WHERE id=?", (name.strip(), pid))

    def set_person_hero(self, pid: int, path: str | None):
        with self.connect() as c:
            c.execute("UPDATE people SET hero_path=? WHERE id=?", (path, pid))

    def people(self):
        with self.connect() as c:
            return c.execute("""
                SELECT p.id,p.name,
                       SUM(CASE WHEN r.is_negative=0 THEN 1 ELSE 0 END) AS pos,
                       SUM(CASE WHEN r.is_negative=1 THEN 1 ELSE 0 END) AS neg,
                       p.manual_threshold,p.hero_path
                FROM people p LEFT JOIN refs r ON r.person_id=p.id
                GROUP BY p.id,p.name ORDER BY p.name COLLATE NOCASE
            """).fetchall()

    def add_ref(self, pid: int, image_path: str | None, emb: np.ndarray, quality: float, negative: bool = False, source: str = "manual"):
        blob, dim = vec_blob(emb)
        with self.connect() as c:
            c.execute("INSERT INTO refs(person_id,image_path,embedding,dim,quality,is_negative,source) VALUES(?,?,?,?,?,?,?)", (pid, image_path, blob, dim, float(quality), int(bool(negative)), source))

    def refs(self, pid: int, negative: bool | None = None):
        q = "SELECT id,image_path,embedding,dim,quality,is_negative,source FROM refs WHERE person_id=?"
        args: list = [pid]
        if negative is not None:
            q += " AND is_negative=?"; args.append(int(negative))
        q += " ORDER BY id"
        with self.connect() as c:
            rows = c.execute(q, args).fetchall()
        return [(r[0], r[1], blob_vec(r[2], r[3]), float(r[4] or .5), bool(r[5]), r[6]) for r in rows]

    def delete_ref(self, rid: int):
        with self.connect() as c:
            c.execute("DELETE FROM refs WHERE id=?", (rid,))

    def cache_get(self, path: Path) -> MediaAnalysis | None:
        try:
            st = path.stat()
        except OSError:
            return None
        with self.connect() as c:
            row = c.execute("SELECT * FROM media_cache WHERE path=?", (str(path),)).fetchone()
        if not row or int(row["size"]) != st.st_size or int(row["mtime_ns"]) != st.st_mtime_ns:
            return None
        return MediaAnalysis(str(path), int(row["size"]), int(row["mtime_ns"]), int(row["phash"] or 0), unpack_faces(row["faces"] or b""), row["kind"] or "image", row["captured_at"], int(row["width"] or 0), int(row["height"] or 0), float(row["duration"] or 0))

    def cache_put(self, m: MediaAnalysis):
        with self.connect() as c:
            c.execute("""
                INSERT INTO media_cache(path,size,mtime_ns,phash,kind,faces,face_count,captured_at,width,height,duration,analyzed_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(path) DO UPDATE SET
                    size=excluded.size,mtime_ns=excluded.mtime_ns,phash=excluded.phash,kind=excluded.kind,
                    faces=excluded.faces,face_count=excluded.face_count,captured_at=excluded.captured_at,
                    width=excluded.width,height=excluded.height,duration=excluded.duration,analyzed_at=CURRENT_TIMESTAMP
            """, (m.path, m.size, m.mtime_ns, str(int(m.phash)), m.kind, pack_faces(m.faces), len(m.faces), m.captured_at, m.width, m.height, m.duration))

    def find_exact_duplicate(self, phash: int, size: int, exclude_path: str) -> MediaAnalysis | None:
        if not phash:
            return None
        with self.connect() as c:
            r = c.execute("SELECT * FROM media_cache WHERE phash=? AND size=? AND path<>? LIMIT 1", (str(int(phash)), int(size), exclude_path)).fetchone()
        if not r:
            return None
        return MediaAnalysis(r["path"], int(r["size"]), int(r["mtime_ns"]), int(r["phash"]), unpack_faces(r["faces"] or b""), r["kind"], r["captured_at"], int(r["width"] or 0), int(r["height"] or 0), float(r["duration"] or 0))

    def set_face_state(self, path: str, idx: int, state: str, bpid, bscore, spid=None, sscore=None, tpid=None, tscore=None):
        with self.connect() as c:
            c.execute("""
                INSERT INTO face_state(path,face_idx,state,best_person,best_score,second_person,second_score,third_person,third_score)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(path,face_idx) DO UPDATE SET
                    state=excluded.state,best_person=excluded.best_person,best_score=excluded.best_score,
                    second_person=excluded.second_person,second_score=excluded.second_score,
                    third_person=excluded.third_person,third_score=excluded.third_score
            """, (path, idx, state, bpid, bscore, spid, sscore, tpid, tscore))

    def mark_state(self, path: str, idx: int, state: str):
        with self.connect() as c:
            c.execute("UPDATE face_state SET state=? WHERE path=? AND face_idx=?", (state, path, idx))

    def review_items(self, limit: int = 1000):
        with self.connect() as c:
            return c.execute("""
                SELECT fs.path,fs.face_idx,fs.state,fs.best_person,fs.best_score,p.name,mc.faces,
                       fs.second_person,fs.second_score,fs.third_person,fs.third_score
                FROM face_state fs
                LEFT JOIN people p ON p.id=fs.best_person
                LEFT JOIN media_cache mc ON mc.path=fs.path
                WHERE fs.state IN ('review','unknown')
                ORDER BY CASE fs.state WHEN 'review' THEN 0 ELSE 1 END,fs.best_score DESC LIMIT ?
            """, (limit,)).fetchall()

    def unknown_faces(self, limit: int = 20000):
        out = []
        for r in self.review_items(limit):
            fs = unpack_faces(r[6] or b"")
            idx = int(r[1])
            if 0 <= idx < len(fs):
                out.append((r[0], idx, fs[idx], r[2], r[3], r[4], r[5]))
        return out

    def top_candidates(self, path: str, idx: int):
        with self.connect() as c:
            r = c.execute("SELECT * FROM face_state WHERE path=? AND face_idx=?", (path, idx)).fetchone()
            if not r:
                return []
            pairs = [(r["best_person"], r["best_score"]), (r["second_person"], r["second_score"]), (r["third_person"], r["third_score"])]
            out = []
            for pid, score in pairs:
                if pid is None or score is None: continue
                p = c.execute("SELECT name FROM people WHERE id=?", (pid,)).fetchone()
                if p: out.append((int(pid), p[0], float(score)))
            return out

    def media_count(self, search: str = "", person_id=None, album_id=None, month: str | None = None) -> int:
        q, args = self._media_query(search, person_id, album_id, month, count=True)
        with self.connect() as c:
            return int(c.execute(q, args).fetchone()[0])

    def _media_query(self, search="", person_id=None, album_id=None, month=None, count=False):
        select = "COUNT(DISTINCT mc.path)" if count else "DISTINCT mc.path,mc.kind,mc.face_count,mc.captured_at,mc.width,mc.height,mc.duration,COALESCE(mf.favorite,0)"
        q = f"SELECT {select} FROM media_cache mc LEFT JOIN media_flags mf ON mf.path=mc.path"
        joins = []; where = ["COALESCE(mf.hidden,0)=0"]; args = []
        if person_id is not None:
            joins.append("JOIN face_state fsp ON fsp.path=mc.path AND fsp.best_person=? AND fsp.state='matched'"); args.append(int(person_id))
        if album_id is not None:
            joins.append("JOIN album_media am ON am.path=mc.path AND am.album_id=?"); args.append(int(album_id))
        if search:
            where.append("mc.path LIKE ?"); args.append(f"%{search}%")
        if month:
            where.append("substr(mc.captured_at,1,7)=?"); args.append(month)
        if joins: q += " " + " ".join(joins)
        if where: q += " WHERE " + " AND ".join(where)
        if not count: q += " ORDER BY COALESCE(mc.captured_at,'') DESC, mc.path COLLATE NOCASE"
        return q, args

    def media_rows(self, limit=500, offset=0, search="", person_id=None, album_id=None, month=None):
        q, args = self._media_query(search, person_id, album_id, month, count=False)
        q += " LIMIT ? OFFSET ?"; args += [int(limit), int(offset)]
        with self.connect() as c:
            return [tuple(r) for r in c.execute(q, args).fetchall()]

    def media_detail(self, path: str):
        with self.connect() as c:
            mc = c.execute("SELECT * FROM media_cache WHERE path=?", (path,)).fetchone()
            if not mc: return None
            faces = c.execute("SELECT * FROM face_state WHERE path=? ORDER BY face_idx", (path,)).fetchall()
            albums = c.execute("SELECT a.id,a.name FROM albums a JOIN album_media am ON am.album_id=a.id WHERE am.path=? ORDER BY a.name", (path,)).fetchall()
            return dict(mc), [dict(x) for x in faces], [tuple(x) for x in albums]

    def timeline_months(self):
        with self.connect() as c:
            return c.execute("SELECT substr(captured_at,1,7) AS month,COUNT(*) FROM media_cache WHERE captured_at IS NOT NULL GROUP BY month ORDER BY month DESC").fetchall()

    def albums(self):
        with self.connect() as c:
            return c.execute("SELECT a.id,a.name,COUNT(am.path) FROM albums a LEFT JOIN album_media am ON am.album_id=a.id GROUP BY a.id ORDER BY a.name").fetchall()

    def add_album(self, name: str) -> int:
        with self.connect() as c:
            return c.execute("INSERT INTO albums(name) VALUES(?)", (name.strip(),)).lastrowid

    def delete_album(self, aid: int):
        with self.connect() as c:
            c.execute("DELETE FROM albums WHERE id=?", (aid,))

    def add_to_album(self, aid: int, paths: list[str]):
        with self.connect() as c:
            c.executemany("INSERT OR IGNORE INTO album_media(album_id,path) VALUES(?,?)", [(aid, p) for p in paths])

    def remove_from_album(self, aid: int, paths: list[str]):
        with self.connect() as c:
            c.executemany("DELETE FROM album_media WHERE album_id=? AND path=?", [(aid, p) for p in paths])

    def set_favorite(self, paths: list[str], value: bool):
        with self.connect() as c:
            c.executemany("INSERT INTO media_flags(path,favorite) VALUES(?,?) ON CONFLICT(path) DO UPDATE SET favorite=excluded.favorite", [(p, int(value)) for p in paths])

    def person_profile(self, pid: int):
        with self.connect() as c:
            p = c.execute("SELECT id,name,manual_threshold,hero_path FROM people WHERE id=?", (pid,)).fetchone()
            if not p: return None
            pos = c.execute("SELECT COUNT(*),COALESCE(AVG(quality),0) FROM refs WHERE person_id=? AND is_negative=0", (pid,)).fetchone()
            neg = c.execute("SELECT COUNT(*) FROM refs WHERE person_id=? AND is_negative=1", (pid,)).fetchone()[0]
            photos = c.execute("SELECT COUNT(DISTINCT path) FROM face_state WHERE best_person=? AND state='matched'", (pid,)).fetchone()[0]
            recent = c.execute("SELECT fs.path,mc.captured_at,MAX(fs.best_score) score FROM face_state fs JOIN media_cache mc ON mc.path=fs.path WHERE fs.best_person=? AND fs.state='matched' GROUP BY fs.path ORDER BY COALESCE(mc.captured_at,'') DESC LIMIT 12", (pid,)).fetchall()
            hero = p[3]
            if not hero:
                h = c.execute("SELECT fs.path,MAX(fs.best_score) s FROM face_state fs WHERE fs.best_person=? AND fs.state='matched' GROUP BY fs.path ORDER BY s DESC LIMIT 1", (pid,)).fetchone()
                hero = h[0] if h else None
            return {"id":p[0],"name":p[1],"manual_threshold":p[2],"hero":hero,"positive":int(pos[0]),"avg_quality":float(pos[1]),"negative":int(neg),"photos":int(photos),"recent":[tuple(x) for x in recent]}

    def confusion(self, pid: int, limit: int = 8):
        with self.connect() as c:
            rows = c.execute("SELECT fs.second_person,p.name,COUNT(*) n,AVG(fs.best_score-fs.second_score) margin FROM face_state fs JOIN people p ON p.id=fs.second_person WHERE fs.best_person=? AND fs.second_person IS NOT NULL GROUP BY fs.second_person ORDER BY n DESC,margin ASC LIMIT ?", (pid, limit)).fetchall()
            return [tuple(x) for x in rows]

    def dashboard(self):
        with self.connect() as c:
            media = c.execute("SELECT COUNT(*),COALESCE(SUM(face_count),0) FROM media_cache").fetchone(); people = c.execute("SELECT COUNT(*) FROM people").fetchone()[0]
            states = dict(c.execute("SELECT state,COUNT(*) FROM face_state GROUP BY state").fetchall()); albums = c.execute("SELECT COUNT(*) FROM albums").fetchone()[0]
            recent = c.execute("SELECT created_at,summary FROM operations ORDER BY id DESC LIMIT 6").fetchall()
            return {"media":int(media[0]),"faces":int(media[1]),"people":int(people),"albums":int(albums),"matched":int(states.get('matched',0)),"review":int(states.get('review',0)),"unknown":int(states.get('unknown',0)),"recent":[tuple(r) for r in recent]}

    def operation_history(self, limit=40):
        with self.connect() as c:
            return c.execute("SELECT id,created_at,source_root,output_root,summary FROM operations ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    def clear_cache(self):
        with self.connect() as c:
            c.execute("DELETE FROM media_cache"); c.execute("DELETE FROM face_state"); c.execute("DELETE FROM album_media")
        try:
            for p in THUMBS.glob("*.jpg"): p.unlink(missing_ok=True)
        except Exception:
            pass

    def cleanup_cache(self):
        with self.connect() as c:
            paths = [r[0] for r in c.execute("SELECT path FROM media_cache").fetchall()]
        dead = [p for p in paths if not Path(p).exists()]
        with self.connect() as c:
            for p in dead:
                c.execute("DELETE FROM media_cache WHERE path=?", (p,)); c.execute("DELETE FROM face_state WHERE path=?", (p,)); c.execute("DELETE FROM album_media WHERE path=?", (p,))
        return len(paths), len(dead)

    def cache_stats(self):
        with self.connect() as c:
            n, faces = c.execute("SELECT COUNT(*),COALESCE(SUM(face_count),0) FROM media_cache").fetchone(); reviews = c.execute("SELECT COUNT(*) FROM face_state WHERE state IN ('review','unknown')").fetchone()[0]
        return int(n), int(faces), int(reviews), DB.stat().st_size if DB.exists() else 0

    def duplicate_groups(self, limit=100):
        with self.connect() as c:
            rows = c.execute("SELECT phash,size,COUNT(*) n,GROUP_CONCAT(path,'\n') paths FROM media_cache WHERE phash<>'0' GROUP BY phash,size HAVING COUNT(*)>1 ORDER BY n DESC LIMIT ?", (limit,)).fetchall()
        return [(r[0], int(r[1]), int(r[2]), str(r[3]).split('\n')) for r in rows]

    def duplicate_summary(self, limit=50000):
        with self.connect() as c:
            rows = c.execute("SELECT path,phash,size FROM media_cache WHERE phash<>'0' LIMIT ?", (limit,)).fetchall()
        exact: dict[tuple[int,int], list[str]] = {}; buckets: dict[int,list[tuple[int,str]]] = {}
        for path, ph, size in rows:
            try: h = int(ph)
            except Exception: continue
            exact.setdefault((h, int(size)), []).append(path); buckets.setdefault((h >> 52) & 0xFFF, []).append((h, path))
        exact_groups = sum(1 for v in exact.values() if len(v) > 1); near_pairs = 0; seen = set()
        for key, vals in buckets.items():
            candidates = list(vals)
            for bit in range(12): candidates.extend(buckets.get(key ^ (1 << bit), []))
            for h, p in vals:
                for h2, p2 in candidates:
                    if p >= p2 or (p, p2) in seen: continue
                    seen.add((p, p2))
                    if (h ^ h2).bit_count() <= 5: near_pairs += 1
        return len(rows), exact_groups, near_pairs

    def log_operation(self, source: Path, output: Path, created: list[str], summary: str = ""):
        with self.connect() as c:
            c.execute("INSERT INTO operations(source_root,output_root,payload,summary) VALUES(?,?,?,?)", (str(source), str(output), json.dumps(created, ensure_ascii=False), summary))

    def undo_last(self):
        with self.connect() as c:
            row = c.execute("SELECT id,payload FROM operations ORDER BY id DESC LIMIT 1").fetchone()
            if not row: return 0, 0
            c.execute("DELETE FROM operations WHERE id=?", (row[0],))
        removed = missed = 0
        for s in reversed(json.loads(row[1])):
            p = Path(s)
            try:
                if p.exists() and p.is_file(): p.unlink(); removed += 1
                else: missed += 1
            except Exception: missed += 1
        return removed, missed
