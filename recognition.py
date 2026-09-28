from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from storage import (
    FaceData, LibraryDB, MediaAnalysis, PersonModel, VIDEO_EXTS,
    media_basic_info, media_frames, model_paths, ort, phash64,
)


class FaceEngine:
    def __init__(self, use_gpu: bool = False):
        det_path, rec_path = model_paths()
        self.det = cv2.FaceDetectorYN.create(str(det_path), "", (320, 320), .78, .3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(rec_path), "")
        self.gpu_session = None
        if use_gpu and ort is not None and sys.platform.startswith("win"):
            try:
                if "DmlExecutionProvider" in ort.get_available_providers():
                    self.gpu_session = ort.InferenceSession(str(rec_path), providers=["DmlExecutionProvider", "CPUExecutionProvider"])
            except Exception:
                self.gpu_session = None

    @staticmethod
    def gpu_available():
        if ort is None:
            return False, "DirectML 모듈 없음"
        try:
            ps = ort.get_available_providers()
            return "DmlExecutionProvider" in ps, ", ".join(ps)
        except Exception as e:
            return False, str(e)

    def detect(self, img: np.ndarray):
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        return [] if faces is None else [f for f in faces if float(f[-1]) >= .78]

    def feature(self, img: np.ndarray, face: np.ndarray):
        aligned = self.rec.alignCrop(img, face)
        if self.gpu_session is not None:
            blob = cv2.dnn.blobFromImage(aligned, 1.0, (112, 112), (0, 0, 0), swapRB=True, crop=False).astype(np.float32)
            inp = self.gpu_session.get_inputs()[0].name
            out = self.gpu_session.run(None, {inp: blob})[0].reshape(-1).astype(np.float32)
        else:
            out = self.rec.feature(aligned).astype(np.float32).reshape(-1)
        n = float(np.linalg.norm(out))
        return (out / n if n > 1e-8 else out), aligned

    def quality(self, img: np.ndarray, face: np.ndarray, aligned: np.ndarray):
        h, w = img.shape[:2]
        _, _, bw, bh = [float(v) for v in face[:4]]
        area = min(1.0, (bw * bh) / (max(1.0, w * h) * .08))
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharp_score = max(0.0, min(1.0, (sharp - 25.0) / 180.0))
        conf = max(0.0, min(1.0, float(face[-1])))
        pts = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
        eyes = (pts[0] + pts[1]) / 2
        mouth = (pts[3] + pts[4]) / 2
        mid = (eyes + mouth) / 2
        frontal = max(0.0, 1.0 - float(np.linalg.norm(pts[2] - mid)) / (max(8.0, bw) * .22))
        return float(max(.05, min(1.0, .32 * area + .28 * sharp_score + .18 * conf + .22 * frontal)))

    def analyze_frame(self, img: np.ndarray):
        out = []
        for f in self.detect(img):
            try:
                v, aligned = self.feature(img, f)
                out.append(FaceData(v, self.quality(img, f, aligned), tuple(float(x) for x in f[:4])))
            except Exception:
                pass
        return out

    def reference(self, path: Path):
        candidates = []; face_count = 0
        for img in media_frames(path, 3):
            fs = self.detect(img); face_count += len(fs)
            for f in fs:
                try:
                    v, aligned = self.feature(img, f); q = self.quality(img, f, aligned)
                    candidates.append((float(f[2] * f[3]) * q, v, q))
                except Exception:
                    pass
        if not candidates:
            raise ValueError("얼굴을 찾지 못함")
        _, v, q = max(candidates, key=lambda x: x[0])
        return v, q, face_count

    def analyze_media(self, path: Path):
        st = path.stat(); frames = media_frames(path, 10); all_faces = []; hashes = []
        for img in frames:
            hashes.append(phash64(img)); all_faces.extend(self.analyze_frame(img))
        if path.suffix.lower() in VIDEO_EXTS and len(all_faces) > 1:
            kept = []
            for f in sorted(all_faces, key=lambda x: x.quality, reverse=True):
                if not kept or max(float(np.dot(f.embedding, k.embedding)) for k in kept) < .94:
                    kept.append(f)
                if len(kept) >= 24:
                    break
            all_faces = kept
        captured, width, height, duration = media_basic_info(path)
        if not width and frames:
            height, width = frames[0].shape[:2]
        return MediaAnalysis(str(path), st.st_size, st.st_mtime_ns,
                             hashes[len(hashes)//2] if hashes else 0, all_faces,
                             "video" if path.suffix.lower() in VIDEO_EXTS else "image",
                             captured, width, height, duration)


def build_models(db: LibraryDB, default_threshold: float = .43):
    models = {}
    for row in db.people():
        pid, name, _, _, manual = row[:5]
        pos = db.refs(pid, False)
        if not pos:
            continue
        mat = np.vstack([r[2] for r in pos]).astype(np.float32)
        q = np.asarray([r[3] for r in pos], dtype=np.float32)
        weights = np.clip(q, .15, 1.0)
        cen = (mat * weights[:, None]).sum(0) / max(1e-8, float(weights.sum()))
        cen /= max(1e-8, float(np.linalg.norm(cen)))
        negs = db.refs(pid, True)
        neg = np.vstack([r[2] for r in negs]).astype(np.float32) if negs else None
        if manual is not None:
            th = float(manual)
        elif len(mat) >= 3:
            sims = mat @ mat.T; np.fill_diagonal(sims, -2)
            best = sims.max(1)
            th = max(.37, min(.54, float(np.quantile(best, .15)) - .035))
            th = max(th, default_threshold - .035)
            if neg is not None and len(neg):
                th = max(th, min(.58, float((mat @ neg.T).max()) + .025))
        else:
            th = default_threshold
        models[int(pid)] = PersonModel(int(pid), str(name), mat, q, neg, cen, float(th))
    return models


class RecognitionIndex:
    """Exact vectorized reference search for large person libraries."""
    def __init__(self, models: dict[int, PersonModel]):
        self.models = models
        self.pids = np.asarray(sorted(models), dtype=np.int32)
        self.pid_to_pos = {int(pid): i for i, pid in enumerate(self.pids.tolist())}
        refs = []; ref_person = []; ref_weight = []; cents = []
        for pid in self.pids:
            m = models[int(pid)]
            refs.append(m.positives)
            ref_person.extend([self.pid_to_pos[int(pid)]] * len(m.positives))
            ref_weight.extend((.88 + .12 * m.pos_quality).tolist())
            cents.append(m.centroid)
        self.refs = np.vstack(refs).astype(np.float32) if refs else np.empty((0, 128), np.float32)
        self.ref_person = np.asarray(ref_person, dtype=np.int32)
        self.ref_weight = np.asarray(ref_weight, dtype=np.float32)
        self.centroids = np.vstack(cents).astype(np.float32) if cents else np.empty((0, 128), np.float32)

    def scores(self, v: np.ndarray):
        n = len(self.pids)
        if not n:
            return np.empty(0, np.float32)
        per = np.full(n, -2.0, dtype=np.float32)
        if len(self.refs):
            rs = (self.refs @ v) * self.ref_weight
            np.maximum.at(per, self.ref_person, rs)
        cs = (self.centroids @ v) * .995
        per = np.maximum(per, cs)
        for i, pid in enumerate(self.pids):
            m = self.models[int(pid)]
            if m.negatives is None or not len(m.negatives):
                continue
            neg = float((m.negatives @ v).max())
            if neg >= float(per[i]) - .01:
                per[i] -= .09
            elif neg > .42:
                per[i] -= max(0.0, (neg - .42) * .35)
        return per

    def top(self, v: np.ndarray, k: int = 3):
        scores = self.scores(v)
        if not len(scores):
            return []
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(self.pids[i]), float(scores[i])) for i in idx]


def match_face(v: np.ndarray, models: dict[int, PersonModel], index: RecognitionIndex | None = None):
    idx = index or RecognitionIndex(models)
    top = idx.top(v, 3)
    while len(top) < 3:
        top.append((None, None))
    return top[0][0], top[0][1], top[1][0], top[1][1], top[2][0], top[2][1]


class Clusterer:
    """Scalable unknown-face clustering with multi-table LSH and safe fallback.

    Multiple short hash tables drastically reduce the chance that two views of
    the same person never become candidates. When the active cluster count is
    still small, a vectorized centroid fallback gives deterministic grouping.
    """
    def __init__(self, threshold: float = .475):
        self.threshold = float(threshold)
        self.rng = np.random.default_rng(2929)

    @staticmethod
    def _centroid(items):
        vs = np.vstack([x[2].embedding for x in items[-80:]]).astype(np.float32)
        qs = np.asarray([max(.1, x[2].quality) for x in items[-80:]], np.float32)
        cen = (vs * qs[:, None]).sum(0) / max(1e-8, float(qs.sum()))
        cen /= max(1e-8, float(np.linalg.norm(cen)))
        return cen.astype(np.float32)

    def cluster(self, items):
        if not items:
            return []
        dim = int(items[0][2].embedding.size)
        tables = 4
        bits_per_table = 9
        planes = self.rng.normal(size=(tables, bits_per_table, dim)).astype(np.float32)
        clusters = []
        buckets = [dict() for _ in range(tables)]

        def keys(v):
            out=[]
            for t in range(tables):
                bits=(planes[t] @ v) > 0; key=0
                for i,b in enumerate(bits):
                    if b: key |= 1 << i
                out.append(key)
            return out

        def candidate_ids(v, ks):
            cand=set()
            for t,k in enumerate(ks):
                cand.update(buckets[t].get(k, ()))
                for i in range(bits_per_table):
                    cand.update(buckets[t].get(k ^ (1 << i), ()))
            # When the number of current identities is modest, checking all
            # centroids is still cheap and prevents early LSH fragmentation.
            if len(clusters) <= 512:
                cand.update(range(len(clusters)))
            elif len(cand) < 4 and clusters:
                # Coarse vectorized fallback: compare against all centroids,
                # then only keep the most promising few for exact handling.
                cents=np.vstack([c['centroid'] for c in clusters]).astype(np.float32)
                sims=cents @ v
                take=min(8,len(sims))
                idx=np.argpartition(-sims,take-1)[:take] if take else []
                cand.update(int(i) for i in idx)
            return cand

        for item in sorted(items, key=lambda x: x[2].quality, reverse=True):
            v=np.asarray(item[2].embedding,dtype=np.float32)
            ks=keys(v)
            cand=candidate_ids(v,ks)
            best=None
            for ci in cand:
                s=float(np.dot(v,clusters[ci]['centroid']))
                if best is None or s>best[0]: best=(s,ci)
            if best and best[0] >= self.threshold:
                ci=best[1]; c=clusters[ci]; c['items'].append(item); c['centroid']=self._centroid(c['items'])
            else:
                ci=len(clusters); clusters.append({'centroid':v.copy(),'items':[item]})
            for t,k in enumerate(ks):
                arr=buckets[t].setdefault(k,[])
                if not arr or arr[-1] != ci:
                    arr.append(ci)

        out=[c for c in clusters if len(c['items']) >= 2]
        out.sort(key=lambda c: len(c['items']), reverse=True)
        return out
