from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from recognition import FaceEngine, RecognitionIndex, build_models, match_face
from storage import LibraryDB, MediaAnalysis, ScanConfig, iter_media, safe_name


def output_copy(src: Path, dst_dir: Path, use_hardlink: bool = False):
    dst_dir.mkdir(parents=True, exist_ok=True)
    cand = dst_dir / src.name; i = 2
    while cand.exists():
        try:
            if cand.stat().st_size == src.stat().st_size:
                return cand, False
        except Exception:
            pass
        cand = dst_dir / f"{src.stem}_{i}{src.suffix}"; i += 1
    if use_hardlink:
        try:
            os.link(src, cand); return cand, True
        except Exception:
            pass
    shutil.copy2(src, cand); return cand, True


def drive_media_hint(path: Path) -> str:
    if not sys.platform.startswith("win"):
        return "unknown"
    try:
        cmd = ["powershell", "-NoProfile", "-Command", "(Get-PhysicalDisk | Select-Object -ExpandProperty MediaType) -join ','"]
        out = subprocess.check_output(cmd, timeout=2, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).decode(errors="ignore").lower()
        if "ssd" in out: return "ssd"
        if "hdd" in out: return "hdd"
    except Exception:
        pass
    return "unknown"


def auto_worker_count(source: Path, gpu: bool) -> tuple[int, str]:
    if gpu:
        return 2, "GPU 모드: 2개 파이프라인"
    cpu = max(2, os.cpu_count() or 4); media = drive_media_hint(source)
    if media == "hdd": return min(3, max(2, cpu // 4)), "HDD 감지: I/O 과부하 방지"
    if media == "ssd": return min(8, max(3, cpu // 2)), "SSD 감지: 병렬 분석 강화"
    return min(6, max(2, cpu // 2)), "자동 균형 설정"


class AnalyzerPool:
    def __init__(self, use_gpu: bool):
        self.local = threading.local(); self.use_gpu = use_gpu
    def engine(self):
        if not hasattr(self.local, "engine"):
            self.local.engine = FaceEngine(self.use_gpu)
        return self.local.engine
    def analyze(self, p: Path):
        return self.engine().analyze_media(p)


class ScanEngine:
    def __init__(self, cfg: ScanConfig, progress=None, status=None, stop_check=None):
        self.cfg = cfg; self.progress = progress or (lambda *a: None); self.status = status or (lambda *a: None); self.stop_check = stop_check or (lambda: False); self.records = []

    def run(self):
        started = time.time(); db = LibraryDB(); models = build_models(db, self.cfg.threshold)
        if not models: raise RuntimeError("등록된 인물과 레퍼런스가 없습니다.")
        index = RecognitionIndex(models)
        if self.cfg.auto_workers:
            workers, why = auto_worker_count(self.cfg.source, self.cfg.use_gpu); self.status(f"파일 목록을 읽는 중… · {why} · 워커 {workers}")
        else:
            workers = self.cfg.workers; self.status(f"파일 목록을 읽는 중… · 수동 워커 {workers}")
        files = list(iter_media(self.cfg.source, self.cfg.recursive, self.cfg.output)); total = len(files)
        stats = {"전체":total,"분류":0,"확인필요":0,"미확인":0,"얼굴없음":0,"오류":0,"캐시":0,"새분석":0,"중복재사용":0}
        created = []; self.cfg.output.mkdir(parents=True, exist_ok=True); pool = AnalyzerPool(self.cfg.use_gpu)
        max_workers = max(1, min(int(workers), 2 if self.cfg.use_gpu else 8)); executor = ThreadPoolExecutor(max_workers=max_workers)
        pending = {}; cached_queue = []; cursor = completed = 0; mem_dups = {}

        def submit_more():
            nonlocal cursor
            while not self.stop_check() and cursor < total and len(pending) < max_workers * 4:
                p = files[cursor]; cursor += 1; cached = db.cache_get(p)
                if cached is not None: cached_queue.append((p, cached))
                else: pending[executor.submit(pool.analyze, p)] = p

        submit_more()
        while (pending or cached_queue or cursor < total) and not self.stop_check():
            burst = min(len(cached_queue), max_workers * 8)
            for _ in range(burst):
                p, a = cached_queue.pop(); completed = self._consume(db, models, index, p, a, "cache", stats, created, completed, total, started)
            submit_more()
            if pending:
                done, _ = wait(list(pending), timeout=.06, return_when=FIRST_COMPLETED)
                for fut in done:
                    p = pending.pop(fut)
                    try:
                        a = fut.result(); key = (a.phash, a.size); dup = mem_dups.get(key) or db.find_exact_duplicate(a.phash, a.size, a.path)
                        if dup is not None and dup.faces and a.phash:
                            a = MediaAnalysis(a.path, a.size, a.mtime_ns, a.phash, dup.faces, a.kind, a.captured_at, a.width, a.height, a.duration); stats["중복재사용"] += 1
                        db.cache_put(a); mem_dups[key] = a; stats["새분석"] += 1
                        completed = self._consume(db, models, index, p, a, "new", stats, created, completed, total, started)
                    except Exception as e:
                        stats["오류"] += 1; completed += 1; self.records.append([str(p), "오류", "", "", "new", p.suffix.lower(), str(e)])
                submit_more()
        executor.shutdown(wait=False, cancel_futures=True)
        summary = f"{stats['전체']:,}개 · 자동 {stats['분류']:,} · 검토 {stats['확인필요']:,} · 미확인 {stats['미확인']:,}"
        if created: db.log_operation(self.cfg.source, self.cfg.output, created, summary)
        try:
            import csv
            with open(self.cfg.output / "분류결과.csv", "w", newline="", encoding="utf-8-sig") as rf:
                wr = csv.writer(rf); wr.writerow(["원본파일","상태","인물","최고유사도","캐시","종류","비고"]); wr.writerows(self.records)
        except Exception: pass
        stats["중단"] = self.stop_check(); stats["초"] = time.time() - started; stats["워커"] = max_workers
        return stats

    def _consume(self, db, models, index, p, a, source, stats, created, completed, total, started):
        if source == "cache": stats["캐시"] += 1
        matched = {}; review = []
        for face_idx, f in enumerate(a.faces):
            bpid, bscore, spid, sscore, tpid, tscore = match_face(f.embedding, models, index)
            if bpid is None:
                db.set_face_state(a.path, face_idx, "unknown", None, None); continue
            th = models[bpid].threshold; margin = bscore - (sscore if sscore is not None else -1)
            if bscore >= th and margin >= .025:
                state = "matched"; matched[bpid] = max(matched.get(bpid, -1), bscore)
            elif bscore >= max(.31, th - .10):
                state = "review"; review.append((bscore, bpid))
            else: state = "unknown"
            db.set_face_state(a.path, face_idx, state, bpid, bscore, spid, sscore, tpid, tscore)

        if not a.faces:
            d, made = output_copy(p, self.cfg.output / "_얼굴없음", self.cfg.hardlink); created.extend([str(d)] if made else []); stats["얼굴없음"] += 1; self.records.append([str(p),"얼굴없음","","",source,a.kind,""])
        elif matched:
            arr = sorted(matched.items(), key=lambda x: x[1], reverse=True); arr = arr if self.cfg.multi else arr[:1]; labels = []
            for pid, score in arr:
                d, made = output_copy(p, self.cfg.output / safe_name(models[pid].name), self.cfg.hardlink); created.extend([str(d)] if made else []); labels.append(f"{models[pid].name}:{score:.3f}")
            stats["분류"] += 1; self.records.append([str(p),"분류"," / ".join(labels),f"{max(matched.values()):.3f}",source,a.kind,""])
        elif review:
            d, made = output_copy(p, self.cfg.output / "_확인필요", self.cfg.hardlink); created.extend([str(d)] if made else []); stats["확인필요"] += 1; best = max(review); self.records.append([str(p),"확인필요",models[best[1]].name,f"{best[0]:.3f}",source,a.kind,""])
        else:
            d, made = output_copy(p, self.cfg.output / "_미확인", self.cfg.hardlink); created.extend([str(d)] if made else []); stats["미확인"] += 1; self.records.append([str(p),"미확인","","",source,a.kind,""])
        completed += 1; elapsed = max(.01, time.time() - started); speed = completed / elapsed; self.progress(completed, total, p.name, speed, stats["캐시"], stats["새분석"]); return completed
