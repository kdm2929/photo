from storage import *
from recognition import *
def output_copy(src,dst_dir,use_hardlink=False):
    dst_dir.mkdir(parents=True,exist_ok=True);cand=dst_dir/src.name;i=2
    while cand.exists():
        try:
            if cand.stat().st_size==src.stat().st_size:return cand,False
        except Exception:pass
        cand=dst_dir/f'{src.stem}_{i}{src.suffix}';i+=1
    if use_hardlink:
        try:os.link(src,cand);return cand,True
        except Exception:pass
    shutil.copy2(src,cand);return cand,True


class AnalyzerPool:
    def __init__(self,use_gpu):self.local=threading.local();self.use_gpu=use_gpu
    def engine(self):
        if not hasattr(self.local,'engine'):self.local.engine=FaceEngine(self.use_gpu)
        return self.local.engine
    def analyze(self,p):return self.engine().analyze_media(p)


class ScanEngine:
    def __init__(self,cfg,progress=None,status=None,stop_check=None):self.cfg=cfg;self.progress=progress or (lambda *a:None);self.status=status or (lambda *a:None);self.stop_check=stop_check or (lambda:False)
    def run(self):
        started=time.time();self.records=[];db=LibraryDB();models=build_models(db,self.cfg.threshold)
        if not models:raise RuntimeError('등록된 인물과 레퍼런스가 없습니다.')
        self.status('파일 목록을 빠르게 읽는 중…');files=list(iter_media(self.cfg.source,self.cfg.recursive,self.cfg.output));total=len(files);stats={'전체':total,'분류':0,'확인필요':0,'미확인':0,'얼굴없음':0,'오류':0,'캐시':0,'새분석':0,'중복재사용':0};created=[];self.cfg.output.mkdir(parents=True,exist_ok=True);pool=AnalyzerPool(self.cfg.use_gpu);max_workers=max(1,min(self.cfg.workers,2 if self.cfg.use_gpu else 8));executor=ThreadPoolExecutor(max_workers=max_workers);pending={};cursor=completed=0;cache_batch=[];mem_dups={}
        def submit_more():
            nonlocal cursor
            while not self.stop_check() and cursor<total and len(pending)<max_workers*3:
                p=files[cursor];cursor+=1;cached=db.cache_get(p)
                if cached is not None:cache_batch.append((p,cached,'cache'))
                else:pending[executor.submit(pool.analyze,p)]=p
        submit_more()
        while (pending or cache_batch or cursor<total) and not self.stop_check():
            while cache_batch:
                p,a,src=cache_batch.pop();completed=self._consume(db,models,p,a,src,stats,created,completed,total,started);submit_more()
                if len(cache_batch)<max_workers:break
            if pending:
                done,_=wait(list(pending),timeout=.08,return_when=FIRST_COMPLETED)
                for fut in done:
                    p=pending.pop(fut)
                    try:
                        a=fut.result();key=(a.phash,a.size);dup=mem_dups.get(key) or db.find_exact_duplicate(a.phash,a.size,a.path)
                        if dup is not None and dup.faces and a.phash:a=MediaAnalysis(a.path,a.size,a.mtime_ns,a.phash,dup.faces,a.kind);stats['중복재사용']+=1
                        db.cache_put(a);mem_dups[key]=a;stats['새분석']+=1;completed=self._consume(db,models,p,a,'new',stats,created,completed,total,started)
                    except Exception:stats['오류']+=1;completed+=1
                submit_more()
        executor.shutdown(wait=False,cancel_futures=True)
        if created:db.log_operation(self.cfg.source,self.cfg.output,created)
        try:
            with open(self.cfg.output/'분류결과.csv','w',newline='',encoding='utf-8-sig') as rf:
                wr=csv.writer(rf);wr.writerow(['원본파일','상태','인물','최고유사도','캐시','종류']);wr.writerows(self.records)
        except Exception:pass
        stats['중단']=self.stop_check();stats['초']=time.time()-started;return stats
    def _consume(self,db,models,p,a,source,stats,created,completed,total,started):
        if source=='cache':stats['캐시']+=1
        matched={};review=[]
        for idx,f in enumerate(a.faces):
            bpid,bscore,spid,sscore=match_face(f.embedding,models)
            if bpid is None:db.set_face_state(a.path,idx,'unknown',None,None);continue
            th=models[bpid].threshold;margin=bscore-(sscore if sscore is not None else -1)
            if bscore>=th and margin>=.025:state='matched';matched[bpid]=max(matched.get(bpid,-1),bscore)
            elif bscore>=max(.31,th-.10):state='review';review.append((bscore,bpid))
            else:state='unknown'
            db.set_face_state(a.path,idx,state,bpid,bscore,spid,sscore)
        if not a.faces:
            d,made=output_copy(p,self.cfg.output/'_얼굴없음',self.cfg.hardlink);created.extend([str(d)] if made else []);stats['얼굴없음']+=1;self.records.append([str(p),'얼굴없음','','',source,a.kind])
        elif matched:
            arr=sorted(matched.items(),key=lambda x:x[1],reverse=True);arr=arr if self.cfg.multi else arr[:1];labels=[]
            for pid,score in arr:
                d,made=output_copy(p,self.cfg.output/safe_name(models[pid].name),self.cfg.hardlink);created.extend([str(d)] if made else []);labels.append(f"{models[pid].name}:{score:.3f}")
            stats['분류']+=1;self.records.append([str(p),'분류',' / '.join(labels),f"{max(matched.values()):.3f}",source,a.kind])
        elif review:
            d,made=output_copy(p,self.cfg.output/'_확인필요',self.cfg.hardlink);created.extend([str(d)] if made else []);stats['확인필요']+=1;best=max(review);self.records.append([str(p),'확인필요',models[best[1]].name,f"{best[0]:.3f}",source,a.kind])
        else:
            d,made=output_copy(p,self.cfg.output/'_미확인',self.cfg.hardlink);created.extend([str(d)] if made else []);stats['미확인']+=1;self.records.append([str(p),'미확인','','',source,a.kind])
        completed+=1;elapsed=max(.01,time.time()-started);speed=completed/elapsed;self.progress(completed,total,p.name,speed,stats['캐시'],stats['새분석']);return completed
