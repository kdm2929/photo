from storage import *
class FaceEngine:
    def __init__(self,use_gpu=False):
        det_path,rec_path=model_paths(); self.det=cv2.FaceDetectorYN.create(str(det_path),'',(320,320),.78,.3,5000); self.rec=cv2.FaceRecognizerSF.create(str(rec_path),''); self.gpu_session=None
        if use_gpu and ort is not None and sys.platform.startswith('win'):
            try:
                if 'DmlExecutionProvider' in ort.get_available_providers():self.gpu_session=ort.InferenceSession(str(rec_path),providers=['DmlExecutionProvider','CPUExecutionProvider'])
            except Exception:self.gpu_session=None
    @staticmethod
    def gpu_available():
        if ort is None:return False,'DirectML 모듈 없음'
        try:
            ps=ort.get_available_providers();return 'DmlExecutionProvider' in ps,', '.join(ps)
        except Exception as e:return False,str(e)
    def detect(self,img):
        h,w=img.shape[:2];self.det.setInputSize((w,h));_,faces=self.det.detect(img);return [] if faces is None else [f for f in faces if float(f[-1])>=.78]
    def feature(self,img,face):
        aligned=self.rec.alignCrop(img,face)
        if self.gpu_session is not None:
            blob=cv2.dnn.blobFromImage(aligned,1.0,(112,112),(0,0,0),swapRB=True,crop=False).astype(np.float32);inp=self.gpu_session.get_inputs()[0].name;out=self.gpu_session.run(None,{inp:blob})[0].reshape(-1).astype(np.float32)
        else:out=self.rec.feature(aligned).astype(np.float32).reshape(-1)
        n=float(np.linalg.norm(out));out=out/n if n>1e-8 else out;return out,aligned
    def quality(self,img,face,aligned):
        h,w=img.shape[:2];x,y,bw,bh=[float(v) for v in face[:4]];area=min(1.0,(bw*bh)/(max(1.0,w*h)*.08));gray=cv2.cvtColor(aligned,cv2.COLOR_BGR2GRAY);sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var());sharp_score=max(0.0,min(1.0,(sharp-25.0)/180.0));conf=max(0.0,min(1.0,float(face[-1])));pts=np.asarray(face[4:14],dtype=np.float32).reshape(5,2);eyes=(pts[0]+pts[1])/2;mouth=(pts[3]+pts[4])/2;mid=(eyes+mouth)/2;nose=pts[2];frontal=max(0.0,1.0-float(np.linalg.norm(nose-mid))/(max(8.0,bw)*.22));return float(max(.05,min(1.0,.32*area+.28*sharp_score+.18*conf+.22*frontal)))
    def analyze_frame(self,img):
        out=[]
        for f in self.detect(img):
            try:v,a=self.feature(img,f);out.append(FaceData(v,self.quality(img,f,a),tuple(float(x) for x in f[:4])))
            except Exception:pass
        return out
    def reference(self,path):
        candidates=[];face_count=0
        for img in media_frames(path,3):
            fs=self.detect(img);face_count+=len(fs)
            for f in fs:
                try:v,a=self.feature(img,f);q=self.quality(img,f,a);candidates.append((float(f[2]*f[3])*q,v,q))
                except Exception:pass
        if not candidates:raise ValueError('얼굴을 찾지 못함')
        _,v,q=max(candidates,key=lambda x:x[0]);return v,q,face_count
    def analyze_media(self,path):
        st=path.stat();frames=media_frames(path,10);all_faces=[];hashes=[]
        for img in frames:hashes.append(phash64(img));all_faces.extend(self.analyze_frame(img))
        if path.suffix.lower() in VIDEO_EXTS and len(all_faces)>1:
            kept=[]
            for f in sorted(all_faces,key=lambda x:x.quality,reverse=True):
                if not kept or max(float(np.dot(f.embedding,k.embedding)) for k in kept)<.94:kept.append(f)
                if len(kept)>=24:break
            all_faces=kept
        return MediaAnalysis(str(path),st.st_size,st.st_mtime_ns,hashes[len(hashes)//2] if hashes else 0,all_faces,'video' if path.suffix.lower() in VIDEO_EXTS else 'image')


def build_models(db,default_threshold=.43):
    models={}
    for pid,name,_,_,manual in db.people():
        pos=db.refs(pid,False)
        if not pos:continue
        mat=np.vstack([r[2] for r in pos]).astype(np.float32);q=np.asarray([r[3] for r in pos],dtype=np.float32);weights=np.clip(q,.15,1.0);cen=(mat*weights[:,None]).sum(0)/weights.sum();cen/=max(1e-8,float(np.linalg.norm(cen)));negs=db.refs(pid,True);neg=np.vstack([r[2] for r in negs]).astype(np.float32) if negs else None
        if manual is not None:th=float(manual)
        elif len(mat)>=3:
            sims=mat@mat.T;np.fill_diagonal(sims,-2);best=sims.max(1);th=max(.37,min(.54,float(np.quantile(best,.15))-.035));th=max(th,default_threshold-.035)
            if neg is not None and len(neg):th=max(th,min(.58,float((mat@neg.T).max())+.025))
        else:th=default_threshold
        models[pid]=PersonModel(pid,name,mat,q,neg,cen,float(th))
    return models


def match_face(v,models):
    scores=[]
    for pid,m in models.items():
        sims=m.positives@v;score=max(float((sims*(.88+.12*m.pos_quality)).max()),float(np.dot(m.centroid,v))*.995)
        if m.negatives is not None and len(m.negatives):
            neg=float((m.negatives@v).max())
            if neg>=score-.01:score-=.09
            elif neg>.42:score-=max(0.0,(neg-.42)*.35)
        scores.append((score,pid,m))
    scores.sort(reverse=True,key=lambda x:x[0])
    if not scores:return None,None,None,None
    b=scores[0];s=scores[1] if len(scores)>1 else (None,None,None);return b[1],b[0],s[1],s[0]


class Clusterer:
    def __init__(self,threshold=.475):self.threshold=threshold;self.rng=np.random.default_rng(2929)
    def cluster(self,items):
        if not items:return []
        dim=items[0][2].embedding.size;planes=self.rng.normal(size=(10,dim)).astype(np.float32);clusters=[];bucket_clusters={}
        def key(v):
            bits=(planes@v)>0;k=0
            for i,b in enumerate(bits):
                if b:k|=1<<i
            return k
        for item in sorted(items,key=lambda x:x[2].quality,reverse=True):
            v=item[2].embedding;k=key(v);cand=[]
            for kk in [k]+[k^(1<<i) for i in range(10)]:cand.extend(bucket_clusters.get(kk,[]))
            best=None
            for ci in set(cand):
                s=float(np.dot(v,clusters[ci]['centroid']))
                if best is None or s>best[0]:best=(s,ci)
            if best and best[0]>=self.threshold:
                c=clusters[best[1]];c['items'].append(item);vs=np.vstack([x[2].embedding for x in c['items'][-40:]]);cen=vs.mean(0);cen/=max(1e-8,float(np.linalg.norm(cen)));c['centroid']=cen;bucket_clusters.setdefault(k,[]).append(best[1])
            else:
                ci=len(clusters);clusters.append({'centroid':v.copy(),'items':[item]});bucket_clusters.setdefault(k,[]).append(ci)
        out=[c for c in clusters if len(c['items'])>=2];out.sort(key=lambda c:len(c['items']),reverse=True);return out
