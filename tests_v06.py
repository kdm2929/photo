from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from recognition import Clusterer, RecognitionIndex, match_face
from storage import FaceData, PersonModel, pack_faces, phash64, safe_name, unpack_faces


def unit(v):
    a=np.asarray(v,dtype=np.float32).reshape(-1)
    n=float(np.linalg.norm(a))
    return a/n if n else a


def test_pack_roundtrip():
    rng=np.random.default_rng(7)
    faces=[]
    for i in range(3):
        v=unit(rng.normal(size=128))
        faces.append(FaceData(v,0.4+i*0.2,(10+i,20+i,80,90)))
    out=unpack_faces(pack_faces(faces))
    assert len(out)==3
    for a,b in zip(faces,out):
        assert np.allclose(a.embedding,b.embedding,atol=1e-6)
        assert abs(a.quality-b.quality)<1e-6
        assert np.allclose(a.bbox,b.bbox)


def test_phash_stability():
    img=np.zeros((240,320,3),np.uint8)
    cv2.rectangle(img,(40,40),(180,200),(255,255,255),-1)
    a=phash64(img)
    b=phash64(cv2.resize(img,(160,120),interpolation=cv2.INTER_AREA))
    assert isinstance(a,int) and isinstance(b,int)
    assert (a^b).bit_count() <= 10


def make_model(pid,name,base,negative=None,threshold=.42):
    rng=np.random.default_rng(pid)
    refs=[]
    for _ in range(5):
        refs.append(unit(base + rng.normal(0,.025,size=128)))
    mat=np.vstack(refs).astype(np.float32)
    cen=unit(mat.mean(0))
    neg=np.vstack([unit(negative)]).astype(np.float32) if negative is not None else None
    return PersonModel(pid,name,mat,np.ones(len(mat),np.float32)*.9,neg,cen,threshold)


def test_recognition_ranking():
    e1=np.zeros(128,np.float32);e1[0]=1
    e2=np.zeros(128,np.float32);e2[1]=1
    models={1:make_model(1,'A',e1),2:make_model(2,'B',e2)}
    idx=RecognitionIndex(models)
    q=unit(e1 + np.random.default_rng(11).normal(0,.02,128))
    top=idx.top(q,2)
    assert top[0][0]==1
    assert top[0][1] > top[1][1]
    bpid,bscore,spid,sscore,*_=match_face(q,models,idx)
    assert bpid==1 and spid==2 and bscore>sscore


def test_negative_penalty():
    e1=np.zeros(128,np.float32);e1[0]=1
    e2=np.zeros(128,np.float32);e2[1]=1
    target=unit(.86*e1+.51*e2)
    a=make_model(1,'A',e1,negative=target)
    b=make_model(2,'B',e2)
    idx=RecognitionIndex({1:a,2:b})
    raw=float((a.positives@target).max())
    score=float(idx.scores(target)[0])
    assert score < raw


def test_clusterer_groups_similar_faces():
    rng=np.random.default_rng(99)
    base1=unit(rng.normal(size=128));base2=unit(rng.normal(size=128))
    items=[]
    for g,base in enumerate((base1,base2)):
        for i in range(7):
            v=unit(base+rng.normal(0,.035,size=128))
            items.append((f'p{g}_{i}.jpg',0,FaceData(v,.8,(0,0,100,100))))
    clusters=Clusterer(.45).cluster(items)
    sizes=sorted([len(c['items']) for c in clusters],reverse=True)
    assert len(sizes)>=2
    assert sizes[0]>=5 and sizes[1]>=5


def test_safe_name():
    assert safe_name('A:B/C*D?')=='A_B_C_D_'
    assert safe_name('...')=='이름없음'


def run_all():
    tests=[
        test_pack_roundtrip,
        test_phash_stability,
        test_recognition_ranking,
        test_negative_penalty,
        test_clusterer_groups_similar_faces,
        test_safe_name,
    ]
    for fn in tests:
        fn()
        print('PASS',fn.__name__)
    print(f'ALL PASS ({len(tests)} tests)')
    return 0


if __name__=='__main__':
    raise SystemExit(run_all())
