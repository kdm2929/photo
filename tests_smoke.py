from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import storage
from recognition import Clusterer, RecognitionIndex, build_models, match_face


def unit(v,dim=128):
    x=np.zeros(dim,np.float32);x[v]=1.0;return x


def run():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); storage.DB=root/'test.sqlite3'; storage.THUMBS=root/'thumbs'; storage.THUMBS.mkdir()
        db=storage.LibraryDB(); a=db.add_person('Alpha');b=db.add_person('Beta')
        for noise,q in [(0,.9),(1,.8),(2,.7)]:
            va=unit(0);va[10+noise]=.05;va/=np.linalg.norm(va);db.add_ref(a,f'a{noise}.jpg',va,q)
            vb=unit(1);vb[20+noise]=.05;vb/=np.linalg.norm(vb);db.add_ref(b,f'b{noise}.jpg',vb,q)
        db.add_ref(a,'negative.jpg',unit(1),.8,True,'test-negative')
        models=build_models(db);assert set(models)=={a,b};idx=RecognitionIndex(models)
        top=idx.top(unit(0),2);assert top[0][0]==a and top[0][1]>.8
        got=match_face(unit(1),models,idx);assert got[0]==b
        media=root/'sample.jpg';media.write_bytes(b'x');face=storage.FaceData(unit(0),.88,(10,10,80,80))
        ma=storage.MediaAnalysis(str(media),1,media.stat().st_mtime_ns,12345,[face],'image','2026-09-28 12:00:00',100,100,0)
        db.cache_put(ma);assert db.cache_get(media) is not None;db.set_face_state(str(media),0,'review',a,.91,b,.72,None,None)
        c=db.top_candidates(str(media),0);assert c[0][0]==a and c[1][0]==b;assert db.media_count()==1 and len(db.media_rows())==1
        album=db.add_album('Concert');db.add_to_album(album,[str(media)]);assert db.media_count(album_id=album)==1
        p=db.person_profile(a);assert p['positive']==3;dash=db.dashboard();assert dash['media']==1 and dash['people']==2
        db.set_favorite([str(media)],True);assert db.media_rows()[0][-1]==1
        items=[]
        for i in range(4):
            f=storage.FaceData(unit(5),.9-i*.05,(0,0,10,10));items.append((f'p{i}',0,f,'unknown',None,None,None))
        clusters=Clusterer(.47).cluster(items);assert clusters and len(clusters[0]['items'])==4
        packed=storage.pack_faces([face]);un=storage.unpack_faces(packed);assert len(un)==1 and np.dot(un[0].embedding,face.embedding)>.99
    print('SMOKE_OK')

if __name__=='__main__':run()
