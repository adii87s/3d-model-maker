from __future__ import annotations

import re, shutil, uuid
from pathlib import Path
import cv2
import numpy as np
import trimesh
import cadquery as cq
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from shapely.geometry import Polygon
from shapely.ops import triangulate

ROOT=Path(__file__).resolve().parent
GENERATED=ROOT/"generated"
GENERATED.mkdir(parents=True,exist_ok=True)

app=FastAPI(title="3D Assembly Maker API",version="0.2.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
app.mount("/generated",StaticFiles(directory=GENERATED),name="generated")
MAX_BYTES=200*1024*1024

def safe_name(name:str)->str:
    return re.sub(r"[^a-zA-Z0-9_.-]+","_",name or "file")

async def save_upload(upload:UploadFile,folder:Path)->Path:
    folder.mkdir(parents=True,exist_ok=True)
    out=folder/f"{uuid.uuid4().hex}_{safe_name(upload.filename or 'upload')}"
    total=0
    with out.open("wb") as f:
        while True:
            chunk=await upload.read(1024*1024)
            if not chunk: break
            total+=len(chunk)
            if total>MAX_BYTES:
                out.unlink(missing_ok=True)
                raise HTTPException(413,"File is larger than 200 MB")
            f.write(chunk)
    return out

def leaf_objects(assy:cq.Assembly):
    out=[]
    for name,item in assy.objects.items():
        obj=getattr(item,"obj",None)
        if obj is not None: out.append((str(name),obj))
    return out

@app.get("/api/health")
def health():
    return {"ok":True,"cadquery":cq.__version__}

@app.post("/api/convert-step")
async def convert_step(file:UploadFile=File(...)):
    if Path(file.filename or "").suffix.lower() not in {".step",".stp"}:
        raise HTTPException(400,"Only STEP/STP files are supported")
    job=GENERATED/uuid.uuid4().hex
    source=await save_upload(file,job)
    glb=job/"model.glb"
    try:
        assy=cq.Assembly.importStep(str(source),unit="MM")
        leaves=leaf_objects(assy)
        if not leaves:
            shape=cq.importers.importStep(str(source),unit="MM")
            assy=cq.Assembly(name=Path(file.filename or "assembly").stem)
            assy.add(shape,name=Path(file.filename or "assembly").stem)
            leaves=leaf_objects(assy)
        assy.export(str(glb))
    except Exception as exc:
        shutil.rmtree(job,ignore_errors=True)
        raise HTTPException(422,f"STEP import failed: {exc}") from exc
    return {"kind":"step","fileName":file.filename,
            "modelUrl":f"/generated/{job.name}/model.glb",
            "assemblyParts":[{"name":name,"index":i} for i,(name,_) in enumerate(leaves)],
            "exactGeometry":True,"units":"mm"}

def image_mask(image:np.ndarray)->np.ndarray:
    if image.shape[2]==4 and np.count_nonzero(image[:,:,3]>20)>0.05*image.shape[0]*image.shape[1]:
        return np.where(image[:,:,3]>20,255,0).astype(np.uint8)
    bgr=image[:,:,:3]; h,w=bgr.shape[:2]
    q=max(4,h//12); qx=max(4,w//12)
    corners=np.concatenate([bgr[:q,:qx].reshape(-1,3),bgr[:q,-qx:].reshape(-1,3),
                            bgr[-q:,:qx].reshape(-1,3),bgr[-q:,-qx:].reshape(-1,3)])
    bg=np.median(corners.astype(np.float32),axis=0)
    diff=np.linalg.norm(bgr.astype(np.float32)-bg,axis=2)
    mask=np.where(diff>max(25.0,float(np.percentile(diff,72))),255,0).astype(np.uint8)
    k=np.ones((7,7),np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask,cv2.MORPH_OPEN,k),cv2.MORPH_CLOSE,k)

def photo_mesh(image:np.ndarray)->trimesh.Trimesh:
    mask=image_mask(image)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError("Could not isolate the product from the image")
    c=max(contours,key=cv2.contourArea); h,w=mask.shape
    if cv2.contourArea(c)<0.01*w*h: raise ValueError("Detected product is too small")
    pts=c[:,0,:].astype(float); pts[:,0]-=w/2; pts[:,1]=-(pts[:,1]-h/2)
    poly=Polygon(pts).buffer(0)
    if poly.is_empty or poly.area<=1: raise ValueError("Photo silhouette is unusable")
    longest=max(poly.bounds[2]-poly.bounds[0],poly.bounds[3]-poly.bounds[1])
    scale=300/max(longest,1)
    poly=Polygon([(x*scale,y*scale) for x,y in poly.exterior.coords])
    height=max(12,0.08*max(poly.bounds[2]-poly.bounds[0],poly.bounds[3]-poly.bounds[1]))
    verts=[]; faces=[]
    def add(x,y,z):
        verts.append([float(x),float(y),float(z)]); return len(verts)-1
    for tri in (t for t in triangulate(poly) if poly.contains(t.representative_point())):
        co=list(tri.exterior.coords)[:3]
        top=[add(x,y,height/2) for x,y in co]; bot=[add(x,y,-height/2) for x,y in co]
        faces += [top,bot[::-1]]
    co=list(poly.exterior.coords)[:-1]
    for a,b in zip(co,co[1:]+co[:1]):
        i1,i2,i3,i4=add(a[0],a[1],height/2),add(b[0],b[1],height/2),add(b[0],b[1],-height/2),add(a[0],a[1],-height/2)
        faces += [[i1,i2,i3],[i1,i3,i4]]
    return trimesh.Trimesh(vertices=np.asarray(verts),faces=np.asarray(faces),process=True)

@app.post("/api/photo-to-3d")
async def photo_to_3d(file:UploadFile=File(...),part_name:str=Form("Photo reconstructed part"),step_number:int=Form(0)):
    if Path(file.filename or "").suffix.lower() not in {".png",".jpg",".jpeg",".webp",".bmp"}:
        raise HTTPException(400,"Unsupported image format")
    job=GENERATED/uuid.uuid4().hex; source=await save_upload(file,job)
    image=cv2.imdecode(np.fromfile(str(source),dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    if image is None:
        shutil.rmtree(job,ignore_errors=True); raise HTTPException(422,"Could not decode image")
    if image.ndim==2: image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGRA)
    elif image.shape[2]==3: image=cv2.cvtColor(image,cv2.COLOR_BGR2BGRA)
    try: photo_mesh(image).export(job/"model.glb")
    except Exception as exc:
        shutil.rmtree(job,ignore_errors=True); raise HTTPException(422,f"Image-to-3D approximation failed: {exc}") from exc
    return {"kind":"photo-approximation","fileName":file.filename,"partName":part_name,
            "stepNumber":step_number,"modelUrl":f"/generated/{job.name}/model.glb",
            "exactGeometry":False,"method":"silhouette-extrusion"}

app.mount("/",StaticFiles(directory=ROOT,html=True),name="web")
