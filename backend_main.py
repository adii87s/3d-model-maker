from __future__ import annotations

import re, shutil, uuid, json
from datetime import datetime, timezone
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
PROJECTS=ROOT/"projects"
GENERATED.mkdir(parents=True,exist_ok=True)
PROJECTS.mkdir(parents=True,exist_ok=True)

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


def project_id_ok(project_id:str)->bool:
    return bool(re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", project_id or ""))

def project_dir(project_id:str)->Path:
    if not project_id_ok(project_id):
        raise HTTPException(400,"Invalid project id")
    return PROJECTS/project_id

def demo_project():
    return {
        "id":"demo-hydep-frame",
        "name":"Hydep Frame Sub-Assembly Demo",
        "sourceMode":"photo",
        "demo":True,
        "createdAt":"2026-09-19T00:00:00+00:00",
        "updatedAt":"2026-09-19T00:00:00+00:00",
        "steps":[
            {"number":1,"name":"Pick & load Frame Base","instruction":"Pick and load the Frame Base on the fixture with the help of the manipulator.","action":"PLACE","partName":"Frame Base","quantity":1,"photoUrl":None},
            {"number":2,"name":"Scan QR Code","instruction":"Scan the QR Code with the hand-held scanner.","action":"SCAN","partName":"Frame Base","quantity":1,"photoUrl":None},
            {"number":3,"name":"Place Frame Gasket Cathode","instruction":"Pick the Frame Gasket Cathode and place it on the frame base.","action":"PLACE","partName":"Frame Gasket Cathode","quantity":1,"photoUrl":None},
            {"number":4,"name":"Place Manifold Gasket","instruction":"Pick the manifold gasket and place it on the frame base.","action":"PLACE","partName":"Manifold Gasket","quantity":2,"photoUrl":None},
            {"number":5,"name":"Place Frame Cover","instruction":"Pick the Frame Cover and place it on the frame base.","action":"PLACE","partName":"Frame Cover","quantity":1,"photoUrl":None},
            {"number":6,"name":"Move barcode to placard","instruction":"Remove the Bar Code from the Frame Cover and stick it on the placard.","action":"LABEL","partName":"Frame Cover","quantity":1,"photoUrl":None},
            {"number":7,"name":"Movement to next station","instruction":"Move the assembly to the next station.","action":"MOVE","partName":"Assembly","quantity":1,"photoUrl":None}
        ],
        "model":{"fileName":None,"modelUrl":None}
    }

def write_project(data:dict)->None:
    folder=project_dir(data["id"])
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"photos").mkdir(parents=True,exist_ok=True)
    (folder/"project.json").write_text(json.dumps(data,indent=2),encoding="utf-8")

def read_project(project_id:str)->dict:
    folder=project_dir(project_id)
    path=folder/"project.json"
    if not path.exists():
        raise HTTPException(404,"Project not found")
    return json.loads(path.read_text(encoding="utf-8"))

def ensure_demo_project():
    data=demo_project()
    path=PROJECTS/data["id"]/ "project.json"
    if not path.exists():
        write_project(data)

ensure_demo_project()

@app.get("/api/projects")
def list_projects():
    items=[]
    for path in PROJECTS.iterdir():
        if not path.is_dir(): continue
        file=path/"project.json"
        if not file.exists(): continue
        try:
            data=json.loads(file.read_text(encoding="utf-8"))
            items.append({
                "id":data.get("id",path.name),
                "name":data.get("name","Untitled"),
                "demo":bool(data.get("demo")),
                "updatedAt":data.get("updatedAt"),
                "stepCount":len(data.get("steps",[])),
                "sourceMode":data.get("sourceMode","photo")
            })
        except Exception:
            continue
    items.sort(key=lambda x:(not x.get("demo",False),x.get("updatedAt") or ""),reverse=False)
    return {"projects":items}

@app.get("/api/projects/{project_id}")
def get_project(project_id:str):
    return read_project(project_id)

@app.post("/api/projects")
async def save_project(
    project:str=Form(...),
    model_file:UploadFile|None=File(default=None),
    photos:list[UploadFile]=File(default=[])
):
    try:
        data=json.loads(project)
    except Exception as exc:
        raise HTTPException(400,"Invalid project JSON") from exc
    if not isinstance(data,dict):
        raise HTTPException(400,"Project must be an object")
    original_id=data.get("id")
    if original_id=="demo-hydep-frame" or not project_id_ok(original_id or ""):
        data["id"]=uuid.uuid4().hex[:16]
        data["demo"]=False
        data["createdAt"]=datetime.now(timezone.utc).isoformat()
    else:
        existing=read_project(original_id)
        data["createdAt"]=existing.get("createdAt",datetime.now(timezone.utc).isoformat())
        data["demo"]=False
    data["updatedAt"]=datetime.now(timezone.utc).isoformat()
    data.setdefault("name","Untitled")
    data.setdefault("steps",[])
    folder=project_dir(data["id"])
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"photos").mkdir(parents=True,exist_ok=True)

    incoming={}
    for upload in photos:
        incoming[Path(upload.filename or "").name]=upload

    for step in data["steps"]:
        key=step.get("photoUploadName")
        if key and key in incoming:
            upload=incoming[key]
            saved=await save_upload(upload,folder/"photos")
            step["photoUrl"]=f"/projects/{data['id']}/photos/{saved.name}"
            step.pop("photoUploadName",None)
        elif step.get("photoUrl"):
            step.pop("photoUploadName",None)
        else:
            step["photoUrl"]=None

    if model_file is not None:
        saved_model=await save_upload(model_file,folder)
        data["model"]={"fileName":model_file.filename,"modelUrl":f"/projects/{data['id']}/{saved_model.name}"}
    else:
        data.setdefault("model",{"fileName":None,"modelUrl":None})

    write_project(data)
    return data

@app.delete("/api/projects/{project_id}")
def delete_project(project_id:str):
    if project_id=="demo-hydep-frame":
        raise HTTPException(400,"The demo project cannot be deleted")
    folder=project_dir(project_id)
    if not folder.exists():
        raise HTTPException(404,"Project not found")
    shutil.rmtree(folder,ignore_errors=True)
    return {"ok":True}

app.mount("/projects",StaticFiles(directory=PROJECTS),name="projects")
app.mount("/",StaticFiles(directory=ROOT,html=True),name="web")
