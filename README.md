# 3D Assembly Maker

A working foundation for the software you described: upload an assembly document, step/product photographs, and optional STEP/STP/GLB models, then build a reviewable animated 3D assembly.

## Current build

- Excel / XLS / CSV / JSON assembly data
- Multiple step/product photographs with manual step mapping
- STEP / STP CAD conversion to GLB using CadQuery/OpenCascade
- GLB / GLTF loading
- Assembly procedure and BOM extraction
- Step-to-part mapping using names, part numbers and part-type rules
- PLACE, SCAN, LABEL, MOVE and INSPECT action detection
- Three.js orbit/zoom, step animation and exploded view
- Current-step reference images and assembly.json export
- Human review before accepting the generated assembly

## Accuracy

STEP/STP is treated as the exact geometry source when available. Photo-only 3D is currently an approximate silhouette extrusion and does not reconstruct hidden surfaces or engineering dimensions from one photograph.

## Run locally

```bash
python -m pip install -r requirements.txt
python start.py
```

Open http://127.0.0.1:8000

Backend endpoints:
- GET /api/health
- POST /api/convert-step
- POST /api/photo-to-3d

## Target pipeline

DOCUMENT -> STEP/BOM EXTRACTION -> PART DATABASE -> CAD/PHOTO MODEL DATABASE -> STEP/PART/REFERENCE MAPPING -> ASSEMBLY DEFINITION -> ANIMATION PLAN -> THREE.JS REVIEWER

The next engineering layer is stronger multi-view image reconstruction and automatic placement/orientation inference without replacing the assembly-definition layer.