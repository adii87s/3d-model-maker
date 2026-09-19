# 3D Assembly Maker

A browser-based foundation for turning assembly documents and product reference photos into a step-by-step animated 3D assembly review.

## Current MVP
- Three.js interactive 3D viewer
- Hydep 7-step demonstration
- Assembly step timeline
- PLACE / SCAN / LABEL / MOVE actions
- Multiple-part quantity handling
- Previous / Assemble Next / Auto Play / Exploded View / Reset
- Upload JSON or Excel/CSV assembly sheets
- Upload multiple reference photos
- Export the generated assembly definition as JSON
- Procedural approximate 3D geometry from part names

## Run locally
python -m http.server 5173
Open http://localhost:5173

## Architecture target
DOCUMENT -> STEP EXTRACTION -> PART DATABASE -> MODEL DATABASE -> ASSEMBLY RELATIONSHIPS -> ANIMATION PLAN -> THREE.JS RENDERER

## Planned engineering layer
FastAPI backend, AI document/vision analyzer, PostgreSQL storage, GLB/GLTF/STEP ingestion, AI-assisted part recognition and placement/orientation, and human review before final generation.

Exact engineering geometry should come from validated CAD/GLB or a validated reconstruction pipeline; the browser MVP labels its generated geometry as approximate.
