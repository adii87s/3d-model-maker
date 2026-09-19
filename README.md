# 3D Assembly Maker

A manual assembly-definition and 3D animation application.

## Current workflow

1. Enter a project name.
2. Optionally upload one STEP/STP/GLB/GLTF model.
3. Create each assembly step manually:
   - Step name
   - Step explanation
   - Action
   - Part/component name
   - Quantity
   - Photo for that step
4. Build the 3D assembly.
5. Review the result step-by-step.
6. Save the assembly to the Project Library so it can be loaded again later.

Excel upload is intentionally not part of the current UI.

## Project Library
Saved assemblies are stored by the local FastAPI backend. The library includes a seeded Hydep Frame Sub-Assembly demo and supports loading and deleting user-created projects. Build 3D also saves the current assembly automatically.

## 3D behavior

- STEP/STP: backend converts CAD to GLB with CadQuery/OpenCascade.
- GLB/GLTF: loaded directly in Three.js.
- Photo-only PLACE steps: backend creates an approximate 3D silhouette from the product photo.
- PLACE steps animate physical geometry.
- SCAN, LABEL, INSPECT, MOVE and NOTE steps remain procedural instructions and do not create fake physical parts.

## Run

```bash
python -m pip install -r requirements.txt
python start.py
```

Open:

`http://127.0.0.1:8000`

## Controls

- Assemble next
- Previous
- Auto play
- Exploded view
- Reset
- Export assembly.json
- Project Library: Save current, Load, Delete, Refresh

## Accuracy

A STEP/STP model is treated as the geometry source. Photo-only reconstruction is approximate and should not be treated as engineering-accurate CAD.

The next development layer is automatic recognition of parts from step photos plus stronger multi-view 3D reconstruction and placement/orientation inference.
