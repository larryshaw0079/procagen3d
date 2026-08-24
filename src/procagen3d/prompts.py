"""Prompts for CLI agents; these are application contracts, not a Codex skill."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .granularity import DEFAULT_GRANULARITY, validate_granularity
from .plan_schema import plan_schema_text
from .reconstruction import DEFAULT_RECONSTRUCTION_MODE, validate_reconstruction_mode


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _mode_contract(reconstruction_mode: str) -> str:
    reconstruction_mode = validate_reconstruction_mode(reconstruction_mode)
    if reconstruction_mode == "procedural":
        return """Reconstruction mode: `procedural`.
- Geometry must be synthesized from primitives, curves, modifiers, and algorithmically generated procedural meshes appropriate to the selected granularity.
- Do not copy, embed, trace, or import the reference mesh, and do not emit large vertex/index dumps.
- The reference GLB and its reports are measurement evidence only.
- `src/program.py` is the standalone replay source of truth in this mode."""
    return """Reconstruction mode: `glb-ref`.
- At build time the verified GLB is host-imported, frame-0/rest-posed, and normalized into the Blender collection `PROCAGEN3D_REFERENCE`; its longest dimension is 2 units and its ground is Z=0.
- `build()` may inspect those Blender objects and may duplicate mesh data, segment, remesh, retopologize, simplify, transform, or procedurally augment them. This is an intentionally glb-ref reconstruction, not an independently procedural claim.
- Do not reset the scene or delete, rename, mutate, hide, move, or relink host-owned reference objects or `PROCAGEN3D_REFERENCE`; cleanup may remove only objects created by your own `build()`.
- `build()` must create candidate geometry as new Blender objects. Put it outside `PROCAGEN3D_REFERENCE`, detach it from reference parents, and do not leave modifiers that depend on reference objects. The host deletes every original reference object before saving and exporting.
- Do not read or import the GLB yourself, embed large geometry dumps in Python, or export the original host-owned objects directly.
- This mode is not standalone-program reconstruction: `src/program.py` plus the provenance-locked `inputs/reference.glb` and the documented host preload contract are the replay source of truth."""


def _granularity_contract(granularity: str) -> str:
    granularity = validate_granularity(granularity)
    if granularity == "coarse":
        return """Granularity: `coarse`.
- Produce a fast semantic blockout with major masses, pose, proportions, and silhouette.
- Simple primitives are appropriate, but keep parts named and editable.
- Surface-distance acceptance is disabled at this level."""
    if granularity == "medium":
        return """Granularity: `medium` (compatibility default).
- Produce compact editable geometry with major secondary forms and identity details.
- Primitives, curves, modifiers, and low-control-count custom meshes may be combined.
- Optimize the six canonical silhouettes and spatial color; exact surface-distance acceptance is disabled."""
    shared = """- A higher triangle count by itself is not progress: do not merely increase primitive segments, bevel segments, or subdivision levels.
- Replace box/sphere/cylinder-only fitting with semantic, surface-conforming custom meshes: contour cages, variable cross-section lofts, planar armor patches, inset/recess geometry, and controlled curves where appropriate.
- Use every canonical view and the measured cross-sections to solve depth, concavity, pose, asymmetry, plane breaks, and transitions between parts.
- Keep topology generated algorithmically and compactly; do not embed or transcribe a large reference vertex/index dump."""
    if granularity == "fine":
        return f"""Granularity: `fine`.
- Match the target surface closely while retaining semantic, editable parts. Generic primitives should be limited to genuinely primitive hidden cores and joints.
{shared}
- The host evaluates deterministic bidirectional 3D surface distance after export. Treat failed mean or p95 surface gates and their worst residual coordinates as primary repair targets."""
    return f"""Granularity: `surface` (maximum surface fit).
- Make the generated surface approach the reference as closely as the selected reconstruction mode permits. Model visible contour changes, hard-surface plane breaks, cavities, overlapping armor, anatomy, clothing layers, and attachments explicitly.
{shared}
- Prefer watertight/manifold shells within each solid semantic part when that matches the subject; preserve intentional gaps between separate parts.
- The host applies the strictest deterministic bidirectional mean and p95 surface-distance gates. Resolve their worst residual coordinates before decorative micro-detail or material polish.
- In procedural mode this remains a best-effort authored approximation from evidence; exact reference-derived topology requires `glb-ref`."""


def initial_prompt(
    *,
    root: Path,
    image: Path,
    user_prompt: str,
    reconstruction_mode: str = DEFAULT_RECONSTRUCTION_MODE,
    granularity: str = DEFAULT_GRANULARITY,
) -> str:
    image_name = _relative(image, root)
    mode_contract = _mode_contract(reconstruction_mode)
    granularity_contract = _granularity_contract(granularity)
    schema_text = plan_schema_text()
    return f"""You are the geometry author inside a GLB-guided Blender asset workspace.

Goal from the user:
{user_prompt or 'Reconstruct the reference subject faithfully as a programmable Blender asset.'}

Evidence available in this workspace:
- Original image: `{image_name}`
- Read-only reference model copy: `inputs/reference.glb`
- Container metadata: `evidence/glb_probe.json`
- Normalized geometry, components, bounds and cross-sections: `evidence/reference_scene.json`
- Exact camera contract: `evidence/camera_contract.json`
- Canonical GLB renders: `evidence/reference_views/front.png`, `back.png`, `left.png`, `right.png`, `top.png`, `iso.png`

Treat node and material names as unreliable: many source GLBs are one anonymous merged mesh. Infer meaningful parts from the image, silhouettes, bounds, sections, and views. Follow the selected mode's reference-use contract exactly.

{mode_contract}

{granularity_contract}

Do not discover, invoke, or depend on a ProcAgen3D/Codex skill. This workspace contract is complete and the retired skill implementation is intentionally out of scope.

Create exactly these two deliverables:

1. `src/plan.json`: valid JSON conforming to the exact JSON Schema below. Set `reconstruction_mode` to `{reconstruction_mode}` and `granularity` to `{granularity}`, matching the selected host configuration. Each part should record a semantic name, shape family, approximate bounds, parent/attachment, and visual role. For a character or hybrid, use `proportions` to describe head-to-body and limb proportions. Empty character-analysis lists are valid when a feature is absent; put every uncertain or hidden feature in `inferred_features`.
2. `src/program.py`: mode-conforming Blender Python source defining a callable `build()` with no arguments. `build()` constructs the complete asset as editable, semantically named Blender geometry. The selected mode contract above states whether this file is standalone or requires the provenance-locked host reference preload.

Authoritative `src/plan.json` JSON Schema (copy its field names and types exactly):
```json
{schema_text}
```

Program contract:
- Use Blender's `bpy` plus safe standard modules such as `math` and `mathutils`.
- Use X for width, Y for depth, Z up, ground at Z=0. Match the normalized bounds in `reference_scene.json`; the reference's longest dimension is 2 units.
- Make recognizable silhouette, proportions, major color blocks, and identity features the priority. Organize parts under named objects/collections.
- Make transforms deterministic. Prefer assigning explicit world matrices built with `mathutils.Matrix.LocRotScale`. If you set location/rotation/scale properties or create/reparent a hierarchy, call `bpy.context.view_layer.update()` before reading or preserving any `matrix_world`; newly authored nested parent chains also require this evaluation before a keep-world reparent.
- For characters, preserve the reference posture and character-relative left/right; solve the head, face, hands/feet, costume silhouette, attachments, and held props as explicit semantic parts. A merged source mesh is not permission to invent hidden anatomy. Do not claim a usable rig unless you actually construct and bind one.
- The host build runs under `blender --background --factory-startup`. In procedural mode the program needs no application-owned inputs; in glb-ref mode it intentionally requires the documented, normalized host collection.
- Do not read files, import/link external assets, access the network, start processes, render, save a .blend, or export a GLB. The application owns those operations.
- Do not invoke Blender during this agent turn. Limit local validation to JSON parsing and Python syntax/static checks; the host pipeline runs Blender after source promotion.
- Do not merely explain or paste code into chat. Inspect the evidence and write both files directly. Finish only after `python -m py_compile src/program.py` would succeed.
"""


def repair_prompt(
    *,
    root: Path,
    user_prompt: str,
    failure: str | None,
    comparison: dict[str, Any] | None,
    iteration: int,
    reconstruction_mode: str = DEFAULT_RECONSTRUCTION_MODE,
    granularity: str = DEFAULT_GRANULARITY,
) -> str:
    comparison_text = json.dumps(comparison, indent=2) if comparison else "No fidelity report was produced."
    mode_contract = _mode_contract(reconstruction_mode)
    granularity_contract = _granularity_contract(granularity)
    schema_text = plan_schema_text()
    return f"""Repair iteration {iteration} for the GLB-guided Blender reconstruction.

Original goal:
{user_prompt or 'Reconstruct the reference subject faithfully as a programmable Blender asset.'}

The current source is `src/program.py`; its plan is `src/plan.json`. Preserve working details and make targeted source edits. The reference evidence and canonical views remain under `evidence/`.

{mode_contract}

{granularity_contract}

The plan must set `reconstruction_mode` to `{reconstruction_mode}` and `granularity` to `{granularity}`, matching the selected host configuration, and conform to this authoritative JSON Schema exactly:
```json
{schema_text}
```

The current compiled candidate views are `artifacts/renders/front.png`, `back.png`, `left.png`, `right.png`, `top.png`, and `iso.png`. Compare them directly with the corresponding `evidence/reference_views/*.png` images when they are present.

Build/static failure (if any):
{failure or 'None; the program built successfully.'}

Deterministic comparison:
{comparison_text}

Fix the highest-impact problem first: a build error, then every failed hard gate. At fine or surface granularity, prioritize bidirectional surface mean/p95 failures and the reported worst residual coordinates; otherwise prioritize the lowest-IoU canonical silhouette, then color/identity details. For a character, explicitly re-check posture, head/body and limb proportions, face placement, left/right asymmetry, clothing layers, attachments, and held props before decorative micro-detail. Make transforms deterministic with explicit `Matrix.LocRotScale` world matrices, or refresh the view layer before reading `matrix_world`; evaluate newly authored nested parent chains before any keep-world reparent. Retain the selected mode's replay/source-of-truth and no-file-I/O/no-self-import/no-export contracts. Do not invoke Blender during this agent turn; the host pipeline performs the build after promotion. Write edits directly to `src/program.py` and update `src/plan.json` whenever its declared granularity or construction plan changes. Do not respond with a tutorial.
"""
