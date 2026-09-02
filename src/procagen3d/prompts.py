"""Prompts for CLI agents; these are application contracts, not a Codex skill."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .granularity import DEFAULT_GRANULARITY, validate_granularity
from .plan_schema import PLAN_SCHEMA, plan_schema_text
from .quality import QualityProfile, resolve_quality_profile
from .reconstruction import DEFAULT_RECONSTRUCTION_MODE, validate_reconstruction_mode
from .stages import RepairTarget


_SAFE_COLLECTION_LINKING_CONTRACT = (
    "Link every data-created Blender object to `bpy.context.scene.collection` or an "
    "explicitly scene-linked collection; never assume `bpy.context.collection` is non-null."
)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _assembly_planning_schema_text() -> str:
    """Return the planning-only schema without the later PBR-stage field."""

    schema = copy.deepcopy(PLAN_SCHEMA)
    schema["properties"].pop("material_plan", None)
    schema["additionalProperties"] = False
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False)


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


def _material_export_contract() -> str:
    return """Material/export contract:
- The host exports `model.glb` without baking Blender procedural shader graphs.
- For every visible color block, use glTF-safe direct constants on unlinked Principled BSDF `Base Color`, `Metallic`, and `Roughness` inputs, and assign that material to the intended geometry.
- Do not connect Noise Texture, Color Ramp, Wave Texture, or other procedural nodes to exported Principled inputs unless `build()` explicitly converts or bakes the result into a self-contained image texture that the GLB exporter embeds.
- Setting only `Material.diffuse_color` does not preserve a linked unsupported `Base Color` graph through GLB export. Without an explicit bake, create variation with additional direct-constant materials, vertex colors, or geometry."""


def _quality_contract(profile: QualityProfile) -> str:
    value = json.dumps(profile.as_dict(), sort_keys=True)
    return f"""Independent quality profile: `{value}`.
- Surface fidelity controls bidirectional distance, visible coverage, surface-area, and normal-aware comparison; it is independent of how many authored details are requested.
- Detail richness requires declared semantic parts to map to real Blender object names and retains enough geometric richness to express the reference's visible forms.
- Material fidelity applies non-compensating spatial-color, palette, and exported glTF material gates. Implicit white is never a substitute for an authored visible color.
- Structural coherence applies topology, component, winding/normal, degeneracy, contact, and unexpected-intersection gates. Use the typed attachment tolerances in the plan; do not leave visually attached parts merely intersecting or floating."""


def initial_prompt(
    *,
    root: Path,
    image: Path,
    user_prompt: str,
    reconstruction_mode: str = DEFAULT_RECONSTRUCTION_MODE,
    granularity: str = DEFAULT_GRANULARITY,
    quality_profile: QualityProfile | None = None,
) -> str:
    image_name = _relative(image, root)
    mode_contract = _mode_contract(reconstruction_mode)
    granularity_contract = _granularity_contract(granularity)
    material_export_contract = _material_export_contract()
    quality_profile = quality_profile or resolve_quality_profile(granularity)
    quality_contract = _quality_contract(quality_profile)
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
- Canonical depth, world-normal, and object-ID diagnostics: `evidence/reference_views/diagnostics/` with `manifest.json`

Treat node and material names as unreliable: many source GLBs are one anonymous merged mesh. Infer meaningful parts from the image, silhouettes, bounds, sections, and views. Follow the selected mode's reference-use contract exactly.

{mode_contract}

{granularity_contract}

{material_export_contract}

{quality_contract}

Do not discover, invoke, or depend on a ProcAgen3D/Codex skill. This workspace contract is complete and the retired skill implementation is intentionally out of scope.

Create exactly these two deliverables:

1. `src/plan.json`: valid JSON conforming to the exact JSON Schema below. Set `reconstruction_mode` to `{reconstruction_mode}`, `granularity` to `{granularity}`, and `quality_profile` to `{json.dumps(quality_profile.as_dict(), sort_keys=True)}`, matching the selected host configuration. Give every part a stable `id`, exact Blender `object_names`, numeric bounds, shape family, visual role, and typed `attachment` containing a declared parent ID, contact region, attachment type, maximum gap/penetration, and minimum contact area. For a character or hybrid, use `proportions` to describe head-to-body and limb proportions. Empty character-analysis lists are valid when a feature is absent; put every uncertain or hidden feature in `inferred_features`.
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


def assembly_planning_prompt(
    *,
    root: Path,
    image: Path,
    user_prompt: str,
    reconstruction_mode: str = DEFAULT_RECONSTRUCTION_MODE,
    granularity: str = DEFAULT_GRANULARITY,
    quality_profile: QualityProfile | None = None,
    export_urdf: bool = False,
    failure: str | None = None,
) -> str:
    """Plan a Procedura-style assembly and create a geometry-free build scaffold."""

    image_name = _relative(image, root)
    mode_contract = _mode_contract(reconstruction_mode)
    granularity_contract = _granularity_contract(granularity)
    quality_profile = quality_profile or resolve_quality_profile(granularity)
    quality_contract = _quality_contract(quality_profile)
    schema_text = _assembly_planning_schema_text()
    articulation_contract = (
        "The user requested URDF output. If and only if this is an explicit mechanical "
        "subject, add `articulation` with `enabled: true`, `mechanical: true`, and a safe "
        "robot name. Make the assembly mates themselves one connected link tree and omit "
        "`articulation.joints` so the host derives URDF origins, axes, rest offsets, and "
        "limits from the same connector solution. Place every revolute or prismatic "
        "child connector on the true motion axis — a wheel bore, hinge pin, or slide — "
        "not at the part modeling origin; the child rotates about that connector, not "
        "about (0,0,0). Otherwise set both `enabled: false` and `mechanical: false`, "
        "and explain the reason in limitations."
        if export_urdf
        else "Include `articulation` only when the reference clearly depicts an explicit mechanical joint; otherwise omit it."
    )
    repair_contract = (
        f"""Strict planning repair: the preserved candidate was rejected by the host for the
exact reasons below. Treat every reported item as authoritative, correct all of them in
`src/plan.json`, and do not spend the retry changing unrelated subject identity or valid fields.

<prior-plan-rejection>
{failure}
</prior-plan-rejection>
"""
        if failure is not None
        else ""
    )
    return f"""You are the planning agent for ProcAgen3D's structured Blender pipeline.

Goal from the user:
{user_prompt or 'Reconstruct the reference subject faithfully as a programmable Blender asset.'}

Evidence:
- Original image: `{image_name}`
- Verified reference: `inputs/reference.glb`
- Container/material evidence: `evidence/glb_probe.json`
- Normalized geometry and cross-sections: `evidence/reference_scene.json`
- Six RGB/depth/normal/object-ID views: `evidence/reference_views/`

{mode_contract}

{granularity_contract}

{quality_contract}

This turn is planning only. Create exactly `src/plan.json` and `src/program.py`.

{repair_contract}

`src/plan.json` must conform to the schema below and must contain a version-1
`assembly` object with these operational fields:
- `placement`: exactly `host-solved`. Part builders author local geometry and the trusted host
  applies the solved connector transform after `build()`.
- `part_order`: every part ID exactly once, in parent-before-child construction order.
- `connectors`: explicit part-local connector frames. Each frame has an origin and
  right-handed orthonormal X/Y/Z axes, an interface class, and shared nominal dimensions.
- `mates`: pair parent and child connectors with a rigid, revolute, prismatic, or spherical
  mate; state fit/clearance and shared nominal dimensions. A rigid mate must omit `rest` and
  `limits` entirely. A revolute or prismatic mate requires a finite scalar `rest`; when it has
  `limits`, their finite scalar `lower` and `upper` values must contain `rest`. A spherical mate
  requires a three-number `rest`; when it has `limits`, `lower` and `upper` must each be three
  numbers and contain `rest` component by component.
- Declare exactly one root part. Its `attachment.type` is `root` and its
  `attachment.parent_id` is the literal sentinel `__root__`, not `world`, an empty string, the
  root part's own ID, or any other alias.
- Every non-root part must be placed by one declared mate. Put receiving holes, sockets,
  windows, and seats in the earlier parent part so later steps never mutate accepted geometry.
- Decompose the visible subject into real semantic parts. Do not use one catch-all part for a
  multi-component object. Every `object_names` entry must be an exact future Blender object name.
- Omit `material_plan` from `src/plan.json` in this planning turn, even though it is an optional
  field in the general schema. Populate only the required legacy `materials` summary array. The
  dedicated material stage is the only stage allowed to create final PBR records and assignments.

{articulation_contract}

Authoritative schema:
```json
{schema_text}
```

`src/program.py` is a safe scaffold, not the complete asset. It must:
- define `PROCAGEN3D_PART_ORDER` from the plan, `PROCAGEN3D_COMPLETED_PARTS = []`,
  and `PROCAGEN3D_PART_BUILDERS = {{}}`;
- define `build()` that calls only the registered builders in completed-part order;
- contain reusable deterministic helpers if useful, but create no geometry yet;
- obey the selected reconstruction mode and define no file/network/process/render/save/export operations.

Use neutral glTF-safe materials during geometry construction. Do not perform final material
extraction or assignment now; a dedicated PBR pass runs after geometry acceptance. Do not invoke
Blender. Write both files directly and finish only after JSON and Python syntax are valid.
"""


def incremental_part_prompt(
    *,
    part: dict[str, Any],
    assembly: dict[str, Any],
    completed_part_ids: list[str],
    part_index: int,
    part_count: int,
    solved_world_transform: list[list[float]] | None = None,
    checkpoint_failure: str | None = None,
) -> str:
    """Author exactly one part while freezing all accepted predecessors."""

    transform_text = (
        json.dumps(solved_world_transform, indent=2)
        if solved_world_transform is not None
        else "No solved matrix is available; use the declared bounds and connector frames exactly."
    )
    related_connectors = [
        item
        for item in assembly.get("connectors", [])
        if isinstance(item, dict) and item.get("part_id") == part.get("id")
    ]
    connector_ids = {item.get("id") for item in related_connectors}
    related_mates = [
        item
        for item in assembly.get("mates", [])
        if isinstance(item, dict)
        and (
            item.get("parent_connector_id") in connector_ids
            or item.get("child_connector_id") in connector_ids
        )
    ]
    return f"""Incremental geometry step {part_index + 1} of {part_count}.

Implement exactly this planned part in `src/program.py`:
```json
{json.dumps(part, indent=2, ensure_ascii=False)}
```

Its connectors and mates are:
```json
{json.dumps({'connectors': related_connectors, 'mates': related_mates}, indent=2, ensure_ascii=False)}
```

Host-solved world transform for this part, when available:
```json
{transform_text}
```

Previously accepted part IDs are `{json.dumps(completed_part_ids)}`. Their builder functions,
object names, geometry, transforms, connector features, and registration order are frozen. Do not
edit them. Add one deterministic builder for `{part.get('id')}` and add only this ID to the two
existing registry bindings. Keep both registries as complete, call-free module-level container
assignments, preserving every accepted entry, for example
`PROCAGEN3D_PART_BUILDERS = {{..., "{part.get('id')}": build_function}}` and
`PROCAGEN3D_COMPLETED_PARTS = [..., "{part.get('id')}"]`. Never register with a module-level
subscript assignment such as `PROCAGEN3D_PART_BUILDERS["{part.get('id')}"] = ...`, and never use
`.append()`, `.extend()`, `.insert()`, or `.update()` at module scope.

Construction contract:
- Create every exact `object_names` entry declared for this part and no future part objects.
- Model all receiving and projecting connector features assigned to this part now.
- Author this part around its own local origin and connector frames. Do not bake the supplied world
  matrix into vertices or Blender object transforms: the trusted build host applies it after
  `build()`. Use the matrix only to understand the resulting world location and orientation.
  If this part has an incoming revolute or prismatic mate, its child connector is the URDF
  link origin: keep that frame on the mechanical axis so the part spins or slides in place.
- Keep geometry semantically editable and deterministic. Use neutral, direct-constant Principled
  materials only; final PBR extraction and assignment happen later.
- At module scope use only imports, call-free plain-name assignments, and helper function
  definitions. Preserve exactly one undecorated, synchronous, no-argument top-level `def build():`;
  the host calls it. Do not use module-level classes, decorators, function or method calls,
  attribute or subscript assignments, or control flow. Do not use dynamic introspection such as
  `getattr`; access known Blender attributes directly.
- {_SAFE_COLLECTION_LINKING_CONTRACT}
- Do not edit `src/plan.json`: its parts, object ownership, connectors, mates, and articulation
  are frozen planning outputs. Never collapse or merge accepted semantic parts.
- Do not read files, invoke Blender, access the network, launch processes, render, save, or export.

{('The previous checkpoint failed: ' + checkpoint_failure + '. Correct only this part and its registration.') if checkpoint_failure else ''}

Write the source directly. Do not provide a tutorial.
"""


def dedicated_material_prompt(
    *,
    plan: dict[str, Any],
    geometry_signature: dict[str, Any],
    failure: str | None = None,
) -> str:
    """Run the isolated PBR extraction and assignment pass."""

    return f"""Dedicated PBR extraction and assignment pass for an accepted Blender assembly.

Use the original image, `evidence/glb_probe.json`, all reference views, and the current compiled
views under `artifacts/renders/`. Build a compact PBR library, then assign it by semantic part and,
where needed, stable subpart/material-slot rules. Update `src/plan.json` field `material_plan` as
`{{"schema_version": 1, "materials": [...], "assignments": [...]}}`. Material records contain
stable IDs, direct glTF-safe base-color RGBA, metallic, roughness, and optional emissive/alpha
values. Assignment records contain `part_id`, `material_id`, and optional stable subpart targeting.
Each assignment target is the pair (`part_id`, `subpart_id`) and must occur exactly once. For any
part, at most one assignment may omit `subpart_id`; that record is the whole-part default. Never
create several whole-part assignments for one part merely by giving them different `object_names`.
Every additional material region must have a unique semantic `subpart_id` and at least one exact,
part-owned `object_names` entry or a bounded visual `selector`. A subpart may override the
whole-part default, but two subpart rules must not claim the same object. Assign every declared
part at least once and reference no object owned by another part.
Keep the legacy required `materials` array as a compact mirror of the library for compatibility.

Edit `src/program.py` only to create and assign these materials after geometry construction.
Do not change vertices, faces, modifiers, transforms, object hierarchy, object names, part builders,
connector geometry, or `PROCAGEN3D_COMPLETED_PARTS`. Unsupported procedural shader graphs are not
allowed unless explicitly baked into an embedded texture; prefer direct Principled constants.
{_SAFE_COLLECTION_LINKING_CONTRACT}

The host will reject any geometry change against this pre-material signature:
```json
{json.dumps(geometry_signature, indent=2, ensure_ascii=False)}
```

Current normalized plan:
```json
{json.dumps(plan, indent=2, ensure_ascii=False)}
```

{('The previous material attempt was rejected: ' + failure + '. Fix only the material code and assignments.') if failure else ''}

Do not invoke Blender or perform file/network/process/render/save/export operations. Write both
source files directly and finish only after syntax and JSON validity checks.
"""


def targeted_repair_prompt(
    *,
    user_prompt: str,
    target: RepairTarget,
    iteration: int,
    reconstruction_mode: str,
    granularity: str,
    quality_profile: QualityProfile,
    geometry_only: bool = False,
) -> str:
    """Create a one-diagnosis/one-edit repair transaction."""

    material_rule = (
        "This is a geometry/assembly repair. Preserve neutral materials and do not perform the PBR pass."
        if geometry_only
        else "Preserve every unrelated material and geometry detail."
    )
    return f"""Targeted repair transaction {iteration}.

Original goal:
{user_prompt or 'Reconstruct the reference subject faithfully as a programmable Blender asset.'}

The deterministic critic selected exactly one problem:
```json
{json.dumps(target.as_dict(), indent=2, ensure_ascii=False)}
```

Repair only that diagnosis in `src/program.py`. The only permitted structured-plan edits are tuning
an existing connector frame or an existing mate's `fit_offset` for the selected placement fix.
Part ownership, bounds, attachment tolerances, dimensions, interface/fit semantics, rest values,
limits, quality settings, connector IDs, and the mate graph are frozen. Use `artifacts/renders/`,
their diagnostics, and surface residuals to localize the selected failure. Do not opportunistically
redesign unrelated regions. {material_rule}

Mode: `{reconstruction_mode}`. Granularity: `{granularity}`.
Quality profile: `{json.dumps(quality_profile.as_dict(), sort_keys=True)}`.

Retain the no-file-I/O, no-self-import, no-network, no-process, no-render/save/export source
contract. Do not invoke Blender during this agent turn. Write the edits directly.
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
    quality_profile: QualityProfile | None = None,
) -> str:
    comparison_text = json.dumps(comparison, indent=2) if comparison else "No fidelity report was produced."
    mode_contract = _mode_contract(reconstruction_mode)
    granularity_contract = _granularity_contract(granularity)
    material_export_contract = _material_export_contract()
    quality_profile = quality_profile or resolve_quality_profile(granularity)
    quality_contract = _quality_contract(quality_profile)
    schema_text = plan_schema_text()
    return f"""Repair iteration {iteration} for the GLB-guided Blender reconstruction.

Original goal:
{user_prompt or 'Reconstruct the reference subject faithfully as a programmable Blender asset.'}

The current source is `src/program.py`; its plan is `src/plan.json`. Preserve working details and make targeted source edits. The reference evidence and canonical views remain under `evidence/`.

{mode_contract}

{granularity_contract}

{material_export_contract}

{quality_contract}

The plan must set `reconstruction_mode` to `{reconstruction_mode}`, `granularity` to `{granularity}`, and `quality_profile` to `{json.dumps(quality_profile.as_dict(), sort_keys=True)}`, matching the selected host configuration, and conform to this authoritative JSON Schema exactly. Every non-root part must name its exact Blender objects and declare its parent, attachment type, contact region, and numeric gap/penetration/contact tolerances:
```json
{schema_text}
```

The current compiled candidate views are `artifacts/renders/front.png`, `back.png`, `left.png`, `right.png`, `top.png`, and `iso.png`. Compare them directly with the corresponding `evidence/reference_views/*.png` images when they are present. Use the candidate `artifacts/renders/diagnostics/` depth, world-normal, and object-ID views beside the reference diagnostics to localize incoherent surfaces, missing parts, and wrong part boundaries. When present, use `artifacts/surface_residuals/` heatmaps to repair the largest visible residual regions first.

Build/static failure (if any):
{failure or 'None; the program built successfully.'}

Deterministic comparison:
{comparison_text}

Fix the highest-impact problem first: a build error, then every failed hard gate. At fine or surface granularity, prioritize bidirectional surface mean/p95 failures and the reported worst residual coordinates; otherwise prioritize the lowest-IoU canonical silhouette, then color/identity details. For a character, explicitly re-check posture, head/body and limb proportions, face placement, left/right asymmetry, clothing layers, attachments, and held props before decorative micro-detail. Make transforms deterministic with explicit `Matrix.LocRotScale` world matrices, or refresh the view layer before reading `matrix_world`; evaluate newly authored nested parent chains before any keep-world reparent. Retain the selected mode's replay/source-of-truth and no-file-I/O/no-self-import/no-export contracts. Do not invoke Blender during this agent turn; the host pipeline performs the build after promotion. Write edits directly to `src/program.py` and update `src/plan.json` whenever its declared granularity or construction plan changes. Do not respond with a tutorial.
"""
