"""Prompts for CLI agents; these are application contracts, not a Codex skill."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def initial_prompt(*, root: Path, image: Path, user_prompt: str) -> str:
    image_name = _relative(image, root)
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

Treat node and material names as unreliable: many source GLBs are one anonymous merged mesh. Infer meaningful parts from the image, silhouettes, bounds, sections, and views. The GLB is measurement evidence only.

Do not discover, invoke, or depend on a ProcAgen3D/Codex skill. This workspace contract is complete and the retired skill implementation is intentionally out of scope.

Create exactly these two deliverables:

1. `src/plan.json`: valid JSON with `subject`, `subject_kind`, `coordinate_frame`, `dimensions`, `parts`, `materials`, `construction_strategy`, `identity_features`, and `limitations`. Classify `subject_kind` as `object`, `character`, `hybrid`, or `scene`. Each part needs a semantic name, shape family, approximate bounds, parent/attachment, and visual role. A character or hybrid must also include `character_analysis` with a non-empty string `pose`, an object `proportions` describing head-to-body and limb proportions, and list fields `facial_landmarks`, `hair_or_headwear`, `clothing_layers`, `held_props`, `left_right_asymmetry`, and `inferred_features`. Empty lists are valid when a feature is absent; put every uncertain or hidden feature in `inferred_features`.
2. `src/program.py`: standalone Blender Python source defining a callable `build()` with no arguments. `build()` constructs the complete asset as editable, semantically named Blender geometry.

Program contract:
- Use Blender's `bpy` plus safe standard modules such as `math` and `mathutils`.
- Geometry must be synthesized from primitives, curves, modifiers, and compact procedural meshes. Do not copy or embed the reference mesh or large vertex/index dumps.
- Use X for width, Y for depth, Z up, ground at Z=0. Match the normalized bounds in `reference_scene.json`; the reference's longest dimension is 2 units.
- Make recognizable silhouette, proportions, major color blocks, and identity features the priority. Organize parts under named objects/collections.
- For characters, preserve the reference posture and character-relative left/right; solve the head, face, hands/feet, costume silhouette, attachments, and held props as explicit semantic parts. A merged source mesh is not permission to invent hidden anatomy. Do not claim a usable rig unless you actually construct and bind one.
- The program must work under `blender --background --factory-startup` and must not need this application installed.
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
) -> str:
    comparison_text = json.dumps(comparison, indent=2) if comparison else "No fidelity report was produced."
    return f"""Repair iteration {iteration} for the GLB-guided Blender reconstruction.

Original goal:
{user_prompt or 'Reconstruct the reference subject faithfully as a programmable Blender asset.'}

The current source is `src/program.py`; its plan is `src/plan.json`. Preserve working details and make targeted source edits. The reference evidence and canonical views remain under `evidence/`.

The current compiled candidate views are `artifacts/renders/front.png`, `back.png`, `left.png`, `right.png`, `top.png`, and `iso.png`. Compare them directly with the corresponding `evidence/reference_views/*.png` images when they are present.

Build/static failure (if any):
{failure or 'None; the program built successfully.'}

Deterministic comparison:
{comparison_text}

Fix the highest-impact problem first: a build error, then global dimensions/grounding, then the lowest-IoU canonical silhouette, then color/identity details. For a character, explicitly re-check posture, head/body and limb proportions, face placement, left/right asymmetry, clothing layers, attachments, and held props before spending effort on generic surface detail. Keep the source standalone and retain the original no-file-I/O/no-reference-import/no-export contract. Do not invoke Blender during this agent turn; the host pipeline performs the build after promotion. Write edits directly to `src/program.py` and update `src/plan.json` only when the construction plan genuinely changes. Do not respond with a tutorial.
"""
