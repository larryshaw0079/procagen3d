from __future__ import annotations

import json
from pathlib import Path

from procagen3d.plan_schema import PLAN_SCHEMA, plan_schema_text
from procagen3d.prompts import (
    assembly_planning_prompt,
    dedicated_material_prompt,
    incremental_part_prompt,
    initial_prompt,
    repair_prompt,
)
from procagen3d.quality import QualityProfile


def test_initial_prompt_routes_characters_through_anatomy_analysis(tmp_path: Path) -> None:
    image = tmp_path / "inputs" / "reference.png"
    image.parent.mkdir()
    image.write_bytes(b"image")

    prompt = initial_prompt(root=tmp_path, image=image, user_prompt="Build this character")

    assert "subject_kind" in prompt
    assert "character_analysis" in prompt
    assert "head-to-body and limb proportions" in prompt
    assert "character-relative left/right" in prompt
    assert "Do not claim a usable rig" in prompt
    assert "Do not invoke Blender during this agent turn" in prompt
    assert plan_schema_text() in prompt
    assert "Set `reconstruction_mode` to `procedural`" in prompt
    assert "`granularity` to `medium`" in prompt
    assert "Granularity: `medium`" in prompt


def test_repair_prompt_rechecks_character_identity_structure(tmp_path: Path) -> None:
    prompt = repair_prompt(
        root=tmp_path,
        user_prompt="Build this character",
        failure="low fidelity",
        comparison={"score": 0.2},
        iteration=1,
    )

    assert "posture" in prompt
    assert "face placement" in prompt
    assert "left/right asymmetry" in prompt
    assert "held props" in prompt
    assert "Do not invoke Blender during this agent turn" in prompt
    assert plan_schema_text() in prompt
    assert "set `reconstruction_mode` to `procedural`" in prompt
    assert "`granularity` to `medium`" in prompt


def test_initial_and_repair_prompts_require_gltf_safe_materials(tmp_path: Path) -> None:
    image = tmp_path / "inputs" / "reference.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    prompts = (
        initial_prompt(root=tmp_path, image=image, user_prompt="Build this scene"),
        repair_prompt(
            root=tmp_path,
            user_prompt="Build this scene",
            failure="colors became white after export",
            comparison={"summary": {"mean_spatial_rgb_similarity": 0.2}},
            iteration=1,
        ),
    )

    for prompt in prompts:
        assert "exports `model.glb` without baking Blender procedural shader graphs" in prompt
        assert "glTF-safe direct constants on unlinked Principled BSDF" in prompt
        assert "unless `build()` explicitly converts or bakes" in prompt
        assert "Setting only `Material.diffuse_color` does not preserve" in prompt
        assert "additional direct-constant materials, vertex colors, or geometry" in prompt


def test_fine_and_surface_prompts_require_surface_conforming_geometry(
    tmp_path: Path,
) -> None:
    image = tmp_path / "inputs" / "reference.png"
    image.parent.mkdir()
    image.write_bytes(b"image")

    fine = initial_prompt(
        root=tmp_path,
        image=image,
        user_prompt="Build this mecha",
        granularity="fine",
    )
    surface = repair_prompt(
        root=tmp_path,
        user_prompt="Build this mecha",
        failure="surface gate failed",
        comparison={"hard_gates": {"failures": ["p95_surface_distance"]}},
        iteration=1,
        granularity="surface",
    )

    for prompt in (fine, surface):
        assert "surface-conforming custom meshes" in prompt
        assert "do not merely increase primitive segments" in prompt
        assert "bidirectional" in prompt
        assert "large reference vertex/index dump" in prompt
    assert "maximum surface fit" in surface
    assert "best-effort authored approximation" in surface


def test_prompts_bind_typed_attachments_diagnostics_and_independent_quality(
    tmp_path: Path,
) -> None:
    image = tmp_path / "inputs" / "reference.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    profile = QualityProfile("balanced", "maximum", "strict", "coherent")
    prompts = (
        initial_prompt(
            root=tmp_path,
            image=image,
            user_prompt="Build it",
            quality_profile=profile,
        ),
        repair_prompt(
            root=tmp_path,
            user_prompt="Build it",
            failure="attachments failed",
            comparison={},
            iteration=1,
            quality_profile=profile,
        ),
    )
    for prompt in prompts:
        assert '"surface_fidelity": "balanced"' in prompt
        assert '"detail_richness": "maximum"' in prompt
        assert "exact Blender `object_names`" in prompt or "exact Blender objects" in prompt
        assert "contact region" in prompt
        assert "gap/penetration" in prompt
        assert "depth" in prompt and "world-normal" in prompt and "object-ID" in prompt


def test_assembly_planning_prompt_states_semantic_plan_constraints(
    tmp_path: Path,
) -> None:
    image = tmp_path / "inputs" / "reference.png"
    image.parent.mkdir()
    image.write_bytes(b"image")

    prompt = assembly_planning_prompt(
        root=tmp_path,
        image=image,
        user_prompt="Build this articulated machine",
        export_urdf=True,
    )
    flat_prompt = " ".join(prompt.split())

    assert "omit `articulation.joints`" in flat_prompt
    assert "child rotates about that connector" in flat_prompt
    assert "rigid mate must omit `rest` and `limits` entirely" in flat_prompt
    assert "revolute or prismatic mate requires a finite scalar `rest`" in flat_prompt
    assert "spherical mate" in flat_prompt and "three-number `rest`" in flat_prompt
    assert "`attachment.parent_id` is the literal sentinel `__root__`" in flat_prompt
    assert "not `world`" in flat_prompt
    assert "Omit `material_plan` from `src/plan.json`" in flat_prompt
    assert "dedicated material stage is the only stage allowed" in flat_prompt
    schema_payload = prompt.split("Authoritative schema:\n```json\n", 1)[1].split(
        "\n```", 1
    )[0]
    planning_schema = json.loads(schema_payload)
    assert "material_plan" not in planning_schema["properties"]
    assert planning_schema["additionalProperties"] is False
    assert "material_plan" in PLAN_SCHEMA["properties"]


def test_assembly_planning_retry_includes_exact_host_rejection(tmp_path: Path) -> None:
    image = tmp_path / "inputs" / "reference.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    rejection = (
        "src/plan.json plan violates the JSON Schema (2 errors):\n"
        "- $.assembly.mates[0]: rigid mates must not declare rest or limits\n"
        "- $.parts[0].attachment.parent_id: a root attachment must use '__root__'"
    )

    prompt = assembly_planning_prompt(
        root=tmp_path,
        image=image,
        user_prompt="Repair the plan",
        failure=rejection,
    )

    assert "Strict planning repair" in prompt
    assert rejection in prompt
    assert "correct all of them" in prompt
    assert "do not spend the retry changing unrelated subject identity" in prompt


def test_dedicated_material_prompt_disambiguates_assignment_targets() -> None:
    prompt = dedicated_material_prompt(
        plan={"parts": [{"id": "body", "object_names": ["Body", "Trim"]}]},
        geometry_signature={"object_count": 2},
    )
    flat_prompt = " ".join(prompt.split())

    assert "pair (`part_id`, `subpart_id`) and must occur exactly once" in flat_prompt
    assert "at most one assignment may omit `subpart_id`" in flat_prompt
    assert "Never create several whole-part assignments" in flat_prompt
    assert "unique semantic `subpart_id`" in flat_prompt
    assert "two subpart rules must not claim the same object" in flat_prompt
    assert "Assign every declared part at least once" in flat_prompt


def test_incremental_and_material_prompts_require_scene_linked_objects() -> None:
    guidance = (
        "Link every data-created Blender object to `bpy.context.scene.collection` or an "
        "explicitly scene-linked collection; never assume `bpy.context.collection` is non-null."
    )
    incremental = incremental_part_prompt(
        part={"id": "body", "object_names": ["Body"]},
        assembly={"connectors": [], "mates": []},
        completed_part_ids=[],
        part_index=0,
        part_count=1,
    )
    materials = dedicated_material_prompt(
        plan={"parts": [{"id": "body", "object_names": ["Body"]}]},
        geometry_signature={"object_count": 1},
    )

    assert guidance in incremental
    assert guidance in materials


def test_incremental_prompt_requires_call_free_literal_registries() -> None:
    prompt = incremental_part_prompt(
        part={"id": "body", "object_names": ["Body"]},
        assembly={"connectors": [], "mates": []},
        completed_part_ids=[],
        part_index=0,
        part_count=1,
    )

    assert '`PROCAGEN3D_PART_BUILDERS = {..., "body": build_function}`' in prompt
    assert '`PROCAGEN3D_COMPLETED_PARTS = [..., "body"]`' in prompt
    assert '`PROCAGEN3D_PART_BUILDERS["body"] = ...`' in prompt
    assert "never use\n`.append()`, `.extend()`, `.insert()`, or `.update()`" in prompt
    assert "Preserve exactly one undecorated, synchronous, no-argument" in prompt
    assert "Do not use module-level classes, decorators, function or method calls" in prompt
    assert "attribute or subscript assignments, or control flow" in prompt
    assert "dynamic introspection such as\n  `getattr`" in prompt
    assert "child connector is the URDF" in prompt
