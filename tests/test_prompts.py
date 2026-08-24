from __future__ import annotations

from pathlib import Path

from procagen3d.prompts import initial_prompt, repair_prompt


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
