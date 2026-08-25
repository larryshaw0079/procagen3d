from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from procagen3d.workspace import Workspace, slugify, workspace_slug, write_json


def _workspace(tmp_path: Path, *, slug: str = "fixture") -> Workspace:
    source = tmp_path / f"{slug}-source"
    source.mkdir()
    image = source / "image.png"
    glb = source / "model.glb"
    image.write_bytes(b"image-data")
    glb.write_bytes(b"glb-data")
    return Workspace.create(
        base=tmp_path / "outputs",
        slug=slug,
        image=image,
        glb=glb,
        prompt="",
        backend="codex",
    )


def test_slugify_is_stable_and_bounded() -> None:
    assert slugify(" Mystic Mouse / Wanderer ") == "mystic-mouse-wanderer"
    assert len(slugify("a" * 100)) == 64
    with pytest.raises(ValueError):
        slugify("---")


def test_workspace_slug_appends_mode_without_duplicating_it() -> None:
    assert workspace_slug("mystic-mouse", reconstruction_mode="procedural") == (
        "mystic-mouse-procedural"
    )
    assert workspace_slug("mystic-mouse", reconstruction_mode="glb-ref") == (
        "mystic-mouse-glb-ref"
    )
    assert workspace_slug("custom-procedural", reconstruction_mode="procedural") == (
        "custom-procedural"
    )
    long_name = "a" * 64
    suffixed = workspace_slug(long_name, reconstruction_mode="procedural")
    assert suffixed.endswith("-procedural")
    assert len(suffixed) == 64


def test_workspace_copies_inputs_and_records_provenance(tmp_path: Path) -> None:
    image = tmp_path / "source.PNG"
    glb = tmp_path / "source.glb"
    image.write_bytes(b"image-data")
    glb.write_bytes(b"glb-data")

    workspace = Workspace.create(
        base=tmp_path / "outputs",
        slug="mouse",
        image=image,
        glb=glb,
        prompt="mouse adventurer",
        backend="codex",
    )

    assert workspace.image_path.name == "reference.png"
    assert workspace.image_path.read_bytes() == b"image-data"
    assert workspace.glb_path.read_bytes() == b"glb-data"
    manifest = json.loads(workspace.manifest_path.read_text(encoding="utf-8"))
    assert manifest["inputs"]["image"]["source"] == str(image.resolve())
    assert manifest["inputs"]["glb"]["sha256"] == hashlib.sha256(b"glb-data").hexdigest()
    assert manifest["reconstruction_mode"] == "procedural"
    assert manifest["granularity"] == "medium"
    assert manifest["quality_profile"] == {
        "surface_fidelity": "off",
        "detail_richness": "standard",
        "material_fidelity": "faithful",
        "structural_coherence": "coherent",
    }
    assert workspace.trajectory_dir(3).name == "iter_03"

    with pytest.raises(FileExistsError):
        Workspace.create(
            base=tmp_path / "outputs",
            slug="mouse",
            image=image,
            glb=glb,
            prompt="",
            backend="codex",
        )


def test_workspace_records_glb_ref_mode(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    glb = tmp_path / "model.glb"
    image.write_bytes(b"image")
    glb.write_bytes(b"glb")

    workspace = Workspace.create(
        base=tmp_path / "outputs",
        slug="derived",
        image=image,
        glb=glb,
        prompt="",
        backend="codex",
        reconstruction_mode="glb-ref",
        granularity="fine",
    )

    assert workspace.manifest()["reconstruction_mode"] == "glb-ref"
    assert workspace.manifest()["granularity"] == "fine"
    assert workspace.manifest()["quality_profile"] == {
        "surface_fidelity": "balanced",
        "detail_richness": "rich",
        "material_fidelity": "faithful",
        "structural_coherence": "coherent",
    }


def test_workspace_records_partial_quality_override(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    glb = tmp_path / "model.glb"
    image.write_bytes(b"image")
    glb.write_bytes(b"glb")

    workspace = Workspace.create(
        base=tmp_path / "outputs",
        slug="strict-color",
        image=image,
        glb=glb,
        prompt="",
        backend="codex",
        granularity="fine",
        quality_profile={"material_fidelity": "strict"},
    )

    assert workspace.manifest()["quality_profile"] == {
        "surface_fidelity": "balanced",
        "detail_richness": "rich",
        "material_fidelity": "strict",
        "structural_coherence": "coherent",
    }


def test_workspace_locate_by_slug_and_update_manifest(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    glb = tmp_path / "model.glb"
    image.write_bytes(b"jpg")
    glb.write_bytes(b"glb")
    base = tmp_path / "outputs"
    workspace = Workspace.create(
        base=base,
        slug="fixture",
        image=image,
        glb=glb,
        prompt="",
        backend="cursor",
    )
    located = Workspace.locate(Path("fixture"), base=base)
    assert located.root == workspace.root
    located.update_manifest(status="prepared", score=0.5)
    assert located.manifest()["status"] == "prepared"
    assert located.manifest()["score"] == 0.5


@pytest.mark.parametrize("schema_version", [None, True, 2, "1"])
def test_workspace_locate_requires_schema_v1(
    tmp_path: Path, schema_version: object
) -> None:
    workspace = _workspace(tmp_path)
    manifest = workspace.manifest()
    manifest["schema_version"] = schema_version
    write_json(workspace.manifest_path, manifest)

    with pytest.raises(ValueError, match="schema_version"):
        Workspace.locate(workspace.root)


@pytest.mark.parametrize("path_kind", ["absolute", "outside-root-inputs"])
def test_workspace_locate_rejects_input_paths_outside_inputs(
    tmp_path: Path, path_kind: str
) -> None:
    workspace = _workspace(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"image-data")
    manifest = workspace.manifest()
    manifest["inputs"]["image"]["path"] = (
        str(outside.resolve()) if path_kind == "absolute" else "evidence/reference.png"
    )
    manifest["inputs"]["image"]["bytes"] = outside.stat().st_size
    manifest["inputs"]["image"]["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    write_json(workspace.manifest_path, manifest)

    with pytest.raises(ValueError, match="inputs directory"):
        Workspace.locate(workspace.root)


def test_workspace_locate_rejects_symlinked_input(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    image_path = workspace.image_path
    external = tmp_path / "external.png"
    external.write_bytes(image_path.read_bytes())
    image_path.unlink()
    try:
        image_path.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="symlink"):
        Workspace.locate(workspace.root)


def test_workspace_locate_rejects_non_file_input(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    image_path = workspace.image_path
    image_path.unlink()
    image_path.mkdir()

    with pytest.raises(ValueError, match="not a regular file"):
        Workspace.locate(workspace.root)


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("image", "bytes"),
        ("image", "sha256"),
        ("glb", "bytes"),
        ("glb", "sha256"),
    ],
)
def test_workspace_locate_verifies_input_provenance(
    tmp_path: Path, kind: str, field: str
) -> None:
    workspace = _workspace(tmp_path)
    manifest = workspace.manifest()
    if field == "bytes":
        manifest["inputs"][kind][field] += 1
        match = "byte count changed"
    else:
        manifest["inputs"][kind][field] = "0" * 64
        match = "SHA-256 changed"
    write_json(workspace.manifest_path, manifest)

    with pytest.raises(ValueError, match=match):
        Workspace.locate(workspace.root)


def test_next_trajectory_iteration_never_reuses_an_existing_path(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    assert workspace.next_trajectory_iteration() == 0

    first = workspace.trajectory_dir(0)
    assert first.name == "iter_00"
    (workspace.root / "trajectories" / "iter_07").mkdir()
    (workspace.root / "trajectories" / "iter_09").write_text("reserved", encoding="utf-8")
    (workspace.root / "trajectories" / "notes").mkdir()

    assert workspace.next_trajectory_iteration() == 10
    with pytest.raises(FileExistsError):
        workspace.trajectory_dir(0)
