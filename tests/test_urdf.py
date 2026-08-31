from __future__ import annotations

from copy import deepcopy
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

import procagen3d.pipeline as pipeline
from procagen3d.urdf import (
    URDFExportError,
    export_urdf,
    plan_to_urdf,
    render_urdf,
)
from procagen3d.workspace import Workspace, sha256, write_json


def _model(tmp_path: Path) -> Path:
    path = tmp_path / "model.glb"
    path.write_bytes(b"minimal-test-glb")
    return path


def _plan() -> dict[str, object]:
    return {
        "subject": "Test Mechanism",
        "subject_kind": "object",
        "parts": [
            {"id": "base"},
            {"id": "arm"},
            {"id": "slider"},
        ],
        "articulation": {
            "enabled": True,
            "mechanical": True,
            "robot_name": "test_mechanism",
            "joints": [
                {
                    "name": "shoulder",
                    "parent": "base",
                    "child": "arm",
                    "type": "revolute",
                    "origin": {"xyz": [0, 0, 0.5], "rpy": [0, 0, 0]},
                    "axis": [0, 1, 0],
                    "limit": {
                        "lower": -1.5,
                        "upper": 1.5,
                        "effort": 12,
                        "velocity": 2,
                    },
                },
                {
                    "name": "extension",
                    "parent": "arm",
                    "child": "slider",
                    "type": "prismatic",
                    "origin": {"xyz": [0, 0, 1], "rpy": [0, 0, 0]},
                    "axis": [0, 0, 2],
                    "limit": {"lower": 0, "upper": 0.4},
                },
            ],
        },
    }


def test_export_is_explicitly_opt_in_and_skips_without_writes(tmp_path: Path) -> None:
    destination = tmp_path / "robot.urdf"
    report = export_urdf({}, tmp_path / "missing.glb", destination)

    assert report.status == "skipped"
    assert report.enabled is False
    assert report.output_path is None
    assert report.as_dict()["link_count"] == 0
    assert not destination.exists()


def test_render_and_atomic_export_are_deterministic(tmp_path: Path) -> None:
    model = _model(tmp_path)
    plan = _plan()
    first = render_urdf(plan, model)
    second = render_urdf(plan, model)
    assert first.xml == second.xml
    assert first.root_link == "base"
    assert first.link_names == ("arm", "base", "slider")
    assert first.joint_names == ("extension", "shoulder")

    root = ET.fromstring(first.xml)
    assert root.attrib == {"name": "test_mechanism"}
    assert [item.attrib["name"] for item in root.findall("link")] == [
        "arm",
        "base",
        "slider",
    ]
    mesh = root.find("./link[@name='base']/visual/geometry/mesh")
    assert mesh is not None
    assert mesh.attrib["filename"] == "model.glb"
    assert root.find("./link[@name='arm']/visual") is None
    axis = root.find("./joint[@name='extension']/axis")
    assert axis is not None
    assert axis.attrib["xyz"] == "0 0 1"

    destination = tmp_path / "nested" / "robot.urdf"
    destination.parent.mkdir()
    destination.write_text("old", encoding="utf-8")
    report = export_urdf(plan, model, destination, enabled=True)
    assert report.status == "exported"
    assert report.output_path == destination.resolve()
    assert report.link_count == 3
    assert report.joint_count == 2
    assert report.bytes_written == len(first.xml.encode())
    assert report.urdf_sha256 == hashlib.sha256(first.xml.encode()).hexdigest()
    assert report.model_sha256 == hashlib.sha256(model.read_bytes()).hexdigest()
    assert destination.read_text(encoding="utf-8") == first.xml
    assert not list(destination.parent.glob(".robot.urdf.*.tmp"))
    assert any("root link only" in warning for warning in report.warnings)
    assert any("normalized" in warning for warning in report.warnings)
    assert any("visual/kinematic only" in warning for warning in report.warnings)
    assert root.find(".//collision") is None
    assert root.find(".//inertial") is None
    assert root.find(".//transmission") is None


def test_package_model_uri_is_xml_escaped(tmp_path: Path) -> None:
    xml = plan_to_urdf(
        _plan(),
        _model(tmp_path),
        package_model_path="package://test_assets/a&b/",
    )
    assert 'filename="package://test_assets/a&amp;b/model.glb"' in xml
    mesh = ET.fromstring(xml).find("./link[@name='base']/visual/geometry/mesh")
    assert mesh is not None
    assert mesh.attrib["filename"] == "package://test_assets/a&b/model.glb"


def test_per_link_meshes_emit_real_articulated_visuals_and_report_mapping(
    tmp_path: Path,
) -> None:
    model = _model(tmp_path)
    meshes = {
        "slider": "package://mechanism/meshes/slider.glb",
        "base": "meshes/base.glb",
        "arm": "package://mechanism/meshes/a&rm.glb",
    }
    document = render_urdf(_plan(), model, link_meshes=meshes)
    root = ET.fromstring(document.xml)

    assert document.link_meshes == (
        ("arm", "package://mechanism/meshes/a&rm.glb"),
        ("base", "meshes/base.glb"),
        ("slider", "package://mechanism/meshes/slider.glb"),
    )
    assert not any("root link only" in warning for warning in document.warnings)
    assert any("visual/kinematic only" in warning for warning in document.warnings)
    assert "a&amp;rm.glb" in document.xml
    for link_name, expected_uri in meshes.items():
        mesh = root.find(f"./link[@name='{link_name}']/visual/geometry/mesh")
        assert mesh is not None
        assert mesh.attrib["filename"] == expected_uri

    destination = tmp_path / "robot.urdf"
    report = export_urdf(
        _plan(),
        model,
        destination,
        enabled=True,
        link_meshes=meshes,
    )
    assert report.link_meshes == document.link_meshes
    assert report.as_dict()["link_meshes"] == {
        "arm": "package://mechanism/meshes/a&rm.glb",
        "base": "meshes/base.glb",
        "slider": "package://mechanism/meshes/slider.glb",
    }


@pytest.mark.parametrize(
    ("meshes", "message"),
    [
        ({"base": "base.glb", "arm": "arm.glb"}, "missing links"),
        (
            {
                "base": "base.glb",
                "arm": "arm.glb",
                "slider": "slider.glb",
                "ghost": "ghost.glb",
            },
            "unknown links",
        ),
        (
            {"base": "base.glb", "arm": "../arm.glb", "slider": "slider.glb"},
            "safe relative path",
        ),
        (
            {"base": "base.glb", "arm": "", "slider": "slider.glb"},
            "non-empty",
        ),
    ],
)
def test_per_link_mesh_contract_rejects_partial_stale_or_unsafe_maps(
    tmp_path: Path, meshes: dict[str, str], message: str
) -> None:
    with pytest.raises(URDFExportError, match=message):
        render_urdf(_plan(), _model(tmp_path), link_meshes=meshes)


def test_absolute_per_link_mesh_path_becomes_file_uri(tmp_path: Path) -> None:
    model = _model(tmp_path)
    base = tmp_path / "base mesh.glb"
    arm = tmp_path / "arm.glb"
    slider = tmp_path / "slider.glb"
    for path in (base, arm, slider):
        path.write_bytes(b"mesh")
    document = render_urdf(
        _plan(),
        model,
        link_meshes={"base": base, "arm": arm, "slider": slider},
    )
    parsed = ET.fromstring(document.xml)
    mesh = parsed.find("./link[@name='base']/visual/geometry/mesh")
    assert mesh is not None
    assert mesh.attrib["filename"] == base.resolve().as_uri()


@pytest.mark.parametrize("subject_kind", ["character", "vehicle", ""])
def test_rejects_non_mechanical_subject_kinds(tmp_path: Path, subject_kind: str) -> None:
    plan = _plan()
    plan["subject_kind"] = subject_kind
    with pytest.raises(URDFExportError, match="subject_kind|mechanical"):
        render_urdf(plan, _model(tmp_path))


@pytest.mark.parametrize(
    ("articulation", "message"),
    [
        ({"enabled": False, "mechanical": True, "joints": []}, "enabled"),
        ({"enabled": True, "mechanical": False, "joints": []}, "mechanical"),
        ({"enabled": True, "mechanical": True, "joints": []}, "at least one joint"),
    ],
)
def test_plan_must_explicitly_opt_in(
    tmp_path: Path, articulation: dict[str, object], message: str
) -> None:
    plan = _plan()
    plan["articulation"] = articulation
    with pytest.raises(URDFExportError, match=message):
        render_urdf(plan, _model(tmp_path))


@pytest.mark.parametrize("bad_name", ["wheel left", "1wheel", "wheel/link", ""])
def test_rejects_unsafe_link_names(tmp_path: Path, bad_name: str) -> None:
    plan = _plan()
    plan["parts"][1]["id"] = bad_name  # type: ignore[index]
    with pytest.raises(URDFExportError, match="safe URDF name|non-empty"):
        render_urdf(plan, _model(tmp_path))


def test_rejects_duplicate_links_and_unknown_joint_references(tmp_path: Path) -> None:
    plan = _plan()
    plan["parts"][1]["id"] = "base"  # type: ignore[index]
    with pytest.raises(URDFExportError, match="duplicate link"):
        render_urdf(plan, _model(tmp_path))

    plan = _plan()
    plan["articulation"]["joints"][0]["parent"] = "missing"  # type: ignore[index]
    with pytest.raises(URDFExportError, match="unknown parent"):
        render_urdf(plan, _model(tmp_path))


def test_rejects_non_tree_joint_graphs(tmp_path: Path) -> None:
    plan = _plan()
    joints = plan["articulation"]["joints"]  # type: ignore[index]
    joints[1]["parent"] = "base"  # type: ignore[index]
    joints[1]["child"] = "arm"  # type: ignore[index]
    with pytest.raises(URDFExportError, match="more than one parent"):
        render_urdf(plan, _model(tmp_path))

    plan = _plan()
    plan["articulation"]["joints"] = [  # type: ignore[index]
        {
            "name": "cycle_a",
            "parent": "base",
            "child": "arm",
            "type": "continuous",
            "origin": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},
            "axis": [0, 0, 1],
        },
        {
            "name": "cycle_b",
            "parent": "arm",
            "child": "base",
            "type": "fixed",
            "origin": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},
        },
    ]
    with pytest.raises(URDFExportError, match="root|disconnected|cycle"):
        render_urdf(plan, _model(tmp_path))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("axis", [0, 0, 0]), "non-zero"),
        (("axis", [1, 2]), "three"),
        (("origin", {"xyz": [0, 0, 0]}), "xyz and rpy"),
        (("limit", {"lower": 1, "upper": -1}), "less than upper"),
        (("limit", {"lower": -1}), "lower and upper"),
    ],
)
def test_validates_axis_origin_and_limits(
    tmp_path: Path, mutation: tuple[str, object], message: str
) -> None:
    plan = _plan()
    plan["articulation"]["joints"][0][mutation[0]] = mutation[1]  # type: ignore[index]
    with pytest.raises(URDFExportError, match=message):
        render_urdf(plan, _model(tmp_path))


def test_continuous_joint_needs_no_positional_bounds(tmp_path: Path) -> None:
    plan = _plan()
    joint = plan["articulation"]["joints"][0]  # type: ignore[index]
    joint["type"] = "continuous"
    joint.pop("limit")
    xml = plan_to_urdf(plan, _model(tmp_path))
    element = ET.fromstring(xml).find("./joint[@name='shoulder']")
    assert element is not None
    assert element.attrib["type"] == "continuous"
    limit = element.find("limit")
    assert limit is not None
    assert "lower" not in limit.attrib
    assert limit.attrib["effort"] == "1"


def test_spherical_joint_requires_explicit_conservative_policy(tmp_path: Path) -> None:
    plan = _plan()
    joint = plan["articulation"]["joints"][0]  # type: ignore[index]
    joint["type"] = "spherical"
    joint.pop("limit")
    joint.pop("axis")
    with pytest.raises(URDFExportError, match="standard URDF cannot represent"):
        render_urdf(plan, _model(tmp_path))

    document = render_urdf(plan, _model(tmp_path), spherical_policy="fixed")
    element = ET.fromstring(document.xml).find("./joint[@name='shoulder']")
    assert element is not None
    assert element.attrib["type"] == "fixed"
    assert any("degraded spherical" in warning for warning in document.warnings)


def test_procedura_assembly_connectors_and_mates_are_supported(tmp_path: Path) -> None:
    plan = _plan()
    plan["parts"] = [{"id": "base"}, {"id": "door"}]
    plan["articulation"]["joints"] = []  # type: ignore[index]
    identity_frame = {
        "origin": [0.25, 0, 0.5],
        "x_axis": [1, 0, 0],
        "y_axis": [0, 1, 0],
        "z_axis": [0, 0, 1],
    }
    child_frame = {**identity_frame, "origin": [0.05, 0, 0.1]}
    plan["assembly"] = {
        "version": 1,
        "part_order": ["base", "door"],
        "connectors": [
            {"id": "base_hinge", "part_id": "base", "frame": identity_frame},
            {"id": "door_hinge", "part_id": "door", "frame": child_frame},
        ],
        "mates": [
            {
                "id": "door_joint",
                "type": "revolute",
                "parent_connector_id": "base_hinge",
                "child_connector_id": "door_hinge",
                "limits": {"lower": -1.57, "upper": 0},
            }
        ],
    }

    document = render_urdf(plan, _model(tmp_path))
    joint = ET.fromstring(document.xml).find("./joint[@name='door_joint']")
    assert joint is not None
    assert joint.attrib["type"] == "revolute"
    assert joint.find("parent").attrib["link"] == "base"  # type: ignore[union-attr]
    assert joint.find("child").attrib["link"] == "door"  # type: ignore[union-attr]
    # Link meshes use part-local coordinates, so the URDF joint origin is the
    # complete solved parent-part -> child-part transform at the mate rest pose.
    assert joint.find("origin").attrib["xyz"] == "0.2 0 0.4"  # type: ignore[union-attr]
    assert joint.find("axis").attrib["xyz"] == "0 0 1"  # type: ignore[union-attr]

    plan["assembly"]["mates"][0]["rest"] = 0.5  # type: ignore[index]
    plan["assembly"]["mates"][0]["limits"] = {  # type: ignore[index]
        "lower": -1.0,
        "upper": 1.0,
    }
    rested = ET.fromstring(render_urdf(plan, _model(tmp_path)).xml).find(
        "./joint[@name='door_joint']"
    )
    assert rested is not None
    assert rested.find("origin").attrib["rpy"] == "0 0 0.5"  # type: ignore[union-attr]
    assert rested.find("limit").attrib["lower"] == "-1.5"  # type: ignore[union-attr]
    assert rested.find("limit").attrib["upper"] == "0.5"  # type: ignore[union-attr]

    per_part = deepcopy(plan)
    connectors = per_part["assembly"].pop("connectors")  # type: ignore[index]
    per_part["parts"][0]["connectors"] = [connectors[0]]  # type: ignore[index]
    per_part["parts"][1]["connectors"] = [connectors[1]]  # type: ignore[index]
    per_part_joint = ET.fromstring(render_urdf(per_part, _model(tmp_path)).xml).find(
        "./joint[@name='door_joint']"
    )
    assert per_part_joint is not None
    assert per_part_joint.find("origin").attrib["rpy"] == "0 0 0.5"  # type: ignore[union-attr]


def test_host_solved_assembly_rejects_explicit_joint_overrides(tmp_path: Path) -> None:
    plan = _plan()
    plan["assembly"] = {"placement": "host-solved"}
    with pytest.raises(URDFExportError, match="must omit articulation.joints"):
        render_urdf(plan, _model(tmp_path))

    plan["articulation"]["joints"] = []  # type: ignore[index]
    plan["joints"] = [
        {
            "name": "unsafe_override",
            "parent": "base",
            "child": "arm",
            "type": "continuous",
            "origin": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},
            "axis": [0, 0, 1],
        }
    ]
    with pytest.raises(URDFExportError, match="must omit top-level joints"):
        render_urdf(plan, _model(tmp_path))


def test_rejects_invalid_connector_frames(tmp_path: Path) -> None:
    plan = _plan()
    plan["parts"] = [{"id": "base"}, {"id": "door"}]
    plan["articulation"]["joints"] = []  # type: ignore[index]
    plan["assembly"] = {
        "connectors": [
            {
                "id": "a",
                "part_id": "base",
                "frame": {
                    "origin": [0, 0, 0],
                    "x_axis": [1, 0, 0],
                    "y_axis": [1, 0, 0],
                    "z_axis": [0, 0, 1],
                },
            },
            {
                "id": "b",
                "part_id": "door",
                "frame": {
                    "origin": [0, 0, 0],
                    "x_axis": [1, 0, 0],
                    "y_axis": [0, 1, 0],
                    "z_axis": [0, 0, 1],
                },
            },
        ],
        "mates": [
            {
                "id": "bad_hinge",
                "type": "revolute",
                "parent_connector_id": "a",
                "child_connector_id": "b",
                "limits": {"lower": -1, "upper": 1},
            }
        ],
    }
    with pytest.raises(URDFExportError, match="orthogonal"):
        render_urdf(plan, _model(tmp_path))


def test_missing_model_and_bad_output_suffix_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(URDFExportError, match="does not exist"):
        render_urdf(_plan(), tmp_path / "missing.glb")

    model = _model(tmp_path)
    with pytest.raises(URDFExportError, match="end in .urdf"):
        export_urdf(_plan(), model, tmp_path / "robot.xml", enabled=True)


def test_pipeline_exports_host_solved_per_link_meshes(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.artifacts_dir.mkdir(parents=True)
    model = workspace.artifacts_dir / "model.glb"
    model.write_bytes(b"compiled-host-solved-model")
    frame = {
        "origin": [0.0, 0.0, 0.0],
        "x_axis": [1.0, 0.0, 0.0],
        "y_axis": [0.0, 1.0, 0.0],
        "z_axis": [0.0, 0.0, 1.0],
    }
    plan = {
        "subject": "hinge",
        "subject_kind": "object",
        "parts": [
            {"id": "base", "object_names": ["Base"]},
            {"id": "arm", "object_names": ["Arm"]},
        ],
        "assembly": {
            "version": 1,
            "placement": "host-solved",
            "part_order": ["base", "arm"],
            "connectors": [
                {
                    "id": "base_hinge",
                    "part_id": "base",
                    "interface": "cylindrical",
                    "role": "female",
                    "frame": frame,
                    "nominal_dimensions": {"diameter": 0.1},
                },
                {
                    "id": "arm_hinge",
                    "part_id": "arm",
                    "interface": "cylindrical",
                    "role": "male",
                    "frame": frame,
                    "nominal_dimensions": {"diameter": 0.1},
                },
            ],
            "mates": [
                {
                    "id": "hinge_joint",
                    "type": "revolute",
                    "parent_connector_id": "base_hinge",
                    "child_connector_id": "arm_hinge",
                    "fit": "clearance",
                    "clearance": 0.001,
                    "fit_offset": [0.0, 0.0, 0.0],
                    "nominal_dimensions": {"diameter": 0.1},
                    "rest": 0.0,
                    "limits": {"lower": -1.0, "upper": 1.0},
                }
            ],
        },
        "articulation": {
            "enabled": True,
            "mechanical": True,
            "joints": [],
        },
    }

    class FakeRuntime:
        stages: list[str] = []

        def run_stage(self, stage, arguments, *, cwd, timeout_s):
            self.stages.append(stage)
            assert stage == "export_urdf_parts"
            output = Path(arguments[arguments.index("--output-dir") + 1])
            output.mkdir(parents=True)
            records = []
            for part_id in ("base", "arm"):
                path = output / f"{part_id}.glb"
                path.write_bytes(f"mesh:{part_id}".encode())
                records.append(
                    {
                        "part_id": part_id,
                        "path": path.name,
                        "sha256": sha256(path),
                    }
                )
            write_json(
                output / "manifest.json",
                {
                    "schema_version": 1,
                    "source_sha256": sha256(model),
                    "parts": records,
                },
            )
            return SimpleNamespace(ok=True, stdout="ready", stderr="")

    runtime = FakeRuntime()
    report = pipeline._export_urdf_artifacts(
        workspace,
        runtime,  # type: ignore[arg-type]
        plan=plan,
        timeout_s=30,
    )

    assert runtime.stages == ["export_urdf_parts"]
    assert report["link_meshes"] == {
        "arm": "urdf_parts/arm.glb",
        "base": "urdf_parts/base.glb",
    }
    assert report["parts_manifest"] == "urdf_parts/manifest.json"
    assert (workspace.artifacts_dir / "urdf_parts" / "manifest.json").is_file()
    urdf = (workspace.artifacts_dir / "model.urdf").read_text(encoding="utf-8")
    assert 'filename="urdf_parts/base.glb"' in urdf
    assert 'filename="urdf_parts/arm.glb"' in urdf
