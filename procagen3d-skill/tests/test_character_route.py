import importlib.util
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "procagen3d.py"
SPEC = importlib.util.spec_from_file_location("procagen3d_driver", MODULE_PATH)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


LANDMARKS = [
    "head_top", "chin", "neck", "pelvis",
    "shoulder_left", "elbow_left", "wrist_left",
    "shoulder_right", "elbow_right", "wrist_right",
    "hip_left", "knee_left", "ankle_left",
    "hip_right", "knee_right", "ankle_right",
]


def valid_character_plan():
    transitions = []
    for joint, side in (
        ("neck", "center"),
        ("shoulder", "left"), ("shoulder", "right"),
        ("elbow", "left"), ("elbow", "right"),
        ("hip", "left"), ("hip", "right"),
        ("knee", "left"), ("knee", "right"),
    ):
        transition_id = f"{joint}_{side}"
        transitions.append({
            "id": transition_id,
            "joint": joint,
            "side": side,
            "mode": "continuous",
            "host_pattern": "BodyCore",
            "connects": ["BodyCore", "BodyCore"],
        })
    return {
        "version": 1,
        "archetype": "humanoid",
        "coverage": "full-body",
        "body_strategy": "continuous-skin",
        "rig_strategy": "segmented",
        "expected_eye_count": 2,
        "proportion_landmarks": [
            {
                "id": name,
                "pattern": f"Fit_{name}",
                "role": name.rsplit("_", 1)[0],
                "side": ("left" if name.endswith("_left") else
                         "right" if name.endswith("_right") else "center"),
            }
            for name in LANDMARKS
        ],
        "proportion_system": {
            "height_heads": 7.5,
            "source": "reference",
            "head_landmarks": ["head_top", "chin"],
        },
        "anatomy_chains": [
            {"id": "spine", "kind": "spine", "side": "center",
             "landmarks": ["pelvis", "neck", "chin", "head_top"]},
            {"id": "arm_left", "kind": "arm", "side": "left",
             "landmarks": ["shoulder_left", "elbow_left", "wrist_left"]},
            {"id": "arm_right", "kind": "arm", "side": "right",
             "landmarks": ["shoulder_right", "elbow_right", "wrist_right"]},
            {"id": "leg_left", "kind": "leg", "side": "left",
             "landmarks": ["hip_left", "knee_left", "ankle_left"]},
            {"id": "leg_right", "kind": "leg", "side": "right",
             "landmarks": ["hip_right", "knee_right", "ankle_right"]},
        ],
        "deformation_layers": [{
            "id": "body_core",
            "layer": "core_volume",
            "pattern": "BodyCore",
            "construction": "implicit-union",
            "crosses_joints": True,
        }],
        "joint_transitions": transitions,
        "facial_regions": [
            {"id": "cranium", "kind": "cranium", "pattern": "BodyCore"},
            {"id": "eyes", "kind": "eye", "pattern": "Eye_*"},
            {"id": "nose", "kind": "nose-muzzle", "pattern": "Nose"},
            {"id": "mouth", "kind": "mouth-jaw", "pattern": "Mouth"},
        ],
    }


def mesh(name, layer="surface_detail", construction="geometry"):
    return {
        "name": name,
        "type": "MESH",
        "parent": "CharacterRoot",
        "bbox_world_min": [-1.0, -0.2, 0.0],
        "bbox_world_max": [1.0, 0.2, 2.0],
        "dimensions": [2.0, 0.4, 2.0],
        "vertex_count": 24,
        "poly_count": 12,
        "base_vertex_count": 24,
        "base_poly_count": 12,
        "materials": ["Test Material"],
        "custom_props": {
            "procagen3d_character_layer": layer,
            "procagen3d_character_construction": construction,
        },
    }


def valid_graph(plan):
    transition_ids = [entry["id"] for entry in plan["joint_transitions"]]
    body = mesh("BodyCore", "core_volume", "implicit-union")
    body["custom_props"]["procagen3d_character_transitions"] = ",".join(
        transition_ids)
    objects = [{
        "name": "CharacterRoot",
        "type": "EMPTY",
        "parent": None,
        "custom_props": {
            "procagen3d_subject_domain": "character",
            "procagen3d_character_routine": "organic-v1",
        },
    }, body, mesh("Eye_Left"), mesh("Eye_Right"), mesh("Nose"), mesh("Mouth")]
    objects.extend({
        "name": f"Fit_{name}",
        "type": "EMPTY",
        "parent": "CharacterRoot",
        "custom_props": {},
    } for name in LANDMARKS)
    return {
        "roots": ["CharacterRoot"],
        "objects": objects,
        "totals": {"objects": len(objects), "meshes": 5, "triangles": 120},
        "joints": [],
        "world_bbox": {"size": [2.0, 0.4, 2.0]},
    }


def valid_fit_spec(plan):
    return {
        "landmarks": [{"id": name} for name in LANDMARKS],
        "pose": {"chains": [
            {"id": entry["id"]}
            for entry in plan["anatomy_chains"]
            if entry["kind"] != "spine"
        ]},
    }


class CharacterRouteTests(unittest.TestCase):
    def test_generic_detail_floors_are_unchanged(self):
        self.assertEqual(
            (420, 90000, 16),
            DRIVER.adaptive_detail_floor("showcase", "extreme", "object"),
        )
        self.assertEqual(
            (48, 60000, 10),
            DRIVER.adaptive_detail_floor("showcase", "extreme", "character"),
        )

    def test_character_plan_schema_accepts_full_humanoid(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "character_plan.json").write_text(
                json.dumps(valid_character_plan()))
            plan, errors = DRIVER.load_character_plan(directory, required=True)
        self.assertIsNotNone(plan)
        self.assertEqual([], errors)

    def test_shipped_character_plan_example_matches_schema(self):
        reference_dir = Path(__file__).resolve().parents[1] / "references"
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "character_plan.json").write_text(
                (reference_dir / "character-plan.example.json").read_text())
            plan, errors = DRIVER.load_character_plan(directory, required=True)
        self.assertIsNotNone(plan)
        self.assertEqual([], errors)

    def test_character_plan_requires_major_humanoid_transitions(self):
        plan = valid_character_plan()
        plan["joint_transitions"] = plan["joint_transitions"][:-1]
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "character_plan.json").write_text(json.dumps(plan))
            _, errors = DRIVER.load_character_plan(directory, required=True)
        self.assertTrue(any("missing humanoid joint transitions" in error
                            for error in errors))

    def test_character_contract_accepts_tagged_connected_surface(self):
        plan = valid_character_plan()
        failures = DRIVER.character_contract_failures(
            valid_graph(plan), plan, valid_fit_spec(plan))
        self.assertEqual([], failures)

    def test_full_check_dispatches_to_valid_character_contract(self):
        plan = valid_character_plan()
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "character_plan.json").write_text(json.dumps(plan))
            Path(directory, "scene_graph.json").write_text(
                json.dumps(valid_graph(plan)))
            args = SimpleNamespace(
                dir=directory,
                tier="quick",
                form="auto",
                subject="auto",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = DRIVER.cmd_check(args)
        self.assertEqual(0, result)

    def test_build_copies_character_sidecar_to_sibling_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory, "source")
            output_dir = Path(directory, "output")
            source_dir.mkdir()
            program = source_dir / "character.py"
            program.write_text("def build():\n    pass\n")
            sidecar = source_dir / "character_plan.json"
            sidecar.write_text(json.dumps(valid_character_plan()))
            args = SimpleNamespace(
                program=str(program), out=str(output_dir), size=64,
                engine="workbench", no_render=True, form_diagnostics=False,
                blender=None,
            )
            original = DRIVER.run_blender
            DRIVER.run_blender = lambda *_args, **_kwargs: 0
            try:
                result = DRIVER.cmd_build(args)
            finally:
                DRIVER.run_blender = original
            self.assertEqual(0, result)
            self.assertEqual(
                sidecar.read_text(),
                Path(output_dir, "character_plan.json").read_text(),
            )

    def test_character_contract_rejects_untagged_joint_host(self):
        plan = valid_character_plan()
        graph = valid_graph(plan)
        graph["objects"][1]["custom_props"].pop(
            "procagen3d_character_transitions")
        failures = DRIVER.character_contract_failures(
            graph, plan, valid_fit_spec(plan))
        self.assertIn("CHARACTER_TRANSITIONS", {tag for tag, _ in failures})

    def test_character_contract_rejects_construction_mismatch(self):
        plan = valid_character_plan()
        graph = valid_graph(plan)
        graph["objects"][1]["custom_props"][
            "procagen3d_character_construction"] = "sculpted-surface"
        failures = DRIVER.character_contract_failures(
            graph, plan, valid_fit_spec(plan))
        self.assertIn("CHARACTER_LAYERS", {tag for tag, _ in failures})

    def test_undeclared_assets_stay_on_object_route(self):
        domain, errors = DRIVER.resolve_subject_domain([], None, "auto")
        self.assertEqual("object", domain)
        self.assertEqual([], errors)

    def test_subject_domain_conflict_is_reported(self):
        root = [{
            "custom_props": {"procagen3d_subject_domain": "character"},
        }]
        domain, errors = DRIVER.resolve_subject_domain(
            root, {"subject_domain": "object"}, "auto")
        self.assertEqual("character", domain)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
