"""Registered image-fit stage: render the contracted camera and score gates."""

import json
import math
from pathlib import Path

import bpy

from .fit_io import (
    finite_values,
    fit_path,
    load_rgba,
    save_mask,
    save_rgba,
    sha256_file,
)
from .fit_measure import (
    bbox_iou,
    fit_camera,
    landmark_uv,
    mask_observation,
    projected_instance,
    reference_mask,
    uv_distance,
)
from .render import setup_engine
from .runtime import FAIL, OK, depsgraph, finish


def fit_gate(gates, gate_id, kind, target, measured, passed, note=""):
    gates.append({
        "id": str(gate_id),
        "kind": kind,
        "target": target,
        "measured": measured,
        "pass": bool(passed),
        "note": note,
    })


def draw_cross(image, uv, color, radius=4):
    height, width = image.shape[:2]
    x = int(round(uv[0] * (width - 1)))
    y = int(round(uv[1] * (height - 1)))
    if not (0 <= x < width and 0 <= y < height):
        return
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    image[y, x0:x1, :3] = color
    image[y0:y1, x, :3] = color


def stage_fit(args):
    import numpy as np

    out_dir = Path(args.out).resolve()
    blend = out_dir / "scene.blend"
    spec_path = Path(args.spec).resolve()
    renders_dir = out_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "fit_report.json"
    artifact_names = (
        "reference_match.png", "reference_overlay.png",
        "reference_mask.png", "render_mask.png",
    )
    report_path.unlink(missing_ok=True)
    for name in artifact_names:
        (renders_dir / name).unlink(missing_ok=True)

    base_report = {
        "procagen3d_fit_version": 1,
        "fit_spec": spec_path.name,
        "passed": False,
        "gates": [],
    }
    try:
        if not blend.is_file():
            raise ValueError(f"{blend} not found (run build first)")
        if not spec_path.is_file():
            raise ValueError(f"fit spec not found: {spec_path}")
        spec = json.loads(spec_path.read_text())
        if not isinstance(spec, dict) or spec.get("version") != 1:
            raise ValueError("fit spec must be a JSON object with version: 1")
        reference_path = fit_path(
            out_dir, spec.get("reference_image"), "reference_image")
        reference_rgba = load_rgba(reference_path)
        height, width = reference_rgba.shape[:2]
        input_hashes = {
            "fit_spec_sha256": sha256_file(spec_path),
            "reference_sha256": sha256_file(reference_path),
            "scene_graph_sha256": sha256_file(out_dir / "scene_graph.json"),
            "scene_blend_sha256": sha256_file(blend),
        }

        bpy.ops.wm.open_mainfile(filepath=str(blend))
        setup_engine(bpy.context.scene, args.engine)
        camera, normalized_camera = fit_camera(
            bpy.context.scene, spec.get("camera", {}), width, height)
        render_path = renders_dir / "reference_match.png"
        bpy.context.scene.render.filepath = str(render_path)
        bpy.ops.render.render(write_still=True)
        rendered_rgba = load_rgba(render_path)
        if rendered_rgba.shape[:2] != reference_rgba.shape[:2]:
            raise ValueError("registered render dimensions do not match reference")

        thresholds = spec.get("thresholds", {})
        if not isinstance(thresholds, dict):
            raise ValueError("thresholds must be an object")
        gates = []
        mask_config = spec.get("mask")
        reference_fg = None
        render_fg = rendered_rgba[..., 3] > 0.5
        overlay = reference_rgba.copy()
        overlay[..., 3] = 1.0
        if mask_config is not None:
            if not isinstance(mask_config, dict):
                raise ValueError("mask must be an object")
            reference_fg, mask_source = reference_mask(
                reference_rgba, mask_config, out_dir)
            if not np.any(render_fg):
                raise ValueError("registered render mask is empty")
            ref_obs = mask_observation(reference_fg)
            render_obs = mask_observation(render_fg)
            intersection = np.logical_and(reference_fg, render_fg).sum()
            union = np.logical_or(reference_fg, render_fg).sum()
            iou = float(intersection / union) if union else 0.0
            bbox_error = max(abs(a - b) for a, b in zip(
                ref_obs["bbox_uv"], render_obs["bbox_uv"]))
            centroid_error = math.hypot(*(
                ref_obs["centroid_uv"][index] - render_obs["centroid_uv"][index]
                for index in range(2)))
            area_ratio_error = abs(
                render_obs["area_fraction"] / ref_obs["area_fraction"] - 1.0)
            min_iou = float(mask_config.get(
                "min_iou", thresholds.get("mask_min_iou", 0.70)))
            max_bbox = float(mask_config.get(
                "max_bbox_error", thresholds.get("bbox_max_error", 0.05)))
            max_centroid = float(mask_config.get(
                "max_centroid_error", thresholds.get("centroid_max_error", 0.04)))
            max_area = float(mask_config.get(
                "max_area_ratio_error", thresholds.get("area_ratio_max_error", 0.25)))
            fit_gate(gates, "mask_iou", "mask", f">= {min_iou:.4f}",
                     round(iou, 6), iou >= min_iou)
            fit_gate(gates, "mask_bbox", "mask", f"<= {max_bbox:.4f}",
                     round(bbox_error, 6), bbox_error <= max_bbox)
            fit_gate(gates, "mask_centroid", "mask", f"<= {max_centroid:.4f}",
                     round(centroid_error, 6), centroid_error <= max_centroid)
            fit_gate(gates, "mask_area_ratio", "mask", f"<= {max_area:.4f}",
                     round(area_ratio_error, 6), area_ratio_error <= max_area)
            save_mask(renders_dir / "reference_mask.png", reference_fg)
            save_mask(renders_dir / "render_mask.png", render_fg)
            overlap = np.logical_and(reference_fg, render_fg)
            reference_only = np.logical_and(reference_fg, ~render_fg)
            render_only = np.logical_and(render_fg, ~reference_fg)
            overlay[overlap, :3] = 0.45 * overlay[overlap, :3] + np.array(
                [0.1, 0.9, 0.2], dtype=np.float32) * 0.55
            overlay[reference_only, :3] = 0.35 * overlay[reference_only, :3] + np.array(
                [1.0, 0.1, 0.1], dtype=np.float32) * 0.65
            overlay[render_only, :3] = 0.35 * overlay[render_only, :3] + np.array(
                [0.1, 0.35, 1.0], dtype=np.float32) * 0.65
            base_report["mask"] = {
                "source": mask_source,
                "reference": ref_obs,
                "render": render_obs,
            }
            if mask_source == "file":
                supplied_mask_path = fit_path(
                    out_dir, mask_config.get("path"), "mask.path")
                input_hashes["mask_sha256"] = sha256_file(supplied_mask_path)
                base_report["mask"]["path"] = supplied_mask_path.name

        dg = depsgraph()
        landmark_records = []
        landmark_map = {}
        default_landmark_error = float(thresholds.get("landmark_max_error", 0.04))
        landmarks = spec.get("landmarks", [])
        if not isinstance(landmarks, list):
            raise ValueError("landmarks must be a list")
        for index, entry in enumerate(landmarks):
            if not isinstance(entry, dict):
                raise ValueError(f"landmarks[{index}] must be an object")
            landmark_id = str(entry.get("id", f"landmark_{index + 1}"))
            if landmark_id in landmark_map:
                raise ValueError(f"duplicate landmark id: {landmark_id}")
            reference_uv = finite_values(
                entry.get("reference_uv"), 2, f"landmark {landmark_id}.reference_uv")
            try:
                render_uv = landmark_uv(bpy.context.scene, camera, dg, entry)
                error = math.hypot(
                    reference_uv[0] - render_uv[0], reference_uv[1] - render_uv[1])
                record = {
                    "id": landmark_id,
                    "reference_uv": reference_uv,
                    "render_uv": render_uv,
                    "error": error,
                }
                landmark_map[landmark_id] = record
                if bool(entry.get("gate", True)):
                    maximum = float(entry.get("max_error", default_landmark_error))
                    fit_gate(gates, landmark_id, "landmark", f"<= {maximum:.4f}",
                             round(error, 6), error <= maximum)
                draw_cross(overlay, reference_uv, (1.0, 0.85, 0.05))
                draw_cross(overlay, render_uv, (0.0, 0.95, 1.0))
            except ValueError as exc:
                record = {"id": landmark_id, "reference_uv": reference_uv,
                          "error": str(exc)}
                fit_gate(gates, landmark_id, "landmark", "resolvable",
                         "unmeasurable", False, str(exc))
            landmark_records.append(record)
        base_report["landmarks"] = landmark_records

        ratio_records = []
        ratios = spec.get("ratios", [])
        if not isinstance(ratios, list):
            raise ValueError("ratios must be a list")
        for index, entry in enumerate(ratios):
            if not isinstance(entry, dict):
                raise ValueError(f"ratios[{index}] must be an object")
            ratio_id = str(entry.get("id", f"ratio_{index + 1}"))
            numerator = entry.get("numerator")
            denominator = entry.get("denominator")
            try:
                if (not isinstance(numerator, list) or len(numerator) != 2
                        or not isinstance(denominator, list) or len(denominator) != 2):
                    raise ValueError("numerator and denominator need two landmark ids")
                points = [landmark_map.get(str(item))
                          for item in numerator + denominator]
                if any(point is None for point in points):
                    raise ValueError("ratio references an unresolved landmark")
                axis = str(entry.get("axis", "distance"))
                ref_num = uv_distance(points[0]["reference_uv"],
                                      points[1]["reference_uv"], axis)
                ref_den = uv_distance(points[2]["reference_uv"],
                                      points[3]["reference_uv"], axis)
                got_num = uv_distance(points[0]["render_uv"],
                                      points[1]["render_uv"], axis)
                got_den = uv_distance(points[2]["render_uv"],
                                      points[3]["render_uv"], axis)
                if ref_den <= 1e-9 or got_den <= 1e-9 or ref_num <= 1e-9:
                    raise ValueError("ratio contains a zero-length segment")
                target_ratio = ref_num / ref_den
                measured_ratio = got_num / got_den
                relative_error = abs(measured_ratio / target_ratio - 1.0)
                maximum = float(entry.get(
                    "max_relative_error", thresholds.get("ratio_max_relative_error", 0.10)))
                record = {
                    "id": ratio_id,
                    "target": target_ratio,
                    "measured": measured_ratio,
                    "relative_error": relative_error,
                }
                fit_gate(gates, ratio_id, "ratio", f"relative error <= {maximum:.4f}",
                         round(relative_error, 6), relative_error <= maximum,
                         f"target={target_ratio:.4f}, measured={measured_ratio:.4f}")
            except ValueError as exc:
                record = {"id": ratio_id, "error": str(exc)}
                fit_gate(gates, ratio_id, "ratio", "measurable", "unmeasurable",
                         False, str(exc))
            ratio_records.append(record)
        base_report["ratios"] = ratio_records

        instance_records = []
        instance_map = {}
        instances = spec.get("instances", [])
        if not isinstance(instances, list):
            raise ValueError("instances must be a list")
        for index, entry in enumerate(instances):
            if not isinstance(entry, dict):
                raise ValueError(f"instances[{index}] must be an object")
            instance_id = str(entry.get("id", f"instance_{index + 1}"))
            if instance_id in instance_map:
                raise ValueError(f"duplicate instance id: {instance_id}")
            pattern = entry.get("pattern")
            reference_bbox = finite_values(
                entry.get("reference_bbox_uv"), 4,
                f"instance {instance_id}.reference_bbox_uv")
            if reference_bbox[0] >= reference_bbox[2] or reference_bbox[1] >= reference_bbox[3]:
                raise ValueError(f"instance {instance_id} has an invalid reference bbox")
            reference_centroid = finite_values(
                entry.get("reference_centroid_uv", [
                    (reference_bbox[0] + reference_bbox[2]) / 2.0,
                    (reference_bbox[1] + reference_bbox[3]) / 2.0,
                ]), 2, f"instance {instance_id}.reference_centroid_uv")
            try:
                observation = projected_instance(
                    bpy.context.scene, camera, dg, str(pattern))
                observation.update({
                    "id": instance_id,
                    "reference_bbox_uv": reference_bbox,
                    "reference_centroid_uv": reference_centroid,
                })
                bbox_error = max(abs(a - b) for a, b in zip(
                    reference_bbox, observation["bbox_uv"]))
                centroid_error = math.hypot(
                    reference_centroid[0] - observation["centroid_uv"][0],
                    reference_centroid[1] - observation["centroid_uv"][1])
                observation["bbox_error"] = bbox_error
                observation["centroid_error"] = centroid_error
                max_bbox = float(entry.get(
                    "max_bbox_error", thresholds.get("instance_bbox_max_error", 0.05)))
                max_centroid = float(entry.get(
                    "max_centroid_error", thresholds.get(
                        "instance_centroid_max_error", 0.04)))
                fit_gate(gates, f"{instance_id}.bbox", "instance",
                         f"<= {max_bbox:.4f}", round(bbox_error, 6),
                         bbox_error <= max_bbox)
                fit_gate(gates, f"{instance_id}.centroid", "instance",
                         f"<= {max_centroid:.4f}", round(centroid_error, 6),
                         centroid_error <= max_centroid)
                instance_map[instance_id] = observation
            except ValueError as exc:
                observation = {"id": instance_id, "pattern": pattern,
                               "reference_bbox_uv": reference_bbox,
                               "reference_centroid_uv": reference_centroid,
                               "error": str(exc)}
                fit_gate(gates, instance_id, "instance", "resolvable",
                         "unmeasurable", False, str(exc))
            instance_records.append(observation)
        base_report["instances"] = instance_records

        relation_records = []
        relations = spec.get("relations", [])
        if not isinstance(relations, list):
            raise ValueError("relations must be a list")
        for index, entry in enumerate(relations):
            if not isinstance(entry, dict):
                raise ValueError(f"relations[{index}] must be an object")
            relation_id = str(entry.get("id", f"relation_{index + 1}"))
            relation_type = str(entry.get("type", "relative_position"))
            try:
                if relation_type == "depth_order":
                    front_id, behind_id = str(entry.get("front")), str(entry.get("behind"))
                    front = instance_map.get(front_id)
                    behind = instance_map.get(behind_id)
                    if front is None or behind is None:
                        raise ValueError("depth_order references an unresolved instance")
                    margin = float(entry.get("min_margin_m", 0.0))
                    measured = behind["camera_depth_m"] - front["camera_depth_m"]
                    passed = measured > margin
                    target = f"> {margin:.4f} m"
                    note = f"{front_id} in front of {behind_id}"
                else:
                    a_id, b_id = str(entry.get("a")), str(entry.get("b"))
                    a = instance_map.get(a_id)
                    b = instance_map.get(b_id)
                    if a is None or b is None:
                        raise ValueError("relation references an unresolved instance")
                    maximum = float(entry.get(
                        "max_error", thresholds.get("relation_max_error", 0.05)))
                    if relation_type == "relative_position":
                        target_delta = [
                            b["reference_centroid_uv"][axis]
                            - a["reference_centroid_uv"][axis] for axis in range(2)]
                        render_delta = [
                            b["centroid_uv"][axis] - a["centroid_uv"][axis]
                            for axis in range(2)]
                        measured = math.hypot(
                            target_delta[0] - render_delta[0],
                            target_delta[1] - render_delta[1])
                        note = f"target_delta={target_delta}, render_delta={render_delta}"
                    elif relation_type == "bbox_iou":
                        target_iou = bbox_iou(
                            a["reference_bbox_uv"], b["reference_bbox_uv"])
                        render_iou = bbox_iou(a["bbox_uv"], b["bbox_uv"])
                        measured = abs(target_iou - render_iou)
                        note = f"target_iou={target_iou:.4f}, render_iou={render_iou:.4f}"
                    else:
                        raise ValueError(
                            "relation.type must be relative_position, bbox_iou, or depth_order")
                    passed = measured <= maximum
                    target = f"error <= {maximum:.4f}"
                record = {
                    "id": relation_id,
                    "type": relation_type,
                    "measured": measured,
                    "pass": passed,
                    "note": note,
                }
                fit_gate(gates, relation_id, "relation", target,
                         round(measured, 6), passed, note)
            except ValueError as exc:
                record = {"id": relation_id, "type": relation_type,
                          "error": str(exc)}
                fit_gate(gates, relation_id, "relation", "measurable",
                         "unmeasurable", False, str(exc))
            relation_records.append(record)
        base_report["relations"] = relation_records

        if not gates:
            raise ValueError(
                "fit spec defines no gates; add mask, landmarks, instances, or relations")
        save_rgba(renders_dir / "reference_overlay.png", overlay)
        passed_count = sum(1 for gate in gates if gate["pass"])
        base_report.update({
            "reference_image": reference_path.name,
            "camera": normalized_camera,
            "gates": gates,
            "summary": {
                "passed": passed_count,
                "total": len(gates),
                "failures": len(gates) - passed_count,
            },
            "passed": passed_count == len(gates),
            "inputs": input_hashes,
        })
        report_path.write_text(json.dumps(base_report, indent=2))
        print(f"ProcAgen3D fit — {reference_path.name}")
        for gate in gates:
            verdict = "PASS" if gate["pass"] else "FAIL"
            note = f"  ({gate['note']})" if gate.get("note") else ""
            print(f"  {gate['id']:<28} target {str(gate['target']):<24} "
                  f"measured {str(gate['measured']):>12}  {verdict}{note}")
        print(f"  -> {passed_count}/{len(gates)} fit gates passed")
        if passed_count != len(gates):
            print(f"{FAIL}:REFERENCE_FIT] {len(gates) - passed_count} fit gate(s) failed")
            finish(1)
        print(f"{OK} registered reference fit passed; overlay -> "
              f"{renders_dir / 'reference_overlay.png'}")
        finish(0)
    except Exception as exc:
        base_report["error"] = str(exc)
        report_path.write_text(json.dumps(base_report, indent=2))
        print(f"{FAIL}:FIT_SPEC] {exc}")
        finish(1)
