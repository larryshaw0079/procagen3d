#!/usr/bin/env python3
"""ProcAgen3D driver — code-native 3D asset generation (arXiv:2607.22738).

Pure Python 3.10+ stdlib. Blender-side stages are dispatched to
blender_stages.py running under Blender's bundled Python; everything else
(check/score/guard/edit-gates) runs under plain python3.

Subcommands:
    build      <program.py> --out DIR    build, export GLB, render views
    render     <dir>                     re-render canonical views
    fit        <dir> --spec FILE         registered image/pose-fit gates
    check      <dir>                     deterministic scene-graph gates
    joints     <dir>                     validate articulation (Blender)
    score      <dir> --spec FILE         measure constraints against spec
    guard      <old.py> <new.py>         doctrine guard for repair iterations
    edit-gates <base_dir> <edited_dir> --target PATTERN
                                         deterministic local-edit gates

Exit code 0 = pass, 1 = at least one failure (read the printed reasons).
"""

import argparse
import difflib
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OK = "[PROCAGEN3D:OK]"


def warn(tag, msg):
    print(f"[PROCAGEN3D:WARN:{tag}] {msg}")


def fail(tag, msg):
    print(f"[PROCAGEN3D:FAIL:{tag}] {msg}")


# ---------------------------------------------------------------- blender glue

def find_blender(explicit=None):
    candidates = [explicit, os.environ.get("PROCAGEN3D_BLENDER"), shutil.which("blender")]
    for cand in candidates:
        if cand and Path(cand).exists():
            return str(cand)
    for hit in sorted(Path.home().glob(".cache/procagen3d/*/blender")):
        if hit.is_file():
            return str(hit)
    sys.exit("ProcAgen3D: Blender not found. Set PROCAGEN3D_BLENDER=/path/to/blender, "
             "put blender on PATH, or install under ~/.cache/procagen3d/.")


def _fatal_signal(code):
    """POSIX signal number if `code` is a signal death, else None."""
    if code is None:
        return None
    if code < 0:
        return -code
    if 128 < code < 160:
        return code - 128
    return None


def _crash_log_from_output(output):
    for line in output.splitlines():
        if line.startswith("Writing:") and ".crash.txt" in line:
            path = Path(line.split("Writing:", 1)[1].strip())
            if path.is_file():
                return path
    return None


def _metal_startup_crash(crash_path):
    if crash_path is None:
        return False
    try:
        text = crash_path.read_text(errors="replace")
    except OSError:
        return False
    return ("supports_barycentric_whitelist" in text
            or "metal_is_supported" in text
            or "GPU_backend_type_selection_detect" in text)


def run_blender(stage_args, blender=None):
    blender_bin = find_blender(blender)
    cmd = [
        blender_bin, "--background", "--factory-startup",
        "--python-exit-code", "1",
        "--python", str(SCRIPT_DIR / "blender_stages.py"), "--", *stage_args,
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    output = proc.stdout or ""
    code = proc.returncode
    for line in output.splitlines():
        m = re.match(r"^PROCAGEN3D_EXIT:(\d+)$", line.strip())
        if m:
            code = int(m.group(1))
        elif line.strip() and not line.startswith(
                ("Blender ", "Read prefs", "Time:", "Saved:", "Info:",
                 "Fra:", "Read blend:", "WARN (gpu", "INFO ",
                 "Color management")) and "| INFO" not in line:
            print(line)
    wrote_crash = "Writing:" in output and ".crash.txt" in output
    sig = _fatal_signal(code)
    if sig is not None or wrote_crash:
        crash_path = _crash_log_from_output(output)
        version = "Blender"
        for line in output.splitlines():
            if line.startswith("Blender "):
                version = line.strip()
                break
        if sig == 11 or _metal_startup_crash(crash_path):
            fail(
                "BLENDER_CRASH",
                f"{version} ({blender_bin}) SIGSEGV'd during Metal GPU "
                "detection, before any ProcAgen3D Python ran. On macOS this "
                "is a Blender 5.x bug when the process cannot see the GPU "
                "(typical of an agent sandbox). Re-run this command with "
                "full/unsandboxed permissions so Metal can initialize — you "
                "do not need to downgrade to 4.x. "
                f"Crash log: {crash_path or '(none)'}")
            return 1
        fail(
            "BLENDER_CRASH",
            f"{version} ({blender_bin}) aborted"
            + (f" with signal {sig}" if sig else f" with exit {code}")
            + (f"; crash log: {crash_path}" if crash_path else ""))
        return 1
    return code


def cmd_build(args):
    program = Path(args.program)
    if not program.exists():
        sys.exit(f"ProcAgen3D: program not found: {program}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = out / "program.py"
    if program.resolve() != kept.resolve():
        shutil.copyfile(program, kept)
    stage = ["build", "--program", str(program), "--out", str(out),
             "--size", str(args.size), "--engine", args.engine]
    if args.no_render:
        stage.append("--no-render")
    if args.form_diagnostics:
        stage.append("--form-diagnostics")
    return run_blender(stage, args.blender)


def cmd_render(args):
    stage = ["render", "--out", args.dir, "--size", str(args.size),
             "--engine", args.engine]
    if args.form_diagnostics:
        stage.append("--form-diagnostics")
    return run_blender(stage, args.blender)


def cmd_fit(args):
    source = Path(args.spec)
    if not source.is_file():
        sys.exit(f"ProcAgen3D: fit spec not found: {source}")
    out = Path(args.dir)
    if not (out / "scene.blend").is_file():
        sys.exit(f"ProcAgen3D: {out / 'scene.blend'} not found (run build first)")
    kept = out / "fit_spec.json"
    if source.resolve() != kept.resolve():
        shutil.copyfile(source, kept)
        try:
            spec = json.loads(source.read_text())
        except json.JSONDecodeError as exc:
            sys.exit(f"ProcAgen3D: invalid fit spec JSON: {exc}")
        if not isinstance(spec, dict):
            sys.exit("ProcAgen3D: fit spec must be a JSON object")

        def copy_dependency(relative, label):
            if not isinstance(relative, str) or not relative:
                return
            rel = Path(relative)
            if rel.is_absolute() or ".." in rel.parts:
                sys.exit(f"ProcAgen3D: {label} must be a safe relative path")
            dependency = source.parent / rel
            if not dependency.is_file():
                sys.exit(f"ProcAgen3D: {label} not found beside fit spec: "
                         f"{dependency}")
            destination = out / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            if dependency.resolve() != destination.resolve():
                shutil.copyfile(dependency, destination)

        copy_dependency(spec.get("reference_image"), "reference_image")
        mask = spec.get("mask")
        if isinstance(mask, dict) and str(mask.get("source", "")).lower() == "file":
            copy_dependency(mask.get("path"), "mask.path")
        if spec.get("version") == 2:
            plan = source.parent / "reconstruction_plan.json"
            if not plan.is_file():
                sys.exit("ProcAgen3D: fit spec version 2 requires "
                         "reconstruction_plan.json beside the fit spec")
            plan_destination = out / "reconstruction_plan.json"
            if plan.resolve() != plan_destination.resolve():
                shutil.copyfile(plan, plan_destination)
    stage = ["fit", "--out", str(out), "--spec", str(kept),
             "--engine", args.engine]
    return run_blender(stage, args.blender)


def cmd_solve_camera(args):
    out = Path(args.dir)
    spec = Path(args.spec)
    if not spec.is_file():
        sys.exit(f"ProcAgen3D: fit spec not found: {spec}")
    if not (out / "scene.blend").is_file():
        sys.exit(f"ProcAgen3D: {out / 'scene.blend'} not found (run build first)")
    kept = out / "fit_spec.json"
    if spec.resolve() != kept.resolve():
        shutil.copyfile(spec, kept)
    stage = ["solve-camera", "--out", str(out), "--spec", str(kept),
             "--max-rms", str(args.max_rms)]
    if args.free_shift:
        stage.append("--free-shift")
    if args.solve_root:
        stage.append("--solve-root")
    if args.fix:
        stage.extend(["--fix", *args.fix])
    return run_blender(stage, args.blender)


def cmd_joints(args):
    stage = ["joints", "--out", args.dir]
    if args.strict:
        stage.append("--strict")
    return run_blender(stage, args.blender)


# ---------------------------------------------------------------- scene graph

def load_graph(dir_path):
    path = Path(dir_path) / "scene_graph.json"
    if not path.exists():
        sys.exit(f"ProcAgen3D: {path} not found (run build first)")
    return json.loads(path.read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# How far past its threshold a gate may sit and still count as "approximate"
# rather than "wrong".  Minimum-style gates (IoU) are judged on the absolute
# shortfall; maximum-style gates (errors, angles) on the ratio.
CAMERA_SOLVE_MAX_RMS = 0.02
NEAR_MISS_MAX_SHARE = 0.25
NEAR_MISS_IOU_SHORTFALL = 0.08
NEAR_MISS_ERROR_MULTIPLE = 2.0
TARGET_RE = re.compile(r"(>=|<=|>|<)\s*([0-9.]+)")


def within_near_miss(gate):
    """True when a failing gate is close enough to its threshold to be residue."""
    match = TARGET_RE.search(str(gate.get("target", "")))
    measured = gate.get("measured")
    if not match or not isinstance(measured, (int, float)):
        return False
    operator, threshold = match.group(1), float(match.group(2))
    if operator.startswith(">"):
        return measured >= threshold - NEAR_MISS_IOU_SHORTFALL
    if threshold <= 1e-9:
        return measured <= NEAR_MISS_ERROR_MULTIPLE * 1e-9
    return measured <= threshold * NEAR_MISS_ERROR_MULTIPLE


def image_fit_errors(dir_path):
    """Validate required, passing, hash-bound fit evidence for image inputs.

    Returns (errors, warnings); a warning is only ever issued for a documented
    single-view shortfall.
    """
    root = Path(dir_path)
    references = sorted(root.glob("reference_[0-9][0-9].*"))
    if not references:
        return [], []
    spec_path = root / "fit_spec.json"
    report_path = root / "fit_report.json"
    errors = []
    warnings = []
    if not spec_path.is_file():
        errors.append("fit_spec.json missing")
    if not report_path.is_file():
        errors.append("fit_report.json missing (run `procagen3d fit`)")
    if errors:
        return errors, warnings
    try:
        spec = json.loads(spec_path.read_text())
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid fit evidence JSON: {exc}"], warnings
    if not isinstance(spec, dict) or not isinstance(report, dict):
        return ["fit spec and report must be JSON objects"], warnings
    reference_name = spec.get("reference_image")
    reference_path = root / reference_name if isinstance(reference_name, str) else None
    if reference_path is None or not reference_path.is_file():
        errors.append(f"fit reference is missing: {reference_name!r}")
    # Version 1 predates local silhouette, pose, and shape-prior evidence.
    # Leaving it available let an image-conditioned asset opt out of every
    # reconstruction gate by declaring the older schema.
    if spec.get("version") != 2:
        errors.append(
            f"image-conditioned assets require fit_spec version 2 (found "
            f"{spec.get('version')!r}) — version 1 has no local silhouette, "
            "pose, or shape-prior contract")
    if not (root / "reconstruction_plan.json").is_file():
        errors.append("image-conditioned assets require reconstruction_plan.json")
    if not report.get("passed"):
        summary = report.get("summary", {})
        message = (f"registered fit did not pass ({summary.get('passed', 0)}/"
                   f"{summary.get('total', '?')} gates)")
        # From a single view some residual is the input's fault, not the
        # model's, and refusing to deliver anything serves nobody.  The escape
        # is available only in exchange for writing down exactly what is wrong.
        policy = report.get("threshold_policy") or {}
        gates = report.get("gates") or []
        failed = [gate for gate in gates if not gate.get("pass")]
        failed_ids = [str(gate.get("id")) for gate in failed]
        structural = {"threshold_policy", "landmark_provenance"}
        # The escape exists for the residue of an ill-posed problem, not for a
        # wrong model.  A reconstruction that misses most of its gates, or
        # misses any one of them by a wide margin, is not "approximate" — it is
        # incorrect, and documenting that in prose does not make it shippable.
        share = len(failed) / len(gates) if gates else 1.0
        blowouts = [f"{gate['id']} {gate.get('measured')} vs {gate.get('target')}"
                    for gate in failed
                    if not within_near_miss(gate)]
        if structural.intersection(failed_ids):
            errors.append(message + " — an integrity failure "
                          f"({sorted(structural.intersection(failed_ids))}) is "
                          "never an evidence limitation")
        elif not policy.get("single_view_reconstruction"):
            errors.append(message)
        elif share > NEAR_MISS_MAX_SHARE:
            errors.append(
                message + f" — {share:.0%} of gates failed, over the "
                f"{NEAR_MISS_MAX_SHARE:.0%} ceiling for a documented "
                "single-view approximation. This is a wrong reconstruction, "
                "not depth ambiguity; fix it or report that it cannot be built")
        elif blowouts:
            errors.append(
                message + " — these miss by too much to call approximate: "
                + "; ".join(blowouts[:5]))
        else:
            note = Path(dir_path) / "limitations.md"
            text = note.read_text() if note.is_file() else ""
            missing = [gate for gate in failed_ids if gate not in text]
            if missing:
                errors.append(
                    message + " — single-view input allows delivering this as "
                    "approximate, but only with limitations.md naming every "
                    f"failing gate; absent from it: {missing[:8]}")
            else:
                warnings.append(
                    message + "; accepted as an approximate single-view "
                    "reconstruction, documented in limitations.md")
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        errors.append("fit report has no input hashes")
        return errors, warnings
    expected = {
        "fit_spec_sha256": spec_path,
        "scene_graph_sha256": root / "scene_graph.json",
        "scene_blend_sha256": root / "scene.blend",
    }
    if reference_path is not None and reference_path.is_file():
        expected["reference_sha256"] = reference_path
    mask = spec.get("mask")
    if isinstance(mask, dict) and str(mask.get("source", "auto")).lower() == "file":
        mask_name = mask.get("path")
        mask_path = root / mask_name if isinstance(mask_name, str) else None
        if mask_path is None or not mask_path.is_file():
            errors.append(f"fit mask is missing: {mask_name!r}")
        else:
            expected["mask_sha256"] = mask_path
    for key, path in expected.items():
        if inputs.get(key) != sha256_file(path):
            errors.append(f"stale fit evidence: {key} does not match {path.name}")
    return errors, warnings


def children_map(graph):
    kids = {}
    for obj in graph["objects"]:
        if obj["parent"]:
            kids.setdefault(obj["parent"], []).append(obj["name"])
    return kids


def subtree(names, kids):
    out = set(names)
    stack = list(names)
    while stack:
        for child in kids.get(stack.pop(), []):
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


def match_objects(graph, pattern, types=None):
    return [o for o in graph["objects"]
            if fnmatch.fnmatchcase(o["name"], pattern)
            and (types is None or o["type"] in types)]


# ---------------------------------------------------------------- check gates

DEFAULT_NAME_RE = re.compile(
    r"^(Cube|Sphere|Icosphere|IcoSphere|Cylinder|Cone|Torus|Plane|Circle|Grid|"
    r"Monkey|Suzanne|Text|Mesh|Empty|Object|Curve|BezierCurve|BezierCircle)"
    r"(\.\d+)?$")
DUPE_SUFFIX_RE = re.compile(r"\.\d{3}$")
JOINT_TYPES = ("revolute", "prismatic", "fixed")
FORM_TOPOLOGIES = {"continuous", "shell", "assembled", "strand", "relief"}
FORM_METHODS = {
    "loft", "sweep", "revolve", "subdivision", "surface-grid", "nurbs",
    "curve", "solidify", "profile-extrude", "primitive-csg", "boolean",
    "analytic-primitive", "decal",
}
TOPOLOGY_METHODS = {
    "continuous": {"loft", "sweep", "revolve", "subdivision",
                   "surface-grid", "nurbs", "analytic-primitive"},
    "shell": {"loft", "sweep", "subdivision", "surface-grid", "nurbs",
              "solidify"},
    "assembled": {"primitive-csg", "profile-extrude", "boolean", "revolve",
                  "analytic-primitive"},
    "strand": {"sweep", "curve", "nurbs"},
    "relief": {"curve", "profile-extrude", "primitive-csg", "boolean",
               "decal"},
}

SHAPE_FAMILIES = {
    "box", "prism", "cylinder", "cone", "sphere", "ellipsoid", "capsule",
    "revolve", "loft", "sweep", "surface-grid", "shell", "strand",
}
SHAPE_FAMILY_METHODS = {
    "box": {"primitive-csg", "profile-extrude", "boolean"},
    "prism": {"primitive-csg", "profile-extrude", "boolean"},
    "cylinder": {"primitive-csg", "revolve", "boolean",
                 "analytic-primitive"},
    "cone": {"primitive-csg", "revolve", "analytic-primitive"},
    "sphere": {"analytic-primitive", "revolve", "subdivision"},
    "ellipsoid": {"analytic-primitive", "loft", "subdivision"},
    "capsule": {"analytic-primitive", "loft", "sweep", "subdivision"},
    "revolve": {"revolve"},
    "loft": {"loft"},
    "sweep": {"sweep"},
    "surface-grid": {"surface-grid", "subdivision", "nurbs"},
    "shell": {"solidify", "surface-grid", "loft", "subdivision", "nurbs"},
    "strand": {"curve", "sweep", "nurbs"},
}
COMPLEXITY_BANDS = {
    "simple": (1, 7),
    "moderate": (8, 15),
    "complex": (16, 27),
    "extreme": (28, None),
}
ADAPTIVE_DETAIL_FLOORS = {
    "standard": {
        "simple": (24, 5000, 4),
        "moderate": (50, 10000, 6),
        "complex": (100, 20000, 8),
        "extreme": (160, 32000, 10),
    },
    "showcase": {
        "simple": (60, 12000, 8),
        "moderate": (140, 28000, 10),
        "complex": (260, 55000, 14),
        "extreme": (420, 90000, 16),
    },
}
DETAIL_PRIORITIES = {"identity", "structural", "secondary", "micro", "inferred"}
OBJECT_REGIONS = {
    "top-left", "top-center", "top-right",
    "middle-left", "middle-center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
}
MIN_OCCUPIED_REGIONS = {
    "simple": 1,
    "moderate": 3,
    "complex": 5,
    "extreme": 6,
}


def load_reconstruction_plan(dir_path):
    """Load the v2 image-reconstruction contract, when present/required."""
    root = Path(dir_path)
    path = root / "reconstruction_plan.json"
    # Any image-conditioned asset needs the plan, not only one that happens to
    # declare a version-2 fit spec.
    required = bool(sorted(root.glob("reference_[0-9][0-9].*")))
    fit_path = root / "fit_spec.json"
    if not required and fit_path.is_file():
        try:
            fit_spec = json.loads(fit_path.read_text())
            required = (isinstance(fit_spec, dict)
                        and fit_spec.get("version") == 2)
        except (OSError, json.JSONDecodeError):
            pass
    if not path.is_file():
        return None, (["reconstruction_plan.json missing"] if required else [])
    try:
        plan = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"invalid reconstruction_plan.json: {exc}"]
    errors = []
    if not isinstance(plan, dict):
        return None, ["reconstruction plan must be a JSON object"]
    if plan.get("version") != 1:
        errors.append("reconstruction plan must be a JSON object with version: 1")
    complexity = plan.get("complexity")
    if not isinstance(complexity, dict):
        errors.append("complexity must be an object")
    else:
        complexity_class = complexity.get("class")
        if complexity_class not in COMPLEXITY_BANDS:
            errors.append(
                "complexity.class must be simple, moderate, complex, or extreme")
        drivers = complexity.get("drivers")
        if (not isinstance(drivers, list) or not drivers
                or any(not isinstance(item, str) or not item.strip() for item in drivers)):
            errors.append("complexity.drivers must be a non-empty string list")
        occupied_regions = complexity.get("occupied_regions")
        if (not isinstance(occupied_regions, list) or not occupied_regions
                or any(region not in OBJECT_REGIONS for region in occupied_regions)):
            errors.append(
                "complexity.occupied_regions must be a non-empty list of "
                "object-centric 3x3 regions")
        elif len(set(occupied_regions)) != len(occupied_regions):
            errors.append("complexity.occupied_regions contains duplicates")
    # The camera must be solved once, on evidence, and then held still.  With a
    # free camera, a wrong viewpoint and a wrong shape are indistinguishable
    # from a single silhouette score, and the shape is what ends up deformed.
    camera_solve = plan.get("camera_solve")
    if not isinstance(camera_solve, dict):
        errors.append("camera_solve must be an object with the solved camera, "
                      "the alternatives you tested, and locked: true")
    else:
        if camera_solve.get("locked") is not True:
            errors.append("camera_solve.locked must be true before synthesis")
        if not isinstance(camera_solve.get("camera"), dict):
            errors.append("camera_solve.camera must hold the solved camera, in "
                          "the same form as fit_spec.camera")
        candidates = camera_solve.get("candidates_tested")
        if (not isinstance(candidates, list) or len(candidates) < 2
                or any(not isinstance(item, str) or not item.strip()
                       for item in candidates)):
            errors.append("camera_solve.candidates_tested must record at least "
                          "two viewpoints you compared, and why the loser lost")
    if not isinstance(plan.get("shape_priors"), list) or not plan.get("shape_priors"):
        errors.append("shape_priors must be a non-empty list")
    if (not isinstance(plan.get("detail_features"), list)
            or not plan.get("detail_features")):
        errors.append("detail_features must be a non-empty list")
    return plan, errors


def form_prop(obj, name):
    props = obj.get("custom_props")
    return props.get(name) if isinstance(props, dict) else None


def mesh_count(obj, base_name, evaluated_name):
    value = obj.get(base_name, obj.get(evaluated_name))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def is_shallow_form_cage(obj):
    """Return true for a box-scale authored cage, with a profile-lathe exception."""
    base_vertices = mesh_count(obj, "base_vertex_count", "vertex_count")
    base_polys = mesh_count(obj, "base_poly_count", "poly_count")
    method = form_prop(obj, "procagen3d_form_method")
    modifiers = obj.get("modifiers") or []
    # A Screw lathe legitimately begins as an open 2D mesh profile: it has no
    # faces until evaluation, but enough profile samples to control curvature.
    profile_lathe = (method == "revolve" and "SCREW" in modifiers
                     and base_vertices is not None and base_vertices >= 5
                     and base_polys == 0)
    if profile_lathe:
        return False
    return ((base_vertices is not None and base_vertices <= 8)
            or (base_polys is not None and base_polys <= 6))


# Measured contradictions between a declared shape family and the geometry the
# program actually emitted.  Each entry is a predicate over the build-time shape
# signature that must be FALSE for the declared family, plus the reason printed
# when it is true.  Only clear contradictions are listed: the point is to make
# an ellipsoid impossible to ship under a `box` label, not to guess intent.
#
# Reference values for a clean primitive:
#   fill_ratio            box 1.00  cylinder 0.79  ellipsoid 0.52  cone 0.26
#   planar_area_fraction  box 1.00  cylinder 0.42  ellipsoid 0.05  loft 0.05
SHAPE_FAMILY_CONTRADICTIONS = {
    "box": [
        (lambda s: s["planar_area_fraction"] < 0.40,
         "no broad planar field: only {planar_area_fraction:.0%} of the surface "
         "lies in its six largest coplanar clusters (a plain box measures 100%, "
         "a bevelled box 82%, a heavily rounded box 51%, an ellipsoid 8%) — "
         "this mesh is a blob, not a box"),
    ],
    "prism": [
        (lambda s: s["planar_area_fraction"] < 0.40,
         "no broad planar field: {planar_area_fraction:.0%} planar area — a "
         "prism is flat-sided by definition"),
    ],
    "cylinder": [
        (lambda s: s["fill_ratio"] > 0.92,
         "fills {fill_ratio:.0%} of its bounding box — that is a box, not a "
         "cylinder (a cylinder fills ~79%)"),
        (lambda s: s["fill_ratio"] < 0.45,
         "fills only {fill_ratio:.0%} of its bounding box — too hollow/tapered "
         "for a cylinder"),
    ],
    "cone": [
        (lambda s: s["fill_ratio"] > 0.62,
         "fills {fill_ratio:.0%} of its bounding box — a cone fills ~26%"),
    ],
    "sphere": [
        (lambda s: s["planar_area_fraction"] > 0.55,
         "{planar_area_fraction:.0%} of the surface is flat — this is a faceted "
         "block wearing a sphere label"),
        (lambda s: s["fill_ratio"] > 0.80,
         "fills {fill_ratio:.0%} of its bounding box — a sphere fills ~52%"),
    ],
    "ellipsoid": [
        (lambda s: s["planar_area_fraction"] > 0.55,
         "{planar_area_fraction:.0%} of the surface is flat — this is a box or "
         "prism, and the plan should say so"),
        (lambda s: s["fill_ratio"] > 0.80,
         "fills {fill_ratio:.0%} of its bounding box — an ellipsoid fills ~52%"),
    ],
    "capsule": [
        (lambda s: s["planar_area_fraction"] > 0.55,
         "{planar_area_fraction:.0%} of the surface is flat — not a capsule"),
    ],
    "loft": [
        (lambda s: s["planar_area_fraction"] > 0.80
         and s["section_variation"] < 0.10,
         "constant section ({section_variation:.2f}) and {planar_area_fraction:.0%} "
         "planar area: this is a straight extrusion of a flat profile, so plan "
         "it as a box or prism instead of softening a block into a loft"),
    ],
    "sweep": [
        (lambda s: s["planar_area_fraction"] > 0.80
         and s["section_variation"] < 0.10,
         "constant section and {planar_area_fraction:.0%} planar area — a "
         "straight prism, not a sweep"),
    ],
    "revolve": [
        (lambda s: s["fill_ratio"] > 0.92,
         "fills {fill_ratio:.0%} of its bounding box — a lathe of any profile "
         "cannot fill a box"),
    ],
    "surface-grid": [
        (lambda s: s["planar_area_fraction"] > 0.85,
         "{planar_area_fraction:.0%} planar area — a flat card, not a shaped "
         "surface grid"),
    ],
}
# Families whose whole justification is curvature.  A macro mass may only claim
# one of these when the geometry is measurably curved, which is what keeps a
# faceted mecha from dissolving into ellipsoids to satisfy a form quota.
CURVED_FAMILIES = {"sphere", "ellipsoid", "capsule", "loft", "sweep",
                   "revolve", "surface-grid", "shell"}
PLATE_FAMILIES = {"shell", "surface-grid", "strand"}
PLATE_TOPOLOGIES = {"shell", "relief", "strand"}
# A primary mass thinner than this against its own longest axis is a cut-out,
# not a volume.  Bodies, limbs, and housings that collapse to a slab are the
# signature of fitting one camera and ignoring depth.
PRIMARY_THIN_RATIO_FLOOR = 0.08
# Ratio of the smallest to largest orthographic silhouette area.  A solid
# subject stays well above this; a bas-relief does not.
VIEW_COLLAPSE_FAIL = 0.10
VIEW_COLLAPSE_WARN = 0.22
INSTANCE_ARRAY_RE = re.compile(r"^(?P<base>.*?[A-Za-z])_?(?P<index>\d{1,3})$")
CONGRUENCE_WARN = 0.08
CONGRUENCE_FAIL = 0.25


REGION_MESH_FLOOR = {"simple": 2, "moderate": 4, "complex": 8, "extreme": 12}
REGION_ROWS = ("bottom", "middle", "top")
REGION_COLUMNS = ("left", "center", "right")


def shape_signature(obj):
    sig = obj.get("shape_signature")
    return sig if isinstance(sig, dict) else None


# An assembled object is one connected solid.  A part that touches nothing is
# floating, and a single reference view hides it completely whenever the gap
# happens to fall behind other geometry.
DETACHED_GAP_FRACTION = 0.01
DETACHED_ISLAND_SHARE = 0.12
# Separate objects in a scene may touch but must not occupy the same space.
# Depth is measured by containment, so a lamp standing on a table reads ~0
# while a lamp sunk into a sofa reads its true burial depth.
INTERPENETRATION_FAIL = 0.10
INTERPENETRATION_WARN = 0.03


def connected_components(meshes, tolerance):
    """Group meshes into islands that touch or nearly touch."""
    boxes = []
    for obj in meshes:
        lo = obj.get("bbox_world_min")
        hi = obj.get("bbox_world_max")
        if isinstance(lo, list) and isinstance(hi, list) and len(lo) >= 3:
            boxes.append((obj["name"], lo, hi))
    parent = {name: name for name, _, _ in boxes}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for index, (name_a, lo_a, hi_a) in enumerate(boxes):
        for name_b, lo_b, hi_b in boxes[index + 1:]:
            if all(lo_a[i] - hi_b[i] <= tolerance
                   and lo_b[i] - hi_a[i] <= tolerance for i in range(3)):
                root_a, root_b = find(name_a), find(name_b)
                if root_a != root_b:
                    parent[root_a] = root_b
    islands = {}
    for name, _, _ in boxes:
        islands.setdefault(find(name), []).append(name)
    return sorted(islands.values(), key=len, reverse=True)


# Bilateral symmetry is the strongest depth constraint a single reference view
# offers.  One image cannot say how far forward a shoulder sits, but it can say
# that the left one sits exactly as far forward as the right.  Tying the pairs
# together removes roughly half the free depth parameters in a humanoid or a
# vehicle, which is where "correct from this camera, wrong in 3D" comes from.
SIDE_TOKENS = (
    (re.compile(r"^(?P<base>.+?)_L(?P<tail>_\d{1,3})?$"), "L"),
    (re.compile(r"^(?P<base>.+?)_R(?P<tail>_\d{1,3})?$"), "R"),
    (re.compile(r"^(?P<base>.+?)_Left(?P<tail>_\d{1,3})?$"), "L"),
    (re.compile(r"^(?P<base>.+?)_Right(?P<tail>_\d{1,3})?$"), "R"),
    (re.compile(r"^Left(?P<base>[A-Z].*?)(?P<tail>_\d{1,3})?$"), "L"),
    (re.compile(r"^Right(?P<base>[A-Z].*?)(?P<tail>_\d{1,3})?$"), "R"),
)
SYMMETRY_AXES = {"x": 0, "y": 1, "z": 2}
SYMMETRY_WARN = 0.004
SYMMETRY_FAIL = 0.015
SYMMETRY_MIN_PAIRS = 3


def mirror_side(name):
    """Split a mesh name into its side-independent base and its side, if any."""
    for pattern, side in SIDE_TOKENS:
        match = pattern.match(name)
        if match:
            return match.group("base") + (match.group("tail") or ""), side
    return None, None


def mirror_pairs(meshes):
    """Group meshes into left/right counterparts by name."""
    sides = {}
    for obj in meshes:
        base, side = mirror_side(obj["name"])
        if side is None:
            continue
        sides.setdefault(base, {})[side] = obj
    return {base: (entry["L"], entry["R"])
            for base, entry in sides.items() if "L" in entry and "R" in entry}


def world_centroid(obj):
    lo = obj.get("bbox_world_min")
    hi = obj.get("bbox_world_max")
    if not (isinstance(lo, list) and isinstance(hi, list)
            and len(lo) >= 3 and len(hi) >= 3):
        return None
    return [(lo[i] + hi[i]) / 2.0 for i in range(3)]


def mirror_mismatch(left, right, axis, origin, major):
    """Positional and size disagreement between a mirrored pair, in object scale.

    Returns (offset per axis, size difference) both normalised, or None when
    either part cannot be measured.
    """
    a = world_centroid(left)
    b = world_centroid(right)
    if a is None or b is None or major <= 1e-9:
        return None
    expected = list(a)
    expected[axis] = 2.0 * origin - a[axis]
    offset = [abs(expected[i] - b[i]) / major for i in range(3)]
    size = 0.0
    sig_a, sig_b = shape_signature(left), shape_signature(right)
    if sig_a and sig_b:
        for key, power in (("volume", 1.0 / 3.0), ("surface_area", 0.5)):
            va, vb = sig_a.get(key), sig_b.get(key)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                la, lb = max(va, 0.0) ** power, max(vb, 0.0) ** power
                mean = (la + lb) / 2.0
                if mean > 1e-12:
                    size = max(size, abs(la - lb) / mean)
    return offset, size


# A rifle, spear, mast, axle, or pipe run is one straight object.  Its parts
# only bend apart when each endpoint is placed independently to hit a position
# in the reference image, because a single view cannot see the depth that the
# bend hides in.
RIGID_AXIS_FAIL_DEG = 5.0
RIGID_AXIS_MIN_ELONGATION = 1.8
# An assembly whose elongated parts disagree by more than this needs a stated
# decision: it either bends at a joint or it is broken.  Silence is not an
# answer, because an opt-in contract is one nobody opts into.
RIGID_AXIS_DECLARE_DEG = 8.0
# Only assemblies that are themselves long and thin are asked the question.  A
# whole standing figure is not (roughly 1.2:1); a rifle, a limb, or a mast is.
RIGID_AXIS_GROUP_ASPECT = 2.5


def elongated_axis(obj):
    signature = shape_signature(obj)
    if signature is None:
        return None
    axis = signature.get("world_axis")
    elongation = signature.get("elongation")
    if (not isinstance(axis, list) or len(axis) != 3
            or not isinstance(elongation, (int, float))
            or elongation < RIGID_AXIS_MIN_ELONGATION):
        return None
    return [float(c) for c in axis]


def axis_disagreement_deg(a, b):
    """Angle between two undirected axes, in degrees."""
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def rigid_axis_spread(members):
    """Worst pairwise long-axis disagreement across an assembly's parts."""
    axes = [(obj["name"], elongated_axis(obj)) for obj in members]
    axes = [(name, axis) for name, axis in axes if axis is not None]
    if len(axes) < 2:
        return None, None
    worst = 0.0
    culprit = ""
    for index, (name_a, axis_a) in enumerate(axes):
        for name_b, axis_b in axes[index + 1:]:
            angle = axis_disagreement_deg(axis_a, axis_b)
            if angle > worst:
                worst, culprit = angle, f"{name_a} vs {name_b}"
    return worst, culprit


def top_two_eigenvalues(points):
    """Two largest covariance eigenvalues, by power iteration with deflation."""
    count = len(points)
    if count < 3:
        return None
    centre = [sum(p[i] for p in points) / count for i in range(3)]
    cov = [[0.0] * 3 for _ in range(3)]
    for point in points:
        d = [point[i] - centre[i] for i in range(3)]
        for i in range(3):
            for j in range(3):
                cov[i][j] += d[i] * d[j]

    def iterate(matrix, seed):
        vector = seed
        value = 0.0
        for _ in range(60):
            nxt = [sum(matrix[i][j] * vector[j] for j in range(3))
                   for i in range(3)]
            length = math.sqrt(sum(c * c for c in nxt))
            if length <= 1e-18:
                return None, 0.0
            vector = [c / length for c in nxt]
            value = length
        return vector, value

    first, lambda1 = iterate(cov, [0.5773, 0.5774, 0.5775])
    if first is None:
        return None
    for i in range(3):
        for j in range(3):
            cov[i][j] -= lambda1 * first[i] * first[j]
    seed = [0.0, 1.0, 0.0] if abs(first[0]) > 0.9 else [1.0, 0.0, 0.0]
    _, lambda2 = iterate(cov, seed)
    return lambda1, lambda2


def group_aspect(members):
    """How long and thin an assembly is, independent of world orientation.

    Built from each part's own axis and length rather than the union bounding
    box: a rod lying diagonally has a fat axis-aligned box and would otherwise
    read as compact.  A radial array such as wheel spokes correctly reads as a
    disc and is not treated as a long assembly at all.
    """
    points = []
    for obj in members:
        centroid = world_centroid(obj)
        axis = elongated_axis(obj)
        signature = shape_signature(obj)
        if centroid is None or axis is None or signature is None:
            continue
        dims = signature.get("local_dims")
        if not isinstance(dims, list) or len(dims) != 3:
            continue
        half = max(dims) / 2.0
        points.append([centroid[i] + axis[i] * half for i in range(3)])
        points.append([centroid[i] - axis[i] * half for i in range(3)])
    eigen = top_two_eigenvalues(points)
    if eigen is None:
        return 0.0
    lambda1, lambda2 = eigen
    if lambda2 <= 1e-15:
        return 999.0
    return math.sqrt(lambda1 / lambda2)


def bent_assemblies(meshes):
    """Elongated assemblies whose parts point in visibly different directions.

    Grouped by transform parent, which is what "assembly" means in a ProcAgen3D
    program.  Each one returned here is a question the plan has to answer: does
    this thing bend at a joint, or is it a straight object that came out bent?
    """
    groups = {}
    for obj in meshes:
        parent = obj.get("parent")
        if parent and elongated_axis(obj) is not None:
            groups.setdefault(parent, []).append(obj)
    out = []
    for parent, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        if group_aspect(members) < RIGID_AXIS_GROUP_ASPECT:
            continue
        spread, culprit = rigid_axis_spread(members)
        if spread is not None and spread >= RIGID_AXIS_DECLARE_DEG:
            out.append((parent, members, spread, culprit))
    return out


CAMERA_LOCK_TOLERANCE = {
    "azimuth_deg": 0.5, "elevation_deg": 0.5, "roll_deg": 0.5,
    "fov_y_deg": 0.5, "distance_m": 1e-3, "ortho_scale_m": 1e-3,
    "shift_x": 1e-3, "shift_y": 1e-3,
}


def camera_drift(locked, actual):
    """Describe how a fit camera differs from the locked solve, if it does."""
    drift = []
    for key in ("projection",):
        if str(locked.get(key, "perspective")) != str(actual.get(key, "perspective")):
            drift.append(f"{key} {locked.get(key)!r}->{actual.get(key)!r}")
    for key, tolerance in CAMERA_LOCK_TOLERANCE.items():
        if key not in locked and key not in actual:
            continue
        a, b = locked.get(key, 0.0), actual.get(key, 0.0)
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            continue
        if abs(float(a) - float(b)) > tolerance:
            drift.append(f"{key} {a:g}->{b:g}")
    for key in ("target_m", "location_m"):
        a, b = locked.get(key), actual.get(key)
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            if any(abs(float(x) - float(y)) > 1e-3 for x, y in zip(a, b)):
                drift.append(f"{key} {a}->{b}")
        elif (a is None) != (b is None):
            drift.append(f"{key} {'set' if b is not None else 'removed'}")
    return "; ".join(drift[:6])


def region_occupancy(graph, meshes):
    """Count meshes whose centre falls in each object-centric 3x3 region.

    The grid is the world bounding box split in thirds across X (left/right)
    and Z (bottom/top), matching the region vocabulary the plan uses.
    """
    box = graph.get("world_bbox") or {}
    low = box.get("min")
    size = box.get("size")
    if not (isinstance(low, list) and isinstance(size, list)
            and len(low) >= 3 and len(size) >= 3):
        return {}
    counts = {}
    for obj in meshes:
        lo = obj.get("bbox_world_min")
        hi = obj.get("bbox_world_max")
        if not (isinstance(lo, list) and isinstance(hi, list)):
            continue
        cell = []
        for axis, names in ((0, REGION_COLUMNS), (2, REGION_ROWS)):
            span = size[axis]
            if span <= 1e-9:
                cell.append(names[1])
                continue
            centre = (lo[axis] + hi[axis]) / 2.0
            index = int((centre - low[axis]) / span * 3.0)
            cell.append(names[min(2, max(0, index))])
        counts[f"{cell[1]}-{cell[0]}"] = counts.get(f"{cell[1]}-{cell[0]}", 0) + 1
    return counts


def shape_family_contradiction(family, signature):
    """Return the reason the geometry contradicts the declared family, or None."""
    for predicate, template in SHAPE_FAMILY_CONTRADICTIONS.get(family, ()):
        try:
            if predicate(signature):
                return template.format(**signature)
        except (KeyError, TypeError, ValueError):
            continue
    return None


def instance_arrays(meshes, minimum=3):
    """Group ``Sword_01``-style siblings by their shared base name."""
    groups = {}
    for obj in meshes:
        match = INSTANCE_ARRAY_RE.match(obj["name"])
        if match:
            groups.setdefault(match.group("base"), []).append(obj)
    return {base: members for base, members in groups.items()
            if len(members) >= minimum}


def congruence_spread(members):
    """Largest relative disagreement in size across array members.

    Measured from volume and surface area rather than bounding-box dimensions,
    because both are rotation-invariant: a builder that bakes each instance's
    rotation into its vertices still produces identical numbers for identical
    parts.  Both are reduced to a characteristic length so the spread reads as
    a percentage of size.
    """
    lengths = []
    for obj in members:
        signature = shape_signature(obj)
        if signature is None:
            return None
        volume = signature.get("volume")
        area = signature.get("surface_area")
        if not isinstance(volume, (int, float)) or not isinstance(area, (int, float)):
            return None
        lengths.append((max(volume, 0.0) ** (1.0 / 3.0), max(area, 0.0) ** 0.5))
    spread = 0.0
    for column in range(2):
        values = [item[column] for item in lengths]
        mean = sum(values) / len(values)
        if mean <= 1e-12:
            continue
        spread = max(spread, (max(values) - min(values)) / mean)
    return spread


def bbox_proxy_volume(obj):
    dims = obj.get("dimensions")
    if (not isinstance(dims, list) or len(dims) < 3
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       for v in dims[:3])):
        return 0.0
    return max(0.0, dims[0]) * max(0.0, dims[1]) * max(0.0, dims[2])


def cmd_check(args):
    graph = load_graph(args.dir)
    objs = graph["objects"]
    meshes = [o for o in objs if o["type"] == "MESH"]
    failures = 0

    fit_errors, fit_warnings = image_fit_errors(args.dir)
    if fit_errors:
        failures += 1
        fail("REFERENCE_FIT", "; ".join(fit_errors))
    for message in fit_warnings:
        warn("REFERENCE_FIT_APPROXIMATE", message)

    # A camera solve that could not converge is a verdict about the model, not
    # a formality: it says no single viewpoint explains the landmarks, so the
    # proportions or the layout are wrong.  Proceeding on the rejected seed is
    # how a scene with a chair facing the wrong way reaches delivery.
    solution_path = Path(args.dir) / "camera_solution.json"
    if solution_path.is_file():
        try:
            solution = json.loads(solution_path.read_text())
        except (OSError, json.JSONDecodeError):
            solution = None
        if isinstance(solution, dict):
            rms = solution.get("rms_uv_error")
            limit = solution.get("max_rms", CAMERA_SOLVE_MAX_RMS)
            if isinstance(rms, (int, float)) and rms > limit:
                failures += 1
                fail("CAMERA_SOLVE", f"camera resection did not converge (RMS "
                     f"{rms:.4f} vs {limit:.4f}): no single viewpoint explains "
                     "the landmarks you read, so the fault is in proportions or "
                     "instance layout, not the camera. Fix those and re-solve — "
                     "do not carry on with the rejected seed")

    reconstruction_plan, plan_errors = load_reconstruction_plan(args.dir)
    if plan_errors:
        failures += 1
        fail("RECONSTRUCTION_PLAN", "; ".join(plan_errors))

    root_names = set(graph.get("roots") or [])
    if not root_names:
        root_names = {o["name"] for o in objs if o.get("parent") is None}
    root_objs = [o for o in objs if o["name"] in root_names]
    profile_values = [form_prop(o, "procagen3d_form_profile")
                      for o in root_objs]
    profile_values = [value for value in profile_values if value is not None]
    child_profiles = [
        (o["name"], form_prop(o, "procagen3d_form_profile"))
        for o in objs if o["name"] not in root_names
        and form_prop(o, "procagen3d_form_profile") is not None
    ]
    if child_profiles:
        if profile_values:
            warn("FORM_PROFILE_SCOPE", "ignoring non-root form profiles: "
                 f"{child_profiles[:6]}")
        else:
            failures += 1
            fail("FORM_PROFILE_SCOPE", "form profile must be declared on a "
                 f"root object, not only on children: {child_profiles[:6]}")
    invalid_profiles = [value for value in profile_values
                        if not isinstance(value, str)
                        or value not in {"rectilinear", "curved", "mixed"}]
    declared_profiles = {value for value in profile_values
                         if isinstance(value, str)
                         and value in {"rectilinear", "curved", "mixed"}}
    if invalid_profiles or len(declared_profiles) > 1:
        failures += 1
        fail("FORM_PROFILE", f"invalid/conflicting root form profiles: "
             f"{sorted(str(value) for value in profile_values)}")
    declared_profile = (next(iter(declared_profiles))
                        if len(declared_profiles) == 1 else None)
    form_profile = (declared_profile
                    if args.form == "auto" and declared_profile else args.form)
    if (args.form != "auto" and declared_profile is not None
            and args.form != declared_profile):
        failures += 1
        fail("FORM_PROFILE", f"CLI --form {args.form} conflicts with declared "
             f"root profile {declared_profile}")

    if not meshes:
        fail("NO_MESHES", "scene contains no mesh objects")
        return 1

    bad_names = [o["name"] for o in objs if DEFAULT_NAME_RE.match(o["name"])]
    if bad_names:
        failures += 1
        fail("DEFAULT_NAMES", f"unnamed primitives: {bad_names[:8]} — every part "
             "must carry a semantic PascalCase name (doctrine)")

    dupes = [o["name"] for o in objs if DUPE_SUFFIX_RE.search(o["name"])]
    if dupes:
        failures += 1
        fail("DUPLICATE_NAMES", f"'.NNN' suffixes indicate name collisions: "
             f"{dupes[:8]} — name repeated instances individually (Spoke_17)")

    empty_meshes = [o["name"] for o in meshes if o.get("poly_count", 0) == 0]
    if empty_meshes:
        failures += 1
        fail("EMPTY_MESHES", f"zero-polygon meshes: {empty_meshes[:8]}")

    degenerate = [o["name"] for o in meshes
                  if min(o.get("dimensions", [1, 1, 1])) < 1e-5]
    if degenerate:
        warn("DEGENERATE", f"near-zero extent meshes: {degenerate[:8]}")

    unmat = [o["name"] for o in meshes if not o.get("materials")]
    if unmat:
        warn("NO_MATERIAL", f"meshes without materials: {unmat[:8]}")

    unapplied = [o["name"] for o in meshes
                 if any(abs(s - 1.0) > 1e-3 for s in o.get("scale", [1, 1, 1]))]
    if unapplied:
        warn("UNAPPLIED_SCALE", f"non-unit object scale (apply transforms or "
             f"size via dimensions): {unapplied[:8]}")

    roots = graph.get("roots", [])
    if len(roots) > 1:
        warn("MULTIPLE_ROOTS", f"{len(roots)} root objects {roots[:8]} — "
             "doctrine wants one root node")

    group_empties = [o for o in objs if o["type"] == "EMPTY"
                     and "procagen3d_joint_type" not in o.get("custom_props", {})]
    if len(meshes) > 12 and not group_empties:
        warn("FLAT_HIERARCHY", f"{len(meshes)} meshes with no semantic group "
             "nodes (Body, Doors, Left_Arm...)")

    names = {o["name"] for o in objs}
    for j in graph.get("joints", []):
        if j.get("type") not in JOINT_TYPES:
            failures += 1
            fail("JOINT_TYPE", f"{j['name']}: type '{j.get('type')}' invalid")
        if j.get("child") not in names:
            failures += 1
            fail("JOINT_CHILD", f"{j['name']}: child '{j.get('child')}' missing")

    totals = graph["totals"]

    # detail floors (references/detail.md): advisory, tier-dependent
    mats = set()
    for o in meshes:
        mats.update(o.get("materials") or [])
    # Structural masses exclude trim strips, tread lugs, seams, and thin glass.
    # Authored/base topology is recorded separately from evaluated topology.
    # The general detail floor tolerates intentionally beveled solids; the
    # curved-form primitive-cage gate below does not let Bevel hide them.
    world_size = graph.get("world_bbox", {}).get("size", [1, 1, 1])
    major_dim = max(s for s in world_size if isinstance(s, (int, float)))
    structural = []
    for o in meshes:
        dims = sorted(o.get("dimensions", [0, 0, 0]))
        if dims[0] >= 0.01 * major_dim and dims[1] >= 0.06 * major_dim:
            structural.append(o)
    boxy = [o["name"] for o in structural
            if o.get("base_poly_count", o.get("poly_count", 0)) <= 6
            and ("BEVEL" not in o.get("modifiers", [])
                 or o.get("poly_count", 0)
                 <= o.get("base_poly_count", o.get("poly_count", 0)))]
    boxy_ratio = len(boxy) / len(structural) if structural else 0.0
    primitive_cages = [o["name"] for o in structural
                       if o.get("base_vertex_count", o.get("vertex_count", 0)) <= 8
                       and o.get("base_poly_count", o.get("poly_count", 0)) <= 6]
    primitive_cage_ratio = (len(primitive_cages) / len(structural)
                            if structural else 0.0)

    # Semantic form contract.  Geometry helpers set these custom properties on
    # silhouette-bearing masses; the checker rejects incompatible topology /
    # construction pairs instead of trying to infer intent from triangle count.
    tagged = [o for o in meshes if any(form_prop(o, key) is not None for key in (
        "procagen3d_form_role", "procagen3d_topology",
        "procagen3d_form_method"))]
    valid_contract_names = set()
    for obj in tagged:
        role = form_prop(obj, "procagen3d_form_role")
        topology = form_prop(obj, "procagen3d_topology")
        method = form_prop(obj, "procagen3d_form_method")
        if (not isinstance(role, str) or not isinstance(topology, str)
                or not isinstance(method, str)
                or role not in ("primary", "secondary")
                or topology not in FORM_TOPOLOGIES or method not in FORM_METHODS):
            failures += 1
            fail("FORM_TAG", f"{obj['name']}: invalid/incomplete form contract "
                 f"role={role!r}, topology={topology!r}, method={method!r}")
        elif method not in TOPOLOGY_METHODS[topology]:
            failures += 1
            fail("FORM_METHOD", f"{obj['name']}: topology '{topology}' cannot "
                 f"use '{method}' — choose a compatible representation "
                 "(references/complex-forms.md)")
        else:
            valid_contract_names.add(obj["name"])

    primary = [o for o in meshes
               if form_prop(o, "procagen3d_form_role") == "primary"]

    # ---- measured geometry gates -------------------------------------------
    # Everything above this point compares one declaration against another.
    # These compare declarations against the mesh that was actually built.
    mismatched_family = []
    for obj in meshes:
        family = form_prop(obj, "procagen3d_shape_family")
        signature = shape_signature(obj)
        if not isinstance(family, str) or signature is None:
            continue
        reason = shape_family_contradiction(family, signature)
        if reason:
            mismatched_family.append(f"{obj['name']} declared {family!r}: {reason}")
    if mismatched_family:
        failures += 1
        fail("SHAPE_FAMILY_MEASURED",
             "declared shape family contradicts the built geometry — "
             + " | ".join(mismatched_family[:6])
             + (f" (+{len(mismatched_family) - 6} more)"
                if len(mismatched_family) > 6 else ""))

    slabs = []
    for obj in primary:
        signature = shape_signature(obj)
        if signature is None:
            continue
        family = form_prop(obj, "procagen3d_shape_family")
        topology = form_prop(obj, "procagen3d_topology")
        if family in PLATE_FAMILIES or topology in PLATE_TOPOLOGIES:
            continue
        if signature["thin_ratio"] < PRIMARY_THIN_RATIO_FLOOR:
            slabs.append(f"{obj['name']} thin_ratio={signature['thin_ratio']:.3f}")
    if slabs:
        failures += 1
        fail("PRIMARY_DEPTH", "primary volumes collapsed to slabs (floor "
             f"{PRIMARY_THIN_RATIO_FLOOR:g}): {slabs[:8]} — a body, limb, or "
             "housing that is paper-thin off-axis fits one camera and no other; "
             "tag it shell/relief only if the reference really shows a plate")

    silhouettes = graph.get("view_silhouettes")
    if isinstance(silhouettes, dict) and len(silhouettes) >= 3:
        areas = {view: entry.get("area_fraction", 0.0)
                 for view, entry in silhouettes.items()
                 if isinstance(entry, dict)}
        widest = max(areas.values()) if areas else 0.0
        if widest > 1e-6:
            thinnest_view = min(areas, key=areas.get)
            collapse = areas[thinnest_view] / widest
            detail = (f"{thinnest_view} silhouette is {collapse:.0%} of the "
                      f"largest view ({', '.join(f'{v}={a:.3f}' for v, a in sorted(areas.items()))})")
            if collapse < VIEW_COLLAPSE_FAIL:
                failures += 1
                fail("VIEW_COLLAPSE", f"the model is a bas-relief: {detail} — "
                     "orthographic views share one scale, so this measures real "
                     "depth, not framing; rebuild the masses with thickness "
                     "instead of fitting the reference camera alone")
            elif collapse < VIEW_COLLAPSE_WARN:
                warn("VIEW_COLLAPSE", f"shallow depth: {detail}")

    # Local evidence must scale with the number of contested masses.  Three
    # regions on a mecha leaves most family and pose decisions untested, and a
    # whole-object mask hides all of them.
    fit_spec_path = Path(args.dir) / "fit_spec.json"
    if fit_spec_path.is_file() and primary:
        try:
            fit_spec = json.loads(fit_spec_path.read_text())
        except (OSError, json.JSONDecodeError):
            fit_spec = None
        if isinstance(fit_spec, dict) and reconstruction_plan is not None:
            locked = (reconstruction_plan.get("camera_solve") or {}).get("camera")
            spec_camera = fit_spec.get("camera")
            if isinstance(locked, dict) and isinstance(spec_camera, dict):
                drift = camera_drift(locked, spec_camera)
                if drift:
                    failures += 1
                    fail("CAMERA_LOCK", "fit camera drifted from the locked "
                         f"solve: {drift} — a camera change is a priors "
                         "revision: re-solve it against the reference and "
                         "update camera_solve, do not tune it to rescue a fit")
        if isinstance(fit_spec, dict) and fit_spec.get("version") == 2:
            declared = len(fit_spec.get("silhouette_regions") or [])
            required = min(8, max(3, math.ceil(len(primary) / 2)))
            if declared < required:
                failures += 1
                fail("FIT_REGION_COVERAGE", f"{len(primary)} primary masses but "
                     f"only {declared} silhouette regions (need {required}) — "
                     "put a tight region on every contested primitive family, "
                     "taper, limb, and attachment")

    # Arrays the plan explicitly calls out are held to their declaration.
    # Everything else is only forced into the open when it is both large and
    # badly inconsistent, so a run of genuinely different seams or major/minor
    # tick marks does not have to be argued with.
    declared_arrays = []
    if reconstruction_plan is not None:
        for entry in reconstruction_plan.get("instance_arrays") or []:
            if isinstance(entry, dict) and isinstance(entry.get("pattern"), str):
                declared_arrays.append(entry)
    # Only objects the fit spec calls separate instances are held apart.  The
    # sub-assemblies of one articulated body are supposed to interpenetrate at
    # every joint, and a shoulder inside an arm socket is correct engineering.
    scene_instances = {}
    spec_path = Path(args.dir) / "fit_spec.json"
    if spec_path.is_file():
        try:
            spec_data = json.loads(spec_path.read_text())
        except (OSError, json.JSONDecodeError):
            spec_data = {}
        for entry in (spec_data.get("instances") or []):
            if isinstance(entry, dict) and isinstance(entry.get("pattern"), str):
                scene_instances[str(entry.get("id"))] = entry["pattern"]

    def instance_of(assembly):
        for instance_id, pattern in scene_instances.items():
            if fnmatch.fnmatchcase(assembly, pattern):
                return instance_id
        return None

    allowed_overlaps = [p for p in
                        ((reconstruction_plan or {}).get("allowed_intersections") or [])
                        if isinstance(p, dict)]
    clashing, grazing = [], []
    for entry in graph.get("assembly_intersections") or []:
        fraction = entry.get("penetration_fraction", 0.0)
        if not isinstance(fraction, (int, float)) or fraction <= INTERPENETRATION_WARN:
            continue
        first, second = instance_of(entry["a"]), instance_of(entry["b"])
        if first is None or second is None or first == second:
            continue
        if any({entry["a"], entry["b"]} == {allowed.get("a"), allowed.get("b")}
               for allowed in allowed_overlaps):
            continue
        detail = (f"{entry['a']} and {entry['b']} overlap by "
                  f"{entry.get('penetration_m', 0):.3f} m "
                  f"({fraction:.0%} of the smaller object)")
        (clashing if fraction > INTERPENETRATION_FAIL else grazing).append(detail)
    if grazing:
        warn("SCENE_CONTACT", "objects clip slightly: " + "; ".join(grazing[:5]))
    if clashing:
        failures += 1
        fail("SCENE_INTERPENETRATION", "separate objects occupy the same space: "
             + "; ".join(clashing[:5]) + " — resting on, leaning against, and "
             "tucked beside all measure near zero here, so this is real burial: "
             "move the object until it sits on the surface. Deliberate overlaps "
             "go in reconstruction_plan.allowed_intersections")

    islands = connected_components(meshes, DETACHED_GAP_FRACTION * major_dim)
    if len(islands) > 1:
        detached_patterns = [p for p in
                             ((reconstruction_plan or {}).get("detached_groups") or [])
                             if isinstance(p, str)]
        # A floating trim strip or button is cosmetic; a floating head, rail, or
        # windshield is a broken assembly.  Only islands carrying a structural
        # mass are hard failures, so the gate does not drown in small parts
        # sitting a fraction of a millimetre proud of their host surface.
        structural_names = {o["name"] for o in structural}
        by_name = {o["name"]: o for o in meshes}
        object_diagonal = math.sqrt(sum(v * v for v in world_size[:3])) or 1.0
        stray, trim = [], []
        for island in islands[1:]:
            if any(fnmatch.fnmatchcase(name, pattern)
                   for name in island for pattern in detached_patterns):
                continue
            carried = sorted(structural_names.intersection(island))
            # A rail or a windshield can be made of parts that are each too
            # slender to count as structural while the assembly plainly is not
            # trim, so weigh the island as a whole too.
            lo = [math.inf] * 3
            hi = [-math.inf] * 3
            for name in island:
                obj = by_name.get(name) or {}
                a, b = obj.get("bbox_world_min"), obj.get("bbox_world_max")
                if isinstance(a, list) and isinstance(b, list):
                    for i in range(3):
                        lo[i] = min(lo[i], a[i])
                        hi[i] = max(hi[i], b[i])
            span = (math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3)))
                    if all(math.isfinite(v) for v in lo + hi) else 0.0)
            share = span / object_diagonal
            if carried or share >= DETACHED_ISLAND_SHARE:
                stray.append(f"{len(island)} part(s) spanning {share:.0%} of the "
                             f"object, including {(carried or sorted(island))[:4]}")
            else:
                trim.append(f"{len(island)}x {sorted(island)[:2]}")
        if trim:
            warn("DETACHED_TRIM", f"{len(trim)} small detached island(s) — "
                 "cosmetic unless the reference shows them touching: "
                 + "; ".join(trim[:6]))
        if stray:
            failures += 1
            fail("DETACHED_PARTS", "structural geometry is in "
                 f"{len(islands)} disconnected islands; these touch nothing "
                 f"(gap over {DETACHED_GAP_FRACTION:.0%} of object size): "
                 + "; ".join(stray[:5]) + " — build the part that joins them "
                 "(a neck, a stem, a mount) or move them into contact; a "
                 "floating assembly can look attached from the one camera you "
                 "fitted. Genuinely separate pieces go in "
                 "reconstruction_plan.detached_groups")

    pairs = mirror_pairs(meshes)
    symmetry = (reconstruction_plan or {}).get("symmetry")
    if len(pairs) >= SYMMETRY_MIN_PAIRS and reconstruction_plan is not None:
        if not isinstance(symmetry, dict):
            failures += 1
            fail("SYMMETRY", f"{len(pairs)} left/right part pairs exist but the "
                 "plan declares no symmetry contract — add "
                 '"symmetry": {"plane": "x", "origin_m": 0.0, "asymmetric": [...]}, '
                 'or "plane": null with a reason if the subject really is not '
                 "bilateral. Mirrored pairs are the only thing a single view "
                 "offers to pin part depth down")
        elif symmetry.get("plane") is not None:
            axis_name = str(symmetry.get("plane", "x")).lower()
            if axis_name not in SYMMETRY_AXES:
                failures += 1
                fail("SYMMETRY", f"symmetry.plane must be x, y, z, or null "
                     f"(got {symmetry.get('plane')!r})")
            else:
                axis = SYMMETRY_AXES[axis_name]
                origin = symmetry.get("origin_m", 0.0)
                if not isinstance(origin, (int, float)):
                    origin = 0.0
                exempt = [p for p in symmetry.get("asymmetric") or []
                          if isinstance(p, str)]
                tolerance = symmetry.get("tolerance", SYMMETRY_FAIL)
                if not isinstance(tolerance, (int, float)):
                    tolerance = SYMMETRY_FAIL
                broken, drifting = [], []
                for base, (left, right) in sorted(pairs.items()):
                    if any(fnmatch.fnmatchcase(left["name"], pattern)
                           or fnmatch.fnmatchcase(right["name"], pattern)
                           for pattern in exempt):
                        continue
                    result = mirror_mismatch(left, right, axis, origin, major_dim)
                    if result is None:
                        continue
                    offset, size = result
                    worst = max(max(offset), size)
                    if worst <= SYMMETRY_WARN:
                        continue
                    along = max(offset[i] for i in range(3) if i != axis)
                    parts = []
                    if along > SYMMETRY_WARN:
                        parts.append(f"sits {along:.1%} of object size apart "
                                     "along the mirror plane (a depth or height "
                                     "difference the reference camera cannot see)")
                    if offset[axis] > SYMMETRY_WARN:
                        parts.append(f"{offset[axis]:.1%} unequal across the "
                                     "mirror plane")
                    if size > SYMMETRY_WARN:
                        parts.append(f"{size:.1%} different in size")
                    detail = f"{base}: " + ", ".join(parts)
                    (broken if worst > tolerance else drifting).append(detail)
                if drifting:
                    warn("SYMMETRY", "mirrored pairs drifting: "
                         + "; ".join(drifting[:6]))
                if broken:
                    failures += 1
                    fail("SYMMETRY", "left/right pairs are not mirror images: "
                         + "; ".join(broken[:8]) + " — build both sides from one "
                         "builder and one set of constants with the side as a "
                         "sign, so a part's depth cannot be chosen independently "
                         "per side; list genuinely one-sided parts in "
                         "symmetry.asymmetric")

    # Rigid single-axis assemblies must actually be straight in 3D.  An
    # articulated chain is elongated too and is supposed to bend, so the plan
    # says which is which — but every bent assembly must be classified, because
    # a contract nobody is required to fill in is a contract nobody fills in.
    declared_axes = []
    for entry in (reconstruction_plan or {}).get("rigid_axes") or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("pattern"), str):
            failures += 1
            fail("RIGID_AXIS", 'rigid_axes entries need {"pattern": ..., '
                 '"rigid": true|false} and, when not rigid, a reason')
            continue
        if entry.get("rigid") is False and not str(entry.get("reason", "")).strip():
            failures += 1
            fail("RIGID_AXIS", f"{entry['pattern']}: a non-rigid assembly needs "
                 "a reason naming the joint that bends")
            continue
        declared_axes.append(entry)
        if entry.get("rigid") is False:
            continue
        members = [o for o in meshes
                   if fnmatch.fnmatchcase(o["name"], entry["pattern"])]
        if len(members) < 2:
            continue
        tolerance = entry.get("max_deviation_deg", RIGID_AXIS_FAIL_DEG)
        if not isinstance(tolerance, (int, float)):
            tolerance = RIGID_AXIS_FAIL_DEG
        spread, culprit = rigid_axis_spread(members)
        if spread is not None and spread > tolerance:
            failures += 1
            fail("RIGID_AXIS", f"{entry['pattern']} is declared one rigid axis "
                 f"but its parts disagree by {spread:.1f}° (limit {tolerance:g}°, "
                 f"worst: {culprit}) — author the assembly from a single origin "
                 "and direction and place every station along it, never by "
                 "positioning each endpoint to match the image separately")

    if reconstruction_plan is not None:
        undeclared_bends = []
        for parent, members, spread, culprit in bent_assemblies(meshes):
            if any(fnmatch.fnmatchcase(obj["name"], entry["pattern"])
                   for entry in declared_axes for obj in members):
                continue
            undeclared_bends.append(
                f"{parent} ({len(members)} elongated parts) bends {spread:.1f}° "
                f"at {culprit}")
        if undeclared_bends:
            failures += 1
            fail("RIGID_AXIS", "long assemblies whose parts point in different "
                 "directions, with no decision on record: "
                 + "; ".join(undeclared_bends[:6]) + " — add a "
                 'reconstruction_plan.rigid_axes entry for each: {"pattern": '
                 '"Rifle_*", "rigid": true} to require one straight axis, or '
                 '{"pattern": "Leg_*", "rigid": false, "reason": "knee joint"} '
                 "when the bend is a real joint. A weapon, mast, or axle that "
                 "bends is the signature of placing each endpoint separately to "
                 "match the image while depth was free")

    incongruent = []
    undeclared = []
    for base, members in sorted(instance_arrays(meshes).items()):
        spread = congruence_spread(members)
        if spread is None:
            continue
        declaration = next(
            (entry for entry in declared_arrays
             if fnmatch.fnmatchcase(f"{base}_01", entry["pattern"])
             or fnmatch.fnmatchcase(base, entry["pattern"])), None)
        summary = f"{base}_* ({len(members)} members) sizes differ by {spread:.0%}"
        if declaration is not None:
            if declaration.get("congruent") is False:
                continue
            tolerance = declaration.get("tolerance", CONGRUENCE_WARN)
            if not isinstance(tolerance, (int, float)):
                tolerance = CONGRUENCE_WARN
            if spread > tolerance:
                incongruent.append(f"{summary} (declared congruent within "
                                   f"{tolerance:.0%})")
            continue
        if len(members) >= 6 and spread > CONGRUENCE_FAIL:
            undeclared.append(summary)
        elif spread > CONGRUENCE_WARN:
            warn("INSTANCE_CONGRUENCE", summary + " — intended variation, or drift?")
    if incongruent:
        failures += 1
        fail("INSTANCE_CONGRUENCE", "arrays the plan declares congruent are not: "
             + "; ".join(incongruent[:6]) + " — build repeated parts from one "
             "builder with shared dimension constants and vary only the "
             "transform")
    if undeclared:
        failures += 1
        fail("INSTANCE_CONGRUENCE", "large instance arrays vary in size without "
             "a decision on record: " + "; ".join(undeclared[:6])
             + " — either build them from one builder with shared constants, or "
             "add a reconstruction_plan.instance_arrays entry with "
             "\"congruent\": false and the reason they legitimately differ")

    if form_profile in ("curved", "mixed"):
        macro = sorted(
            structural,
            key=bbox_proxy_volume,
            reverse=True,
        )[:12]
        contracted_macro = [o for o in macro if o["name"] in valid_contract_names]
        contract_ratio = len(contracted_macro) / len(macro) if macro else 1.0
        required_contract_ratio = (
            0.75 if form_profile == "curved" or reconstruction_plan is not None
            else 0.60
        )
        if contract_ratio < required_contract_ratio:
            failures += 1
            untagged = [o["name"] for o in macro
                        if o["name"] not in valid_contract_names]
            fail("FORM_TAG_COVERAGE", f"only {len(contracted_macro)}/{len(macro)} "
                 f"largest structural masses have valid form contracts "
                 f"(need {required_contract_ratio:.0%}); untagged: {untagged[:8]}")
        # A mass only counts as shaped when the built surface is measurably
        # curved.  Declaring `continuous` over an eight-vertex cage, or over a
        # flat-sided block, buys nothing.
        def measurably_curved(obj):
            signature = shape_signature(obj)
            if signature is None:
                return not is_shallow_form_cage(obj)
            return signature["planar_area_fraction"] < 0.55

        macro_shaped = [
            o for o in macro
            if o["name"] in valid_contract_names
            and form_prop(o, "procagen3d_topology") in ("continuous", "shell")
            and not is_shallow_form_cage(o)
            and measurably_curved(o)
        ]
        mixed_shaped = [
            o for o in structural
            if o["name"] in valid_contract_names
            and form_prop(o, "procagen3d_topology") in ("continuous", "shell")
            and not is_shallow_form_cage(o)
            and measurably_curved(o)
        ]
        mixed_assembled = [
            o for o in structural
            if o["name"] in valid_contract_names
            and form_prop(o, "procagen3d_topology") == "assembled"
        ]
        # Promised curvature must be real.  Note the direction: this never asks
        # for more curved mass, only that whatever was declared curved is.
        # Converting correct blocks into lofts to hit a quota is the failure
        # this replaces, not the behaviour it rewards.
        broken_promises = [
            f"{o['name']} ({shape_signature(o)['planar_area_fraction']:.0%} planar)"
            for o in structural
            if o["name"] in valid_contract_names
            and form_prop(o, "procagen3d_topology") in ("continuous", "shell")
            and shape_signature(o) is not None
            and not measurably_curved(o)
        ]
        if broken_promises:
            failures += 1
            fail("FORM_PROMISED_CURVATURE", "masses declared continuous/shell are "
                 f"flat-sided blocks: {broken_promises[:8]} — either build the "
                 "curvature or retag them assembled and plan them as box/prism")
        if form_profile == "curved":
            macro_volume = sum(bbox_proxy_volume(o) for o in macro)
            shaped_volume = sum(bbox_proxy_volume(o) for o in macro_shaped)
            macro_shaped_ratio = shaped_volume / macro_volume if macro_volume else 1.0
            if macro_shaped_ratio < 0.25:
                failures += 1
                fail("FORM_PROFILE_EVIDENCE", "declared form profile 'curved' but "
                     f"only {macro_shaped_ratio:.0%} of the top structural "
                     "envelope is measurably curved — if the reference is a "
                     "faceted, assembled object the fix is to declare "
                     "'rectilinear' or 'mixed', NOT to inflate correct blocks "
                     "into lofts")
        elif not mixed_shaped or not mixed_assembled:
            message = ("mixed means evidence-backed coexistence, not a "
                       "continuous-form quota: structural masses need at least "
                       f"one continuous/shell ({len(mixed_shaped)}) and one "
                       f"assembled ({len(mixed_assembled)}) representative")
            if reconstruction_plan is not None:
                failures += 1
                fail("FORM_MIXED_FAMILIES", message)
            else:
                warn("FORM_MIXED_FAMILIES_LEGACY", message
                     + "; legacy asset has no reconstruction plan")
        if not primary:
            failures += 1
            fail("FORM_CONTRACT", f"--form {form_profile} requires primary masses "
                 "tagged with procagen3d_form_role/topology/form_method; rebuild "
                 "from references/complex-forms.md")
        else:
            shallow = []
            weak_sections = []
            for obj in primary:
                topology = form_prop(obj, "procagen3d_topology")
                if topology not in ("continuous", "shell"):
                    continue
                if is_shallow_form_cage(obj):
                    shallow.append(obj["name"])
                method = form_prop(obj, "procagen3d_form_method")
                section_count = form_prop(obj, "procagen3d_section_count")
                if method in ("loft", "sweep") and (
                        not isinstance(section_count, int) or section_count < 4):
                    weak_sections.append(
                        f"{obj['name']} section_count={section_count!r}")
                planes = obj.get("base_axis_plane_counts")
                if planes and sum(count >= 5 for count in planes) < 2:
                    weak_sections.append(
                        f"{obj['name']} planes={planes}")
            if shallow:
                failures += 1
                fail("FORM_SHALLOW", "continuous/shell primary masses still use an "
                     f"eight-vertex/six-face cage: {shallow[:8]}")
            if weak_sections:
                warn("FORM_SECTIONS", "continuous forms have weak authored "
                     f"cross-section variation: {weak_sections[:6]}")

        if form_profile == "curved" and primitive_cage_ratio > 0.35:
            warn("FORM_PRIMITIVES", f"{primitive_cage_ratio:.0%} of structural "
                 "meshes are eight-vertex/six-face cages (limit 35%; e.g. "
                 f"{primitive_cages[:6]})")

    complexity_class = None
    if reconstruction_plan is not None and not plan_errors:
        complexity = reconstruction_plan["complexity"]
        complexity_class = complexity["class"]
        occupied_regions = set(complexity["occupied_regions"])
        shape_prior_ids = set()
        shape_prior_patterns = set()
        planned_shape_names = set()
        for index, entry in enumerate(reconstruction_plan["shape_priors"]):
            if not isinstance(entry, dict):
                failures += 1
                fail("SHAPE_PRIOR", f"shape_priors[{index}] must be an object")
                continue
            prior_id = entry.get("id")
            if not isinstance(prior_id, str) or not prior_id.strip():
                failures += 1
                fail("SHAPE_PRIOR", f"shape_priors[{index}].id must be a "
                     "non-empty string")
                continue
            if prior_id in shape_prior_ids:
                failures += 1
                fail("SHAPE_PRIOR", f"duplicate shape prior id: {prior_id}")
                continue
            shape_prior_ids.add(prior_id)
            pattern = entry.get("pattern")
            family = entry.get("family")
            confidence = entry.get("confidence")
            evidence = entry.get("evidence")
            rejected = entry.get("rejected_alternatives")
            edge_treatment = entry.get("edge_treatment")
            schema_errors = []
            if (not isinstance(pattern, str) or not pattern
                    or re.search(r"[A-Za-z0-9]", pattern) is None):
                schema_errors.append(
                    "pattern must contain a semantic alphanumeric literal")
            elif pattern in shape_prior_patterns:
                schema_errors.append(f"duplicate pattern {pattern!r}")
            if family not in SHAPE_FAMILIES:
                schema_errors.append(f"unknown family {family!r}")
            if confidence not in ("high", "medium", "low"):
                schema_errors.append("confidence must be high, medium, or low")
            if (not isinstance(evidence, list) or not evidence
                    or any(not isinstance(item, str) or not item.strip()
                           for item in evidence)):
                schema_errors.append("evidence must be a non-empty string list")
            if (not isinstance(rejected, list) or not rejected
                    or any(not isinstance(item, str) or not item.strip()
                           for item in rejected)):
                schema_errors.append(
                    "rejected_alternatives must be a non-empty string list")
            elif family in SHAPE_FAMILIES and any(re.match(
                    rf"^\s*{re.escape(family)}(?:\s|:|$)", item,
                    flags=re.IGNORECASE) for item in rejected):
                schema_errors.append(
                    "chosen family cannot also be a rejected alternative")
            if not isinstance(edge_treatment, str) or not edge_treatment.strip():
                schema_errors.append("edge_treatment must be a non-empty string")
            if schema_errors:
                failures += 1
                fail("SHAPE_PRIOR", f"{prior_id}: " + "; ".join(schema_errors))
                continue
            shape_prior_patterns.add(pattern)
            matched = [obj for obj in meshes
                       if fnmatch.fnmatchcase(obj["name"], pattern)]
            if not matched:
                failures += 1
                fail("SHAPE_PRIOR", f"{prior_id}: pattern {pattern!r} matches "
                     "no renderable mesh")
                continue
            planned_shape_names.update(obj["name"] for obj in matched)
            wrong_family = [
                f"{obj['name']}={form_prop(obj, 'procagen3d_shape_family')!r}"
                for obj in matched
                if form_prop(obj, "procagen3d_shape_family") != family
            ]
            if wrong_family:
                failures += 1
                fail("SHAPE_PRIOR", f"{prior_id}: planned family {family!r} does "
                     f"not match program tags {wrong_family[:8]}")
            wrong_method = [
                f"{obj['name']}={form_prop(obj, 'procagen3d_form_method')!r}"
                for obj in matched
                if form_prop(obj, "procagen3d_form_method")
                not in SHAPE_FAMILY_METHODS[family]
            ]
            if wrong_method:
                failures += 1
                fail("SHAPE_METHOD", f"{prior_id}: family {family!r} is "
                     f"incompatible with construction {wrong_method[:8]}")

        uncovered_primary = [obj["name"] for obj in primary
                             if obj["name"] not in planned_shape_names]
        if uncovered_primary:
            failures += 1
            fail("SHAPE_PRIOR_COVERAGE", "primary masses missing from the "
                 f"shape-prior plan: {uncovered_primary[:10]}")
        plan_macro = sorted(structural, key=bbox_proxy_volume, reverse=True)[:12]
        covered_macro = [obj for obj in plan_macro
                         if obj["name"] in planned_shape_names]
        plan_macro_ratio = len(covered_macro) / len(plan_macro) if plan_macro else 1.0
        if plan_macro_ratio < 0.75:
            failures += 1
            fail("SHAPE_PRIOR_COVERAGE", f"only {len(covered_macro)}/"
                 f"{len(plan_macro)} largest structural masses occur in "
                 "shape_priors (need 75%); uncovered: "
                 f"{[obj['name'] for obj in plan_macro if obj not in covered_macro][:8]}")

        feature_ids = set()
        feature_patterns = set()
        visible_feature_count = 0
        identity_count = 0
        planned_visible_regions = set()
        missing_features = []
        for index, entry in enumerate(reconstruction_plan["detail_features"]):
            if not isinstance(entry, dict):
                failures += 1
                fail("DETAIL_PLAN", f"detail_features[{index}] must be an object")
                continue
            feature_id = entry.get("id")
            if not isinstance(feature_id, str) or not feature_id.strip():
                failures += 1
                fail("DETAIL_PLAN", f"detail_features[{index}].id must be a "
                     "non-empty string")
                continue
            if feature_id in feature_ids:
                failures += 1
                fail("DETAIL_PLAN", f"duplicate detail feature id: {feature_id}")
                continue
            feature_ids.add(feature_id)
            pattern = entry.get("pattern")
            priority = entry.get("priority")
            region = entry.get("region")
            minimum = entry.get("min_count", 1)
            required_value = entry.get("required", priority != "inferred")
            if (not isinstance(pattern, str) or not pattern
                    or re.search(r"[A-Za-z0-9]", pattern) is None
                    or priority not in DETAIL_PRIORITIES
                    or not isinstance(region, str) or not region.strip()
                    or not isinstance(minimum, int) or isinstance(minimum, bool)
                    or minimum < 1 or not isinstance(required_value, bool)):
                failures += 1
                fail("DETAIL_PLAN", f"{feature_id}: requires non-empty pattern/"
                     "region, a valid priority, integer min_count >= 1, and a "
                     "boolean required value")
                continue
            if pattern in feature_patterns:
                failures += 1
                fail("DETAIL_PLAN", f"{feature_id}: duplicate feature pattern "
                     f"{pattern!r}; each visible group needs a distinct semantic "
                     "mesh pattern")
                continue
            feature_patterns.add(pattern)
            if priority != "inferred" and region not in OBJECT_REGIONS:
                failures += 1
                fail("DETAIL_PLAN", f"{feature_id}: visible feature region "
                     f"{region!r} is not an object-centric 3x3 region")
                continue
            if priority != "inferred" and region not in occupied_regions:
                failures += 1
                fail("DETAIL_PLAN", f"{feature_id}: region {region!r} is not "
                     "declared in complexity.occupied_regions")
                continue
            if priority != "inferred" and not required_value:
                failures += 1
                fail("DETAIL_PLAN", f"{feature_id}: every visible/non-inferred "
                     "feature group must be required")
                continue
            if priority != "inferred":
                visible_feature_count += 1
                planned_visible_regions.add(region)
            if priority == "identity":
                identity_count += 1
            required = required_value
            count = sum(1 for obj in meshes
                        if fnmatch.fnmatchcase(obj["name"], pattern))
            if args.tier != "quick" and required and count < minimum:
                missing_features.append(
                    f"{feature_id} {count}/{minimum} ({pattern})")
        if args.tier != "quick" and missing_features:
            failures += 1
            fail("DETAIL_COVERAGE", "required reference feature groups missing: "
                 + "; ".join(missing_features[:12]))
        lower, upper = COMPLEXITY_BANDS[complexity_class]
        if visible_feature_count < lower or (
                upper is not None and visible_feature_count > upper):
            failures += 1
            interval = f"{lower}+" if upper is None else f"{lower}..{upper}"
            fail("COMPLEXITY_CLASS", f"{complexity_class} requires {interval} "
                 f"visible feature groups; plan declares {visible_feature_count}")
        minimum_regions = MIN_OCCUPIED_REGIONS[complexity_class]
        if len(occupied_regions) < minimum_regions:
            failures += 1
            fail("COMPLEXITY_REGIONS", f"{complexity_class} requires at least "
                 f"{minimum_regions} occupied object regions; plan declares "
                 f"{len(occupied_regions)}")
        uncovered_regions = sorted(occupied_regions - planned_visible_regions)
        if uncovered_regions:
            failures += 1
            fail("DETAIL_REGIONS", "occupied regions without a required visible "
                 f"feature group: {uncovered_regions}")

        # Measured region density.  Name patterns prove only that meshes were
        # named after the plan; this counts the meshes that physically occupy
        # each declared region, so an empty torso cannot be paid for with two
        # hundred panel slivers somewhere else.
        if args.tier != "quick":
            occupancy = region_occupancy(graph, meshes)
            floor = REGION_MESH_FLOOR[complexity_class]
            sparse = [f"{region} has {occupancy.get(region, 0)} meshes"
                      for region in sorted(occupied_regions)
                      if occupancy.get(region, 0) < floor]
            if sparse:
                failures += 1
                fail("REGION_DENSITY", f"{complexity_class} plan declares these "
                     f"regions occupied but the geometry is nearly empty there "
                     f"(floor {floor} meshes): {sparse[:8]} — spend the budget on "
                     "the empty region, not on more repetition elsewhere")
        identity_minimum = {
            "simple": 1, "moderate": 2, "complex": 3, "extreme": 4,
        }[complexity_class]
        if identity_count < identity_minimum:
            failures += 1
            fail("DETAIL_IDENTITY", f"{complexity_class} plan needs at least "
                 f"{identity_minimum} identity feature groups; found {identity_count}")

    floors = {"standard": (40, 8000, 6), "showcase": (150, 25000, 12)}
    if complexity_class is not None and args.tier in ADAPTIVE_DETAIL_FLOORS:
        floors = {
            **floors,
            args.tier: ADAPTIVE_DETAIL_FLOORS[args.tier][complexity_class],
        }
    if args.tier in floors:
        mesh_floor, tri_floor, mat_floor = floors[args.tier]
        misses = []
        if totals["meshes"] < mesh_floor:
            misses.append(f"meshes {totals['meshes']}/{mesh_floor}")
        if totals["triangles"] < tri_floor:
            misses.append(f"tris {totals['triangles']}/{tri_floor}")
        if len(mats) < mat_floor:
            misses.append(f"materials {len(mats)}/{mat_floor}")
        if boxy_ratio > 0.4:
            misses.append(f"raw unbeveled boxes {boxy_ratio:.0%} — bevel or "
                          f"chamfer them, do not curve them (e.g. {boxy[:4]})")
        if misses:
            message = (f"{args.tier} floors not met"
                       + (f" for {complexity_class} complexity" if complexity_class
                          else "")
                       + ": " + "; ".join(misses)
                       + " — fix the named form/detail deficit "
                       "(references/detail.md; references/complex-forms.md)")
            if reconstruction_plan is not None and args.tier == "showcase":
                failures += 1
                fail("LOW_DETAIL", message)
            else:
                warn("LOW_DETAIL", message)

    world = graph.get("world_bbox", {}).get("size", ["?"] * 3)
    print(f"{OK if not failures else '[PROCAGEN3D:SUMMARY]'} {totals['meshes']} meshes, "
          f"{totals['triangles']} tris, {len(mats)} materials, "
          f"{len(graph.get('joints', []))} joints, "
          f"world size {world}, {failures} failure(s)")
    return 1 if failures else 0


# ---------------------------------------------------------------- mini YAML

def _strip_comment(line):
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def _scalar(tok):
    tok = tok.strip()
    if not tok:
        return None
    if len(tok) >= 2 and tok[0] in "\"'" and tok[-1] == tok[0]:
        return tok[1:-1]
    if tok.startswith("[") and tok.endswith("]"):
        inner = tok[1:-1].strip()
        return [_scalar(t) for t in inner.split(",")] if inner else []
    low = tok.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~"):
        return None
    for cast in (int, float):
        try:
            return cast(tok)
        except ValueError:
            pass
    return tok


def parse_simple_yaml(text):
    """Strict-subset YAML: 2-space indent maps, '- ' lists, plain scalars,
    flow lists like [1, 2]. Enough for ProcAgen3D spec files."""
    items = []
    for raw in text.splitlines():
        line = _strip_comment(raw.replace("\t", "  "))
        if line.strip():
            items.append([len(line) - len(line.lstrip()), line.strip()])
    pos = [0]

    def parse(indent):
        if pos[0] >= len(items):
            return None
        return (parse_list if items[pos[0]][1].startswith("- ")
                or items[pos[0]][1] == "-" else parse_map)(indent)

    def parse_map(indent):
        out = {}
        while pos[0] < len(items):
            ind, line = items[pos[0]]
            if ind != indent or line.startswith("- "):
                if ind > indent:
                    raise ValueError(f"bad indent near: {line!r}")
                break
            if ":" not in line:
                raise ValueError(f"expected 'key: value' near: {line!r}")
            key, _, rest = line.partition(":")
            pos[0] += 1
            if rest.strip():
                out[_scalar(key)] = _scalar(rest)
            elif pos[0] < len(items) and items[pos[0]][0] > indent:
                out[_scalar(key)] = parse(items[pos[0]][0])
            else:
                out[_scalar(key)] = None
        return out

    def parse_list(indent):
        out = []
        while pos[0] < len(items):
            ind, line = items[pos[0]]
            if ind != indent or not (line.startswith("- ") or line == "-"):
                if ind > indent:
                    raise ValueError(f"bad indent near: {line!r}")
                break
            content = line[2:].strip()
            if not content:
                pos[0] += 1
                out.append(parse(items[pos[0]][0])
                           if pos[0] < len(items) and items[pos[0]][0] > indent
                           else None)
            elif ":" in content and content[0] not in "\"'[":
                items[pos[0]] = [ind + 2, content]
                out.append(parse_map(ind + 2))
            else:
                pos[0] += 1
                out.append(_scalar(content))
        return out

    return parse(items[0][0]) if items else {}


def load_spec(path):
    text = Path(path).read_text()
    if str(path).endswith(".json"):
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        return parse_simple_yaml(text)


# ---------------------------------------------------------------- scorer

def expand_to_meshes(matched, graph, kids):
    """A matched group empty stands for the union of its mesh descendants."""
    by_name = {o["name"]: o for o in graph["objects"]}
    names = subtree([o["name"] for o in matched], kids)
    return [by_name[n] for n in names
            if by_name[n]["type"] == "MESH" and "bbox_world_min" in by_name[n]]


def bbox_union(meshes):
    lo = [min(m["bbox_world_min"][i] for m in meshes) for i in range(3)]
    hi = [max(m["bbox_world_max"][i] for m in meshes) for i in range(3)]
    return lo, hi


def center_of(obj, graph, kids):
    meshes = expand_to_meshes([obj], graph, kids)
    if meshes:
        lo, hi = bbox_union(meshes)
        return [(lo[i] + hi[i]) / 2 for i in range(3)]
    return obj["origin_world"]


def parse_tolerance(c, value):
    tol = c.get("tolerance")
    if isinstance(tol, str) and tol.endswith("%"):
        frac = float(tol[:-1]) / 100
        return value * (1 - frac), value * (1 + frac), f"{value:g} ±{tol}"
    if isinstance(tol, (int, float)):
        return value - tol, value + tol, f"{value:g} ±{tol:g}"
    return value, value, f"{value:g} (exact)"


def measure_constraint(c, graph, kids):
    """Returns (target_str, measured_str, passed, note)."""
    measure = c.get("measure")
    pattern = c.get("pattern", "*")

    if measure == "count":
        n = len(match_objects(graph, pattern, types={"MESH"}))
        target = int(c.get("equals", c.get("target", 0)))
        return f"= {target}", str(n), n == target, ""

    if measure == "exists":
        n = len(match_objects(graph, pattern))
        return ">= 1", str(n), n >= 1, ""

    if measure == "joint":
        joints = [j for j in graph.get("joints", [])
                  if fnmatch.fnmatchcase(j["name"], pattern)
                  and ("type" not in c or j.get("type") == c["type"])]
        want_type = f" {c['type']}" if "type" in c else ""
        if "equals" in c:
            target = int(c["equals"])
            return (f"={target}{want_type} joint(s)", str(len(joints)),
                    len(joints) == target, "")
        min_count = int(c.get("min_count", 1))
        return (f">={min_count}{want_type} joint(s)", str(len(joints)),
                len(joints) >= min_count, "")

    if measure in ("dimension", "distance", "symmetry"):
        if measure == "dimension":
            matched = match_objects(graph, pattern)
            if not matched:
                return "?", "unmeasurable", False, f"no object matches '{pattern}'"
            metric = c.get("metric", "extent_max")
            if metric == "origin_z":
                if len(matched) > 1:
                    return "?", "ambiguous", False, f"'{pattern}' matches {len(matched)} objects"
                got = matched[0]["origin_world"][2]
            else:
                meshes = expand_to_meshes(matched, graph, kids)
                if not meshes:
                    return "?", "unmeasurable", False, f"'{pattern}' has no mesh geometry"
                lo, hi = bbox_union(meshes)
                size = [hi[i] - lo[i] for i in range(3)]
                got = {
                    "extent_x": size[0], "extent_y": size[1], "extent_z": size[2],
                    "height": size[2], "extent_max": max(size),
                    "extent_min": min(size), "diameter_xy": max(size[0], size[1]),
                    "top_z": hi[2], "bottom_z": lo[2],
                }.get(metric)
                if got is None:
                    return "?", "?", False, f"unknown metric '{metric}'"
        elif measure == "distance":
            names = c.get("between", [])
            found = [next((o for o in graph["objects"]
                           if fnmatch.fnmatchcase(o["name"], n)), None)
                     for n in names]
            if len(found) != 2 or None in found:
                return "?", "unmeasurable", False, f"'between' needs 2 resolvable names: {names}"
            a, b = (o["origin_world"] for o in found)
            axis = c.get("axis")
            if axis in ("x", "y", "z"):
                got = abs(a["xyz".index(axis)] - b["xyz".index(axis)])
            else:
                got = sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5
        else:  # symmetry
            pair = c.get("pair", [])
            found = [next((o for o in graph["objects"]
                           if fnmatch.fnmatchcase(o["name"], n)), None)
                     for n in pair]
            if len(found) != 2 or None in found:
                return "?", "unmeasurable", False, f"'pair' needs 2 resolvable names: {pair}"
            plane = "xyz".index(c.get("plane", "x"))
            ca = center_of(found[0], graph, kids)
            cb = center_of(found[1], graph, kids)
            mirrored = list(ca)
            mirrored[plane] = -mirrored[plane]
            got = sum((mirrored[i] - cb[i]) ** 2 for i in range(3)) ** 0.5
            tol = c.get("tolerance", 0.02)
            return (f"mirror gap <= {tol:g}", f"{got:.4f}", got <= tol, "")

        if "range" in c:
            lo_t, hi_t = float(c["range"][0]), float(c["range"][1])
            desc = f"[{lo_t:g}, {hi_t:g}]"
        else:
            value = float(c.get("value", c.get("target", 0)))
            lo_t, hi_t, desc = parse_tolerance(c, value)
        return desc, f"{got:.4f}", lo_t - 1e-9 <= got <= hi_t + 1e-9, ""

    return "?", "?", False, f"unknown measure '{measure}'"


def cmd_score(args):
    graph = load_graph(args.dir)
    spec = load_spec(args.spec)
    kids = children_map(graph)
    constraints = spec.get("constraints", [])
    if not constraints:
        sys.exit(f"ProcAgen3D: no constraints found in {args.spec}")
    rows, passed = [], 0
    for c in constraints:
        cid = c.get("id", c.get("measure", "?"))
        target, measured, ok_flag, note = measure_constraint(c, graph, kids)
        rows.append({"id": cid, "target": target, "measured": measured,
                     "pass": ok_flag, "note": note})
        passed += ok_flag
    width = max(len(r["id"]) for r in rows)
    twidth = max(len(r["target"]) for r in rows)
    print(f"ProcAgen3D score — {spec.get('object', Path(args.spec).stem)}")
    for r in rows:
        verdict = "PASS" if r["pass"] else "FAIL"
        note = f"  ({r['note']})" if r["note"] else ""
        print(f"  {r['id']:<{width}}  target {r['target']:<{twidth}}  "
              f"measured {r['measured']:>10}  {verdict}{note}")
    print(f"  -> {passed}/{len(rows)} constraints satisfied")
    report = {"object": spec.get("object"), "passed": passed,
              "total": len(rows), "constraints": rows}
    (Path(args.dir) / "score_report.json").write_text(json.dumps(report, indent=2))
    return 0 if passed == len(rows) else 1


# ---------------------------------------------------------------- guard

PART_FUNC_RE = re.compile(r"^def\s+(build_[A-Za-z0-9_]+)\s*\(", re.MULTILINE)


def cmd_guard(args):
    old = Path(args.old).read_text()
    new = Path(args.new).read_text()
    failures = 0

    if "def build(" not in new:
        failures += 1
        fail("ENTRY_POINT_LOST", "corrected program dropped def build()")

    if len(new) < 0.85 * len(old) and not args.allow_shrink:
        failures += 1
        fail("SOURCE_SHRINK", f"program shrank {len(old)} -> {len(new)} bytes "
             f"(> 15%) — a repair must not gut the program; pass "
             f"--allow-shrink only if this is an intended rewrite")

    dropped = (set(PART_FUNC_RE.findall(old)) - set(PART_FUNC_RE.findall(new))
               - set(args.allow_drop or []))
    if dropped:
        failures += 1
        fail("DROPPED_PART_FUNCS", f"{sorted(dropped)} vanished — repairs must "
             f"preserve parts; pass --allow-drop NAME if a rename/merge is intended")

    old_joints = old.count("add_joint(") - old.count("def add_joint(")
    new_joints = new.count("add_joint(") - new.count("def add_joint(")
    if new_joints < old_joints:
        warn("FEWER_JOINTS", f"joint declarations {old_joints} -> {new_joints}")

    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), n=0))
    changed = sum(1 for l in diff if l[:1] in "+-" and l[:3] not in ("+++", "---"))
    if failures == 0:
        print(f"{OK} guard passed ({changed} changed lines)")
    return 1 if failures else 0


# ---------------------------------------------------------------- edit gates

def graph_maps(dir_path):
    graph = load_graph(dir_path)
    return graph, {o["name"]: o for o in graph["objects"]}, children_map(graph)


def mesh_signature(o):
    return (o.get("bbox_world_min"), o.get("bbox_world_max"),
            o.get("vertex_count"), o.get("origin_world"))


def cmd_edit_gates(args):
    base_graph, base_by, base_kids = graph_maps(args.base)
    edit_graph, edit_by, edit_kids = graph_maps(args.edited)
    tol = args.tol
    failures = 0

    def gate(name, ok_flag, detail=""):
        nonlocal failures
        state = "PASS" if ok_flag else "FAIL"
        print(f"  {name:<24} {state}{('  ' + detail) if detail else ''}")
        failures += 0 if ok_flag else 1

    print(f"ProcAgen3D edit gates — target '{args.target}'")

    glb_ok = (Path(args.edited) / "model.glb").exists()
    gate("artifact_validity", glb_ok, "" if glb_ok else "edited model.glb missing")

    base_matched = [o["name"] for o in match_objects(base_graph, args.target)]
    edit_matched = [o["name"] for o in match_objects(edit_graph, args.target)]
    mode = args.mode
    if mode == "auto":
        mode = "modify" if base_matched else "add"
    addressable = bool(base_matched) if mode == "modify" else bool(edit_matched)
    gate("target_addressability", addressable,
         f"mode={mode}, base={len(base_matched)}, edited={len(edit_matched)}")

    base_src = (Path(args.base) / "program.py")
    edit_src = (Path(args.edited) / "program.py")
    src_changed = (base_src.exists() and edit_src.exists()
                   and base_src.read_text() != edit_src.read_text())
    target_set_base = subtree(base_matched, base_kids)
    target_set_edit = subtree(edit_matched, edit_kids)
    if mode == "add":
        geom_changed = bool(set(edit_matched) - set(base_by))
    else:
        geom_changed = any(
            n not in edit_by or mesh_signature(base_by[n]) != mesh_signature(edit_by[n])
            for n in target_set_base)
    gate("source_and_glb_change", src_changed and geom_changed,
         f"source_changed={src_changed}, target_geometry_changed={geom_changed}")

    non_target = [n for n in base_by if n not in target_set_base]
    missing = [n for n in non_target if n not in edit_by]
    reparented = [n for n in non_target if n in edit_by
                  and base_by[n]["parent"] != edit_by[n]["parent"]
                  and base_by[n]["parent"] not in target_set_base]
    gate("hierarchy_preservation", not missing and not reparented,
         f"missing={missing[:5]}, reparented={reparented[:5]}"
         if (missing or reparented) else "")

    offenders = []
    for n in non_target:
        b = base_by[n]
        e = edit_by.get(n)
        if e is None or b["type"] != "MESH":
            continue
        if b.get("vertex_count") != e.get("vertex_count"):
            offenders.append(f"{n} (topology)")
            continue
        for key in ("bbox_world_min", "bbox_world_max", "origin_world"):
            if any(abs(x - y) > tol for x, y in zip(b.get(key, []), e.get(key, []))):
                offenders.append(f"{n} ({key})")
                break
    gate("non_target_locality", not offenders,
         f"moved/changed: {offenders[:10]}" if offenders else f"tol={tol:g}")

    added = sorted(set(edit_by) - set(base_by))
    removed = sorted(set(base_by) - set(edit_by))
    print(f"  nodes added: {added[:10] or 'none'}; removed: {removed[:10] or 'none'}")
    print(f"  -> {5 - failures}/5 gates passed")
    return 1 if failures else 0


# ---------------------------------------------------------------- entrypoint

def main():
    parser = argparse.ArgumentParser(
        prog="procagen3d", description="ProcAgen3D code-native 3D asset pipeline")
    parser.add_argument("--blender", help="path to blender executable")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="build program, export GLB, render views")
    p.add_argument("program")
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--engine", default="workbench",
                   choices=["workbench", "eevee", "cycles"])
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--form-diagnostics", action="store_true",
                   help="also render a neutral clay six-view form sheet")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("render", help="re-render canonical views")
    p.add_argument("dir")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--engine", default="workbench",
                   choices=["workbench", "eevee", "cycles"])
    p.add_argument("--form-diagnostics", action="store_true",
                   help="also render a neutral clay six-view form sheet")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser(
        "fit", help="render and score registered silhouette/pose reference fit")
    p.add_argument("dir")
    p.add_argument("--spec", required=True,
                   help="fit_spec.json (copied into the asset directory)")
    p.add_argument("--engine", default="workbench",
                   choices=["workbench", "eevee", "cycles"])
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser(
        "solve-camera",
        help="resect the reference camera from observed image landmarks")
    p.add_argument("dir")
    p.add_argument("--spec", required=True, help="fit_spec.json with landmarks")
    p.add_argument("--max-rms", type=float, default=0.02,
                   help="fail above this RMS reprojection error in uv units")
    p.add_argument("--free-shift", action="store_true",
                   help="also solve lens shift (only for a cropped reference)")
    p.add_argument("--fix", nargs="*", default=[],
                   help="camera parameters to hold at their declared values")
    p.add_argument("--solve-root", action="store_true",
                   help="also estimate a rigid root lean (pitch/yaw/roll): if "
                        "this drops the residual, the subject is not upright")
    p.set_defaults(func=cmd_solve_camera)

    p = sub.add_parser("check", help="deterministic scene-graph gates")
    p.add_argument("dir")
    p.add_argument("--tier", choices=["quick", "standard", "showcase"],
                   default="standard",
                   help="detail-floor tier (references/detail.md)")
    p.add_argument("--form", choices=["auto", "rectilinear", "curved", "mixed"],
                   default="auto",
                   help="primary-form profile (references/complex-forms.md)")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("joints", help="validate articulation")
    p.add_argument("dir")
    p.add_argument("--strict", action="store_true",
                   help="include the joint's parent part in sweep collisions")
    p.set_defaults(func=cmd_joints)

    p = sub.add_parser("score", help="measure spec constraints against the build")
    p.add_argument("dir")
    p.add_argument("--spec", required=True)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("guard", help="doctrine guard between repair iterations")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--allow-shrink", action="store_true")
    p.add_argument("--allow-drop", action="append")
    p.set_defaults(func=cmd_guard)

    p = sub.add_parser("edit-gates", help="deterministic local-edit gates")
    p.add_argument("base")
    p.add_argument("edited")
    p.add_argument("--target", required=True)
    p.add_argument("--mode", default="auto", choices=["auto", "modify", "add"])
    p.add_argument("--tol", type=float, default=1e-4)
    p.set_defaults(func=cmd_edit_gates)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
