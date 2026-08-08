#!/usr/bin/env python3
"""ProcAgen3D driver — code-native 3D asset generation (arXiv:2607.22738).

Pure Python 3.10+ stdlib. Blender-side stages are dispatched to
blender_stages.py running under Blender's bundled Python; everything else
(check/score/guard/edit-gates) runs under plain python3.

Subcommands:
    build      <program.py> --out DIR    build, export GLB, render views
    render     <dir>                     re-render canonical views
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
import json
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


def run_blender(stage_args, blender=None):
    cmd = [
        find_blender(blender), "--background", "--factory-startup",
        "--python-exit-code", "1",
        "--python", str(SCRIPT_DIR / "blender_stages.py"), "--", *stage_args,
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    code = proc.returncode
    for line in proc.stdout.splitlines():
        m = re.match(r"^PROCAGEN3D_EXIT:(\d+)$", line.strip())
        if m:
            code = int(m.group(1))
        elif line.strip() and not line.startswith(
                ("Blender ", "Read prefs", "Time:", "Saved:", "Info:",
                 "Fra:", "Read blend:", "WARN (gpu", "INFO ",
                 "Color management")) and "| INFO" not in line:
            print(line)
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
    "decal",
}
TOPOLOGY_METHODS = {
    "continuous": {"loft", "sweep", "revolve", "subdivision",
                   "surface-grid", "nurbs"},
    "shell": {"loft", "sweep", "subdivision", "surface-grid", "nurbs",
              "solidify"},
    "assembled": {"primitive-csg", "profile-extrude", "boolean", "revolve"},
    "strand": {"sweep", "curve", "nurbs"},
    "relief": {"curve", "profile-extrude", "primitive-csg", "boolean",
               "decal"},
}


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
    if form_profile in ("curved", "mixed"):
        macro = sorted(
            structural,
            key=bbox_proxy_volume,
            reverse=True,
        )[:12]
        contracted_macro = [o for o in macro if o["name"] in valid_contract_names]
        contract_ratio = len(contracted_macro) / len(macro) if macro else 1.0
        required_contract_ratio = 0.75 if form_profile == "curved" else 0.60
        if contract_ratio < required_contract_ratio:
            failures += 1
            untagged = [o["name"] for o in macro
                        if o["name"] not in valid_contract_names]
            fail("FORM_TAG_COVERAGE", f"only {len(contracted_macro)}/{len(macro)} "
                 f"largest structural masses have valid form contracts "
                 f"(need {required_contract_ratio:.0%}); untagged: {untagged[:8]}")
        macro_shaped = [
            o for o in macro
            if o["name"] in valid_contract_names
            and form_prop(o, "procagen3d_topology") in ("continuous", "shell")
            and not is_shallow_form_cage(o)
        ]
        macro_volume = sum(bbox_proxy_volume(o) for o in macro)
        shaped_volume = sum(bbox_proxy_volume(o) for o in macro_shaped)
        macro_shaped_ratio = shaped_volume / macro_volume if macro_volume else 1.0
        required_macro_shaped = 0.50 if form_profile == "curved" else 0.30
        if macro_shaped_ratio < required_macro_shaped:
            failures += 1
            fail("FORM_MACRO_COVERAGE", f"genuine continuous/shell forms cover "
                 f"only {macro_shaped_ratio:.0%} of the top structural envelope "
                 f"(need {required_macro_shaped:.0%}; {len(macro_shaped)}/"
                 f"{len(macro)} masses); a small token loft cannot excuse a "
                 "box-built body")
        if not primary:
            failures += 1
            fail("FORM_CONTRACT", f"--form {form_profile} requires primary masses "
                 "tagged with procagen3d_form_role/topology/form_method; rebuild "
                 "from references/complex-forms.md")
        else:
            shaped = [o for o in primary
                      if form_prop(o, "procagen3d_topology") in
                      ("continuous", "shell")]
            shaped_ratio = len(shaped) / len(primary)
            required = 0.50 if form_profile == "curved" else 0.30
            if shaped_ratio < required:
                failures += 1
                fail("FORM_COVERAGE", f"{form_profile} target has only "
                     f"{len(shaped)}/{len(primary)} ({shaped_ratio:.0%}) primary "
                     "masses routed as continuous/shell forms")

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

        primitive_limit = 0.35 if form_profile == "curved" else 0.45
        if primitive_cage_ratio > primitive_limit:
            warn("FORM_PRIMITIVES", f"{primitive_cage_ratio:.0%} of structural "
                 f"meshes are eight-vertex/six-face cages (limit "
                 f"{primitive_limit:.0%}; e.g. {primitive_cages[:6]})")

    floors = {"standard": (40, 8000, 6), "showcase": (150, 25000, 12)}
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
            misses.append(f"unbeveled-box meshes {boxy_ratio:.0%} "
                          f"(e.g. {boxy[:4]})")
        if misses:
            warn("LOW_DETAIL", f"{args.tier} floors not met: "
                 + "; ".join(misses)
                 + " — fix the named form/detail deficit "
                 "(references/detail.md; references/complex-forms.md)")

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
