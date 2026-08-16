"""Deterministic scene-graph gates (names, joints, form contract, detail floors)."""

import re

from .fit_evidence import image_fit_errors
from .graph import load_graph
from .tags import OK, fail, warn


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

    fit_errors = image_fit_errors(args.dir)
    if fit_errors:
        failures += 1
        fail("REFERENCE_FIT", "; ".join(fit_errors))

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
