"""Measure spec.yaml constraints against a built scene graph."""

import fnmatch
import json
import sys
from pathlib import Path

from .graph import children_map, load_graph, match_objects, subtree
from .yaml_lite import load_spec


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
