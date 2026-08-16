"""Deterministic local-edit gates against a base vs edited scene graph."""

from pathlib import Path

from .graph import children_map, load_graph, match_objects, subtree


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
