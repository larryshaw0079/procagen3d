"""Load scene_graph.json and walk the object tree."""

import fnmatch
import hashlib
import json
import sys
from pathlib import Path


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
