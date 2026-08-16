"""Load spec files: JSON, PyYAML if present, otherwise a strict YAML subset."""

import json
from pathlib import Path


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
