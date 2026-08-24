"""Pure-stdlib inspection of binary glTF (GLB) reference models.

The probe is deliberately conservative: scene wrappers and generated names are
not semantic evidence.  A file with one mesh, one primitive, and one material
is reported as a merged drawable even when it is nested below named nodes.

Only metadata and rest-pose geometry are inspected here.  Blender remains the
authority for evaluated meshes, generated normals, rendering, and formats that
require extensions such as Draco or meshopt.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import itertools
import json
import math
import re
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import unquote_to_bytes


GLB_MAGIC = b"glTF"
GLB_VERSION = 2
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
MAX_DECODED_POSITION_COUNT = 2_000_000

_COMPONENTS: dict[int, tuple[str, int]] = {
    5120: ("b", 1),   # BYTE
    5121: ("B", 1),   # UNSIGNED_BYTE
    5122: ("h", 2),   # SHORT
    5123: ("H", 2),   # UNSIGNED_SHORT
    5124: ("i", 4),   # tolerated for diagnostics; not a core glTF accessor type
    5125: ("I", 4),   # UNSIGNED_INT
    5126: ("f", 4),   # FLOAT
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
_TRIANGLE_MODES = {4, 5, 6}
_GENERIC_NAME = re.compile(
    r"^(?:world|root|scene|node[-_. ]?\d*|mesh[-_. ]?\d*|geometry[-_. ]?\d+|"
    r"object[-_. ]?\d+|primitive[-_. ]?\d+)$",
    re.IGNORECASE,
)


class GLBProbeError(ValueError):
    """Raised when a GLB container or referenced byte range is invalid."""


def _is_data_uri(value: Any) -> bool:
    return isinstance(value, str) and value[:5].lower() == "data:"


def _decode_data_uri(uri: str, *, label: str) -> bytes:
    if not _is_data_uri(uri):
        raise GLBProbeError(f"{label} is not a data URI")
    metadata, separator, encoded = uri[5:].partition(",")
    if not separator:
        raise GLBProbeError(f"{label} data URI has no comma separator")
    tokens = metadata.split(";") if metadata else []
    is_base64 = bool(tokens and tokens[-1].lower() == "base64")
    payload = unquote_to_bytes(encoded)
    if not is_base64:
        return payload
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise GLBProbeError(f"{label} data URI has invalid base64 payload") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chunk_name(chunk_type: int) -> str:
    try:
        return chunk_type.to_bytes(4, "little").decode("ascii", errors="replace")
    except OverflowError:  # pragma: no cover - struct already constrains this to u32
        return hex(chunk_type)


def parse_glb(path: Path) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    """Parse and structurally validate a GLB v2 container.

    The returned binary payload excludes the eight-byte BIN chunk header but
    includes the chunk's alignment padding, as required by glTF.
    """

    path = path.expanduser().resolve()
    if not path.is_file():
        raise GLBProbeError(f"GLB does not exist: {path}")
    data = path.read_bytes()
    if len(data) < 12:
        raise GLBProbeError("GLB is shorter than its 12-byte header")
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != GLB_MAGIC:
        raise GLBProbeError("file does not start with the binary glTF magic")
    if version != GLB_VERSION:
        raise GLBProbeError(f"unsupported GLB version {version}; expected 2")
    if declared_length != len(data):
        raise GLBProbeError(
            f"header length {declared_length} does not match file size {len(data)}"
        )

    cursor = 12
    json_payload: bytes | None = None
    binary_payload: bytes | None = None
    chunks: list[dict[str, Any]] = []
    while cursor < len(data):
        if cursor % 4:
            raise GLBProbeError(f"chunk header at byte {cursor} is not four-byte aligned")
        if cursor + 8 > len(data):
            raise GLBProbeError("truncated GLB chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", data, cursor)
        cursor += 8
        end = cursor + chunk_length
        if end > len(data):
            raise GLBProbeError("GLB chunk extends beyond the declared file length")
        if chunk_length % 4:
            raise GLBProbeError(
                f"{_chunk_name(chunk_type)!r} chunk length {chunk_length} is not four-byte aligned"
            )
        payload = data[cursor:end]
        cursor = end
        chunks.append(
            {
                "index": len(chunks),
                "type": _chunk_name(chunk_type),
                "type_code": chunk_type,
                "bytes": chunk_length,
            }
        )
        if chunk_type == JSON_CHUNK:
            if json_payload is not None:
                raise GLBProbeError("GLB contains more than one JSON chunk")
            if len(chunks) != 1:
                raise GLBProbeError("the JSON chunk must be the first GLB chunk")
            json_payload = payload.rstrip(b" \t\r\n\x00")
        elif chunk_type == BIN_CHUNK:
            if binary_payload is not None:
                raise GLBProbeError("GLB contains more than one BIN chunk")
            binary_payload = payload

    if json_payload is None:
        raise GLBProbeError("GLB contains no JSON chunk")
    try:
        document = json.loads(json_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GLBProbeError(f"invalid JSON chunk: {exc}") from exc
    if not isinstance(document, dict):
        raise GLBProbeError("the glTF JSON root must be an object")

    binary = binary_payload or b""
    buffers = document.get("buffers", [])
    if isinstance(buffers, list) and buffers:
        first = buffers[0]
        if isinstance(first, dict) and "uri" not in first:
            required = int(first.get("byteLength", 0))
            if required > len(binary):
                raise GLBProbeError(
                    f"buffer 0 declares {required} bytes but the BIN chunk has {len(binary)}"
                )
    return document, binary, {
        "version": version,
        "declared_length": declared_length,
        "json_chunk_bytes": len(json_payload),
        "bin_chunk_bytes": len(binary),
        "chunks": chunks,
    }


def _as_list(document: dict[str, Any], key: str) -> list[Any]:
    value = document.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise GLBProbeError(f"glTF field {key!r} must be an array")
    return value


def _record_at(items: Sequence[Any], index: int, label: str) -> dict[str, Any]:
    if not 0 <= index < len(items):
        raise GLBProbeError(f"missing {label} {index}")
    record = items[index]
    if not isinstance(record, dict):
        raise GLBProbeError(f"{label} {index} must be an object")
    return record


def _resource_uri(record: dict[str, Any], *, label: str) -> str | None:
    if "uri" not in record:
        return None
    uri = record["uri"]
    if not isinstance(uri, str) or not uri:
        raise GLBProbeError(f"{label} URI must be a non-empty string")
    return uri


def _resolve_buffer_payloads(
    document: dict[str, Any], binary: bytes
) -> dict[int, bytes]:
    payloads: dict[int, bytes] = {}
    buffers = _as_list(document, "buffers")
    for buffer_index, value in enumerate(buffers):
        record = _record_at(buffers, buffer_index, "buffer")
        uri = _resource_uri(record, label=f"buffer {buffer_index}")
        if uri is None:
            if buffer_index != 0:
                raise GLBProbeError(
                    f"buffer {buffer_index} has no URI; only buffer 0 may use the GLB BIN chunk"
                )
            payload = binary
        elif _is_data_uri(uri):
            payload = _decode_data_uri(uri, label=f"buffer {buffer_index}")
        else:
            continue

        try:
            declared_length = int(record["byteLength"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise GLBProbeError(
                f"buffer {buffer_index} byteLength must be a non-negative integer"
            ) from exc
        if declared_length < 0:
            raise GLBProbeError(
                f"buffer {buffer_index} byteLength must be a non-negative integer"
            )
        if declared_length > len(payload):
            source = "data URI" if uri is not None else "BIN chunk"
            raise GLBProbeError(
                f"buffer {buffer_index} declares {declared_length} bytes but its {source} "
                f"has {len(payload)}"
            )
        payloads[buffer_index] = payload[:declared_length]
    return payloads


def _view_range(
    document: dict[str, Any], buffer_payloads: dict[int, bytes], view_index: int
) -> tuple[bytes, int, int, dict[str, Any]]:
    views = _as_list(document, "bufferViews")
    view = _record_at(views, view_index, "bufferView")
    buffer_index = int(view.get("buffer", 0))
    buffers = _as_list(document, "buffers")
    buffer_record = _record_at(buffers, buffer_index, "buffer")
    if buffer_index not in buffer_payloads:
        uri = _resource_uri(buffer_record, label=f"buffer {buffer_index}")
        if uri is not None and not _is_data_uri(uri):
            raise GLBProbeError(
                f"bufferView {view_index} refers to external buffer URI {uri!r}"
            )
        raise GLBProbeError(f"bufferView {view_index} has no decodable embedded buffer")
    payload = buffer_payloads[buffer_index]
    start = int(view.get("byteOffset", 0))
    length = int(view.get("byteLength", 0))
    end = start + length
    if start < 0 or length < 0 or end > len(payload):
        raise GLBProbeError(f"bufferView {view_index} exceeds embedded buffer {buffer_index}")
    return payload, start, end, view


def _normalise_integer(value: int, component_type: int) -> float:
    if component_type == 5120:
        return max(value / 127.0, -1.0)
    if component_type == 5121:
        return value / 255.0
    if component_type == 5122:
        return max(value / 32767.0, -1.0)
    if component_type == 5123:
        return value / 65535.0
    if component_type == 5124:
        return max(value / 2147483647.0, -1.0)
    if component_type == 5125:
        return value / 4294967295.0
    return float(value)


def _decode_values_from_view(
    document: dict[str, Any],
    buffer_payloads: dict[int, bytes],
    *,
    view_index: int,
    byte_offset: int,
    count: int,
    component_type: int,
    component_count: int,
    normalized: bool,
    label: str,
) -> list[tuple[float | int, ...]]:
    component = _COMPONENTS.get(component_type)
    if component is None:
        raise GLBProbeError(f"{label} has unsupported componentType {component_type}")
    fmt, component_bytes = component
    element_bytes = component_bytes * component_count
    payload, start, end, view = _view_range(document, buffer_payloads, view_index)
    stride = int(view.get("byteStride", element_bytes))
    if stride < element_bytes:
        raise GLBProbeError(
            f"{label} byteStride {stride} is smaller than element size {element_bytes}"
        )
    if stride % component_bytes:
        raise GLBProbeError(
            f"{label} byteStride {stride} is not aligned to its {component_bytes}-byte component"
        )
    base = start + byte_offset
    if byte_offset < 0:
        raise GLBProbeError(f"{label} has a negative byteOffset")
    required_end = base if count == 0 else base + stride * (count - 1) + element_bytes
    if base < start or required_end > end:
        raise GLBProbeError(f"{label} exceeds bufferView {view_index}")
    unpack = struct.Struct("<" + fmt * component_count).unpack_from
    values: list[tuple[float | int, ...]] = []
    for item_index in range(count):
        value = unpack(payload, base + item_index * stride)
        if normalized and component_type != 5126:
            values.append(tuple(_normalise_integer(v, component_type) for v in value))
        else:
            values.append(tuple(value))
    return values


def decode_accessor(
    document: dict[str, Any], binary: bytes, accessor_index: int
) -> list[tuple[float | int, ...]]:
    """Decode a glTF accessor, including interleaving and sparse overrides."""

    return _decode_accessor(
        document,
        _resolve_buffer_payloads(document, binary),
        accessor_index,
    )


def _decode_accessor(
    document: dict[str, Any],
    buffer_payloads: dict[int, bytes],
    accessor_index: int,
) -> list[tuple[float | int, ...]]:

    accessors = _as_list(document, "accessors")
    accessor = _record_at(accessors, accessor_index, "accessor")
    count = int(accessor.get("count", 0))
    if count < 0:
        raise GLBProbeError(f"accessor {accessor_index} has a negative count")
    component_type = int(accessor.get("componentType", 0))
    type_name = accessor.get("type")
    component_count = _TYPE_COMPONENTS.get(str(type_name))
    if component_count is None:
        raise GLBProbeError(f"accessor {accessor_index} has unsupported type {type_name!r}")
    if component_type not in _COMPONENTS:
        raise GLBProbeError(
            f"accessor {accessor_index} has unsupported componentType {component_type}"
        )
    normalized = bool(accessor.get("normalized", False))
    view_index = accessor.get("bufferView")
    if view_index is None:
        values: list[tuple[float | int, ...]] = [
            tuple(0.0 for _ in range(component_count)) for _ in range(count)
        ]
    else:
        values = _decode_values_from_view(
            document,
            buffer_payloads,
            view_index=int(view_index),
            byte_offset=int(accessor.get("byteOffset", 0)),
            count=count,
            component_type=component_type,
            component_count=component_count,
            normalized=normalized,
            label=f"accessor {accessor_index}",
        )

    sparse = accessor.get("sparse")
    if sparse is None:
        return values
    if not isinstance(sparse, dict):
        raise GLBProbeError(f"accessor {accessor_index} sparse field must be an object")
    sparse_count = int(sparse.get("count", 0))
    if sparse_count < 0 or sparse_count > count:
        raise GLBProbeError(
            f"accessor {accessor_index} has invalid sparse count {sparse_count}"
        )
    indices = sparse.get("indices")
    replacements = sparse.get("values")
    if not isinstance(indices, dict) or not isinstance(replacements, dict):
        raise GLBProbeError(f"accessor {accessor_index} has incomplete sparse metadata")
    sparse_index_type = int(indices.get("componentType", 0))
    if sparse_index_type not in (5121, 5123, 5125):
        raise GLBProbeError(
            f"accessor {accessor_index} sparse indices use invalid componentType {sparse_index_type}"
        )
    decoded_indices = _decode_values_from_view(
        document,
        buffer_payloads,
        view_index=int(indices["bufferView"]),
        byte_offset=int(indices.get("byteOffset", 0)),
        count=sparse_count,
        component_type=sparse_index_type,
        component_count=1,
        normalized=False,
        label=f"accessor {accessor_index} sparse indices",
    )
    decoded_values = _decode_values_from_view(
        document,
        buffer_payloads,
        view_index=int(replacements["bufferView"]),
        byte_offset=int(replacements.get("byteOffset", 0)),
        count=sparse_count,
        component_type=component_type,
        component_count=component_count,
        normalized=normalized,
        label=f"accessor {accessor_index} sparse values",
    )
    previous = -1
    for encoded_index, replacement in zip(decoded_indices, decoded_values, strict=True):
        target = int(encoded_index[0])
        if not 0 <= target < count:
            raise GLBProbeError(
                f"accessor {accessor_index} sparse index {target} is out of range"
            )
        if target <= previous:
            raise GLBProbeError(
                f"accessor {accessor_index} sparse indices must be strictly increasing"
            )
        values[target] = replacement
        previous = target
    return values


def _validate_accessor_layout(
    document: dict[str, Any], buffer_payloads: dict[int, bytes], accessor_index: int
) -> None:
    """Validate every referenced byte range without materialising its values."""

    accessor = _record_at(_as_list(document, "accessors"), accessor_index, "accessor")
    count = int(accessor.get("count", 0))
    if count < 0:
        raise GLBProbeError(f"accessor {accessor_index} has a negative count")
    component_type = int(accessor.get("componentType", 0))
    component = _COMPONENTS.get(component_type)
    if component is None:
        raise GLBProbeError(
            f"accessor {accessor_index} has unsupported componentType {component_type}"
        )
    component_count = _TYPE_COMPONENTS.get(str(accessor.get("type")))
    if component_count is None:
        raise GLBProbeError(
            f"accessor {accessor_index} has unsupported type {accessor.get('type')!r}"
        )

    view_index = accessor.get("bufferView")
    if view_index is not None:
        _, component_bytes = component
        element_bytes = component_bytes * component_count
        _, start, end, view = _view_range(document, buffer_payloads, int(view_index))
        stride = int(view.get("byteStride", element_bytes))
        if stride < element_bytes or stride % component_bytes:
            raise GLBProbeError(
                f"accessor {accessor_index} has invalid byteStride {stride} for "
                f"{element_bytes}-byte elements"
            )
        byte_offset = int(accessor.get("byteOffset", 0))
        base = start + byte_offset
        required_end = base if count == 0 else base + stride * (count - 1) + element_bytes
        if byte_offset < 0 or base < start or required_end > end:
            raise GLBProbeError(f"accessor {accessor_index} exceeds bufferView {view_index}")
    elif accessor.get("sparse") is None and count:
        raise GLBProbeError(
            f"accessor {accessor_index} has values but neither bufferView nor sparse storage"
        )

    sparse = accessor.get("sparse")
    if sparse is None:
        return
    if not isinstance(sparse, dict):
        raise GLBProbeError(f"accessor {accessor_index} sparse field must be an object")
    sparse_count = int(sparse.get("count", 0))
    if sparse_count < 0 or sparse_count > count:
        raise GLBProbeError(
            f"accessor {accessor_index} has invalid sparse count {sparse_count}"
        )
    indices = sparse.get("indices")
    replacements = sparse.get("values")
    if not isinstance(indices, dict) or not isinstance(replacements, dict):
        raise GLBProbeError(f"accessor {accessor_index} has incomplete sparse metadata")
    sparse_index_type = int(indices.get("componentType", 0))
    if sparse_index_type not in (5121, 5123, 5125):
        raise GLBProbeError(
            f"accessor {accessor_index} sparse indices use invalid componentType {sparse_index_type}"
        )
    _validate_tightly_packed_range(
        document,
        buffer_payloads,
        view_index=int(indices["bufferView"]),
        byte_offset=int(indices.get("byteOffset", 0)),
        count=sparse_count,
        element_bytes=_COMPONENTS[sparse_index_type][1],
        label=f"accessor {accessor_index} sparse indices",
    )
    _validate_tightly_packed_range(
        document,
        buffer_payloads,
        view_index=int(replacements["bufferView"]),
        byte_offset=int(replacements.get("byteOffset", 0)),
        count=sparse_count,
        element_bytes=component[1] * component_count,
        label=f"accessor {accessor_index} sparse values",
    )


def _validate_tightly_packed_range(
    document: dict[str, Any],
    buffer_payloads: dict[int, bytes],
    *,
    view_index: int,
    byte_offset: int,
    count: int,
    element_bytes: int,
    label: str,
) -> None:
    _, start, end, view = _view_range(document, buffer_payloads, view_index)
    if "byteStride" in view:
        raise GLBProbeError(f"{label} bufferView may not declare byteStride")
    base = start + byte_offset
    required_end = base + count * element_bytes
    if byte_offset < 0 or base < start or required_end > end:
        raise GLBProbeError(f"{label} exceeds bufferView {view_index}")


def _finite_vec3(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        converted = [float(component) for component in value]
    except (TypeError, ValueError):
        return None
    return converted if all(math.isfinite(component) for component in converted) else None


def _position_bounds(
    document: dict[str, Any], buffer_payloads: dict[int, bytes], accessor_index: int
) -> tuple[list[float], list[float], str]:
    accessor = _record_at(_as_list(document, "accessors"), accessor_index, "accessor")
    if accessor.get("type") != "VEC3":
        raise GLBProbeError(f"POSITION accessor {accessor_index} is not VEC3")
    minimum = _finite_vec3(accessor.get("min"))
    maximum = _finite_vec3(accessor.get("max"))
    if minimum is not None and maximum is not None:
        if any(minimum[axis] > maximum[axis] for axis in range(3)):
            raise GLBProbeError(f"POSITION accessor {accessor_index} has inverted min/max")
        return minimum, maximum, "declared"
    count = int(accessor.get("count", 0))
    if count <= 0:
        raise GLBProbeError(f"POSITION accessor {accessor_index} has no vertices")
    if count > MAX_DECODED_POSITION_COUNT:
        raise GLBProbeError(
            f"POSITION accessor {accessor_index} has {count} vertices without min/max; "
            f"fallback decoding is limited to {MAX_DECODED_POSITION_COUNT}"
        )
    values = _decode_accessor(document, buffer_payloads, accessor_index)
    points = [tuple(float(component) for component in value) for value in values]
    if not all(all(math.isfinite(component) for component in point) for point in points):
        raise GLBProbeError(f"POSITION accessor {accessor_index} contains non-finite values")
    return (
        [min(point[axis] for point in points) for axis in range(3)],
        [max(point[axis] for point in points) for axis in range(3)],
        "decoded",
    )


def _identity() -> list[float]:
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def _node_matrix(node: dict[str, Any]) -> list[float]:
    matrix = node.get("matrix")
    if matrix is not None:
        if not isinstance(matrix, list) or len(matrix) != 16:
            raise GLBProbeError("node matrix must contain 16 numbers")
        converted = [float(value) for value in matrix]
        if not all(math.isfinite(value) for value in converted):
            raise GLBProbeError("node matrix contains non-finite values")
        return converted

    translation = node.get("translation", [0.0, 0.0, 0.0])
    rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    scale = node.get("scale", [1.0, 1.0, 1.0])
    if not (isinstance(translation, list) and len(translation) == 3):
        raise GLBProbeError("node translation must contain three numbers")
    if not (isinstance(rotation, list) and len(rotation) == 4):
        raise GLBProbeError("node rotation must contain four numbers")
    if not (isinstance(scale, list) and len(scale) == 3):
        raise GLBProbeError("node scale must contain three numbers")
    tx, ty, tz = (float(value) for value in translation)
    qx, qy, qz, qw = (float(value) for value in rotation)
    sx, sy, sz = (float(value) for value in scale)
    values = (tx, ty, tz, qx, qy, qz, qw, sx, sy, sz)
    if not all(math.isfinite(value) for value in values):
        raise GLBProbeError("node transform contains non-finite values")
    q_length = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if q_length <= 1e-12:
        raise GLBProbeError("node rotation quaternion has zero length")
    qx, qy, qz, qw = (value / q_length for value in (qx, qy, qz, qw))
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return [
        (1 - 2 * (yy + zz)) * sx,
        (2 * (xy + wz)) * sx,
        (2 * (xz - wy)) * sx,
        0.0,
        (2 * (xy - wz)) * sy,
        (1 - 2 * (xx + zz)) * sy,
        (2 * (yz + wx)) * sy,
        0.0,
        (2 * (xz + wy)) * sz,
        (2 * (yz - wx)) * sz,
        (1 - 2 * (xx + yy)) * sz,
        0.0,
        tx,
        ty,
        tz,
        1.0,
    ]


def _multiply(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        sum(a[k * 4 + row] * b[column * 4 + k] for k in range(4))
        for column in range(4)
        for row in range(4)
    ]


def _apply(matrix: Sequence[float], point: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
        matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
        matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
    )


def _merge_bounds(
    bounds: Iterable[tuple[Sequence[float], Sequence[float]]], *, space: str
) -> dict[str, Any] | None:
    collected = list(bounds)
    if not collected:
        return None
    minimum = [min(item[0][axis] for item in collected) for axis in range(3)]
    maximum = [max(item[1][axis] for item in collected) for axis in range(3)]
    return {
        "space": space,
        "min": minimum,
        "max": maximum,
        "center": [(minimum[axis] + maximum[axis]) / 2.0 for axis in range(3)],
        "size": [maximum[axis] - minimum[axis] for axis in range(3)],
    }


def _transform_bounds(
    matrix: Sequence[float], minimum: Sequence[float], maximum: Sequence[float]
) -> tuple[list[float], list[float]]:
    points = [
        _apply(matrix, corner)
        for corner in itertools.product(
            (minimum[0], maximum[0]),
            (minimum[1], maximum[1]),
            (minimum[2], maximum[2]),
        )
    ]
    return (
        [min(point[axis] for point in points) for axis in range(3)],
        [max(point[axis] for point in points) for axis in range(3)],
    )


def _triangle_count(mode: int, element_count: int) -> int:
    if mode == 4:
        return element_count // 3
    if mode in (5, 6):
        return max(0, element_count - 2)
    return 0


def _meaningful_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.strip()
    if not name or _GENERIC_NAME.fullmatch(name):
        return None
    return name


def _semantic_assessment(
    *,
    mesh_count: int,
    primitive_count: int,
    material_count: int,
    drawable_names: list[str],
) -> dict[str, Any]:
    if mesh_count == 1 and primitive_count == 1 and material_count <= 1:
        status = "insufficient"
        boundary = "merged-single-drawable"
        reason = (
            "One mesh, one primitive, and at most one material expose no reliable part boundary. "
            "Wrapper nodes and generated names do not change that conclusion."
        )
    elif len(set(drawable_names)) >= 2:
        status = "rich"
        boundary = "named-drawables"
        reason = "Multiple non-generated drawable names provide explicit part hypotheses."
    else:
        status = "partial"
        boundary = "drawable-or-material-boundaries"
        reason = (
            "Multiple drawable or material boundaries exist, but metadata does not establish "
            "reliable semantic labels."
        )
    return {
        "status": status,
        "reliable_boundary": boundary,
        "reason": reason,
        "meaningful_drawable_names": sorted(set(drawable_names)),
        "claims_allowed": [
            "container and drawable inventory",
            "explicit mesh, primitive, and material boundaries",
            "measured rest-pose bounds",
        ],
        "claims_forbidden_without_render_confirmation": [
            "semantic labels inferred from generated names",
            "part boundaries inside a merged primitive",
            "rig or articulation inferred from static geometry",
        ],
    }


def probe_glb(path: Path) -> dict[str, Any]:
    """Return a deterministic, JSON-serialisable GLB evidence report."""

    path = path.expanduser().resolve()
    document, binary, container = parse_glb(path)
    nodes = _as_list(document, "nodes")
    meshes = _as_list(document, "meshes")
    accessors = _as_list(document, "accessors")
    materials = _as_list(document, "materials")
    textures = _as_list(document, "textures")
    images = _as_list(document, "images")
    skins = _as_list(document, "skins")
    animations = _as_list(document, "animations")
    scenes = _as_list(document, "scenes")
    buffers = _as_list(document, "buffers")
    warnings: list[str] = []

    buffer_payloads = _resolve_buffer_payloads(document, binary)
    external_buffers = []
    for index in range(len(buffers)):
        record = _record_at(buffers, index, "buffer")
        uri = _resource_uri(record, label=f"buffer {index}")
        if uri is not None and not _is_data_uri(uri):
            external_buffers.append({"index": index, "uri": uri})

    external_images = []
    for index in range(len(images)):
        record = _record_at(images, index, "image")
        uri = _resource_uri(record, label=f"image {index}")
        if uri is None:
            continue
        if _is_data_uri(uri):
            _decode_data_uri(uri, label=f"image {index}")
        else:
            external_images.append({"index": index, "uri": uri})
    if external_buffers:
        warnings.append("external buffers are not self-contained in the GLB")
    if external_images:
        warnings.append("external images are not self-contained in the GLB")

    external_buffer_indices = {record["index"] for record in external_buffers}

    def accessor_uses_external_buffer(accessor: dict[str, Any]) -> bool:
        referenced_views: list[int] = []
        if accessor.get("bufferView") is not None:
            referenced_views.append(int(accessor["bufferView"]))
        sparse = accessor.get("sparse")
        if isinstance(sparse, dict):
            for key in ("indices", "values"):
                record = sparse.get(key)
                if isinstance(record, dict) and record.get("bufferView") is not None:
                    referenced_views.append(int(record["bufferView"]))
        views = _as_list(document, "bufferViews")
        for view_index in referenced_views:
            view = _record_at(views, view_index, "bufferView")
            if int(view.get("buffer", 0)) in external_buffer_indices:
                return True
        return False

    for accessor_index, accessor_value in enumerate(accessors):
        accessor = _record_at(accessors, accessor_index, "accessor")
        if not accessor_uses_external_buffer(accessor):
            _validate_accessor_layout(document, buffer_payloads, accessor_index)

    extensions_used = [str(value) for value in document.get("extensionsUsed", [])]
    extensions_required = [str(value) for value in document.get("extensionsRequired", [])]
    geometry_extensions = [
        extension
        for extension in extensions_required
        if "draco" in extension.lower() or "meshopt" in extension.lower()
    ]
    if geometry_extensions:
        warnings.append(
            "compressed geometry requires the Blender reference stage: "
            + ", ".join(geometry_extensions)
        )

    mesh_records: list[dict[str, Any]] = []
    mesh_local_bounds: dict[int, tuple[list[float], list[float]]] = {}
    primitive_count = 0
    total_vertices = 0
    total_triangles = 0
    primitives_with_normals = 0
    drawable_names: list[str] = []
    used_materials: set[int] = set()
    for mesh_index, mesh_value in enumerate(meshes):
        mesh = _record_at(meshes, mesh_index, "mesh")
        primitives = mesh.get("primitives", [])
        if not isinstance(primitives, list):
            raise GLBProbeError(f"mesh {mesh_index} primitives must be an array")
        primitive_records: list[dict[str, Any]] = []
        bounds_for_mesh: list[tuple[list[float], list[float]]] = []
        meaningful_mesh_name = _meaningful_name(mesh.get("name"))
        if meaningful_mesh_name:
            drawable_names.append(meaningful_mesh_name)
        for primitive_index, primitive_value in enumerate(primitives):
            if not isinstance(primitive_value, dict):
                raise GLBProbeError(
                    f"mesh {mesh_index} primitive {primitive_index} must be an object"
                )
            primitive = primitive_value
            primitive_count += 1
            attributes = primitive.get("attributes", {})
            if not isinstance(attributes, dict):
                raise GLBProbeError(
                    f"mesh {mesh_index} primitive {primitive_index} attributes must be an object"
                )
            position_index = attributes.get("POSITION")
            material_index = primitive.get("material")
            if material_index is not None:
                material_index = int(material_index)
                if not 0 <= material_index < len(materials):
                    raise GLBProbeError(
                        f"mesh {mesh_index} primitive {primitive_index} refers to "
                        f"missing material {material_index}"
                    )
                used_materials.add(material_index)
            position_bounds: dict[str, Any] | None = None
            vertex_count = 0
            if position_index is None:
                warnings.append(
                    f"mesh {mesh_index} primitive {primitive_index} has no POSITION accessor"
                )
            elif geometry_extensions:
                warnings.append(
                    f"mesh {mesh_index} primitive {primitive_index} bounds deferred to Blender"
                )
            else:
                position_accessor = _record_at(
                    accessors, int(position_index), "POSITION accessor"
                )
                vertex_count = int(position_accessor.get("count", 0))
                minimum, maximum, source = _position_bounds(
                    document, buffer_payloads, int(position_index)
                )
                position_bounds = {"min": minimum, "max": maximum, "source": source}
                bounds_for_mesh.append((minimum, maximum))
            total_vertices += vertex_count
            if "NORMAL" in attributes:
                primitives_with_normals += 1
            index_accessor = primitive.get("indices")
            element_count = vertex_count
            if index_accessor is not None:
                record = _record_at(accessors, int(index_accessor), "index accessor")
                element_count = int(record.get("count", 0))
            mode = int(primitive.get("mode", 4))
            triangles = _triangle_count(mode, element_count)
            total_triangles += triangles
            primitive_record = {
                "index": primitive_index,
                "mode": mode,
                "attributes": sorted(str(name) for name in attributes),
                "material": material_index,
                "indices": index_accessor,
                "vertex_count": vertex_count,
                "element_count": element_count,
                "triangle_count": triangles,
                "has_normals": "NORMAL" in attributes,
                "has_texcoords": any(str(name).startswith("TEXCOORD_") for name in attributes),
            }
            if position_bounds is not None:
                primitive_record["position_bounds"] = position_bounds
            if primitive.get("targets"):
                primitive_record["morph_target_count"] = len(primitive["targets"])
                warnings.append(
                    f"mesh {mesh_index} primitive {primitive_index} has morph targets; "
                    "the stdlib probe reports only base geometry"
                )
            primitive_records.append(primitive_record)
        merged = _merge_bounds(bounds_for_mesh, space="mesh-local")
        if merged is not None:
            mesh_local_bounds[mesh_index] = (merged["min"], merged["max"])
        mesh_record: dict[str, Any] = {
            "index": mesh_index,
            "name": mesh.get("name"),
            "primitive_count": len(primitives),
            "primitives": primitive_records,
        }
        if merged is not None:
            mesh_record["bounds"] = merged
        mesh_records.append(mesh_record)

    if primitive_count and primitives_with_normals < primitive_count:
        warnings.append(
            f"{primitive_count - primitives_with_normals} of {primitive_count} primitives lack NORMAL; "
            "normals must be generated by Blender before oriented surface analysis"
        )
    if not meshes:
        warnings.append("GLB contains no meshes")
    if not materials:
        warnings.append("GLB contains no materials")
    if not skins:
        warnings.append("GLB contains no skin")

    node_records: list[dict[str, Any]] = []
    for node_index, node_value in enumerate(nodes):
        node = _record_at(nodes, node_index, "node")
        mesh_index = node.get("mesh")
        if mesh_index is not None and not 0 <= int(mesh_index) < len(meshes):
            raise GLBProbeError(f"node {node_index} refers to missing mesh {mesh_index}")
        meaningful_node_name = _meaningful_name(node.get("name"))
        if mesh_index is not None and meaningful_node_name:
            drawable_names.append(meaningful_node_name)
        children = node.get("children", [])
        if not isinstance(children, list):
            raise GLBProbeError(f"node {node_index} children must be an array")
        for child in children:
            if not 0 <= int(child) < len(nodes):
                raise GLBProbeError(f"node {node_index} refers to missing child {child}")
        node_records.append(
            {
                "index": node_index,
                "name": node.get("name"),
                "mesh": mesh_index,
                "skin": node.get("skin"),
                "children": [int(child) for child in children],
                "local_matrix": _node_matrix(node),
            }
        )

    scene_value = document.get("scene", 0)
    default_scene = int(0 if scene_value is None else scene_value) if scenes else None
    if default_scene is not None and not 0 <= default_scene < len(scenes):
        raise GLBProbeError(f"default scene {default_scene} does not exist")
    child_indices = {
        child for record in node_records for child in record.get("children", [])
    }
    if scenes:
        scene = _record_at(scenes, default_scene or 0, "scene")
        roots = scene.get("nodes", [])
        if not isinstance(roots, list):
            raise GLBProbeError("scene nodes must be an array")
        root_indices = [int(index) for index in roots]
    else:
        root_indices = [index for index in range(len(nodes)) if index not in child_indices]
        if nodes:
            warnings.append("GLB has no scene record; inferred roots from the node graph")
    for root in root_indices:
        if not 0 <= root < len(nodes):
            raise GLBProbeError(f"scene refers to missing root node {root}")

    instances: list[dict[str, Any]] = []
    world_bounds: list[tuple[list[float], list[float]]] = []

    def walk(node_index: int, parent: Sequence[float], path_stack: tuple[int, ...]) -> None:
        if node_index in path_stack:
            cycle = " -> ".join(str(index) for index in (*path_stack, node_index))
            raise GLBProbeError(f"node graph contains a cycle: {cycle}")
        node_record = node_records[node_index]
        world = _multiply(parent, node_record["local_matrix"])
        mesh_index = node_record.get("mesh")
        if mesh_index is not None:
            instance: dict[str, Any] = {
                "node": node_index,
                "node_name": node_record.get("name"),
                "mesh": int(mesh_index),
                "world_matrix": world,
                "path": [*path_stack, node_index],
            }
            local = mesh_local_bounds.get(int(mesh_index))
            if local is not None:
                minimum, maximum = _transform_bounds(world, local[0], local[1])
                instance["bounds"] = _merge_bounds(
                    [(minimum, maximum)], space="world"
                )
                world_bounds.append((minimum, maximum))
            instances.append(instance)
        for child in node_record.get("children", []):
            walk(child, world, (*path_stack, node_index))

    for root in root_indices:
        walk(root, _identity(), ())

    # Mesh nodes omitted from a malformed/disconnected scene are still useful
    # inventory, but not folded into the default-scene world bounds.
    reached_nodes = {instance["node"] for instance in instances}
    omitted_mesh_nodes = [
        record["index"]
        for record in node_records
        if record.get("mesh") is not None and record["index"] not in reached_nodes
    ]
    if omitted_mesh_nodes:
        warnings.append(
            "mesh-bearing nodes are not reachable from the default scene: "
            + ", ".join(str(index) for index in omitted_mesh_nodes)
        )

    semantic = _semantic_assessment(
        mesh_count=len(meshes),
        primitive_count=primitive_count,
        material_count=len(used_materials),
        drawable_names=drawable_names,
    )
    self_contained = not external_buffers and not external_images
    measurement_readiness = "requires-blender" if geometry_extensions else "pass"
    reference_readiness = "pass" if meshes and instances and self_contained else "reject"
    return {
        "schema_version": 1,
        "kind": "glb-reference",
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "container": container,
        "asset": {
            "version": (document.get("asset") or {}).get("version")
            if isinstance(document.get("asset"), dict)
            else None,
            "generator": (document.get("asset") or {}).get("generator")
            if isinstance(document.get("asset"), dict)
            else None,
        },
        "extensions": {
            "used": extensions_used,
            "required": extensions_required,
            "geometry_requires_blender": geometry_extensions,
        },
        "scene": {
            "default_scene": default_scene,
            "scene_count": len(scenes),
            "node_count": len(nodes),
            "mesh_count": len(meshes),
            "mesh_instance_count": len(instances),
            "primitive_count": primitive_count,
            "material_count": len(materials),
            "texture_count": len(textures),
            "image_count": len(images),
            "skin_count": len(skins),
            "animation_count": len(animations),
            "vertex_count": total_vertices,
            "triangle_count": total_triangles,
            "primitives_with_normals": primitives_with_normals,
        },
        "bounds": _merge_bounds(world_bounds, space="world"),
        "nodes": node_records,
        "instances": instances,
        "meshes": mesh_records,
        "materials": [
            {
                "index": index,
                "name": material.get("name") if isinstance(material, dict) else None,
                "alpha_mode": material.get("alphaMode", "OPAQUE")
                if isinstance(material, dict)
                else None,
                "double_sided": bool(material.get("doubleSided", False))
                if isinstance(material, dict)
                else False,
            }
            for index, material in enumerate(materials)
        ],
        "images": [
            {
                "index": index,
                "name": image.get("name") if isinstance(image, dict) else None,
                "mime_type": image.get("mimeType") if isinstance(image, dict) else None,
                "buffer_view": image.get("bufferView") if isinstance(image, dict) else None,
                "uri": image.get("uri") if isinstance(image, dict) else None,
            }
            for index, image in enumerate(images)
        ],
        "semantic_decomposition": semantic,
        "self_contained": self_contained,
        "measurement_readiness": measurement_readiness,
        "reference_readiness": reference_readiness,
        "warnings": sorted(set(warnings)),
    }


def write_probe(path: Path, output: Path) -> dict[str, Any]:
    report = probe_glb(path)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a GLB reference with no third-party dependencies")
    parser.add_argument("glb", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        report = write_probe(args.glb, args.out) if args.out else probe_glb(args.glb)
    except (OSError, GLBProbeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
