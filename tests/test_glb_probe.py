from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path

import pytest

from procagen3d.glb_probe import (
    BIN_CHUNK,
    JSON_CHUNK,
    GLBProbeError,
    decode_accessor,
    parse_glb,
    probe_glb,
    write_probe,
)


def _pad(payload: bytes, byte: bytes) -> bytes:
    return payload + byte * (-len(payload) % 4)


def _write_glb(path: Path, document: dict, binary: bytes = b"") -> Path:
    json_payload = _pad(
        json.dumps(document, separators=(",", ":")).encode("utf-8"), b" "
    )
    chunks = [struct.pack("<II", len(json_payload), JSON_CHUNK), json_payload]
    if binary:
        binary_payload = _pad(binary, b"\x00")
        chunks.extend([struct.pack("<II", len(binary_payload), BIN_CHUNK), binary_payload])
    body = b"".join(chunks)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
    return path


def _triangle_document(binary_length: int, *, include_bounds: bool = False) -> dict:
    position_accessor: dict = {
        "bufferView": 0,
        "componentType": 5126,
        "count": 3,
        "type": "VEC3",
    }
    if include_bounds:
        position_accessor.update({"min": [0, 0, 0], "max": [1, 2, 1]})
    return {
        "asset": {"version": "2.0", "generator": "unit-test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "world", "translation": [10, 0, 0], "children": [1]},
            {"name": "geometry_0", "scale": [2, 3, 4], "mesh": 0},
        ],
        "meshes": [
            {
                "name": "geometry_0",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [{}],
        "buffers": [{"byteLength": binary_length}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 48, "byteStride": 16},
            {"buffer": 0, "byteOffset": 48, "byteLength": 6},
        ],
        "accessors": [
            position_accessor,
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR",
            },
        ],
    }


def _triangle_glb(tmp_path: Path, *, include_bounds: bool = False) -> Path:
    positions = b"".join(
        struct.pack("<3fI", *point, 0xDEADBEEF)
        for point in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 1.0))
    )
    indices = struct.pack("<3H", 0, 1, 2)
    binary = positions + indices
    return _write_glb(
        tmp_path / "triangle.glb",
        _triangle_document(len(binary), include_bounds=include_bounds),
        binary,
    )


def test_parse_and_probe_interleaved_geometry_in_world_space(tmp_path: Path) -> None:
    glb = _triangle_glb(tmp_path)

    document, binary, container = parse_glb(glb)
    assert container["version"] == 2
    assert [chunk["type"] for chunk in container["chunks"]] == ["JSON", "BIN\x00"]
    assert decode_accessor(document, binary, 0) == [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 2.0, 1.0),
    ]

    report = probe_glb(glb)
    assert report["reference_readiness"] == "pass"
    assert report["self_contained"] is True
    assert report["scene"]["vertex_count"] == 3
    assert report["scene"]["triangle_count"] == 1
    assert report["scene"]["mesh_instance_count"] == 1
    assert report["meshes"][0]["primitives"][0]["position_bounds"]["source"] == "decoded"
    assert report["bounds"]["min"] == pytest.approx([10.0, 0.0, 0.0])
    assert report["bounds"]["max"] == pytest.approx([12.0, 6.0, 4.0])
    assert report["semantic_decomposition"]["status"] == "insufficient"
    assert report["semantic_decomposition"]["reliable_boundary"] == "merged-single-drawable"


def test_parent_and_child_rotations_are_composed(tmp_path: Path) -> None:
    glb = _triangle_glb(tmp_path, include_bounds=True)
    document, binary, _ = parse_glb(glb)
    half = math.sqrt(0.5)
    document["nodes"][0] = {
        "name": "world",
        "translation": [5, 7, 0],
        "rotation": [0, 0, half, half],
        "children": [1],
    }
    document["nodes"][1] = {"name": "geometry_0", "mesh": 0}
    rotated = _write_glb(tmp_path / "rotated.glb", document, binary[:54])

    bounds = probe_glb(rotated)["bounds"]
    assert bounds["min"] == pytest.approx([3.0, 7.0, 0.0], abs=1e-6)
    assert bounds["max"] == pytest.approx([5.0, 8.0, 1.0], abs=1e-6)


def test_sparse_position_accessor_is_decoded(tmp_path: Path) -> None:
    binary = b"\x01\x00\x00\x00" + struct.pack("<3f", 2.0, 3.0, 4.0)
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 1},
            {"buffer": 0, "byteOffset": 4, "byteLength": 12},
        ],
        "accessors": [
            {
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "sparse": {
                    "count": 1,
                    "indices": {"bufferView": 0, "componentType": 5121},
                    "values": {"bufferView": 1},
                },
            }
        ],
    }
    glb = _write_glb(tmp_path / "sparse.glb", document, binary)

    parsed, payload, _ = parse_glb(glb)
    assert decode_accessor(parsed, payload, 0) == [
        (0.0, 0.0, 0.0),
        (2.0, 3.0, 4.0),
        (0.0, 0.0, 0.0),
    ]
    report = probe_glb(glb)
    assert report["bounds"]["min"] == pytest.approx([0.0, 0.0, 0.0])
    assert report["bounds"]["max"] == pytest.approx([2.0, 3.0, 4.0])


def test_invalid_sparse_indices_fail_closed(tmp_path: Path) -> None:
    binary = b"\x02\x01\x00\x00" + struct.pack("<6f", *range(6))
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 2},
            {"buffer": 0, "byteOffset": 4, "byteLength": 24},
        ],
        "accessors": [
            {
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "sparse": {
                    "count": 2,
                    "indices": {"bufferView": 0, "componentType": 5121},
                    "values": {"bufferView": 1},
                },
            }
        ],
    }
    glb = _write_glb(tmp_path / "bad-sparse.glb", document, binary)
    parsed, payload, _ = parse_glb(glb)

    with pytest.raises(GLBProbeError, match="strictly increasing"):
        decode_accessor(parsed, payload, 0)


def test_declared_bounds_do_not_hide_out_of_range_accessor(tmp_path: Path) -> None:
    binary = struct.pack("<3f", 0, 0, 0)
    document = _triangle_document(len(binary), include_bounds=True)
    document["bufferViews"] = [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(binary), "byteStride": 16},
        {"buffer": 0, "byteOffset": 0, "byteLength": 6},
    ]
    glb = _write_glb(tmp_path / "bad-range.glb", document, binary)

    with pytest.raises(GLBProbeError, match="accessor 0 exceeds"):
        probe_glb(glb)


def test_external_resources_are_reported_as_not_self_contained(tmp_path: Path) -> None:
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "buffers": [{"uri": "mesh.bin", "byteLength": 36}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [1, 1, 0],
            }
        ],
        "images": [{"uri": "texture.png"}],
    }
    glb = _write_glb(tmp_path / "external.glb", document)

    report = probe_glb(glb)
    assert report["self_contained"] is False
    assert report["reference_readiness"] == "reject"
    assert any("external buffers" in warning for warning in report["warnings"])
    assert any("external images" in warning for warning in report["warnings"])


def test_data_uri_buffer_is_decoded_and_self_contained(tmp_path: Path) -> None:
    positions = b"".join(
        struct.pack("<3fI", *point, 0xDEADBEEF)
        for point in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 1.0))
    )
    binary = positions + struct.pack("<3H", 0, 1, 2)
    document = _triangle_document(len(binary))
    document["buffers"][0]["uri"] = (
        "data:application/octet-stream;base64,"
        + base64.b64encode(binary).decode("ascii")
    )
    glb = _write_glb(tmp_path / "data-buffer.glb", document)

    parsed, bin_chunk, _ = parse_glb(glb)
    assert bin_chunk == b""
    assert decode_accessor(parsed, bin_chunk, 0) == [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 2.0, 1.0),
    ]

    report = probe_glb(glb)
    assert report["self_contained"] is True
    assert report["reference_readiness"] == "pass"
    assert report["bounds"]["min"] == pytest.approx([10.0, 0.0, 0.0])
    assert report["bounds"]["max"] == pytest.approx([12.0, 6.0, 4.0])
    assert not any("external buffers" in warning for warning in report["warnings"])


def test_data_uri_image_is_self_contained(tmp_path: Path) -> None:
    source = _triangle_glb(tmp_path, include_bounds=True)
    document, binary, _ = parse_glb(source)
    document["images"] = [
        {
            "name": "embedded-pixel",
            "uri": (
                "data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
                "+A8AAQUBAScY42YAAAAASUVORK5CYII="
            ),
        }
    ]
    declared_length = document["buffers"][0]["byteLength"]
    glb = _write_glb(
        tmp_path / "data-image.glb",
        document,
        binary[:declared_length],
    )

    report = probe_glb(glb)

    assert report["scene"]["image_count"] == 1
    assert report["images"][0]["uri"].startswith("data:image/png;base64,")
    assert report["self_contained"] is True
    assert report["reference_readiness"] == "pass"
    assert not any("external images" in warning for warning in report["warnings"])


@pytest.mark.parametrize(
    ("buffer_uri", "image_uri"),
    [
        ("mesh.bin", "texture.png"),
        ("https://example.invalid/mesh.bin", "https://example.invalid/texture.png"),
    ],
)
def test_non_data_uris_remain_external(
    tmp_path: Path,
    buffer_uri: str,
    image_uri: str,
) -> None:
    document = _triangle_document(54, include_bounds=True)
    document["buffers"][0]["uri"] = buffer_uri
    document["images"] = [{"uri": image_uri}]
    glb = _write_glb(tmp_path / "ordinary-external.glb", document)

    report = probe_glb(glb)

    assert report["self_contained"] is False
    assert report["reference_readiness"] == "reject"
    assert any("external buffers" in warning for warning in report["warnings"])
    assert any("external images" in warning for warning in report["warnings"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: b"NOPE" + data[4:], "magic"),
        (
            lambda data: data[:8] + struct.pack("<I", len(data) + 4) + data[12:],
            "header length",
        ),
    ],
)
def test_malformed_headers_are_rejected(tmp_path: Path, mutation, message: str) -> None:
    valid = _triangle_glb(tmp_path).read_bytes()
    path = tmp_path / f"bad-{message.replace(' ', '-')}.glb"
    path.write_bytes(mutation(valid))

    with pytest.raises(GLBProbeError, match=message):
        parse_glb(path)


def test_write_probe_is_atomic_json_output(tmp_path: Path) -> None:
    glb = _triangle_glb(tmp_path)
    output = tmp_path / "evidence" / "glb_probe.json"

    report = write_probe(glb, output)

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert not output.with_suffix(".json.tmp").exists()


def test_multiple_meaningful_drawables_can_be_semantically_rich(tmp_path: Path) -> None:
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"name": "Head", "mesh": 0},
            {"name": "Torso", "mesh": 1, "translation": [0, -1, 0]},
        ],
        "meshes": [
            {"name": "Head", "primitives": [{"attributes": {"POSITION": 0}}]},
            {"name": "Torso", "primitives": [{"attributes": {"POSITION": 0}}]},
        ],
        "buffers": [{"byteLength": len(positions)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions)}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [1, 1, 0],
            }
        ],
    }
    glb = _write_glb(tmp_path / "multipart.glb", document, positions)

    report = probe_glb(glb)
    assert report["semantic_decomposition"]["status"] == "rich"
    assert report["scene"]["mesh_instance_count"] == 2


def test_supplied_reference_glbs_are_merged_and_probeable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    fixtures = sorted((project_root / "assets" / "3d_glb").glob("*/object_0.glb"))
    if not fixtures:
        pytest.skip("assets/3d_glb fixtures are not present")

    for fixture in fixtures:
        report = probe_glb(fixture)
        assert report["reference_readiness"] == "pass", fixture
        assert report["scene"]["mesh_count"] == 1, fixture
        assert report["scene"]["primitive_count"] == 1, fixture
        assert report["semantic_decomposition"]["status"] == "insufficient", fixture
        assert report["scene"]["primitives_with_normals"] == 0, fixture
        assert any("lack NORMAL" in warning for warning in report["warnings"]), fixture
