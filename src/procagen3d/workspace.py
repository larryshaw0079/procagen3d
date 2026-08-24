"""Workspace lifecycle and reproducible input provenance."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .reconstruction import DEFAULT_RECONSTRUCTION_MODE, validate_reconstruction_mode


_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TRAJECTORY_RE = re.compile(r"^iter_(\d+)$")


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    value = value[:64].rstrip("-")
    if not value:
        raise ValueError("cannot derive a non-empty workspace slug")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Workspace:
    root: Path

    @classmethod
    def create(
        cls,
        *,
        base: Path,
        slug: str,
        image: Path,
        glb: Path,
        prompt: str,
        backend: str,
        reconstruction_mode: str = DEFAULT_RECONSTRUCTION_MODE,
    ) -> "Workspace":
        reconstruction_mode = validate_reconstruction_mode(reconstruction_mode)
        if not _SLUG_RE.fullmatch(slug):
            raise ValueError(f"invalid slug {slug!r}; use lowercase letters, digits, '-' or '_'")
        image = image.expanduser().resolve()
        glb = glb.expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"reference image not found: {image}")
        if not glb.is_file():
            raise FileNotFoundError(f"reference GLB not found: {glb}")
        root = base.expanduser().resolve() / slug
        if root.exists():
            raise FileExistsError(f"workspace already exists: {root}; use `procagen3d run {root}`")
        for directory in (
            root / "inputs",
            root / "evidence" / "reference_views",
            root / "src",
            root / "artifacts" / "renders",
            root / "trajectories",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        image_target = root / "inputs" / f"reference{image.suffix.lower()}"
        glb_target = root / "inputs" / "reference.glb"
        shutil.copy2(image, image_target)
        shutil.copy2(glb, glb_target)
        manifest = {
            "schema_version": 1,
            "slug": slug,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "backend": backend,
            "reconstruction_mode": reconstruction_mode,
            "inputs": {
                "image": {
                    "path": str(image_target.relative_to(root)),
                    "source": str(image),
                    "sha256": sha256(image_target),
                    "bytes": image_target.stat().st_size,
                },
                "glb": {
                    "path": str(glb_target.relative_to(root)),
                    "source": str(glb),
                    "sha256": sha256(glb_target),
                    "bytes": glb_target.stat().st_size,
                },
            },
        }
        write_json(root / "manifest.json", manifest)
        return cls(root)

    @classmethod
    def locate(cls, value: Path, *, base: Path | None = None) -> "Workspace":
        candidate = value.expanduser()
        if not candidate.is_absolute() and base is not None:
            by_slug = base.expanduser().resolve() / candidate
            if by_slug.exists():
                candidate = by_slug
        candidate = candidate.resolve()
        workspace = cls(candidate)
        manifest = workspace.manifest()
        workspace._validated_input_path("image", manifest=manifest)
        workspace._validated_input_path("glb", manifest=manifest)
        return workspace

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def image_path(self) -> Path:
        return self._validated_input_path("image")

    @property
    def glb_path(self) -> Path:
        return self._validated_input_path("glb")

    @property
    def evidence_dir(self) -> Path:
        return self.root / "evidence"

    @property
    def src_dir(self) -> Path:
        return self.root / "src"

    @property
    def program_path(self) -> Path:
        return self.src_dir / "program.py"

    @property
    def plan_path(self) -> Path:
        return self.src_dir / "plan.json"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    def manifest(self) -> dict[str, Any]:
        if self.manifest_path.is_symlink():
            raise ValueError(f"workspace manifest must not be a symlink: {self.manifest_path}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"not a ProcAgen3D workspace: {self.root}")
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"workspace manifest is invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("workspace manifest must contain a JSON object")
        schema_version = value.get("schema_version")
        if type(schema_version) is not int or schema_version != 1:
            raise ValueError(
                f"unsupported workspace schema_version {schema_version!r}; expected 1"
            )
        return value

    def _validated_input_path(
        self,
        kind: str,
        *,
        manifest: dict[str, Any] | None = None,
    ) -> Path:
        manifest = manifest if manifest is not None else self.manifest()
        inputs = manifest.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("workspace manifest field 'inputs' must be an object")
        record = inputs.get(kind)
        if not isinstance(record, dict):
            raise ValueError(f"workspace manifest input {kind!r} must be an object")

        path_value = record.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"workspace manifest input {kind!r} has an invalid path")
        relative = Path(path_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"workspace manifest input {kind!r} must stay under the inputs directory"
            )
        if not relative.parts or relative.parts[0] != "inputs":
            raise ValueError(
                f"workspace manifest input {kind!r} must stay under the inputs directory"
            )

        root = self.root.expanduser().resolve()
        candidate = root / relative
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError(
                    f"workspace manifest input {kind!r} must not use symlinks: {current}"
                )

        inputs_root = root / "inputs"
        if not inputs_root.is_dir():
            raise FileNotFoundError(f"workspace inputs directory not found: {inputs_root}")
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"workspace input {kind!r} not found: {candidate}") from exc
        try:
            resolved.relative_to(inputs_root.resolve(strict=True))
        except ValueError as exc:
            raise ValueError(
                f"workspace manifest input {kind!r} escapes the inputs directory"
            ) from exc
        if not resolved.is_file():
            raise ValueError(f"workspace input {kind!r} is not a regular file: {resolved}")

        recorded_bytes = record.get("bytes")
        if type(recorded_bytes) is not int or recorded_bytes < 0:
            raise ValueError(f"workspace manifest input {kind!r} has invalid byte provenance")
        actual_bytes = resolved.stat().st_size
        if actual_bytes != recorded_bytes:
            raise ValueError(
                f"workspace input {kind!r} byte count changed: "
                f"expected {recorded_bytes}, found {actual_bytes}"
            )

        recorded_sha256 = record.get("sha256")
        if not isinstance(recorded_sha256, str) or not _SHA256_RE.fullmatch(recorded_sha256):
            raise ValueError(f"workspace manifest input {kind!r} has invalid SHA-256 provenance")
        actual_sha256 = sha256(resolved)
        if actual_sha256 != recorded_sha256:
            raise ValueError(
                f"workspace input {kind!r} SHA-256 changed: "
                f"expected {recorded_sha256}, found {actual_sha256}"
            )
        return resolved

    def update_manifest(self, **updates: Any) -> None:
        manifest = self.manifest()
        manifest.update(updates)
        write_json(self.manifest_path, manifest)

    def next_trajectory_iteration(self) -> int:
        trajectories = self.root / "trajectories"
        trajectories.mkdir(parents=True, exist_ok=True)
        existing = []
        for path in trajectories.iterdir():
            match = _TRAJECTORY_RE.fullmatch(path.name)
            if match is not None:
                existing.append(int(match.group(1)))
        return max(existing, default=-1) + 1

    def trajectory_dir(self, iteration: int) -> Path:
        if type(iteration) is not int or iteration < 0:
            raise ValueError("trajectory iteration must be a non-negative integer")
        directory = self.root / "trajectories" / f"iter_{iteration:02d}"
        directory.mkdir(parents=True, exist_ok=False)
        return directory
