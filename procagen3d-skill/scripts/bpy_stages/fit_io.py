"""Image and path helpers for the registered image-fit stage."""

import hashlib
import math
from pathlib import Path

import bpy


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_path(out_dir, value, label):
    """Resolve a fit artifact inside the asset output directory."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    root = Path(out_dir).resolve()
    candidate = (root / value).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"{label} must stay inside {root}: {value!r}")
    if not candidate.is_file():
        raise ValueError(f"{label} not found: {candidate}")
    return candidate


def finite_values(value, count, label):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{label} must contain {count} numbers")
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain {count} numbers") from exc
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{label} must contain finite numbers")
    return values


def load_rgba(path):
    """Load an image as a top-down float RGBA numpy array."""
    import numpy as np

    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = (int(value) for value in image.size)
        if width <= 0 or height <= 0:
            raise ValueError(f"image has invalid dimensions: {path}")
        pixels = np.array(image.pixels[:], dtype=np.float32)
        rgba = pixels.reshape(height, width, 4)
        return np.flipud(rgba).copy()
    finally:
        bpy.data.images.remove(image)


def save_rgba(path, rgba):
    """Save a top-down float RGBA numpy array without external imaging deps."""
    import numpy as np

    height, width, channels = rgba.shape
    if channels != 4:
        raise ValueError("save_rgba expects HxWx4 pixels")
    image = bpy.data.images.new(
        "ProcAgen3D_" + Path(path).stem, width=width, height=height, alpha=True)
    try:
        stored = np.flipud(np.clip(rgba, 0.0, 1.0)).astype(np.float32)
        image.pixels = stored.ravel().tolist()
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        image.save()
    finally:
        bpy.data.images.remove(image)


def save_mask(path, mask):
    import numpy as np

    value = mask.astype(np.float32)
    rgba = np.empty((*value.shape, 4), dtype=np.float32)
    rgba[..., :3] = value[..., None]
    rgba[..., 3] = 1.0
    save_rgba(path, rgba)
