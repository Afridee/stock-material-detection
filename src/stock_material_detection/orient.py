"""Load a phone photo with its EXIF orientation applied to the pixels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_oriented(path: Path) -> Image.Image:
    """RGB image rotated per EXIF, so boxes are in the upright frame.

    Raises ``OSError`` (Pillow's ``UnidentifiedImageError`` included) when the
    file is not a readable image.
    """
    with Image.open(path) as im:
        im.load()
        oriented = ImageOps.exif_transpose(im)
        return oriented.convert("RGB")


def to_bgr(image: Image.Image) -> np.ndarray:
    """Pillow RGB to the BGR array OpenCV and Ultralytics expect."""
    return np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
