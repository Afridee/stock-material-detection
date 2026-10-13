"""Crop helpers: padded box crops, the stick filter band, and lighter color.

Every function takes and returns BGR ``uint8`` arrays.
"""

from __future__ import annotations

import cv2
import numpy as np

BOX_PADDING = 0.04
"""Fraction of the box width/height added on each side before cropping."""

FILTER_BAND_FRACTION = 0.3
"""Share of the stick length kept at the printed filter end."""

HIST_BINS = (18, 8)
"""Hue × saturation bins for the lighter color histogram."""


def crop_box(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    padding: float = BOX_PADDING,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Padded ``image[y1:y2, x1:x2]`` and the crop's top-left offset."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(width, int(np.ceil(x2 + pad_x)))
    bottom = min(height, int(np.ceil(y2 + pad_y)))
    return image[top:bottom, left:right], (left, top)


def polygon_mask(shape: tuple[int, ...], polygon: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    """Rasterize an image-space polygon into a mask for a crop at ``offset``."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    local = np.round(polygon - np.asarray(offset, dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(mask, [local], 255)
    return mask


def foreground_mask(crop: np.ndarray) -> np.ndarray:
    """Object mask from Lab distance to the crop's border color.

    The padded crop's border is mostly the surface the item lies on, so its
    median color stands in for the background.
    """
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    distance = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(distance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _color_variation(region: np.ndarray) -> float:
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return float(lab.std(axis=0).sum())


def filter_band(
    crop: np.ndarray,
    mask: np.ndarray | None = None,
    fraction: float = FILTER_BAND_FRACTION,
) -> np.ndarray:
    """The printed filter end of one stick, rotated horizontal.

    ``cv2.minAreaRect`` on the stick outline gives its axis. The stick is
    rotated so that axis is horizontal, then the end with more color
    variation is kept. The white paper shaft is dropped. The kept end is
    flipped if needed so the tip always faces right.
    """
    if mask is None:
        mask = foreground_mask(crop)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop
    (cx, cy), (w, h), angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    if w < 2 or h < 2:
        return crop
    if w < h:
        w, h = h, w
        angle += 90.0

    # Pad so the rotation does not clip the stick ends.
    margin = int(np.ceil(np.hypot(*crop.shape[:2]) / 2))
    padded = cv2.copyMakeBorder(crop, margin, margin, margin, margin, cv2.BORDER_REPLICATE)
    center = (cx + margin, cy + margin)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        padded, rotation, (padded.shape[1], padded.shape[0]), flags=cv2.INTER_LINEAR
    )
    stick = cv2.getRectSubPix(rotated, (int(round(w)), int(round(h))), center)

    band_len = max(1, int(round(stick.shape[1] * fraction)))
    left = stick[:, :band_len]
    right = stick[:, -band_len:]
    if _color_variation(left) > _color_variation(right):
        return np.ascontiguousarray(left[:, ::-1])
    return np.ascontiguousarray(right)


def color_histogram(crop: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """L1-normalized hue × saturation histogram, flattened to float32."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, list(HIST_BINS), [0, 180, 0, 256])
    hist = hist.flatten().astype(np.float32)
    total = float(hist.sum())
    return hist / total if total > 0 else hist


def histogram_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram intersection of two L1-normalized histograms, in [0, 1]."""
    return float(np.minimum(a, b).sum())
