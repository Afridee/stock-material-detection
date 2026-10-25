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

BAND_TIE_RATIO = 0.05
"""Relative print-detail gap below which both stick ends are kept.

``abs(left - right) / max(left, right)``. A pale filter and a plain shaft
can fall inside this, so the caller matches both instead of guessing.
"""

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


def _print_detail(band: np.ndarray) -> float:
    """Mean Lab change between neighbouring columns of a horizontal band.

    Only change along the stick counts. The stick's edges, its shading, and
    any surface caught above or below it run lengthwise and would otherwise
    outweigh a pale filter print.
    """
    lab = cv2.cvtColor(band, cv2.COLOR_BGR2LAB).astype(np.float32)
    if lab.shape[1] < 2:
        return 0.0
    return float(np.abs(np.diff(lab, axis=1)).mean(axis=(0, 1)).sum())


def _oriented_ends(
    crop: np.ndarray,
    mask: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Both stick ends, each with its tip facing right, plus their print detail.

    ``None`` when the outline is missing or too small to orient.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    (cx, cy), (w, h), angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    if w < 2 or h < 2:
        return None
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
    left = np.ascontiguousarray(stick[:, :band_len][:, ::-1])
    right = np.ascontiguousarray(stick[:, -band_len:])
    return left, right, _print_detail(left), _print_detail(right)


def filter_bands(
    crop: np.ndarray,
    mask: np.ndarray | None = None,
    fraction: float = FILTER_BAND_FRACTION,
    tie_ratio: float = BAND_TIE_RATIO,
) -> list[np.ndarray]:
    """Filter end of one stick, or both ends when their print detail is a tie.

    The stick is rotated horizontal. Each kept end is flipped so its tip
    faces right. The end with more print detail is first. Both are returned when
    ``abs(left - right) / max(left, right)`` is below ``tie_ratio``.
    """
    if mask is None:
        mask = foreground_mask(crop)
    ends = _oriented_ends(crop, mask, fraction)
    if ends is None:
        return [crop]
    left, right, left_var, right_var = ends
    ordered = [left, right] if left_var > right_var else [right, left]
    peak = max(left_var, right_var)
    tie = peak == 0 or abs(left_var - right_var) / peak < tie_ratio
    return ordered if tie else ordered[:1]


def filter_band(
    crop: np.ndarray,
    mask: np.ndarray | None = None,
    fraction: float = FILTER_BAND_FRACTION,
) -> np.ndarray:
    """The stronger filter end of one stick. See ``filter_bands`` for a tie."""
    return filter_bands(crop, mask, fraction)[0]


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
