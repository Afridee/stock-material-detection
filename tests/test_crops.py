"""Crop helpers on synthetic images."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from stock_material_detection.crops import (
    FILTER_BAND_FRACTION,
    color_histogram,
    crop_box,
    filter_band,
    histogram_similarity,
)

STICK_LENGTH = 300
STICK_WIDTH = 30
FILTER_LENGTH = 80


def _synthetic_stick(angle: float, filter_at_start: bool) -> np.ndarray:
    """A white stick on a dark surface, rotated, with a striped colored filter at one end."""
    canvas = np.full((420, 420, 3), 50, dtype=np.uint8)
    stick = np.full((STICK_WIDTH, STICK_LENGTH, 3), 245, dtype=np.uint8)
    filter_part = np.zeros((STICK_WIDTH, FILTER_LENGTH, 3), dtype=np.uint8)
    for x in range(FILTER_LENGTH):
        filter_part[:, x] = (40, 160, 230) if (x // 6) % 2 else (60, 200, 90)
    if filter_at_start:
        stick[:, :FILTER_LENGTH] = filter_part
    else:
        stick[:, -FILTER_LENGTH:] = filter_part

    y0 = (canvas.shape[0] - STICK_WIDTH) // 2
    x0 = (canvas.shape[1] - STICK_LENGTH) // 2
    canvas[y0 : y0 + STICK_WIDTH, x0 : x0 + STICK_LENGTH] = stick
    center = (canvas.shape[1] / 2, canvas.shape[0] / 2)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        canvas, rotation, (canvas.shape[1], canvas.shape[0]), borderValue=(50, 50, 50)
    )


def _saturation(image: np.ndarray) -> float:
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].mean())


@pytest.mark.parametrize("angle", [0.0, 30.0, 75.0, -50.0])
@pytest.mark.parametrize("filter_at_start", [True, False])
def test_filter_band_keeps_one_end(angle: float, filter_at_start: bool) -> None:
    image = _synthetic_stick(angle, filter_at_start)

    band = filter_band(image)

    height, width = band.shape[:2]
    expected = STICK_LENGTH * FILTER_BAND_FRACTION
    assert width < STICK_LENGTH * 0.5, "kept the whole stick, not one end"
    assert abs(width - expected) <= 15
    assert abs(height - STICK_WIDTH) <= 8
    # The colored filter end, not the white paper shaft.
    assert _saturation(band) > 80


def test_crop_box_pads_and_clamps() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    crop, offset = crop_box(image, (10.0, 20.0, 110.0, 70.0), padding=0.1)
    assert offset == (0, 15)
    assert crop.shape[:2] == (60, 120)

    edge, edge_offset = crop_box(image, (150.0, 50.0, 200.0, 100.0), padding=0.1)
    assert edge_offset == (145, 45)
    assert edge.shape[:2] == (55, 55)


def test_histogram_similarity_separates_colors() -> None:
    blue = np.zeros((20, 20, 3), dtype=np.uint8)
    blue[:] = (200, 60, 20)
    red = np.zeros((20, 20, 3), dtype=np.uint8)
    red[:] = (20, 40, 210)

    assert histogram_similarity(color_histogram(blue), color_histogram(blue)) == pytest.approx(1.0)
    assert histogram_similarity(color_histogram(blue), color_histogram(red)) < 0.1
