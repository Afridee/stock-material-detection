"""EXIF orientation is applied to the pixels the detector sees."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from stock_material_detection.orient import load_oriented, to_bgr

RED = (255, 0, 0)
BLUE = (0, 0, 255)


def _write_rotated_jpeg(path: Path, orientation: int) -> None:
    """40×20 stored image: left half red, right half blue."""
    im = Image.new("RGB", (40, 20), BLUE)
    im.paste(Image.new("RGB", (20, 20), RED), (0, 0))
    exif = Image.Exif()
    exif[0x0112] = orientation
    im.save(path, quality=100, exif=exif)


def _close(pixel: tuple[int, ...], color: tuple[int, int, int]) -> bool:
    return all(abs(a - b) < 40 for a, b in zip(pixel, color))


def test_orientation_6_rotates_pixels(tmp_path: Path) -> None:
    path = tmp_path / "phone.jpg"
    _write_rotated_jpeg(path, orientation=6)

    image = load_oriented(path)

    # Orientation 6 displays the stored image turned 90° clockwise, so the
    # stored left (red) half ends up on top.
    assert image.size == (20, 40)
    assert _close(image.getpixel((10, 5)), RED)
    assert _close(image.getpixel((10, 35)), BLUE)


def test_no_exif_keeps_pixels(tmp_path: Path) -> None:
    path = tmp_path / "flat.jpg"
    _write_rotated_jpeg(path, orientation=1)

    image = load_oriented(path)

    assert image.size == (40, 20)
    assert _close(image.getpixel((5, 10)), RED)


def test_to_bgr_swaps_channels(tmp_path: Path) -> None:
    bgr = to_bgr(Image.new("RGB", (2, 2), RED))
    assert tuple(bgr[0, 0]) == (0, 0, 255)


def test_unreadable_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(OSError):
        load_oriented(path)
