"""Pack size from the face outline, for brands sold in more than one size."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from stock_material_detection.crops import face_aspect
from stock_material_detection.detector import Detection
from stock_material_detection.gallery import Gallery, Match
from stock_material_detection.pipeline import Item, _identify, aggregate, pick_pack_size

BLACK_DIAMOND = {"20": 1.58, "10": 2.03}


def _rect(long: float, short: float, angle: float, center=(400.0, 300.0)) -> np.ndarray:
    return cv2.boxPoints((center, (short, long), angle)).astype(np.float32)


@pytest.mark.parametrize("angle", [0, 17, 45, 88])
def test_face_aspect_ignores_rotation(angle: float) -> None:
    assert face_aspect(_rect(300, 150, angle)) == pytest.approx(2.0, rel=1e-3)


def test_face_aspect_needs_a_polygon() -> None:
    assert face_aspect(None) is None
    assert face_aspect(np.array([[0, 0], [10, 0]], dtype=np.float32)) is None


@pytest.mark.parametrize(
    ("aspect", "size"),
    [
        (1.574, "20"),
        (1.591, "20"),
        (1.994, "10"),
        (2.054, "10"),
        (1.80, None),
        (3.5, None),
        (None, None),
    ],
)
def test_pick_pack_size(aspect: float | None, size: str | None) -> None:
    assert pick_pack_size(aspect, BLACK_DIAMOND) == size


def test_no_sizes_means_no_size() -> None:
    assert pick_pack_size(1.6, {}) is None


def test_gallery_reads_pack_sizes(tmp_path) -> None:
    (tmp_path / "black_diamond").mkdir()
    (tmp_path / "black_diamond" / "pack_sizes.json").write_text('{"20": 1.58, "10": 2.03}')
    (tmp_path / "avon_blue").mkdir()
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "pack_sizes.json").write_text("[1.5]")

    gallery = Gallery(tmp_path)

    assert gallery.pack_sizes("black_diamond") == BLACK_DIAMOND
    assert gallery.pack_sizes("avon_blue") == {}
    assert gallery.pack_sizes("broken") == {}


class _FakeGallery:
    empty = False

    def __init__(self, sizes: dict[str, float]) -> None:
        self._sizes = sizes

    def embed(self, crop: np.ndarray) -> np.ndarray:
        return np.ones((1, 2), dtype=np.float32)

    def match(self, vectors: np.ndarray, hist: np.ndarray | None = None) -> Match:
        return Match(sku="black_diamond", similarity=0.95, runner_up=0.5, ref="black_diamond/pack.jpg")

    def wordmark(self, sku: str) -> str:
        return "BLACK DIAMOND"

    def pack_sizes(self, sku: str) -> dict[str, float]:
        return self._sizes


def _identified(long: float, short: float, sizes: dict[str, float] = BLACK_DIAMOND) -> Item:
    polygon = _rect(long, short, 3)
    x1, y1 = polygon.min(axis=0)
    x2, y2 = polygon.max(axis=0)
    det = Detection("cigarette_pack", 0.8, (float(x1), float(y1), float(x2), float(y2)), polygon)
    item = Item(category=det.category, score=det.score, box=det.box)
    image = np.full((600, 800, 3), 90, dtype=np.uint8)
    _identify(item, det, image, _FakeGallery(sizes), None, min_similarity=0.7, debug=None)
    return item


def test_twenty_and_ten_packs_count_on_separate_lines() -> None:
    items = [_identified(316, 200), _identified(406, 200), _identified(400, 197)]

    assert [it.sku for it in items] == ["black_diamond_20", "black_diamond_10", "black_diamond_10"]
    assert not any(it.flagged for it in items)
    assert items[0].public_dict()["packSize"] == "20"
    counts = {(line.sku, line.count) for line in aggregate(items)}
    assert counts == {("black_diamond_20", 1), ("black_diamond_10", 2)}


def test_unclear_size_keeps_brand_and_flags() -> None:
    item = _identified(360, 200)

    assert item.sku == "black_diamond"
    assert item.pack_size is None
    assert item.flagged
    assert item.public_dict()["faceAspect"] == pytest.approx(1.8, abs=0.01)


def test_single_size_brand_is_unchanged() -> None:
    item = _identified(360, 200, sizes={})

    assert item.sku == "black_diamond"
    assert item.face_aspect is None
    assert not item.flagged
