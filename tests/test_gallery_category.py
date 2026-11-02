"""``category.txt`` re-labels a matched box for products with no detector prompt."""

from __future__ import annotations

import numpy as np

from stock_material_detection.detector import Detection
from stock_material_detection.gallery import Gallery, Match
from stock_material_detection.pipeline import Item, _identify, aggregate, gallery_confirms


def test_gallery_reads_category(tmp_path) -> None:
    (tmp_path / "cyxg_x87").mkdir()
    (tmp_path / "cyxg_x87" / "category.txt").write_text("earphones\n")
    (tmp_path / "avon_blue").mkdir()

    gallery = Gallery(tmp_path)

    assert gallery.category("cyxg_x87") == "earphones"
    assert gallery.category("avon_blue") is None


class _FakeGallery:
    empty = False

    def __init__(self, similarity: float = 0.95) -> None:
        self._similarity = similarity

    def embed(self, crop: np.ndarray) -> np.ndarray:
        return np.ones((1, 2), dtype=np.float32)

    def match(self, vectors: np.ndarray, hist: np.ndarray | None = None) -> Match:
        return Match(sku="cyxg_x87", similarity=self._similarity, runner_up=0.5, ref="cyxg_x87/box.jpg")

    def wordmark(self, sku: str) -> str:
        return "CYXG"

    def pack_sizes(self, sku: str) -> dict[str, float]:
        return {"20": 1.58}

    def category(self, sku: str) -> str | None:
        return "earphones"


def _identified(category: str, similarity: float = 0.95, score: float = 0.9) -> Item:
    det = Detection(category, score, (100.0, 100.0, 300.0, 400.0), None)
    item = Item(category=det.category, score=det.score, box=det.box)
    image = np.full((600, 800, 3), 200, dtype=np.uint8)
    _identify(item, det, image, _FakeGallery(similarity), None, min_similarity=0.7, debug=None)
    return item


def test_matched_box_takes_gallery_category() -> None:
    item = _identified("cigarette_pack")

    assert (item.sku, item.category) == ("cyxg_x87", "earphones")
    assert item.pack_size is None
    assert not item.flagged
    assert [(line.sku, line.category, line.count) for line in aggregate([item])] == [
        ("cyxg_x87", "earphones", 1)
    ]


def test_unmatched_box_keeps_detector_category() -> None:
    item = _identified("cigarette_pack", similarity=0.5)

    assert (item.sku, item.category) == ("unknown", "cigarette_pack")


def test_stick_matching_a_categorised_sku_stays_unknown() -> None:
    item = _identified("cigarette_stick")

    assert (item.sku, item.category) == ("unknown", "cigarette_stick")
    assert item.flagged


def test_weak_box_is_not_relabelled_by_a_middling_match() -> None:
    weak = _identified("cigarette_pack", similarity=0.78, score=0.1)
    confident = _identified("cigarette_pack", similarity=0.9, score=0.1)

    assert (weak.sku, weak.category, weak.flagged) == ("unknown", "cigarette_pack", True)
    assert (confident.sku, confident.category) == ("cyxg_x87", "earphones")


def test_fallback_box_needs_a_gallery_sku_of_its_category() -> None:
    det = Detection("earphones", 0.1, (0.0, 0.0, 10.0, 10.0), None, fallback=True)
    gallery = _FakeGallery()

    confirmed = Item(category="earphones", score=0.1, box=det.box, sku="cyxg_x87")
    unknown = Item(category="earphones", score=0.1, box=det.box)
    other = Item(category="earphones", score=0.1, box=det.box, sku="avon_blue")
    gallery.category = lambda sku: "earphones" if sku == "cyxg_x87" else None

    assert gallery_confirms(confirmed, det, gallery)
    assert not gallery_confirms(unknown, det, gallery)
    assert not gallery_confirms(other, det, gallery)
