"""One photo in, counts per SKU out.

orient -> detect -> crop -> (stick filter band | lighter color) -> gallery
match -> optional wordmark OCR -> group by ``(sku, category)``.

Thresholds
----------
These are conservative starting points, not tuned against labels (there are
none yet). Raise or lower them once a gallery and a labeled audit set exist.

``MIN_SCORE`` (detector.py, ``--min-score``)
    Detector boxes below this are dropped.
``REVIEW_SCORE``
    Kept boxes below this are flagged for review, unless the gallery match
    is at least ``STRONG_SIMILARITY``.
``STRONG_SIMILARITY``
    A gallery cosine this high confirms the object on its own, so a weak
    detector score alone does not flag it. Zero-shot text-prompt scores
    on these packs run low even when the box is right.
``NESTED_OVERLAP``
    A box with at least this share of its area inside a higher-scoring box
    of the same category is a fragment (half a pack, a slid-out sleeve)
    and is dropped before counting.
``MIN_SIMILARITY`` (``--min-similarity``)
    Best gallery cosine below this gives ``sku: unknown``.
``MIN_MARGIN``
    Top-1 minus top-2 SKU cosine below this is flagged.
``MIN_COLOR_SIMILARITY``
    Lighter histogram intersection with the matched SKU below this is flagged.
``BAND_TIE_RATIO`` (crops.py)
    Stick ends closer than this are both matched. Different SKUs flag the stick.
"""

from __future__ import annotations

# Before cv2, torch, and paddle.
from stock_material_detection.cpu_threads import configure_cpu_threads

configure_cpu_threads()

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from stock_material_detection.crops import (
    color_histogram,
    crop_box,
    filter_bands,
    polygon_mask,
)
from stock_material_detection.detector import MIN_SCORE, Detection, Detector
from stock_material_detection.gallery import Gallery, Match, WordmarkReader, ocr_enabled
from stock_material_detection.orient import load_oriented, to_bgr

logger = logging.getLogger(__name__)

UNKNOWN = "unknown"
REVIEW_SCORE = 0.30
STRONG_SIMILARITY = 0.85
NESTED_OVERLAP = 0.8
MIN_SIMILARITY = 0.70
MIN_MARGIN = 0.02
MIN_COLOR_SIMILARITY = 0.40
INSIDE_PACK_OVERLAP = 0.6
"""Share of a stick box inside a pack box that marks it as still in the pack."""

OCR_CATEGORIES = {"cigarette_pack", "lighter"}


@dataclass
class Item:
    """One detection after identification."""

    category: str
    score: float
    box: tuple[float, float, float, float]
    sku: str = UNKNOWN
    similarity: float | None = None
    flagged: bool = True
    inside_pack: bool = False
    wordmark: str | None = None
    color_similarity: float | None = None

    def public_dict(self) -> dict:
        data = {
            "category": self.category,
            "sku": self.sku,
            "score": round(self.score, 4),
            "similarity": None if self.similarity is None else round(self.similarity, 4),
            "box": [round(v, 1) for v in self.box],
            "flagged": self.flagged,
        }
        if self.category == "cigarette_stick":
            data["insidePack"] = self.inside_pack
        if self.color_similarity is not None:
            data["colorSimilarity"] = round(self.color_similarity, 4)
        if self.wordmark is not None:
            data["wordmark"] = self.wordmark
        return data


class CountLine(BaseModel):
    sku: str
    category: str
    count: int
    flagged: bool


class DetectResult(BaseModel):
    image: str | None = None
    counts: list[CountLine] = Field(default_factory=list)
    detections: list[dict] = Field(default_factory=list)
    failReasons: list[str] = Field(default_factory=list)

    def public_dict(self) -> dict:
        data = self.model_dump(mode="json")
        if not data["failReasons"]:
            del data["failReasons"]
        return data


def _overlap_of(inner: tuple[float, ...], outer: tuple[float, ...]) -> float:
    """Share of ``inner``'s area that lies inside ``outer``."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def drop_nested(detections: list[Detection], threshold: float = NESTED_OVERLAP) -> list[Detection]:
    """Drop boxes mostly inside a higher-scoring box of the same category.

    NMS only compares IoU, so half a pack or its slid-out inner sleeve
    survives next to the full pack box and would be counted again. Order
    is preserved.
    """
    ranked = sorted(range(len(detections)), key=lambda i: -detections[i].score)
    keep: list[int] = []
    for index in ranked:
        det = detections[index]
        nested = any(
            detections[k].category == det.category and _overlap_of(det.box, detections[k].box) >= threshold
            for k in keep
        )
        if not nested:
            keep.append(index)
    return [detections[i] for i in sorted(keep)]


def mark_inside_packs(items: list[Item], threshold: float = INSIDE_PACK_OVERLAP) -> None:
    packs = [it.box for it in items if it.category == "cigarette_pack"]
    for item in items:
        if item.category != "cigarette_stick":
            continue
        item.inside_pack = any(_overlap_of(item.box, pack) >= threshold for pack in packs)


def aggregate(items: list[Item]) -> list[CountLine]:
    """Count each pack, loose stick, lighter, ashtray, and speaker as 1.

    Sticks still inside a pack are not counted. A line is flagged when any
    of its detections is flagged, and ``unknown`` is always flagged.
    """
    lines: dict[tuple[str, str], CountLine] = {}
    for item in items:
        if item.category == "cigarette_stick" and item.inside_pack:
            continue
        key = (item.sku, item.category)
        line = lines.get(key)
        if line is None:
            line = lines[key] = CountLine(sku=item.sku, category=item.category, count=0, flagged=False)
        line.count += 1
        line.flagged = line.flagged or item.flagged or item.sku == UNKNOWN
    return sorted(lines.values(), key=lambda line: (line.category, line.sku))


_detector: Detector | None = None
_galleries: dict[Path, Gallery] = {}
_ocr: WordmarkReader | None = None


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector


def get_gallery(gallery_dir: Path | None = None) -> Gallery:
    from stock_material_detection.gallery import default_gallery_dir

    key = (gallery_dir or default_gallery_dir()).resolve()
    if key not in _galleries:
        _galleries[key] = Gallery(key)
    return _galleries[key]


def get_ocr() -> WordmarkReader | None:
    global _ocr
    if not ocr_enabled():
        return None
    if _ocr is None:
        _ocr = WordmarkReader()
    return _ocr


def is_weak(score: float, similarity: float | None) -> bool:
    """Low detector score that the gallery match does not make up for."""
    return score < REVIEW_SCORE and (similarity is None or similarity < STRONG_SIMILARITY)


def _resolve_end_matches(
    matches: list[Match],
    min_similarity: float,
) -> tuple[Match | None, bool]:
    """Best end by similarity, and whether two named ends disagree on SKU.

    An end names a SKU only at or above ``min_similarity``. A weaker other
    end does not count as a disagreement.
    """
    if not matches:
        return None, False
    best = max(matches, key=lambda match: match.similarity)
    disagree = best.similarity >= min_similarity and any(
        match.similarity >= min_similarity and match.sku != best.sku for match in matches
    )
    return best, disagree


def _identify(
    item: Item,
    det: Detection,
    image: np.ndarray,
    gallery: Gallery,
    ocr: WordmarkReader | None,
    *,
    min_similarity: float,
    debug: tuple[Path, int] | None,
) -> None:
    """Fill ``sku``, ``similarity``, and ``flagged`` on ``item`` from its crop."""
    crop, offset = crop_box(image, det.box)
    mask = polygon_mask(crop.shape, det.polygon, offset) if det.polygon is not None else None

    hist = None
    ends_disagree = False
    if det.category == "cigarette_stick":
        queries = filter_bands(crop, mask)
    else:
        queries = [crop]
        if det.category == "lighter":
            hist = color_histogram(crop, mask)

    if debug is not None:
        out_dir, index = debug
        cv2.imwrite(str(out_dir / f"{index:02d}_{det.category}.jpg"), crop)
        if det.category == "cigarette_stick":
            for suffix, band in zip(("", "_alt"), queries):
                if band is crop:
                    continue
                cv2.imwrite(str(out_dir / f"{index:02d}_{det.category}_band{suffix}.jpg"), band)

    queries = [query for query in queries if query.size]
    if gallery.empty or not queries:
        item.flagged = True
        return

    if det.category == "cigarette_stick" and len(queries) > 1:
        found = [gallery.match(gallery.embed(band)) for band in queries]
        match, ends_disagree = _resolve_end_matches(
            [hit for hit in found if hit is not None],
            min_similarity,
        )
    else:
        match = gallery.match(gallery.embed(queries[0]), hist)

    if match is None:
        item.flagged = True
        return

    item.similarity = match.similarity
    item.color_similarity = match.color_similarity
    if match.similarity < min_similarity:
        item.flagged = True
        return

    item.sku = match.sku
    weak = is_weak(det.score, match.similarity)
    thin_gap = match.margin is not None and match.margin < MIN_MARGIN
    off_color = match.color_similarity is not None and match.color_similarity < MIN_COLOR_SIMILARITY
    if ocr is not None and det.category in OCR_CATEGORIES:
        item.wordmark = ocr.check(crop, gallery.wordmark(match.sku))
    item.flagged = weak or thin_gap or off_color or ends_disagree or item.wordmark == "mismatch"


def run_detect(
    image_path: Path,
    *,
    min_score: float = MIN_SCORE,
    min_similarity: float = MIN_SIMILARITY,
    gallery_dir: Path | None = None,
    debug_dir: Path | None = None,
) -> DetectResult:
    """Run the full pipeline on one photo."""
    result = DetectResult(image=image_path.name)
    try:
        oriented = load_oriented(image_path)
    except OSError:
        logger.warning("unreadable image %s", image_path)
        result.failReasons = ["unreadable_image"]
        return result

    image = to_bgr(oriented)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "oriented.jpg"), image)

    detections = drop_nested(get_detector().detect(image, min_score=min_score))
    gallery = get_gallery(gallery_dir)
    ocr = get_ocr()

    items = [Item(category=det.category, score=det.score, box=det.box) for det in detections]
    mark_inside_packs(items)
    for index, (item, det) in enumerate(zip(items, detections)):
        if item.inside_pack:
            # Not counted, so there is nothing to review or identify.
            item.flagged = False
            continue
        _identify(
            item,
            det,
            image,
            gallery,
            ocr,
            min_similarity=min_similarity,
            debug=None if debug_dir is None else (debug_dir, index),
        )

    if debug_dir is not None:
        _save_overlay(image, items, debug_dir / "boxes.jpg")

    result.counts = aggregate(items)
    result.detections = [item.public_dict() for item in items]
    return result


_OVERLAY_COLORS = {
    "cigarette_pack": (0, 200, 255),
    "cigarette_stick": (255, 255, 255),
    "lighter": (255, 120, 0),
    "ashtray": (0, 255, 0),
    "speaker": (255, 0, 255),
}


def _save_overlay(image: np.ndarray, items: list[Item], path: Path) -> None:
    canvas = image.copy()
    for index, item in enumerate(items):
        x1, y1, x2, y2 = (int(round(v)) for v in item.box)
        color = (128, 128, 128) if item.inside_pack else _OVERLAY_COLORS.get(item.category, (0, 0, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{index:02d} {item.category} {item.score:.2f}"
        cv2.putText(canvas, label, (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.imwrite(str(path), canvas)
