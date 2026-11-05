"""One photo in, counts per SKU out.

orient -> detect -> crop -> (stick filter band | lighter color) -> gallery
match -> optional wordmark OCR -> group by ``(sku, category)``.

Thresholds
----------
These are conservative starting points, not tuned against a labeled audit
set (there is none yet). Raise or lower them once that set exists.

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
``FRAGMENT_OVERLAP``
    An unnamed box with at least this share of its area inside a named box,
    whose closest gallery image is that box's SKU, is a split-off piece of
    it and is dropped. Measured: 0.43 for the end of a USB lighter.
``MIN_SIMILARITY`` (``--min-similarity``)
    Best gallery cosine below this gives ``sku: unknown``.
``MIN_MARGIN``
    Top-1 minus top-2 SKU cosine below this is flagged.
``MIN_COLOR_SIMILARITY``
    Lighter histogram intersection with the matched SKU below this is flagged.
``BAND_TIE_RATIO`` (crops.py)
    Stick ends closer than this are both matched. Different SKUs flag the stick.
``SIZE_TOLERANCE``
    For SKUs with ``pack_sizes.json``, a pack face aspect ratio further than
    this (relative) from every listed size is flagged and keeps the brand SKU.
    Measured 20s and 10s Black Diamond faces sit within 0.02 of their listed
    value; an edge-on or squashed pack lands outside.
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
    face_aspect,
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
FRAGMENT_OVERLAP = 0.3
MIN_SIMILARITY = 0.70
MIN_MARGIN = 0.02
MIN_COLOR_SIMILARITY = 0.40
INSIDE_PACK_OVERLAP = 0.6
"""Share of a stick box inside a pack box that marks it as still in the pack."""
SIZE_TOLERANCE = 0.07

OCR_CATEGORIES = {"cigarette_pack", "lighter", "earphones"}


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
    rival_sku: str | None = None
    """Other SKU in a close call: the runner-up, or the other stick end's SKU."""
    rival_similarity: float | None = None
    match_ref: str | None = None
    """Best-matching gallery image (``sku/file.jpg``)."""
    rival_ref: str | None = None
    face_aspect: float | None = None
    """Pack face long/short side, only for SKUs with more than one pack size."""
    pack_size: str | None = None

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
        if self.face_aspect is not None:
            data["faceAspect"] = round(self.face_aspect, 3)
        if self.pack_size is not None:
            data["packSize"] = self.pack_size
        if self.match_ref is not None:
            data["matchRef"] = self.match_ref
        if self.rival_sku is not None:
            data["rivalSku"] = self.rival_sku
            data["rivalSimilarity"] = (
                None if self.rival_similarity is None else round(self.rival_similarity, 4)
            )
            if self.rival_ref is not None:
                data["rivalRef"] = self.rival_ref
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


def drop_split_fragments(items: list[Item], threshold: float = FRAGMENT_OVERLAP) -> list[Item]:
    """Drop unnamed boxes that are a split-off piece of a named neighbour.

    The detector sometimes boxes one end of an object separately under
    another category (the end of a USB lighter as "lighter"), so
    ``drop_nested`` misses it. The piece is too weak to be named, but its
    closest gallery image is the neighbour's SKU. Order is preserved.
    """
    named = [it for it in items if it.sku != UNKNOWN]
    keep = []
    for item in items:
        ref_sku = item.match_ref.split("/", 1)[0] if item.match_ref else None
        split = (
            item.sku == UNKNOWN
            and ref_sku is not None
            and any(
                other.match_ref and other.match_ref.split("/", 1)[0] == ref_sku
                and _overlap_of(item.box, other.box) >= threshold
                for other in named
            )
        )
        if not split:
            keep.append(item)
    return keep


def mark_inside_packs(items: list[Item], threshold: float = INSIDE_PACK_OVERLAP) -> None:
    packs = [it.box for it in items if it.category == "cigarette_pack"]
    for item in items:
        if item.category != "cigarette_stick":
            continue
        item.inside_pack = any(_overlap_of(item.box, pack) >= threshold for pack in packs)


def aggregate(items: list[Item]) -> list[CountLine]:
    """Count each pack, loose stick, lighter, ashtray, speaker, earphones box, matchbox, cigarette holder, keyring watch, electric lighter, wallet, and bag as 1.

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


def gallery_confirms(item: Item, det: Detection, gallery: Gallery) -> bool:
    """A fallback box counts only when it matched a SKU of its own category.

    The fallback prompt also fires weakly on clumps of loose sticks, which
    then match a stick band; those are dropped rather than counted.
    """
    return item.sku != UNKNOWN and gallery.category(item.sku) == det.category


def pick_pack_size(
    aspect: float | None,
    sizes: dict[str, float],
    tolerance: float = SIZE_TOLERANCE,
) -> str | None:
    """Size whose listed face aspect is nearest, or ``None`` when none is within ``tolerance``."""
    if aspect is None or not sizes:
        return None
    size, listed = min(sizes.items(), key=lambda entry: abs(np.log(aspect / entry[1])))
    return size if abs(aspect - listed) / listed <= tolerance else None


def _resolve_end_matches(
    matches: list[Match],
    min_similarity: float,
) -> tuple[Match | None, Match | None]:
    """Best end by similarity, and the best end naming a different SKU, if any.

    An end names a SKU only at or above ``min_similarity``. A weaker other
    end does not count as a disagreement.
    """
    if not matches:
        return None, None
    best = max(matches, key=lambda match: match.similarity)
    if best.similarity < min_similarity:
        return best, None
    rivals = [
        match for match in matches if match.similarity >= min_similarity and match.sku != best.sku
    ]
    return best, max(rivals, key=lambda match: match.similarity, default=None)


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
    end_rival: Match | None = None
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
        match, end_rival = _resolve_end_matches(
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
    item.match_ref = match.ref
    if match.similarity < min_similarity:
        item.flagged = True
        return
    weak = is_weak(det.score, match.similarity)
    if gallery.category(match.sku) and (det.category == "cigarette_stick" or weak):
        # Re-labelling overrides the detector, so it needs a confident match.
        # Weak frame-edge and pack-edge boxes reach 0.74-0.80 on the bagged
        # electric lighter references; real re-labelled products score 0.86+.
        item.flagged = True
        return

    item.sku = match.sku
    if det.category != "cigarette_stick":
        item.category = gallery.category(match.sku) or item.category
    size_unsure = False
    sizes = gallery.pack_sizes(match.sku) if item.category == "cigarette_pack" else {}
    if sizes:
        item.face_aspect = face_aspect(det.polygon)
        item.pack_size = pick_pack_size(item.face_aspect, sizes)
        if item.pack_size is None:
            size_unsure = True
        else:
            item.sku = f"{match.sku}_{item.pack_size}"
    thin_gap = match.margin is not None and match.margin < MIN_MARGIN
    if end_rival is not None:
        item.rival_sku, item.rival_similarity = end_rival.sku, end_rival.similarity
        item.rival_ref = end_rival.ref
    elif thin_gap:
        item.rival_sku, item.rival_similarity = match.runner_up_sku, match.runner_up
        item.rival_ref = match.runner_up_ref
    off_color = match.color_similarity is not None and match.color_similarity < MIN_COLOR_SIMILARITY
    if ocr is not None and item.category in OCR_CATEGORIES:
        item.wordmark = ocr.check(crop, gallery.wordmark(match.sku))
    item.flagged = (
        weak
        or thin_gap
        or off_color
        or size_unsure
        or end_rival is not None
        or item.wordmark == "mismatch"
    )


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
    items = [
        item for item, det in zip(items, detections) if not det.fallback or gallery_confirms(item, det, gallery)
    ]
    items = drop_split_fragments(items)

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
    "earphones": (255, 255, 0),
    "matchbox": (0, 128, 255),
    "cigarette_holder": (0, 180, 180),
    "keyring_digital_watch": (180, 0, 255),
    "electric_lighter": (255, 60, 60),
    "wallet": (60, 200, 60),
    "bag": (200, 120, 60),
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
