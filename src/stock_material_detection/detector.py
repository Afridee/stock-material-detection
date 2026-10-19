"""Open-vocabulary detector for the five physical categories.

YOLOE with fixed text prompts is the v1 detector because there are no box
labels to train on. If the YOLOE weights cannot load, YOLO-World runs with
the same prompts. Brand names never go into the prompts: brand comes from
the gallery match on the crop.

Text prompts score these packs low: half the face is a graphic health
warning, and the open slide packs are L-shaped. YOLOE also takes visual
prompts, so ``visual_prompts/<weights>.npz`` (built by
``scripts/build_visual_prompts.py`` from in-scene photos) adds one visual
class per category next to the text classes. Class-agnostic NMS keeps the
best-scoring box for each object whichever prompt produced it.
"""

from __future__ import annotations

# Before ultralytics, which imports cv2 and torch.
from stock_material_detection.cpu_threads import (
    apply_opencv_thread_limit,
    apply_torch_thread_limit,
    configure_cpu_threads,
)

configure_cpu_threads()

import contextlib
import functools
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CATEGORIES = ("cigarette_pack", "cigarette_stick", "lighter", "ashtray", "speaker")

# Text prompt -> category. Each category has its plain name plus a few
# physical descriptions; class-agnostic NMS keeps the best-scoring one per
# object. No brand names.
PROMPTS: dict[str, str] = {
    "cigarette pack": "cigarette_pack",
    "cigarette box": "cigarette_pack",
    "product box": "cigarette_pack",
    "carton": "cigarette_pack",
    "cigarette": "cigarette_stick",
    "cigarette stick": "cigarette_stick",
    "white stick": "cigarette_stick",
    "lighter": "lighter",
    "cigarette lighter": "lighter",
    "pocket lighter": "lighter",
    "ashtray": "ashtray",
    "ash tray": "ashtray",
    "speaker": "speaker",
    "bluetooth speaker": "speaker",
}

DEFAULT_YOLOE_WEIGHTS = "yoloe-11l-seg.pt"
DEFAULT_WORLD_WEIGHTS = "yolov8l-worldv2.pt"
# Zero-shot text-prompt scores run low (a clear pack often scores 0.3-0.9,
# a clear stick 0.1-0.4). Boxes below this are dropped; the pipeline flags
# anything below its review score instead of trusting it.
MIN_SCORE = 0.05
NMS_IOU = 0.5
# Portrait phone photos letterbox to 540×960 here. At 640 a stick is a few
# pixels thick.
IMAGE_SIZE = 960

REPO_ROOT = Path(__file__).resolve().parents[2]
VISUAL_PROMPT_PREFIX = "visual:"


def weights_dir() -> Path:
    return Path(os.getenv("STOCK_WEIGHTS_DIR", REPO_ROOT / "weights")).resolve()


def visual_prompts_dir() -> Path:
    return Path(os.getenv("STOCK_VISUAL_PROMPTS_DIR", REPO_ROOT / "visual_prompts")).resolve()


def visual_prompts_path(yoloe_weights: str) -> Path:
    """Prompt embeddings are tied to the checkpoint that produced them."""
    return visual_prompts_dir() / f"{Path(yoloe_weights).stem}.npz"


def load_visual_prompts(path: Path, *, embed_dim: int | None = None) -> tuple[list[str], np.ndarray]:
    """``(categories, embeddings)`` from an ``.npz``, or ``([], empty)`` when unusable.

    Rows are unit vectors, one per category. Unknown categories and a
    dimension mismatch mean the file was built for another model and it is
    skipped rather than trusted.
    """
    empty: tuple[list[str], np.ndarray] = ([], np.zeros((0, embed_dim or 0), dtype=np.float32))
    if not path.is_file():
        return empty
    try:
        with np.load(path, allow_pickle=False) as data:
            categories = [str(c) for c in data["categories"]]
            vpe = np.asarray(data["vpe"], dtype=np.float32)
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("ignoring visual prompts %s: %s", path, exc)
        return empty
    if vpe.ndim != 2 or vpe.shape[0] != len(categories) or not categories:
        logger.warning("ignoring visual prompts %s: shape %s for %d categories", path, vpe.shape, len(categories))
        return empty
    if embed_dim is not None and vpe.shape[1] != embed_dim:
        logger.warning("ignoring visual prompts %s: dim %d, model has %d", path, vpe.shape[1], embed_dim)
        return empty
    unknown = sorted(set(categories) - set(CATEGORIES))
    if unknown:
        logger.warning("ignoring visual prompts %s: unknown categories %s", path, unknown)
        return empty
    return categories, vpe


@dataclass
class Detection:
    category: str
    score: float
    box: tuple[float, float, float, float]
    """xyxy in the oriented image."""
    polygon: np.ndarray | None = field(default=None, repr=False)
    """Segmentation outline in image pixels (N×2), when the model has masks."""


class Detector:
    """Loads once. ``detect`` takes a BGR array of the oriented photo."""

    def __init__(
        self,
        yoloe_weights: str | None = None,
        world_weights: str | None = None,
        visual_prompts: Path | None | bool = None,
    ) -> None:
        """``visual_prompts``: ``None`` uses ``visual_prompts/<weights>.npz`` when
        present, ``False`` disables visual prompts, a path loads that file."""
        apply_torch_thread_limit()
        apply_opencv_thread_limit()
        self._prompts = list(PROMPTS)
        # Class index -> category, text classes first then visual classes.
        self._categories: list[str] = [PROMPTS[p] for p in self._prompts]
        self.visual_categories: list[str] = []
        yoloe_weights = yoloe_weights or os.getenv("STOCK_YOLOE_WEIGHTS", DEFAULT_YOLOE_WEIGHTS)
        world_weights = world_weights or os.getenv("STOCK_WORLD_WEIGHTS", DEFAULT_WORLD_WEIGHTS)
        if visual_prompts is None:
            visual_prompts = visual_prompts_path(yoloe_weights)
        elif visual_prompts is False:
            visual_prompts = None

        wdir = weights_dir()
        wdir.mkdir(parents=True, exist_ok=True)
        # Ultralytics downloads weights and the prompt text encoder into the
        # working directory, so load from inside the weights folder.
        with contextlib.chdir(wdir):
            try:
                self.model = self._load_yoloe(yoloe_weights, visual_prompts)
                self.backend = f"yoloe:{Path(yoloe_weights).name}"
            except Exception:
                logger.exception("YOLOE weights %s failed to load, using YOLO-World", yoloe_weights)
                self.model = self._load_world(world_weights)
                self.backend = f"yolo-world:{Path(world_weights).name}"
        logger.info(
            "detector ready (%s, visual prompts: %s)",
            self.backend,
            ", ".join(self.visual_categories) or "none",
        )

    def _load_yoloe(self, weights: str, visual_prompts: Path | None):
        import torch
        from ultralytics import YOLOE

        model = YOLOE(weights)
        tpe = model.get_text_pe(self._prompts)
        names = list(self._prompts)
        embeddings = tpe
        categories: list[str] = []
        if visual_prompts is not None:
            categories, vpe = load_visual_prompts(visual_prompts, embed_dim=int(tpe.shape[-1]))
            if categories:
                names += [VISUAL_PROMPT_PREFIX + c for c in categories]
                embeddings = torch.cat([tpe, torch.from_numpy(vpe)[None].to(tpe)], dim=1)
        model.set_classes(names, embeddings)
        self._categories += categories
        self.visual_categories = categories
        return model

    def _load_world(self, weights: str):
        from ultralytics import YOLOWorld

        model = YOLOWorld(weights)
        model.set_classes(self._prompts)
        return model

    def visual_prompt_embedding(self, image_bgr: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
        """Unit prompt embedding for the objects in ``boxes`` (xyxy pixels) of one image.

        YOLOE only. All boxes are treated as one class; average the result
        over reference images and save it with ``load_visual_prompts``'s
        layout to use it as a visual class.
        """
        if not self.backend.startswith("yoloe:"):
            raise RuntimeError("visual prompts need the YOLOE backend")
        if not boxes:
            raise ValueError("at least one box is required")
        import torch
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        predictor = YOLOEVPSegPredictor(
            overrides=dict(
                model=None,
                task="segment",
                mode="predict",
                imgsz=IMAGE_SIZE,
                device="cpu",
                save=False,
                batch=1,
                verbose=False,
            )
        )
        predictor.set_prompts({"bboxes": np.asarray(boxes, dtype=np.float32), "cls": np.zeros(len(boxes), dtype=int)})
        predictor.setup_model(model=self.model.model, verbose=False)
        predictor.model.names = {0: "object0"}
        vpe = predictor.get_vpe(image_bgr)[0, 0]
        return torch.nn.functional.normalize(vpe.float(), dim=-1).cpu().numpy()

    def detect(self, image_bgr: np.ndarray, *, min_score: float = MIN_SCORE) -> list[Detection]:
        # The visual prompt raises the score of objects the text prompts
        # already found; it does not add objects on its own (on its own it
        # also fires on ashtrays and whole frames, and weak text support
        # down at 0.02 still let frame-sized boxes through). With visual
        # classes, NMS is multi-label and per class so one anchor yields
        # both its text row and its visual row; a visual box is kept only
        # when a text box of its category at or above ``min_score`` overlaps
        # it, then class-agnostic NMS picks the best row per object.
        visual = bool(self.visual_categories)
        kwargs = dict(
            source=image_bgr,
            conf=min_score,
            iou=NMS_IOU,
            agnostic_nms=not visual,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False,
        )
        if visual:
            # YOLOE.predict does not pass ``predictor`` on, so use the base method.
            from ultralytics.engine.model import Model

            results = Model.predict(self.model, predictor=_multi_label_predictor(), **kwargs)
        else:
            results = self.model.predict(**kwargs)
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        polygons = result.masks.xy if result.masks is not None else [None] * len(xyxy)

        if visual:
            keep = self._support_visual_boxes(xyxy, scores, classes)
            xyxy, scores, classes = xyxy[keep], scores[keep], classes[keep]
            polygons = [polygons[i] for i in keep]

        detections: list[Detection] = []
        for box, score, cls, poly in zip(xyxy, scores, classes, polygons):
            category = self._categories[cls]
            detections.append(
                Detection(
                    category=category,
                    score=float(score),
                    box=tuple(float(v) for v in box),
                    polygon=None if poly is None or len(poly) < 3 else np.asarray(poly, dtype=np.float32),
                )
            )
        return detections

    def _support_visual_boxes(self, xyxy: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> list[int]:
        """Indices to keep: unsupported visual rows go, then class-agnostic NMS."""
        n_text = len(self._prompts)
        categories = np.array([self._categories[c] for c in classes])
        is_visual = classes >= n_text
        keep = np.ones(len(xyxy), dtype=bool)
        for i in np.flatnonzero(is_visual):
            same = np.flatnonzero(~is_visual & (categories == categories[i]))
            keep[i] = bool(len(same)) and bool((box_iou(xyxy[i], xyxy[same]) >= NMS_IOU).any())
        candidates = np.flatnonzero(keep)
        return [int(candidates[i]) for i in nms(xyxy[candidates], scores[candidates], NMS_IOU)]


@functools.cache
def _multi_label_predictor():
    """Segmentation predictor whose NMS emits one row per class above ``conf``.

    Cached so ``Model.predict`` sees the same class each call and keeps its
    predictor instead of rebuilding it.
    """
    from ultralytics.models.yolo.segment import SegmentationPredictor
    from ultralytics.utils import nms as nms_module

    class MultiLabelSegmentationPredictor(SegmentationPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            original = nms_module.non_max_suppression
            nms_module.non_max_suppression = functools.partial(original, multi_label=True)
            try:
                return super().postprocess(preds, img, orig_imgs, **kwargs)
            finally:
                nms_module.non_max_suppression = original

    return MultiLabelSegmentationPredictor


def box_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box with each row of ``others``."""
    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-6)


def nms(xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy class-agnostic NMS; indices into ``xyxy`` in descending score order."""
    order = list(np.argsort(-scores))
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        if not order:
            break
        rest = np.array(order)
        order = [int(j) for j in rest[box_iou(xyxy[i], xyxy[rest]) < iou_threshold]]
    return keep
