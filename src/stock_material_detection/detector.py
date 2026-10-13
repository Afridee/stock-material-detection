"""Open-vocabulary detector for the five physical categories.

YOLOE with fixed text prompts is the v1 detector because there are no box
labels to train on. If the YOLOE weights cannot load, YOLO-World runs with
the same prompts. Brand names never go into the prompts: brand comes from
the gallery match on the crop.
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
MIN_SCORE = 0.10
NMS_IOU = 0.5
# Portrait phone photos letterbox to 540×960 here. At 640 a stick is a few
# pixels thick.
IMAGE_SIZE = 960

REPO_ROOT = Path(__file__).resolve().parents[2]


def weights_dir() -> Path:
    return Path(os.getenv("STOCK_WEIGHTS_DIR", REPO_ROOT / "weights")).resolve()


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
    ) -> None:
        apply_torch_thread_limit()
        apply_opencv_thread_limit()
        self._prompts = list(PROMPTS)
        yoloe_weights = yoloe_weights or os.getenv("STOCK_YOLOE_WEIGHTS", DEFAULT_YOLOE_WEIGHTS)
        world_weights = world_weights or os.getenv("STOCK_WORLD_WEIGHTS", DEFAULT_WORLD_WEIGHTS)

        wdir = weights_dir()
        wdir.mkdir(parents=True, exist_ok=True)
        # Ultralytics downloads weights and the prompt text encoder into the
        # working directory, so load from inside the weights folder.
        with contextlib.chdir(wdir):
            try:
                self.model = self._load_yoloe(yoloe_weights)
                self.backend = f"yoloe:{Path(yoloe_weights).name}"
            except Exception:
                logger.exception("YOLOE weights %s failed to load, using YOLO-World", yoloe_weights)
                self.model = self._load_world(world_weights)
                self.backend = f"yolo-world:{Path(world_weights).name}"
        logger.info("detector ready (%s)", self.backend)

    def _load_yoloe(self, weights: str):
        from ultralytics import YOLOE

        model = YOLOE(weights)
        model.set_classes(self._prompts, model.get_text_pe(self._prompts))
        return model

    def _load_world(self, weights: str):
        from ultralytics import YOLOWorld

        model = YOLOWorld(weights)
        model.set_classes(self._prompts)
        return model

    def detect(self, image_bgr: np.ndarray, *, min_score: float = MIN_SCORE) -> list[Detection]:
        results = self.model.predict(
            source=image_bgr,
            conf=min_score,
            iou=NMS_IOU,
            agnostic_nms=True,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        polygons = result.masks.xy if result.masks is not None else [None] * len(xyxy)

        detections: list[Detection] = []
        for box, score, cls, poly in zip(xyxy, scores, classes, polygons):
            category = PROMPTS[self._prompts[cls]]
            detections.append(
                Detection(
                    category=category,
                    score=float(score),
                    box=tuple(float(v) for v in box),
                    polygon=None if poly is None or len(poly) < 3 else np.asarray(poly, dtype=np.float32),
                )
            )
        return detections
