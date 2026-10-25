"""SKU gallery: SigLIP 2 embeddings of reference crops, ranked by cosine.

Adding an SKU
-------------
Create ``gallery/<sku>/`` and put tight reference crops of that SKU in it
(``.jpg``, ``.jpeg``, ``.png``). The folder name is the ``sku`` in the JSON.
One SKU folder can hold every physical form of the product: pack faces,
stick filter bands, lighter bodies. A crop is only compared against crops,
and counts are grouped by ``(sku, category)``, so a pack and a loose stick of
the same SKU land on separate lines.

``detect --debug-dir DIR`` writes each detection crop and each stick filter
band. Those files are the right shape to copy into ``gallery/<sku>/``.

Optional ``gallery/<sku>/wordmark.txt`` holds the Latin wordmark printed on
the product (for example ``AVON``). Without it, the first word of the folder
name is used. The wordmark is only read when ``STOCK_OCR=1``.

Vectors are cached in ``gallery/.embeddings.npz`` and rebuilt when a file
is added, removed, or changed.
"""

from __future__ import annotations

# Before torch and paddle.
from stock_material_detection.cpu_threads import (
    apply_torch_thread_limit,
    configure_cpu_threads,
    resolve_num_threads,
)

configure_cpu_threads()

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from stock_material_detection.crops import color_histogram, histogram_similarity

logger = logging.getLogger(__name__)

MODEL_ID = "google/siglip2-base-patch16-224"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CACHE_NAME = ".embeddings.npz"
WORDMARK_FILE = "wordmark.txt"
LETTERBOX_FILL = (128, 128, 128)
EMBED_BATCH = 16
QUARTER_TURNS = (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)

REPO_ROOT = Path(__file__).resolve().parents[2]


def default_gallery_dir() -> Path:
    return Path(os.getenv("STOCK_GALLERY_DIR", REPO_ROOT / "gallery")).resolve()


def iter_gallery(gallery_dir: Path) -> list[tuple[str, Path]]:
    """``(sku, path)`` for every reference image, sorted for a stable cache."""
    if not gallery_dir.is_dir():
        return []
    entries: list[tuple[str, Path]] = []
    for sku_dir in sorted(gallery_dir.iterdir()):
        if not sku_dir.is_dir() or sku_dir.name.startswith((".", "_")):
            continue
        for path in sorted(sku_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                entries.append((sku_dir.name, path))
    return entries


def letterbox(image: Image.Image) -> Image.Image:
    """Pad to a square so the model's square resize does not stretch the crop.

    Filter bands and lighters are long and thin.
    """
    im = image.convert("RGB")
    side = max(im.size)
    return ImageOps.pad(im, (side, side), color=LETTERBOX_FILL, centering=(0.5, 0.5))


def bgr_to_pil(crop: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


class Embedder:
    """SigLIP 2 image tower. Loads once; returns L2-normalized float32 rows."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        apply_torch_thread_limit()
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self.model_id = model_id
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval()

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        torch = self._torch
        rows: list[np.ndarray] = []
        for start in range(0, len(images), EMBED_BATCH):
            batch = [letterbox(im) for im in images[start : start + EMBED_BATCH]]
            inputs = self.processor(images=batch, return_tensors="pt")
            with torch.inference_mode():
                out = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            feats = out if isinstance(out, torch.Tensor) else out.pooler_output
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            rows.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(rows) if rows else np.zeros((0, 0), dtype=np.float32)


@dataclass
class Match:
    sku: str
    similarity: float
    runner_up: float | None
    """Best similarity of any other SKU, or ``None`` when the gallery has one SKU."""
    color_similarity: float | None = None
    runner_up_sku: str | None = None

    @property
    def margin(self) -> float | None:
        return None if self.runner_up is None else self.similarity - self.runner_up


class Gallery:
    """Reference vectors per SKU. The embedder is only loaded if there are references."""

    def __init__(self, gallery_dir: Path | None = None, embedder: Embedder | None = None) -> None:
        self.dir = (gallery_dir or default_gallery_dir()).resolve()
        self.entries = iter_gallery(self.dir)
        self._embedder = embedder
        self.skus: np.ndarray = np.array([], dtype=object)
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.hists = np.zeros((0, 0), dtype=np.float32)
        if self.entries:
            self._load_or_build()
        logger.info(
            "gallery %s: %d SKUs, %d reference images",
            self.dir,
            len(set(self.skus.tolist())),
            len(self.entries),
        )

    @property
    def empty(self) -> bool:
        return len(self.entries) == 0

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def _manifest(self) -> np.ndarray:
        rows = []
        for sku, path in self.entries:
            stat = path.stat()
            rows.append(f"{sku}/{path.name}|{stat.st_size}|{stat.st_mtime_ns}")
        return np.array(rows)

    def _load_or_build(self) -> None:
        cache = self.dir / CACHE_NAME
        manifest = self._manifest()
        if cache.is_file():
            try:
                with np.load(cache, allow_pickle=False) as data:
                    if str(data["model_id"]) == MODEL_ID and np.array_equal(data["manifest"], manifest):
                        self.skus = data["skus"].astype(object)
                        self.vectors = data["vectors"]
                        self.hists = data["hists"]
                        return
            except (OSError, KeyError, ValueError):
                logger.warning("ignoring unreadable gallery cache %s", cache)

        logger.info("embedding %d gallery images", len(self.entries))
        images: list[Image.Image] = []
        hists: list[np.ndarray] = []
        for _, path in self.entries:
            with Image.open(path) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB")
            images.append(rgb)
            hists.append(color_histogram(cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)))
        self.skus = np.array([sku for sku, _ in self.entries], dtype=object)
        self.vectors = self.embedder.embed(images)
        self.hists = np.stack(hists)
        np.savez(
            cache,
            model_id=np.array(MODEL_ID),
            manifest=manifest,
            skus=self.skus.astype(str),
            vectors=self.vectors,
            hists=self.hists,
        )

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """One row per quarter turn of ``crop``.

        Items lie in any direction but references are upright, and SigLIP is
        not rotation invariant.
        """
        views = [crop if turn is None else cv2.rotate(crop, turn) for turn in QUARTER_TURNS]
        return self.embedder.embed([bgr_to_pil(view) for view in views])

    def match(self, vectors: np.ndarray, hist: np.ndarray | None = None) -> Match | None:
        """Best SKU by max cosine over its references and the query's rows.

        ``None`` if the gallery is empty.
        """
        if self.empty:
            return None
        sims = (self.vectors @ np.atleast_2d(vectors).T).max(axis=1)
        best: dict[str, int] = {}
        for idx, sku in enumerate(self.skus):
            if sku not in best or sims[idx] > sims[best[sku]]:
                best[sku] = idx
        ranked = sorted(best.items(), key=lambda item: -sims[item[1]])
        top_sku, top_idx = ranked[0]
        runner_up = float(sims[ranked[1][1]]) if len(ranked) > 1 else None
        runner_up_sku = str(ranked[1][0]) if len(ranked) > 1 else None
        color = None
        if hist is not None:
            refs = self.hists[self.skus == top_sku]
            color = max(histogram_similarity(hist, ref) for ref in refs)
        return Match(
            sku=str(top_sku),
            similarity=float(sims[top_idx]),
            runner_up=runner_up,
            color_similarity=color,
            runner_up_sku=runner_up_sku,
        )

    def wordmark(self, sku: str) -> str:
        path = self.dir / sku / WORDMARK_FILE
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        return re.split(r"[_\-\s]+", sku)[0]


def ocr_enabled() -> bool:
    return os.getenv("STOCK_OCR", "0") == "1"


def _latin(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


class WordmarkReader:
    """PaddleOCR on pack and lighter crops. Confirms a gallery hit; never picks one."""

    MIN_LETTERS = 3

    def __init__(self) -> None:
        from paddleocr import PaddleOCR

        threads = resolve_num_threads()
        try:
            self._ocr = PaddleOCR(
                lang="en",
                device="cpu",
                cpu_threads=threads,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            )
            self._v3 = True
        except (TypeError, ValueError):
            self._ocr = PaddleOCR(lang="en", use_gpu=False, cpu_threads=threads, use_angle_cls=True)
            self._v3 = False

    def read(self, crop: np.ndarray) -> str:
        """All Latin text on the crop, upper-cased, with spaces and symbols dropped."""
        texts: list[str] = []
        if self._v3:
            for result in self._ocr.predict(crop):
                texts.extend(result["rec_texts"])
        else:
            for page in self._ocr.ocr(crop, cls=True) or []:
                for line in page or []:
                    texts.append(line[1][0])
        return _latin(" ".join(texts))

    def check(self, crop: np.ndarray, wordmark: str) -> str | None:
        """``confirmed`` / ``mismatch``, or ``None`` when no Latin text is readable.

        Packs lie in any direction, so the crop is tried in all four
        quarter turns until the wordmark is found.
        """
        target = _latin(wordmark)
        if not target:
            return None
        read_any = False
        views = (
            crop,
            cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.rotate(crop, cv2.ROTATE_180),
        )
        for view in views:
            text = self.read(view)
            if target in text:
                return "confirmed"
            read_any = read_any or len(re.sub(r"[0-9]", "", text)) >= self.MIN_LETTERS
        return "mismatch" if read_any else None
