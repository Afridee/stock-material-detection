#!/usr/bin/env python3
"""Build YOLOE visual prompt embeddings from in-scene photos.

There are no box labels, so references are bootstrapped: the text-prompt
detector runs on a sample of photos and only its confident whole-object
boxes become visual prompts. One embedding per category is the mean over
reference images, saved to ``visual_prompts/<weights>.npz`` where
``Detector`` picks it up.

Tight crops (the gallery) do not work as references: the prompt box is
then the whole letterboxed frame and the embedding matches whole images.
Use photos with the objects on a surface, like ``reference-images/``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np

from stock_material_detection.detector import (
    CATEGORIES,
    DEFAULT_YOLOE_WEIGHTS,
    Detector,
    load_visual_prompts,
    visual_prompts_path,
)
from stock_material_detection.gallery import IMAGE_SUFFIXES
from stock_material_detection.orient import load_oriented, to_bgr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHOTOS = ROOT / "reference-images"
# Only clear whole objects should become references. Zero-shot pack scores
# above 0.5 were whole packs on every photo checked; sticks run lower.
DEFAULT_MIN_SCORE = {"cigarette_pack": 0.5, "cigarette_stick": 0.4, "lighter": 0.4, "ashtray": 0.4, "speaker": 0.4}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLOE visual prompt embeddings from in-scene photos.")
    parser.add_argument("--photos", type=Path, default=DEFAULT_PHOTOS, help=f"Photo folder (default: {DEFAULT_PHOTOS})")
    parser.add_argument(
        "--category",
        dest="categories",
        action="append",
        choices=CATEGORIES,
        help="Category to build (repeatable; default: cigarette_pack)",
    )
    parser.add_argument("--images", type=int, default=40, help="Reference photos to use per category (default: 40)")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for picking photos (default: 0)")
    parser.add_argument("--output", type=Path, default=None, help="Output .npz (default: visual_prompts/<weights>.npz)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    categories = args.categories or ["cigarette_pack"]
    if not args.photos.is_dir():
        raise SystemExit(f"Missing photo folder: {args.photos}")
    files = sorted(p for p in args.photos.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"No images under {args.photos}")
    random.Random(args.seed).shuffle(files)

    # Text prompts only: the current visual prompts must not pick their own references.
    detector = Detector(visual_prompts=False)
    if not detector.backend.startswith("yoloe:"):
        raise SystemExit(f"Visual prompts need YOLOE, got {detector.backend}")
    weights = detector.backend.split(":", 1)[1]
    output = args.output or visual_prompts_path(weights)

    vectors: dict[str, list[np.ndarray]] = {c: [] for c in categories}
    refs: dict[str, list[dict]] = {c: [] for c in categories}
    for path in files:
        if all(len(vectors[c]) >= args.images for c in categories):
            break
        try:
            image = to_bgr(load_oriented(path))
        except OSError:
            continue
        detections = detector.detect(image)
        for category in categories:
            if len(vectors[category]) >= args.images:
                continue
            boxes = [d.box for d in detections if d.category == category and d.score >= DEFAULT_MIN_SCORE[category]]
            if not boxes:
                continue
            vectors[category].append(detector.visual_prompt_embedding(image, boxes))
            refs[category].append({"file": path.name, "boxes": [[round(v) for v in b] for b in boxes]})
            print(f"[{category} {len(vectors[category])}/{args.images}] {path.name}: {len(boxes)} box(es)", flush=True)

    kept = [c for c in categories if vectors[c]]
    if not kept:
        raise SystemExit("No confident boxes found; nothing to save.")
    for category in categories:
        if category not in kept:
            print(f"Warning: no confident {category} boxes, category skipped", file=sys.stderr)
    rows = {c: np.mean(vectors[c], axis=0) for c in kept}
    min_score: dict = {c: DEFAULT_MIN_SCORE[c] for c in kept}
    references: dict = {c: refs[c] for c in kept}
    # Categories not rebuilt keep their rows (the stick prompt is hand-picked
    # and cannot be bootstrapped: confident stick boxes are mostly litter).
    old_categories, old_vpe = load_visual_prompts(output, embed_dim=len(rows[kept[0]]))
    if old_categories:
        with np.load(output, allow_pickle=False) as data:
            old_meta = json.loads(str(data["meta"])) if "meta" in data else {}
        for c, row in zip(old_categories, old_vpe):
            if c not in rows:
                rows[c] = row
                min_score[c] = old_meta.get("min_score", {}).get(c)
                references[c] = old_meta.get("references", {}).get(c)
                print(f"Keeping existing {c} prompt from {output}")
    order = [c for c in CATEGORIES if c in rows]
    vpe = np.stack([rows[c] for c in order]).astype(np.float32)
    vpe /= np.linalg.norm(vpe, axis=1, keepdims=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        categories=np.array(order),
        vpe=vpe,
        meta=json.dumps(
            {
                "weights": weights,
                "built": date.today().isoformat(),
                "photos": str(args.photos),
                "seed": args.seed,
                "min_score": min_score,
                "references": references,
            }
        ),
    )
    print(f"Wrote {output}: " + ", ".join(f"{c} ({len(vectors[c])} photos)" for c in kept))
    return 0


if __name__ == "__main__":
    sys.exit(main())
