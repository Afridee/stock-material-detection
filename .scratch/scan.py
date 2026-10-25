import json
from pathlib import Path

from stock_material_detection.detector import Detector
from stock_material_detection.gallery import IMAGE_SUFFIXES
from stock_material_detection.orient import load_oriented, to_bgr

det = Detector(visual_prompts=False)
files = sorted(p for p in Path("reference-images").rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
out = []
for i, p in enumerate(files):
    try:
        img = to_bgr(load_oriented(p))
    except OSError:
        continue
    ds = det.detect(img, min_score=0.1)
    out.append({
        "file": p.name,
        "shape": img.shape[:2],
        "sticks": [[round(d.score, 3), [round(v) for v in d.box], None if d.polygon is None else d.polygon.round().astype(int).tolist()] for d in ds if d.category == "cigarette_stick"],
        "packs": [[round(d.score, 3), [round(v) for v in d.box]] for d in ds if d.category == "cigarette_pack" and d.score >= 0.3],
    })
    if i % 100 == 0:
        print(i, flush=True)
Path(".scratch/scan.json").write_text(json.dumps(out))
