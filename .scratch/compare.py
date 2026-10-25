import sys
from pathlib import Path

from stock_material_detection.detector import Detector
from stock_material_detection.orient import load_oriented, to_bgr

photos = sorted(p for p in Path("totest").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
for npz in sys.argv[1:]:
    det = Detector(visual_prompts=Path(npz).resolve())
    print(f"\n=== {npz} ({', '.join(det.visual_categories)})")
    for p in photos:
        ds = det.detect(to_bgr(load_oriented(p)))
        row = sorted(((d.category.split("_")[-1], round(d.score, 2), tuple(round(v) for v in d.box)) for d in ds), key=lambda r: (r[0], -r[1]))
        print(p.name)
        for r in row:
            print("   ", r)
