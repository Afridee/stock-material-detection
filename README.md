# Stock Material Detection

Counts stock materials in one phone photo of items on a surface. Output is a count per SKU, plus a flag when a human should review that line.

1. Pillow EXIF rotate, so boxes are in the upright frame
2. Ultralytics YOLOE (open-vocabulary) boxes five physical categories: `cigarette_pack`, `cigarette_stick`, `lighter`, `ashtray`, `speaker`. Text prompts plus, for packs, a visual prompt built from in-scene photos (`visual_prompts/`)
3. Each box is cropped. Sticks are cut down to their printed filter band. Lighters also get a color histogram
4. SigLIP 2 (`google/siglip2-base-patch16-224`) embeds the crop and ranks it against `gallery/<sku>/` reference crops
5. Optional PaddleOCR confirms the printed wordmark on pack and lighter crops (`STOCK_OCR=1`)
6. Counts are grouped by `(sku, category)`

Brand, variant, filter pattern, and lighter print are not detector classes. A new SKU is a folder of reference crops, not a new detector class.

The CLI is for local eval. `POST /detect` is the same pipeline over HTTP.

## Setup

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/). CPU only.

```bash
cd /Volumes/T7-Dev/stock-material-detection
uv sync                # add --extra ocr for PaddleOCR
```

First run downloads YOLOE (`weights/yoloe-11l-seg.pt` and its text encoder, about 0.65GB) and, once the gallery has images, SigLIP 2 (about 0.4GB, Hugging Face cache). With OCR on, PaddleOCR downloads its models to `~/.paddlex`.

## Commands

```bash
uv run python -m stock_material_detection detect --image "SAMPLE IMAGES/BR00108-01.jpg" --pretty

uv run python -m stock_material_detection detect --image photo.jpg --debug-dir debug/photo
```

| Flag | Description |
|------|-------------|
| `--image PATH` | Required input photo |
| `--pretty` | Pretty-print JSON |
| `--min-score F` | Drop detector boxes below this. Default `0.05` |
| `--min-similarity F` | Gallery cosine below this gives `sku: unknown`. Default `0.70` |
| `--gallery DIR` | Gallery folder. Default `gallery/` or `STOCK_GALLERY_DIR` |
| `--debug-dir DIR` | Write `oriented.jpg` (the pixels the detector saw), `boxes.jpg` (overlay), each crop, and each stick filter band |

Exit code is `0` on success and `1` when the image is unreadable.

### JSON

```json
{
  "image": "BR00108-01.jpg",
  "counts": [
    {"sku": "unknown", "category": "cigarette_pack", "count": 2, "flagged": true}
  ],
  "detections": [
    {"category": "cigarette_pack", "sku": "unknown", "score": 0.9056, "similarity": null,
     "box": [194.6, 727.7, 570.8, 1013.9], "flagged": true}
  ]
}
```

- `box` is xyxy in the oriented image.
- `score` is the detector score. `similarity` is the best gallery cosine, `null` when the gallery is empty.
- Stick detections also carry `insidePack`. Lighters carry `colorSimilarity` once matched. With OCR on, packs and lighters carry `wordmark`: `confirmed`, `mismatch`, or absent when no Latin text was readable.
- Close calls carry `rivalSku` and `rivalSimilarity`: the runner-up SKU when it is within the margin of the match, or the SKU named by the other end of a stick.
- `failReasons: ["unreadable_image"]` appears only when the file is not a readable image.

### Client PDF report

Runs `detect` on photo folders, writes HTML in a temp directory, converts it to PDF with Chrome, and keeps only the PDF. Each photo gets an annotated box overlay, the count lines per SKU, every detection with its crop, and a plain-language reason for each review flag. The front pages hold totals per SKU across all photos, the most common review reasons, and the thresholds used.

```bash
uv run python scripts/generate_client_report.py
```

Defaults: `SAMPLE IMAGES/` and `totest/` (whichever exist), output `reports/stock_material_detection_report.pdf`.

```bash
uv run python scripts/generate_client_report.py \
  --dir "/path/to/photos" \
  --min-score 0.05 --min-similarity 0.70 \
  --output reports/stock_material_detection_report.pdf
```

`--dir` can be repeated. `--min-score`, `--min-similarity`, and `--gallery` mean the same as on `detect`. Requires Google Chrome for the HTML → PDF step.

### Visual prompts

Text prompts alone score these packs low (0.1–0.7 on a clear pack): half the face is a graphic health warning and the open slide packs are L-shaped. YOLOE also accepts visual prompts, so `visual_prompts/<weights>.npz` adds one visual class per category next to the text classes. The visual class only re-scores objects the text prompts already found: a visual box is kept when a text box of the same category (at or above `--min-score`) overlaps it by IoU ≥ 0.5, otherwise it is dropped, and class-agnostic NMS then keeps the best box per object. On its own the pack prompt also fires on ashtrays and whole frames; with the gate, the set of boxes is the same as text-only and only the scores change. With the shipped file, whole packs score 0.36–0.98 instead of 0.06–0.68 on the same photos, and loose sticks on `totest/` score 0.50–0.93 instead of 0.12–0.55. Sticks still inside a pack keep their low scores.

The pack prompt is built without labels: the text-prompt detector runs on a sample of in-scene photos and only its confident whole-object boxes (pack score ≥ 0.5) become references. The embedding is the mean over reference photos. Tight crops like the gallery do not work as references.

The stick prompt cannot be bootstrapped that way: of the 30 stick boxes at score ≥ 0.4 in `reference-images/`, 28 were litter, pack edges, or fingers, and the resulting prompt changed nothing useful. The shipped stick row is the mean over 14 hand-picked loose-stick boxes from 11 photos; the files and boxes are in the `.npz` `meta` under `references`. Rebuilding other categories keeps rows the script did not rebuild, so the stick row survives a default run.

```bash
uv run python scripts/build_visual_prompts.py                       # cigarette_pack from reference-images/, 40 photos
uv run python scripts/build_visual_prompts.py --photos /path/to/photos --category cigarette_pack --category lighter --images 60
```

Rebuild after changing `STOCK_YOLOE_WEIGHTS`; the file is tied to the checkpoint and is ignored (with a warning) when its dimension does not match. `STOCK_VISUAL_PROMPTS_DIR` moves the folder; `Detector(visual_prompts=False)` runs text prompts only.

## Counting rules

- A pack counts as 1, closed or open.
- A loose stick counts as 1. A stick whose box is at least 60% inside a pack box is still in the pack and is not counted.
- Each lighter, ashtray, and speaker counts as 1.
- A box with at least 80% of its area inside a higher-scoring box of the same category (half a pack, a slid-out inner sleeve) is a fragment and is dropped before counting.
- No gallery hit gives `sku: unknown` and `flagged: true`. The pipeline never guesses a brand.
- A line is flagged when any of its detections is flagged.

A detection is flagged when any of these hold. The values are conservative starting points, not tuned against labels (there are none yet). Revisit them once a gallery and a labeled audit set exist.

| Constant | Default | Flags when |
|----------|---------|------------|
| `REVIEW_SCORE` | `0.30` | Detector score is below it and the gallery cosine is below `STRONG_SIMILARITY` (`0.85`) |
| `MIN_SIMILARITY` | `0.70` | Best gallery cosine is below it (also sets `unknown`) |
| `MIN_MARGIN` | `0.02` | Top SKU beats the runner-up SKU by less than this |
| `MIN_COLOR_SIMILARITY` | `0.40` | Lighter histogram overlap with the matched SKU is below it |
| wordmark | | OCR read Latin text that does not contain the SKU's wordmark |

## Gallery

`gallery/` starts empty, so every `sku` is `unknown` until references are added. That is the correct result.

To add an SKU, create `gallery/<sku>/` and put tight reference crops in it (`.jpg`, `.jpeg`, `.png`). The folder name becomes the `sku`. One folder can hold every form of the product: pack faces, stick filter bands, and lighter bodies. Crops from `--debug-dir` are the right shape to copy in. For OCR, add `gallery/<sku>/wordmark.txt` with the printed Latin wordmark (for example `AVON`). Without it, the first word of the folder name is used.

Vectors are cached in `gallery/.embeddings.npz` and rebuilt when a reference file changes.

`reference-images/` holds 1036 unlabeled close-ups. It is not the gallery: do not copy it into `gallery/` or sort it automatically.

## HTTP API

```bash
uv run python -m uvicorn stock_material_detection.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -s -X POST http://localhost:8000/detect \
  -H "X-API-Key: $SERVICE_API_KEY" \
  -F "image=@SAMPLE IMAGES/BR00114-02.jpg"
```

| Call | Behavior |
|------|----------|
| `POST /detect` | Multipart `image`, JPEG or PNG, max 25MB. Same JSON as the CLI |
| `GET /health` | Liveness, no auth |
| `X-API-Key` | Required only when `SERVICE_API_KEY` is set |

Models load once at startup (`STOCK_WARMUP=0` defers that to the first request). One lock wraps the whole pipeline, so one photo is in the model at a time. Run one uvicorn process with no extra workers.

## CPU threads

Inference defaults to 1 CPU thread. Set `STOCK_NUM_THREADS` to a positive integer to override it. Before OpenCV, PyTorch, or Paddle loads, any unset `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` is set to that count; a value already in the environment is left as-is. PyTorch intra-op and inter-op threads, OpenCV, and PaddleOCR's `cpu_threads` use the same count.

On an Apple Silicon laptop at 1 thread, the detector takes about 0.45 s per photo and a cold CLI call about 4 to 5 s including model load.

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `STOCK_NUM_THREADS` | `1` | CPU threads for inference |
| `STOCK_OCR` | `0` | `1` enables the PaddleOCR wordmark check |
| `STOCK_GALLERY_DIR` | `gallery/` | Gallery folder |
| `STOCK_WEIGHTS_DIR` | `weights/` | Detector weights and text encoder |
| `STOCK_YOLOE_WEIGHTS` | `yoloe-11l-seg.pt` | YOLOE checkpoint |
| `STOCK_VISUAL_PROMPTS_DIR` | `visual_prompts/` | Folder with `<weights>.npz` visual prompt embeddings |
| `STOCK_WORLD_WEIGHTS` | `yolov8l-worldv2.pt` | YOLO-World fallback if YOLOE cannot load |
| `SERVICE_API_KEY` | unset | Enables the `X-API-Key` check |
| `STOCK_WARMUP` | `1` | `0` loads models on the first request |

## Tests

No model download. Covers thread defaults, EXIF orientation, the stick filter band on a synthetic rotated stick, the counting rules, and the API with the pipeline stubbed.

```bash
uv run pytest
```

## Smoke test and known limits

`smoke-results/` has the `detect` JSON for the four photos in `SAMPLE IMAGES/`, run with an empty gallery.

| Photo | In the photo | Detected |
|-------|--------------|----------|
| `BR00038-01.jpg` | 1 open pack, 5 loose sticks | 1 pack, 0 sticks |
| `BR00038-02.jpg` | 1 open pack, 5 loose sticks | 1 pack, 1 stick |
| `BR00108-01.jpg` | 2 packs, 4 lighters | 7 "packs" (2 real packs, lighter bodies and heads), 2 "sticks" (paint marks) |
| `BR00114-02.jpg` | 1 ashtray, 2 packs (1 open), 1 speaker box | 1 ashtray, 2 packs, no speaker |

Packs are reliable (scores 0.3 to 0.9). Sticks inside open packs are correctly not counted. The weak part is the zero-shot detector on the other categories: no prompt wording made YOLOE score these sleeved lighters as "lighter", loose sticks mostly score under 0.05, and the speaker box reads as a plain box. Wrong boxes are flagged (low score, `unknown`), not silently trusted, but counts for sticks and lighters are not usable yet. Fixing that needs either box labels for a fine-tuned detector or a second category check on each crop; both are outside this v1.
