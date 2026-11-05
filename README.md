# Stock Material Detection

Counts stock materials in one phone photo of items on a surface. Output is a count per SKU, plus a flag when a human should review that line.

1. Pillow EXIF rotate, so boxes are in the upright frame
2. Ultralytics YOLOE (open-vocabulary) boxes the physical categories: `cigarette_pack`, `cigarette_stick`, `lighter`, `ashtray`, `speaker`, `cigarette_holder`, `keyring_digital_watch`, `bag`, plus gallery-confirmed fallback prompts for `matchbox`, `electric_lighter`, and `wallet`. Text prompts plus visual prompts for packs and sticks, built from in-scene photos (`visual_prompts/`)
3. Each box is cropped. Sticks are cut down to their printed filter band. Lighters also get a color histogram
4. SigLIP 2 (`google/siglip2-base-patch16-224`) embeds the crop and ranks it against `gallery/<sku>/` reference crops
5. Optional PaddleOCR confirms the printed wordmark on pack, lighter, and earphones crops (`STOCK_OCR=1`)
6. Counts are grouped by `(sku, category)`

Brand, variant, filter pattern, and lighter print are not detector classes. A new SKU is a folder of reference crops, not a new detector class.

A brand sold in several pack sizes with the same artwork (Black Diamond 20s and 10s) stays one gallery folder. `gallery/<sku>/pack_sizes.json` lists each size's front face aspect ratio, long side over short side, for example `{"20": 1.58, "10": 2.03}`. The pack's segmentation outline is fitted with a rotated rectangle, and the nearest size within 7% names the SKU (`black_diamond_20`). Otherwise the pack keeps the brand SKU and is flagged. An opened slide pack with its sleeve pulled out, or a pack seen edge-on, is flagged this way.

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
- Stick detections also carry `insidePack`. Lighters carry `colorSimilarity` once matched. With OCR on, packs, lighters, and earphones carry `wordmark`: `confirmed`, `mismatch`, or absent when no Latin text was readable.
- Packs of a SKU with `pack_sizes.json` carry `faceAspect`, plus `packSize` when it matched a listed size.
- Close calls carry `rivalSku` and `rivalSimilarity`: the runner-up SKU when it is within the margin of the match, or the SKU named by the other end of a stick.
- A gallery hit also carries `matchRef` (`sku/file.jpg`): the reference image with the highest cosine. Close calls add `rivalRef` for the other image.
- `failReasons: ["unreadable_image"]` appears only when the file is not a readable image.

### Client PDF report

Runs `detect` on photo folders, writes HTML in a temp directory, converts it to PDF with Chrome, and keeps only the PDF. Each photo gets an annotated box overlay, the count lines per SKU, every detection with its crop and the closest gallery image (both images on a close call), and a plain-language reason for each review flag. The front pages hold totals per SKU across all photos, the most common review reasons, and the thresholds used.

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

### Capacity PDF

Measures what one `POST /detect` costs on this machine: model load, steady-state latency, CPU time, and resident memory. Same `run_detect` path as the API. Writes HTML, prints it to PDF with Chrome, and keeps both.

```bash
uv run python scripts/benchmark_capacity.py
```

Defaults: `SAMPLE IMAGES/` and `totest/` (whichever exist), output `reports/processing_capacity.pdf`. The HTML, a Markdown copy, and `reports/processing_capacity.json` are written beside the PDF.

```bash
uv run python scripts/benchmark_capacity.py --from-json reports/processing_capacity.json
uv run python scripts/benchmark_capacity.py \
  --dir "/path/to/photos" --limit 8 --skip-ocr
```

`--from-json` reprints the PDF from a previous measurement without loading the models. `--dir` can be repeated. `--skip-ocr` leaves out the `STOCK_OCR=1` run. `--skip-sweep` skips the extra `STOCK_NUM_THREADS` comparison. Requires Google Chrome. Run it on the machine you will deploy to; a PDF from a laptop is not the production budget.

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
- Each lighter, ashtray, speaker, earphones box, matchbox, cigarette holder, keyring with digital watch, electric lighter, wallet, and bag counts as 1.
- A box with at least 80% of its area inside a higher-scoring box of the same category (half a pack, a slid-out inner sleeve) is a fragment and is dropped before counting.
- An unnamed box with at least 30% of its area inside a named box, whose closest gallery image is from that box's SKU, is a split-off piece of the same object (the end of a USB lighter boxed separately as "lighter") and is not counted.
- No gallery hit gives `sku: unknown` and `flagged: true`. The pipeline never guesses a brand.
- A line is flagged when any of its detections is flagged.

A detection is flagged when any of these hold. The values are conservative starting points, not tuned against a labeled audit set (there is none yet).

| Constant | Default | Flags when |
|----------|---------|------------|
| `REVIEW_SCORE` | `0.30` | Detector score is below it and the gallery cosine is below `STRONG_SIMILARITY` (`0.85`) |
| `MIN_SIMILARITY` | `0.70` | Best gallery cosine is below it (also sets `unknown`) |
| `MIN_MARGIN` | `0.02` | Top SKU beats the runner-up SKU by less than this |
| `MIN_COLOR_SIMILARITY` | `0.40` | Lighter histogram overlap with the matched SKU is below it |
| wordmark | | OCR read Latin text that does not contain the SKU's wordmark |

## Gallery

`gallery/` holds the reference crops. Cigarette brands in it are Aris Apple, Aris Strawberry, Avon Blue, Black Diamond, Esse Change Mango, Esse Prime, Maxim Prime, Maxim X Bolt, and Maxim Double Burst. The other folders are the products below. With no reference images, every `sku` is `unknown`. That is the correct result.

To add an SKU, create `gallery/<sku>/` and put tight reference crops in it (`.jpg`, `.jpeg`, `.png`). The folder name becomes the `sku`. One folder can hold every form of the product: pack faces, stick filter bands, and lighter bodies. Crops from `--debug-dir` are the right shape to copy in. For OCR on a pack, lighter, or earphones SKU, add `gallery/<sku>/wordmark.txt` with the printed Latin wordmark (for example `AVON`). Without it, the first word of the folder name is used. The check runs after gallery relabeling, so a `wordmark.txt` on an ashtray, bag, holder, electric lighter, speaker, or wallet is not read.

A product with no detector category of its own, such as boxed earphones, is still boxed (usually as `cigarette_pack`, via the `product box` prompt). Add `gallery/<sku>/category.txt` naming its category (for example `earphones`) and a matched box is re-labelled to it. Unmatched boxes keep the detector's category. Re-labelling overrides the detector, so a weak box (score under `REVIEW_SCORE` and cosine under `STRONG_SIMILARITY`) that matches such a SKU stays `unknown` and flagged, and a stick box that matches one (a USB cable, a case edge) always does. `gallery/cyxg_x87/` is the example. Text prompts like "earbuds" were tried and fire on the earbuds pictured on the box, not the box itself, so the gallery decides instead.

Some products are not boxed by any of the main prompts at all. `FALLBACK_PROMPTS` in `detector.py` holds prompts for those (`match tube` → `matchbox`). A fallback box is kept only where no main-prompt box overlaps it (IoU ≥ 0.5) or holds 80% of it, and then only when the gallery matches it to a SKU whose `category.txt` names the same category. Otherwise it is dropped, not flagged. Every tube-like prompt tried (`matchbox`, `cylindrical tube`, `match tube`) outscores the real prompt on loose sticks, round ashtrays, or packs, so these cannot compete in NMS; unconfirmed `match tube` boxes were clumps of loose sticks. `gallery/branded_premium_matchbox/` is the example: the tube scores 0.13 on `match tube` when nothing else boxes it, and when the pack prompts do box it the gallery re-labels it the same way as earphones.

The cigarette holder is a main prompt. The words "cigarette holder" stay under 0.05 on the Avon tube and score about 0.10 on loose sticks. `perfume bottle` boxes the tube (about 0.22 with the wordmark facing the camera, about 0.52 on the plain side) and stayed under 0.05 on the sample packs, sticks, and matchbox. It cannot be a fallback prompt: the plain-side photo is also a speaker box at about 0.08, and a fallback row is dropped wherever a main box overlaps. `gallery/cigarette_holder/` holds both angles, and its `category.txt` is `cigarette_holder`, so a speaker or pack box that wins the same object is still re-labelled on a gallery hit. A real perfume bottle in a later photo will be labelled `cigarette_holder` too.

The keyring with digital watch (a carabiner clip with a round LED watch face) is a main prompt too. "keyring", "keychain", "carabiner", and "digital watch" stay under 0.01 on it. `stopwatch` boxes it at about 0.74 with the display dark and about 0.87 with it lit. It also fires at about 0.07 on the LED battery display printed on the earphones box, so `PROMPT_MIN_SCORES` in `detector.py` drops `stopwatch` boxes under 0.30. `gallery/keyring_digital_watch/` holds both display states and a bagged back, with `category.txt` set to `keyring_digital_watch`. A real stopwatch in a later photo will be labelled a keyring watch too.

The paper ashtray (a printed card wall around a blue sand base) needs no prompt of its own. From the side, `ashtray` boxes it at about 0.60. Seen top-down, `stopwatch` wins it at about 0.49 (a keyring watch). `gallery/paper_ashtray/` holds both views, with `category.txt` set to `ashtray`, so the top-down box is re-labelled on a gallery hit.

The mini speaker is a white carton with a red "music" wordmark. On `SAMPLE IMAGES/BR00114-02.jpg` the pack prompts box it (`cigarette_pack`, score 0.09). `gallery/mini_sound_box/` holds four faces (wordmark, feature icons, ports, and the grille), with `category.txt` set to `speaker`, so that box is relabelled. It matches `face_wordmark.jpg` at 0.86 and is not flagged. Before the folder existed, two of the faces matched `cyxg_x87`.

The electric lighter (a flat blue rechargeable USB lighter with the AVON wordmark) scores under 0.05 on every lighter wording tried: "electric lighter", "usb lighter", "rechargeable lighter", "arc lighter". Unwrapped, the pack prompts box it at 0.55 to 0.90. In its plastic bag nothing main boxes it. `blue box` (0.60 to 0.78 unwrapped) and `blue plastic case` (about 0.56 bagged) are fallback prompts for `electric_lighter`. They are not main prompts because they also fire at 0.56 on a pack and 0.80 on the matchbox tube. Fallback rows are suppressed per category, so a blue-case box does not hide the `match tube` box on the same tube. `gallery/electric_lighter/` holds both faces, bare and bagged, with `category.txt` set to `electric_lighter`. Real crops match at 0.91 to 0.98. Pack edges, frame edges, and the ashtray base reach 0.74 to 0.80 on the bagged references, which is why weak boxes need a strong match before they are re-labelled.

The branded wallet comes in a flat card box with a graffiti "X FACTOR UNLOCKED" print. Every main prompt, "wallet", "purse", "billfold", and "card holder" score 0.00 on it. `sign` boxes it at about 0.65 and stayed under 0.05 on the other sample photos, so it is a fallback prompt for `wallet`. `gallery/branded_wallet/` holds the box front, with `category.txt` set to `wallet`. It matches at 0.99 on its own photo. That is the only photo so far, so the back of the box and other angles are untested.

The same wallet out of its box (black leather bifold, the brand logo embossed on one face) also scores 0.00 on every wallet wording and on `sign`. `leather passport cover` boxes it at 0.56 to 0.61 and stayed under 0.14 on the other sample photos, so it is a second `wallet` fallback prompt. `passport cover` and `black passport cover` were rejected because they fire at 0.56 to 0.72 on the speaker and earphones boxes. With the logo side up, a pack prompt boxes it first (0.81) and the gallery re-labels that box. `gallery/branded_wallet_unboxed/` holds the plain face and the logo face, with `category.txt` set to `wallet`. Both match at 0.99 or higher on their own photos. Before the folder existed, both photos matched the bagged electric lighter at 0.71 to 0.79.

The branded waist bag (dark denim, "X FORCE" print) scores nothing on `bag`, `waist bag`, `fanny pack`, `belt bag`, `bum bag`, `pouch`, or `handbag`. `denim bag` boxes it at about 0.87 unwrapped and 0.14 to 0.41 in its plastic bag, and `wrapped denim bag` at 0.19 to 0.52 in the bag. Both are main prompts for `bag`, because in the second bagged photo `perfume bottle` boxes the same object at 0.08, and a fallback row would have been dropped there. On the other sample photos they stay under 0.30 and lose NMS to the pack boxes. `PROMPT_MIN_SCORES` drops bag boxes under 0.15, which removes a 0.05 box on a leg at the frame edge. `gallery/force_waist_bag/` holds the unwrapped front and two bagged fronts, with `category.txt` set to `bag`. Before the second bagged crop was added, that photo matched the other two at 0.98. Only the front has been photographed so far.

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

Models load once at startup (`STOCK_WARMUP=0` defers that to the first request). One lock wraps the whole pipeline, so one photo is in the model at a time. Run one uvicorn process with no extra workers. RAM, CPU, and how many of those processes to run are measured by `scripts/benchmark_capacity.py`.

## CPU threads

Inference defaults to 1 CPU thread. Set `STOCK_NUM_THREADS` to a positive integer to override it. Before OpenCV, PyTorch, or Paddle loads, any unset `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` is set to that count; a value already in the environment is left as-is. PyTorch intra-op and inter-op threads, OpenCV, and PaddleOCR's `cpu_threads` use the same count.

That pin does not cover the detect call. Ultralytics `select_device("cpu")` sets PyTorch intra-op threads to `min(8, cpu_count - 1)` on every CPU predict. OpenCV stays at `STOCK_NUM_THREADS`. Latency and the cores actually used are in the capacity PDF, not in this default of 1.

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `STOCK_NUM_THREADS` | `1` | CPU threads for inference |
| `LOG_LEVEL` | `WARNING` on the CLI, `INFO` on the API | Python logging level |
| `STOCK_OCR` | `0` | `1` enables the PaddleOCR wordmark check |
| `STOCK_GALLERY_DIR` | `gallery/` | Gallery folder |
| `STOCK_WEIGHTS_DIR` | `weights/` | Detector weights and text encoder |
| `STOCK_YOLOE_WEIGHTS` | `yoloe-11l-seg.pt` | YOLOE checkpoint |
| `STOCK_VISUAL_PROMPTS_DIR` | `visual_prompts/` | Folder with `<weights>.npz` visual prompt embeddings |
| `STOCK_WORLD_WEIGHTS` | `yolov8l-worldv2.pt` | YOLO-World fallback if YOLOE cannot load |
| `SERVICE_API_KEY` | unset | Enables the `X-API-Key` check |
| `STOCK_WARMUP` | `1` | `0` loads models on the first request |

## Tests

No model download. Covers thread defaults, EXIF orientation, stick filter bands, color histograms, the counting rules (pack size, nested fragments, split-off pieces, and stick-end disagreements), gallery match refs, category relabeling, visual-prompt gating and fallback prompts, and the API with the pipeline stubbed.

```bash
uv run pytest
```

## Smoke test and known limits

`smoke-results/` has the `detect` JSON for the four photos in `SAMPLE IMAGES/`, regenerated on 3 November 2026 with the gallery in this repo and the shipped visual prompts.

| Photo | In the photo | Counted |
|-------|--------------|---------|
| `BR00038-01.jpg` | 1 open pack, 5 loose sticks | 2 Aris Apple packs (1 flagged). The stick inside the pack is not counted |
| `BR00038-02.jpg` | 1 open pack, 5 loose sticks | 1 Aris Apple pack, 1 Aris Apple stick (flagged) |
| `BR00108-01.jpg` | 2 packs, 4 lighters | 4 Avon Blue packs and 4 unknown packs, 3 sticks, 1 unknown speaker. No lighter |
| `BR00114-02.jpg` | 1 ashtray, 2 packs (1 open), 1 speaker box | 1 paper ashtray, 1 Aris Apple pack, 1 Maxim Double Burst pack (flagged), 1 Mini Sound Box |

The two clear Avon packs in `BR00108-01.jpg` score 0.99 and 0.97 and match at 0.98 and 0.94. None of the other boxes on that photo is a `lighter`. The extra boxes the gallery names are flagged: the score is low, or the crop matches a stick band. Loose sticks on the open Aris packs are still mostly missed. One stick sits inside the pack and is not counted. The one loose stick that is counted is flagged because Aris Strawberry is within 0.02.

`BR00114-02.jpg` names every object. The ashtray and the mini speaker stay unflagged on gallery cosines of 0.86, which clears `STRONG_SIMILARITY` and excuses detector scores of 0.18 and 0.09. The open pack is flagged: its score is 0.15 and its cosine is 0.80.

A gallery hit can also name the wrong box. The second pack on `BR00038-01.jpg` matches an Aris Apple stick band at 0.79 and is flagged because Avon Blue is 0.01 behind. Stick and lighter counts on these four photos are still weak. The 13 October files were an empty-gallery run: every SKU was `unknown`, the speaker was missed, and `BR00038-01.jpg` had no stick.
