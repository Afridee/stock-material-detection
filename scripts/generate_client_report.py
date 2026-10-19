#!/usr/bin/env python3
"""Count stock in photo folders and write a client PDF report.

Runs the detect pipeline on every photo, builds HTML in a temp directory,
prints it to PDF with Chrome, then keeps only the PDF.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import date
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from stock_material_detection.detector import CATEGORIES, MIN_SCORE
from stock_material_detection.gallery import (
    IMAGE_SUFFIXES,
    default_gallery_dir,
    iter_gallery,
    ocr_enabled,
)
from stock_material_detection.pipeline import (
    MIN_COLOR_SIMILARITY,
    MIN_MARGIN,
    MIN_SIMILARITY,
    NESTED_OVERLAP,
    REVIEW_SCORE,
    STRONG_SIMILARITY,
    UNKNOWN,
    DetectResult,
    is_weak,
    run_detect,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHOTO_DIRS = [ROOT / "SAMPLE IMAGES", ROOT / "totest"]
DEFAULT_OUTPUT = ROOT / "reports" / "stock_material_detection_report.pdf"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

PREVIEW_MAX_SIDE = 1100
CROP_MAX_SIDE = 160

CATEGORY_META = {
    "cigarette_pack": {
        "title": "Cigarette pack",
        "rule": "Each pack counts as 1, whether it is closed or open.",
    },
    "cigarette_stick": {
        "title": "Loose cigarette stick",
        "rule": "Each loose stick counts as 1. A stick that is still inside a pack is not counted separately.",
    },
    "lighter": {
        "title": "Lighter",
        "rule": "Each lighter counts as 1.",
    },
    "ashtray": {
        "title": "Ashtray",
        "rule": "Each ashtray counts as 1.",
    },
    "speaker": {
        "title": "Speaker",
        "rule": "Each speaker counts as 1.",
    },
}

REASON_COPY = {
    "no_gallery": (
        "No reference gallery",
        "No SKU references were available, so the brand could not be identified. "
        "The item is counted as Unknown.",
    ),
    "no_match": (
        "Could not compare against references",
        "The crop could not be compared with the SKU references (for example an empty crop). "
        "The item is counted as Unknown.",
    ),
    "below_similarity": (
        "No close SKU match",
        "The item did not look close enough to any SKU reference, so it is counted as Unknown "
        "rather than guessed.",
    ),
    "low_score": (
        "Weak detection",
        "The detector was not confident this is really the item it labelled, and the SKU "
        "match was not strong enough to confirm it on its own. It may be a false detection.",
    ),
    "close_call": (
        "Close call between similar SKUs",
        "Two SKUs scored almost the same, or the two ends of a stick pointed to different SKUs. "
        "The best match is reported but should be confirmed.",
    ),
    "off_color": (
        "Lighter colour does not match",
        "The lighter's colour differs from the matched SKU's references.",
    ),
    "wordmark_mismatch": (
        "Printed text does not match",
        "Text read from the item does not contain the matched SKU's wordmark.",
    ),
}

OUTCOME_LABEL = {
    "clean": "All lines identified",
    "review": "Needs review",
    "empty": "Nothing detected",
    "unreadable": "Unreadable image",
}

# RGB for the annotated overlay.
BOX_COLORS = {
    "clean": (27, 127, 78),
    "review": (180, 35, 24),
    "inside": (140, 140, 140),
}


def collect_images(folders: list[Path]) -> list[Path]:
    images: list[Path] = []
    for folder in folders:
        if not folder.is_dir():
            raise SystemExit(f"Missing photo folder: {folder}")
        found = sorted(
            p
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")
        )
        if not found:
            raise SystemExit(f"No images found under {folder}")
        images.extend(found)
    return images


def source_label(path: Path) -> str:
    try:
        return str(path.parent.relative_to(ROOT))
    except ValueError:
        return str(path.parent)


def category_title(category: str) -> str:
    meta = CATEGORY_META.get(category)
    return meta["title"] if meta else category.replace("_", " ").title()


def sku_title(sku: str) -> str:
    if sku == UNKNOWN:
        return "Unknown"
    return sku.replace("_", " ").title()


def _placeholder_jpeg(side: int = 320) -> bytes:
    im = Image.new("RGB", (side, side), (220, 220, 220))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=85, optimize=True)
    return buf.getvalue()


def _to_jpeg(im: Image.Image, max_side: int) -> bytes:
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = BytesIO()
    im.save(buf, "JPEG", quality=85, optimize=True)
    return buf.getvalue()


def preview_jpeg(src: Path, max_side: int) -> bytes | None:
    try:
        if not src.is_file() or src.stat().st_size == 0:
            return None
        with Image.open(src) as im:
            return _to_jpeg(im, max_side)
    except (OSError, Image.UnidentifiedImageError) as exc:
        print(f"Warning: no preview for {src} ({exc})", flush=True)
        return None


def data_uri_jpeg(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def detection_status(det: dict) -> str:
    if det.get("insidePack"):
        return "inside"
    return "review" if det["flagged"] else "clean"


def annotated_jpeg(oriented: Path, detections: list[dict], max_side: int = PREVIEW_MAX_SIDE) -> bytes | None:
    """Oriented photo with one labelled box per detection, downscaled for the PDF."""
    try:
        with Image.open(oriented) as im:
            canvas = im.convert("RGB")
    except (OSError, Image.UnidentifiedImageError):
        return None
    scale = min(1.0, max_side / max(canvas.size))
    if scale < 1.0:
        canvas = canvas.resize(
            (round(canvas.width * scale), round(canvas.height * scale)),
            Image.Resampling.LANCZOS,
        )
    draw = ImageDraw.Draw(canvas)
    long_side = max(canvas.size)
    # The image is shown at roughly a third of this size in the PDF.
    width = max(3, round(long_side / 220))
    font_size = max(16, round(long_side / 34))
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # very old Pillow
        font = ImageFont.load_default()

    for index, det in enumerate(detections):
        x1, y1, x2, y2 = (v * scale for v in det["box"])
        status = detection_status(det)
        color = BOX_COLORS[status]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        if status == "inside":
            label = f"#{index + 1:02d} in pack"
        else:
            label = f"#{index + 1:02d} {sku_title(det['sku'])}"
        tx1, ty1, tx2, ty2 = draw.textbbox((0, 0), label, font=font)
        tw, th = tx2 - tx1, ty2 - ty1
        pad = max(2, font_size // 5)
        lx = min(max(0, x1), canvas.width - tw - 2 * pad)
        ly = y1 - th - 2 * pad
        if ly < 0:
            ly = y1 + width
        draw.rectangle((lx, ly, lx + tw + 2 * pad, ly + th + 2 * pad), fill=color)
        draw.text((lx + pad - tx1, ly + pad - ty1), label, fill=(255, 255, 255), font=font)
    return _to_jpeg(canvas, max_side)


def review_reasons(det: dict, *, gallery_empty: bool) -> list[str]:
    """Best-effort reason codes for a flagged detection, from its public JSON."""
    if det.get("insidePack"):
        return []
    reasons: list[str] = []
    if det["sku"] == UNKNOWN:
        if det["similarity"] is None:
            reasons.append("no_gallery" if gallery_empty else "no_match")
        else:
            reasons.append("below_similarity")
    if is_weak(det["score"], det["similarity"]):
        reasons.append("low_score")
    if det.get("wordmark") == "mismatch":
        reasons.append("wordmark_mismatch")
    color = det.get("colorSimilarity")
    if color is not None and color < MIN_COLOR_SIMILARITY:
        reasons.append("off_color")
    if det["flagged"] and not reasons:
        reasons.append("close_call")
    return reasons


def outcome(result: DetectResult) -> str:
    if result.failReasons:
        return "unreadable"
    if not result.counts:
        return "empty"
    if any(line.flagged for line in result.counts):
        return "review"
    return "clean"


def pct(n: int, total: int) -> str:
    if not total:
        return "0%"
    return f"{round(100 * n / total)}%"


def fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_html(rows: list[dict], ctx: dict) -> str:
    total = len(rows)
    n_clean = sum(1 for r in rows if r["outcome"] == "clean")
    n_review = sum(1 for r in rows if r["outcome"] == "review")
    n_empty = sum(1 for r in rows if r["outcome"] == "empty")
    n_unreadable = sum(1 for r in rows if r["outcome"] == "unreadable")
    items_total = sum(line["count"] for r in rows for line in r["counts"])
    lines_total = sum(len(r["counts"]) for r in rows)
    lines_flagged = sum(1 for r in rows for line in r["counts"] if line["flagged"])

    # Totals per (sku, category) across every photo.
    totals: dict[tuple[str, str], dict] = defaultdict(lambda: {"count": 0, "photos": 0, "flagged": 0})
    for r in rows:
        for line in r["counts"]:
            key = (line["sku"], line["category"])
            totals[key]["count"] += line["count"]
            totals[key]["photos"] += 1
            totals[key]["flagged"] += int(line["flagged"])
    total_rows = "".join(
        f"<tr><td>{html.escape(sku_title(sku))}</td><td>{html.escape(category_title(cat))}</td>"
        f"<td class='num'>{t['count']}</td><td class='num'>{t['photos']}</td>"
        f"<td class='num'>{t['flagged']}</td></tr>"
        for (sku, cat), t in sorted(
            totals.items(), key=lambda kv: (kv[0][0] == UNKNOWN, kv[0][1], kv[0][0])
        )
    ) or "<tr><td colspan='5'>No items were detected in any photo.</td></tr>"

    reason_counter: Counter[str] = Counter()
    for r in rows:
        for det in r["detections"]:
            for code in det["reasons"]:
                reason_counter[REASON_COPY[code][0]] += 1
    reason_rows = "".join(
        f"<tr><td>{html.escape(title)}</td><td class='num'>{count}</td></tr>"
        for title, count in reason_counter.most_common(8)
    ) or "<tr><td colspan='2'>No detections were flagged for review.</td></tr>"

    cards = []
    index_rows = []
    for i, r in enumerate(rows, start=1):
        out = r["outcome"]
        badge_text = OUTCOME_LABEL[out]

        if r["counts"]:
            count_rows = "".join(
                f"<tr><td>{html.escape(sku_title(line['sku']))}</td>"
                f"<td>{html.escape(category_title(line['category']))}</td>"
                f"<td class='num'>{line['count']}</td>"
                f"<td><span class='pill {'review' if line['flagged'] else 'clean'}'>"
                f"{'Review' if line['flagged'] else 'OK'}</span></td></tr>"
                for line in r["counts"]
            )
            counts_html = f"""
              <table class="counts">
                <thead><tr><th>SKU</th><th>Item</th><th class="num">Count</th><th>Status</th></tr></thead>
                <tbody>{count_rows}</tbody>
              </table>"""
        elif out == "unreadable":
            counts_html = '<p class="warn-note">The file could not be read as an image. Nothing was counted.</p>'
        else:
            counts_html = '<p class="warn-note">No items were detected in this photo.</p>'

        det_rows = []
        for j, det in enumerate(r["detections"], start=1):
            status = detection_status(det)
            if status == "inside":
                status_html = '<span class="pill inside">In pack · not counted</span>'
            elif status == "review":
                status_html = '<span class="pill review">Review</span>'
            else:
                status_html = '<span class="pill clean">OK</span>'
            notes = [REASON_COPY[code][0] for code in det["reasons"]]
            if det.get("wordmark") == "confirmed":
                notes.append("Printed text confirmed")
            note_html = html.escape("; ".join(notes)) if notes else ""
            thumb = (
                f'<img class="thumb" src="{det["thumb"]}" alt="">'
                if det.get("thumb")
                else '<span class="thumb empty"></span>'
            )
            det_rows.append(
                f"""
                <tr>
                  <td class="num">{j:02d}</td>
                  <td>{thumb}</td>
                  <td>{html.escape(category_title(det["category"]))}</td>
                  <td>{html.escape(sku_title(det["sku"]))}</td>
                  <td class="num">{fmt(det["score"])}</td>
                  <td class="num">{fmt(det["similarity"])}</td>
                  <td>{status_html}<div class="det-note">{note_html}</div></td>
                </tr>"""
            )
        detections_html = (
            f"""
              <h3>Detections</h3>
              <table class="dets">
                <thead>
                  <tr><th>#</th><th>Crop</th><th>Item</th><th>SKU</th>
                  <th class="num">Detector</th><th class="num">Match</th><th>Status</th></tr>
                </thead>
                <tbody>{"".join(det_rows)}</tbody>
              </table>"""
            if det_rows
            else ""
        )

        reason_items = []
        seen: set[str] = set()
        for det in r["detections"]:
            for code in det["reasons"]:
                if code in seen:
                    continue
                seen.add(code)
                title, detail = REASON_COPY[code]
                reason_items.append(
                    f'<div class="fail-item"><strong>{html.escape(title)}</strong>'
                    f"<p>{html.escape(detail)}</p></div>"
                )
        if reason_items:
            why_html = "".join(reason_items)
        elif out == "clean":
            why_html = '<p class="ok-note">Every counted item matched a known SKU with no review flags.</p>'
        elif out == "unreadable":
            why_html = '<p class="warn-note">Replace the file with a readable JPEG or PNG and run again.</p>'
        else:
            why_html = '<p class="warn-note">The detector found none of the tracked item types in this photo.</p>'

        n_items = sum(line["count"] for line in r["counts"])
        n_flag = sum(1 for line in r["counts"] if line["flagged"])

        cards.append(
            f"""
            <article class="card" id="photo-{i}">
              <div class="card-head">
                <div>
                  <div class="photo-id">Photo {i:02d} of {total:02d}</div>
                  <h2>{html.escape(r["filename"])}</h2>
                  <div class="meta">{html.escape(r["source"])}</div>
                </div>
                <div class="badge {out}">{html.escape(badge_text)}</div>
              </div>
              <div class="card-body">
                <div class="photo-wrap">
                  <img src="{r["asset"]}" alt="{html.escape(r["filename"])}">
                </div>
                <div class="detail">
                  <div class="score-line">
                    <div class="score-num">{n_items}</div>
                    <div class="score-label">item{"" if n_items == 1 else "s"} counted across
                      {len(r["counts"])} line{"" if len(r["counts"]) == 1 else "s"}
                      {f"· <strong>{n_flag}</strong> line{'' if n_flag == 1 else 's'} to review" if n_flag else ""}
                    </div>
                  </div>
                  <h3>Counts</h3>
                  {counts_html}
                  <h3>Why this result</h3>
                  {why_html}
                </div>
              </div>
              {detections_html}
            </article>
            """
        )

        index_rows.append(
            f"""
            <tr>
              <td><a href="#photo-{i}">{i:02d}</a></td>
              <td><a href="#photo-{i}">{html.escape(r["filename"])}</a></td>
              <td>{html.escape(r["source"])}</td>
              <td class="num">{n_items}</td>
              <td class="num">{len(r["counts"])}</td>
              <td class="num">{n_flag}</td>
              <td><span class="pill {out}">{html.escape(OUTCOME_LABEL[out])}</span></td>
            </tr>
            """
        )

    category_rows = "".join(
        f"<div><dt>{html.escape(CATEGORY_META[c]['title'])}</dt><dd>{html.escape(CATEGORY_META[c]['rule'])}</dd></div>"
        for c in CATEGORIES
        if c in CATEGORY_META
    )
    gallery_skus = ctx["gallery_skus"]
    if gallery_skus:
        gallery_txt = (
            f"{len(gallery_skus)} SKUs with {ctx['gallery_refs']} reference crops: "
            + ", ".join(sku_title(s) for s in gallery_skus)
            + "."
        )
    else:
        gallery_txt = "No SKU references were loaded, so every item is reported as Unknown."

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stock Material Detection Report</title>
  <style>
    :root {{
      --ink: #1c2430;
      --muted: #5b6777;
      --line: #d9e0ea;
      --bg: #f4f6f9;
      --card: #ffffff;
      --clean: #1b7f4e;
      --clean-bg: #e7f6ee;
      --review: #b42318;
      --review-bg: #fdecea;
      --empty: #9a6700;
      --empty-bg: #fff6dd;
      --inside: #5b6777;
      --inside-bg: #eef1f5;
      --brand: #12325c;
      --accent: #2b6cb0;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.45;
    }}
    .page {{ max-width: 1100px; margin: 0 auto; padding: 28px 20px 64px; }}
    header.hero {{
      background: var(--brand);
      color: #fff;
      border-radius: 16px;
      padding: 28px 32px;
      margin-bottom: 24px;
    }}
    header.hero h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: -0.02em; }}
    header.hero p {{ margin: 0; color: #d7e3f4; max-width: 80ch; }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin: 20px 0 24px;
    }}
    .kpi {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px 16px;
    }}
    .kpi .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .kpi .value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
    .kpi.clean .value {{ color: var(--clean); }}
    .kpi.review .value {{ color: var(--review); }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 20px 22px;
      margin-bottom: 20px;
    }}
    h2, h3 {{ margin: 0 0 10px; }}
    h2 {{ font-size: 20px; color: var(--brand); }}
    h3 {{ font-size: 14px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin-top: 14px; }}
    .legend {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px 24px;
      font-size: 14px;
      margin: 0;
    }}
    .legend dt {{ font-weight: 700; }}
    .legend dd {{ margin: 0 0 8px; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 8px 6px; border-bottom: 1px solid var(--line); vertical-align: middle; }}
    th {{ font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }}
    td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    a {{ color: var(--accent); text-decoration: none; }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 600;
      white-space: nowrap;
    }}
    .pill.clean, .badge.clean {{ background: var(--clean-bg); color: var(--clean); }}
    .pill.review, .badge.review {{ background: var(--review-bg); color: var(--review); }}
    .pill.empty, .badge.empty, .pill.unreadable, .badge.unreadable {{ background: var(--empty-bg); color: var(--empty); }}
    .pill.inside {{ background: var(--inside-bg); color: var(--inside); }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px;
      margin: 0 0 18px;
      break-inside: avoid;
      page-break-inside: avoid;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 14px;
    }}
    .photo-id {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
    .card-head h2 {{ margin: 2px 0 4px; font-size: 18px; color: var(--ink); }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    .badge {{
      border-radius: 999px;
      padding: 8px 12px;
      font-weight: 700;
      white-space: nowrap;
      font-size: 13px;
    }}
    .card-body {{
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 18px;
      align-items: start;
    }}
    .photo-wrap {{
      background: #0f1724;
      border-radius: 10px;
      overflow: hidden;
      min-height: 220px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .photo-wrap img {{
      max-width: 100%;
      max-height: 440px;
      display: block;
      object-fit: contain;
    }}
    .score-line {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 4px; }}
    .score-num {{ font-size: 32px; font-weight: 800; color: var(--brand); }}
    .score-label {{ color: var(--muted); font-size: 14px; }}
    table.counts td, table.counts th {{ padding: 6px 6px; }}
    table.dets {{ margin-top: 4px; }}
    table.dets td {{ font-size: 12px; }}
    .thumb {{
      display: block;
      width: 56px;
      height: 56px;
      object-fit: contain;
      background: #eef1f5;
      border-radius: 6px;
    }}
    .thumb.empty {{ background: repeating-linear-gradient(45deg, #eef1f5, #eef1f5 6px, #e3e8ef 6px, #e3e8ef 12px); }}
    .det-note {{ color: var(--muted); font-size: 11px; margin-top: 2px; }}
    .fail-item {{
      background: var(--review-bg);
      border-radius: 8px;
      padding: 8px 10px;
      margin: 0 0 8px;
    }}
    .fail-item p, .ok-note, .warn-note {{ margin: 4px 0 0; font-size: 13px; color: var(--muted); }}
    .ok-note {{
      background: var(--clean-bg);
      color: var(--clean);
      padding: 8px 10px;
      border-radius: 8px;
    }}
    .warn-note {{
      background: var(--empty-bg);
      color: var(--empty);
      padding: 8px 10px;
      border-radius: 8px;
    }}
    .note {{ color: var(--muted); font-size: 13px; }}
    footer {{ color: var(--muted); font-size: 12px; margin-top: 12px; }}
    @media print {{
      body {{ background: #fff; }}
      .page {{ max-width: none; padding: 0; }}
      header.hero {{ border-radius: 0; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
      .card, .panel, .kpi, .badge, .pill, .fail-item, .ok-note, .warn-note, .thumb {{
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
      }}
      a {{ color: inherit; }}
      .card {{ page-break-before: always; }}
      @page {{ size: A4; margin: 12mm; }}
    }}
    @media (max-width: 800px) {{
      .kpis {{ grid-template-columns: 1fr 1fr; }}
      .card-body, .legend {{ grid-template-columns: 1fr; }}
    }}
    @media print {{
      /* A4 is narrower than 800px, so keep the two-column card on paper. */
      .card-body {{ grid-template-columns: 300px 1fr; }}
      .legend {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <h1>Stock Material Detection Report</h1>
      <p>
        Offline evaluation of {total} stock photo{"" if total == 1 else "s"} on {date.today():%d %B %Y}.
        Each photo is scanned for cigarette packs, loose sticks, lighters, ashtrays, and speakers;
        every item found is matched against the SKU reference gallery and counted per SKU.
        Items the system cannot identify with confidence are counted as <strong>Unknown</strong>
        and flagged for a human to review rather than guessed.
      </p>
    </header>

    <section class="kpis">
      <div class="kpi"><div class="label">Photos reviewed</div><div class="value">{total}</div></div>
      <div class="kpi"><div class="label">Items counted</div><div class="value">{items_total}</div></div>
      <div class="kpi clean"><div class="label">Photos fully identified</div><div class="value">{n_clean}</div></div>
      <div class="kpi review"><div class="label">Photos needing review</div><div class="value">{n_review}</div></div>
    </section>

    <section class="panel">
      <h2>How to read this report</h2>
      <p>Each photo produces a list of <strong>count lines</strong>, one per SKU and item type
      (for example “Avon Blue · Cigarette pack · 2”). A line is marked <span class="pill review">Review</span>
      when at least one of its items was uncertain; otherwise it is <span class="pill clean">OK</span>.
      The annotated photo shows every detection: green boxes are confidently identified, red boxes need review,
      and grey boxes are sticks still inside a pack (not counted).</p>
      <h3>Counting rules</h3>
      <dl class="legend">{category_rows}</dl>
      <h3>Reference gallery</h3>
      <p class="note">{html.escape(gallery_txt)}</p>
      <h3>Thresholds used</h3>
      <dl class="legend">
        <div><dt>Minimum detector score · {ctx["min_score"]:.2f}</dt><dd>Boxes below this were dropped before counting.</dd></div>
        <div><dt>Review score · {REVIEW_SCORE:.2f}</dt><dd>Detections below this are kept but flagged as weak, unless the SKU match is at least {STRONG_SIMILARITY:.2f}.</dd></div>
        <div><dt>Nested box overlap · {NESTED_OVERLAP:.2f}</dt><dd>A box with this share of its area inside a stronger box of the same item type is a fragment and is not counted.</dd></div>
        <div><dt>Minimum SKU similarity · {ctx["min_similarity"]:.2f}</dt><dd>Below this the item is counted as Unknown.</dd></div>
        <div><dt>Minimum SKU margin · {MIN_MARGIN:.2f}</dt><dd>If the top two SKUs are closer than this, the item is flagged.</dd></div>
        <div><dt>Lighter colour similarity · {MIN_COLOR_SIMILARITY:.2f}</dt><dd>Lighters whose colour is further from the matched SKU are flagged.</dd></div>
        <div><dt>Printed-text check</dt><dd>{"On: pack and lighter text was read and compared with the SKU wordmark." if ctx["ocr"] else "Off for this run."}</dd></div>
      </dl>
      <p class="note">Lines: <strong>{lines_total}</strong> in total, <strong>{lines_flagged}</strong> flagged for review
      ({pct(lines_flagged, lines_total)}).
      {f"Photos with nothing detected: <strong>{n_empty}</strong>." if n_empty else ""}
      {f"Unreadable files: <strong>{n_unreadable}</strong>." if n_unreadable else ""}</p>
    </section>

    <section class="panel">
      <h2>Totals across all photos</h2>
      <table>
        <thead><tr><th>SKU</th><th>Item</th><th class="num">Count</th><th class="num">Photos</th><th class="num">Lines to review</th></tr></thead>
        <tbody>{total_rows}</tbody>
      </table>
    </section>

    <section class="panel">
      <h2>Most common review reasons</h2>
      <table>
        <thead><tr><th>Reason</th><th class="num">Detections</th></tr></thead>
        <tbody>{reason_rows}</tbody>
      </table>
    </section>

    <section class="panel">
      <h2>Photo index</h2>
      <table>
        <thead>
          <tr>
            <th>#</th><th>File</th><th>Source</th><th class="num">Items</th><th class="num">Lines</th><th class="num">To review</th><th>Result</th>
          </tr>
        </thead>
        <tbody>
          {"".join(index_rows)}
        </tbody>
      </table>
    </section>

    <h2 style="margin: 8px 4px 12px;">Photo-by-photo results</h2>
    {"".join(cards)}

    <footer>
      Generated by the stock material detection eval (YOLOE open-vocabulary boxes + SigLIP 2 gallery match).
      Gallery: {html.escape(str(ctx["gallery_dir"]))}. Detector score ≥ {ctx["min_score"]:.2f}, SKU similarity ≥ {ctx["min_similarity"]:.2f}.
    </footer>
  </div>
</body>
</html>
"""


def chrome_pdf(html_path: Path, pdf_path: Path) -> None:
    if not CHROME.exists():
        raise SystemExit("Google Chrome not found; cannot convert HTML to PDF.")
    cmd = [
        str(CHROME),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--no-first-run",
        "--disable-extensions",
        f"--print-to-pdf={pdf_path}",
        "--virtual-time-budget=180000",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count stock in photo folders and write a client PDF report."
    )
    parser.add_argument(
        "--dir",
        dest="dirs",
        type=Path,
        action="append",
        help="Photo folder (repeatable). Defaults to 'SAMPLE IMAGES/' and 'totest/' when present.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PDF output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_SCORE,
        help=f"Drop detector boxes below this score (default: {MIN_SCORE})",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=MIN_SIMILARITY,
        help=f"Gallery cosine below this gives sku 'unknown' (default: {MIN_SIMILARITY})",
    )
    parser.add_argument(
        "--gallery",
        type=Path,
        default=None,
        help="Gallery folder (default: ./gallery or STOCK_GALLERY_DIR)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dirs:
        folders = args.dirs
    else:
        folders = [d for d in DEFAULT_PHOTO_DIRS if d.is_dir()]
        if not folders:
            raise SystemExit(
                "No default photo folders found; pass --dir. Looked for: "
                + ", ".join(str(d) for d in DEFAULT_PHOTO_DIRS)
            )
    images = collect_images(folders)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    gallery_dir = (args.gallery or default_gallery_dir()).resolve()
    gallery_entries = iter_gallery(gallery_dir)
    gallery_skus = sorted({sku for sku, _ in gallery_entries})
    ctx = {
        "gallery_dir": gallery_dir,
        "gallery_skus": gallery_skus,
        "gallery_refs": len(gallery_entries),
        "min_score": args.min_score,
        "min_similarity": args.min_similarity,
        "ocr": ocr_enabled(),
    }

    rows: list[dict] = []
    print(
        f"Scoring {len(images)} photos (gallery: {len(gallery_skus)} SKUs, "
        f"min_score={args.min_score}, min_similarity={args.min_similarity})",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="stock_report_") as tmp:
        tmp_dir = Path(tmp)
        for i, image in enumerate(images, start=1):
            print(f"[{i}/{len(images)}] {image}", flush=True)
            debug_dir = tmp_dir / f"debug_{i:03d}"
            result = run_detect(
                image,
                min_score=args.min_score,
                min_similarity=args.min_similarity,
                gallery_dir=gallery_dir,
                debug_dir=debug_dir,
            )
            detections = []
            for index, det in enumerate(result.detections):
                crop = preview_jpeg(debug_dir / f"{index:02d}_{det['category']}.jpg", CROP_MAX_SIDE)
                detections.append(
                    {
                        **det,
                        "reasons": review_reasons(det, gallery_empty=not gallery_entries),
                        "thumb": data_uri_jpeg(crop) if crop else None,
                    }
                )
            asset = annotated_jpeg(debug_dir / "oriented.jpg", result.detections)
            if asset is None:
                asset = preview_jpeg(image, PREVIEW_MAX_SIDE) or _placeholder_jpeg()
            rows.append(
                {
                    "filename": image.name,
                    "source": source_label(image),
                    "asset": data_uri_jpeg(asset),
                    "outcome": outcome(result),
                    "counts": [line.model_dump() for line in result.counts],
                    "detections": detections,
                }
            )

        html_path = tmp_dir / "stock_material_detection_report.html"
        pdf_tmp = tmp_dir / "stock_material_detection_report.pdf"
        html_path.write_text(render_html(rows, ctx), encoding="utf-8")
        print(f"Wrote HTML {html_path} ({html_path.stat().st_size} bytes)", flush=True)
        print("Converting HTML to PDF via Chrome…", flush=True)
        chrome_pdf(html_path, pdf_tmp)
        shutil.copy2(pdf_tmp, output)
        print(f"Wrote {output}", flush=True)

    summary = {
        "total": len(rows),
        "items": sum(line["count"] for r in rows for line in r["counts"]),
        "clean": sum(1 for r in rows if r["outcome"] == "clean"),
        "review": sum(1 for r in rows if r["outcome"] == "review"),
        "empty": sum(1 for r in rows if r["outcome"] == "empty"),
        "unreadable": sum(1 for r in rows if r["outcome"] == "unreadable"),
        "pdf": str(output),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
