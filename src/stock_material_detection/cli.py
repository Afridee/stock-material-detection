"""Typer CLI for stock material detection."""

from __future__ import annotations

# Before pipeline (cv2, torch).
from stock_material_detection.cpu_threads import configure_cpu_threads

configure_cpu_threads()

import json
import logging
import os
from pathlib import Path
from typing import Optional

import typer

from stock_material_detection.detector import MIN_SCORE
from stock_material_detection.pipeline import MIN_SIMILARITY, run_detect

app = typer.Typer(
    name="stock_material_detection",
    help="Count stock materials in one photo (offline).",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "WARNING"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


@app.command("detect")
def detect_cmd(
    image: Path = typer.Option(..., "--image", exists=True, readable=True, dir_okay=False, help="Input photo"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON"),
    min_score: float = typer.Option(MIN_SCORE, "--min-score", help="Drop detector boxes below this score"),
    min_similarity: float = typer.Option(
        MIN_SIMILARITY, "--min-similarity", help="Gallery cosine below this gives sku 'unknown'"
    ),
    gallery: Optional[Path] = typer.Option(
        None, "--gallery", file_okay=False, help="Gallery folder (default: ./gallery or STOCK_GALLERY_DIR)"
    ),
    debug_dir: Optional[Path] = typer.Option(
        None, "--debug-dir", help="Write the oriented photo, box overlay, crops, and filter bands here"
    ),
) -> None:
    """Count items in one photo. Exit 0 on success, 1 if the image is unreadable."""
    result = run_detect(
        image,
        min_score=min_score,
        min_similarity=min_similarity,
        gallery_dir=gallery,
        debug_dir=debug_dir,
    )
    typer.echo(json.dumps(result.public_dict(), indent=2 if pretty else None))
    raise typer.Exit(code=1 if result.failReasons else 0)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
