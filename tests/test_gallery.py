"""Gallery ranking. Uses fake vectors so SigLIP is not loaded."""

from __future__ import annotations

import numpy as np

from stock_material_detection.gallery import Gallery


def _gallery_with_vectors(tmp_path, rows: list[tuple[str, str, list[float]]]) -> Gallery:
    gallery = Gallery(tmp_path)
    gallery.entries = [(sku, tmp_path / sku / name) for sku, name, _ in rows]
    gallery.skus = np.array([sku for sku, _, _ in rows], dtype=object)
    gallery.vectors = np.array([vec for _, _, vec in rows], dtype=np.float32)
    return gallery


def test_match_records_best_and_runner_up_refs(tmp_path) -> None:
    gallery = _gallery_with_vectors(
        tmp_path,
        [
            ("avon_blue", "pack.jpg", [0.90, 0.0]),
            ("avon_blue", "stick_06_band.jpg", [0.99, 0.0]),
            ("black_diamond", "pack.jpg", [0.94, 0.0]),
        ],
    )

    match = gallery.match(np.array([[1.0, 0.0]], dtype=np.float32))

    assert match is not None
    assert match.sku == "avon_blue"
    assert match.ref == "avon_blue/stick_06_band.jpg"
    assert match.runner_up_sku == "black_diamond"
    assert match.runner_up_ref == "black_diamond/pack.jpg"


def test_match_has_no_runner_up_when_gallery_has_one_sku(tmp_path) -> None:
    gallery = _gallery_with_vectors(
        tmp_path,
        [("avon_blue", "pack.jpg", [1.0, 0.0])],
    )

    match = gallery.match(np.array([[1.0, 0.0]], dtype=np.float32))

    assert match is not None
    assert match.ref == "avon_blue/pack.jpg"
    assert match.runner_up is None
    assert match.runner_up_ref is None
