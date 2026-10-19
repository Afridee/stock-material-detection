"""Visual prompt file loading and box gating. No model is loaded."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stock_material_detection.detector import PROMPTS, Detector, box_iou, load_visual_prompts, nms


def _write(path: Path, categories: list[str], vpe: np.ndarray) -> Path:
    np.savez(path, categories=np.array(categories), vpe=vpe.astype(np.float32), meta="{}")
    return path


def test_loads_categories_and_rows(tmp_path: Path) -> None:
    path = _write(tmp_path / "m.npz", ["cigarette_pack", "lighter"], np.eye(2, 512))

    categories, vpe = load_visual_prompts(path, embed_dim=512)

    assert categories == ["cigarette_pack", "lighter"]
    assert vpe.shape == (2, 512)
    assert vpe.dtype == np.float32


def test_missing_file_gives_nothing(tmp_path: Path) -> None:
    categories, vpe = load_visual_prompts(tmp_path / "absent.npz", embed_dim=512)

    assert categories == []
    assert vpe.shape == (0, 512)


def test_other_model_dimension_is_ignored(tmp_path: Path) -> None:
    path = _write(tmp_path / "m.npz", ["cigarette_pack"], np.ones((1, 256)))

    assert load_visual_prompts(path, embed_dim=512)[0] == []


def test_unknown_category_is_ignored(tmp_path: Path) -> None:
    path = _write(tmp_path / "m.npz", ["cigarette_pack", "vape"], np.ones((2, 512)))

    assert load_visual_prompts(path, embed_dim=512)[0] == []


def test_row_count_must_match_categories(tmp_path: Path) -> None:
    path = _write(tmp_path / "m.npz", ["cigarette_pack"], np.ones((2, 512)))

    assert load_visual_prompts(path, embed_dim=512)[0] == []


def test_box_iou() -> None:
    box = np.array([0, 0, 10, 10], dtype=float)
    others = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)

    assert np.allclose(box_iou(box, others), [1.0, 1 / 3, 0.0])


def test_nms_keeps_best_per_object_in_score_order() -> None:
    xyxy = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    scores = np.array([0.5, 0.9, 0.7])

    assert nms(xyxy, scores, 0.5) == [1, 2]


def _bare_detector(visual: list[str]) -> Detector:
    det = Detector.__new__(Detector)
    det._prompts = list(PROMPTS)
    det._categories = [PROMPTS[p] for p in det._prompts] + visual
    det.visual_categories = visual
    return det


def test_visual_box_needs_an_overlapping_text_box_of_its_category() -> None:
    det = _bare_detector(["cigarette_pack"])
    n_text = len(PROMPTS)
    pack_text = det._prompts.index("product box")
    ashtray_text = det._prompts.index("ashtray")
    xyxy = np.array(
        [
            [100, 100, 300, 400],  # text pack
            [102, 98, 298, 402],  # visual pack on the same object -> kept, wins NMS
            [500, 500, 700, 700],  # text ashtray
            [500, 500, 700, 700],  # visual pack on the ashtray -> dropped
            [0, 0, 1000, 1000],  # visual pack on the whole frame -> dropped
        ],
        dtype=float,
    )
    scores = np.array([0.20, 0.90, 0.30, 0.70, 0.15])
    classes = np.array([pack_text, n_text, ashtray_text, n_text, n_text])

    assert det._support_visual_boxes(xyxy, scores, classes) == [1, 2]


def test_shipped_file_is_valid() -> None:
    shipped = Path(__file__).resolve().parents[1] / "visual_prompts" / "yoloe-11l-seg.npz"

    categories, vpe = load_visual_prompts(shipped, embed_dim=512)

    assert "cigarette_pack" in categories
    assert np.allclose(np.linalg.norm(vpe, axis=1), 1.0, atol=1e-4)
