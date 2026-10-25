"""Counting rules. No detector or gallery is loaded."""

from __future__ import annotations

from stock_material_detection.detector import Detection
from stock_material_detection.gallery import Match
from stock_material_detection.pipeline import (
    UNKNOWN,
    Item,
    _resolve_end_matches,
    aggregate,
    drop_nested,
    is_weak,
    mark_inside_packs,
)


def _item(category: str, box, sku: str = "brand_a", flagged: bool = False) -> Item:
    return Item(category=category, score=0.9, box=box, sku=sku, similarity=0.95, flagged=flagged)


def _counts(items: list[Item]) -> dict[tuple[str, str], tuple[int, bool]]:
    return {(line.sku, line.category): (line.count, line.flagged) for line in aggregate(items)}


def test_pack_counts_as_one_and_inner_sticks_are_ignored() -> None:
    items = [
        _item("cigarette_pack", (100, 100, 300, 400)),
        _item("cigarette_stick", (120, 110, 140, 200)),
        _item("cigarette_stick", (150, 110, 170, 200)),
    ]
    mark_inside_packs(items)

    assert [it.inside_pack for it in items[1:]] == [True, True]
    assert _counts(items) == {("brand_a", "cigarette_pack"): (1, False)}


def test_loose_sticks_are_counted() -> None:
    items = [
        _item("cigarette_pack", (100, 100, 300, 400)),
        _item("cigarette_stick", (400, 100, 700, 130)),
        _item("cigarette_stick", (400, 140, 700, 170)),
        _item("cigarette_stick", (280, 380, 600, 410)),  # barely touches the pack
    ]
    mark_inside_packs(items)

    assert not any(it.inside_pack for it in items[1:])
    assert _counts(items) == {
        ("brand_a", "cigarette_pack"): (1, False),
        ("brand_a", "cigarette_stick"): (3, False),
    }


def test_marked_inside_stick_is_skipped() -> None:
    inside = _item("cigarette_stick", (0, 0, 10, 10))
    inside.inside_pack = True
    loose = _item("cigarette_stick", (50, 0, 60, 10))

    assert _counts([inside, loose]) == {("brand_a", "cigarette_stick"): (1, False)}


def test_each_lighter_ashtray_speaker_counts_once() -> None:
    items = [
        _item("lighter", (0, 0, 10, 40)),
        _item("lighter", (20, 0, 30, 40)),
        _item("ashtray", (100, 100, 200, 200), sku="tray"),
        _item("speaker", (300, 300, 400, 400), sku="spk"),
    ]

    assert _counts(items) == {
        ("brand_a", "lighter"): (2, False),
        ("tray", "ashtray"): (1, False),
        ("spk", "speaker"): (1, False),
    }


def test_unknown_is_flagged() -> None:
    items = [
        _item("cigarette_pack", (0, 0, 10, 10), sku=UNKNOWN, flagged=False),
        _item("cigarette_pack", (20, 0, 30, 10), sku=UNKNOWN, flagged=False),
    ]

    assert _counts(items) == {(UNKNOWN, "cigarette_pack"): (2, True)}


def test_one_flagged_detection_flags_its_line() -> None:
    items = [
        _item("cigarette_pack", (0, 0, 10, 10)),
        _item("cigarette_pack", (20, 0, 30, 10), flagged=True),
        _item("lighter", (40, 0, 50, 10)),
    ]

    counts = _counts(items)
    assert counts[("brand_a", "cigarette_pack")] == (2, True)
    assert counts[("brand_a", "lighter")] == (1, False)


def _match(sku: str, similarity: float) -> Match:
    return Match(sku=sku, similarity=similarity, runner_up=similarity - 0.1)


def test_disagreeing_stick_ends_keep_the_higher_score() -> None:
    best, rival = _resolve_end_matches(
        [_match("black_diamond", 0.80), _match("avon_blue", 0.88)],
        min_similarity=0.70,
    )

    assert best is not None
    assert best.sku == "avon_blue"
    assert best.similarity == 0.88
    assert rival is not None
    assert rival.sku == "black_diamond"


def test_a_weak_other_end_is_not_a_disagreement() -> None:
    best, rival = _resolve_end_matches(
        [_match("avon_blue", 0.88), _match("black_diamond", 0.60)],
        min_similarity=0.70,
    )

    assert best is not None
    assert best.sku == "avon_blue"
    assert rival is None


def test_agreeing_stick_ends_are_not_flagged() -> None:
    best, rival = _resolve_end_matches(
        [_match("black_diamond", 0.84), _match("black_diamond", 0.81)],
        min_similarity=0.70,
    )

    assert best is not None
    assert best.sku == "black_diamond"
    assert rival is None


def test_pack_fragments_inside_a_stronger_pack_are_dropped() -> None:
    # Photo 1 of totest/: half a pack and a slid-out sleeve next to the full boxes.
    full_a = Detection("cigarette_pack", 0.68, (350, 2210, 1651, 3024))
    full_b = Detection("cigarette_pack", 0.57, (293, 337, 1487, 1130))
    half_b = Detection("cigarette_pack", 0.10, (853, 345, 1485, 1094))
    sleeve_a = Detection("cigarette_pack", 0.06, (1297, 2210, 1653, 2953))
    stick_in_b = Detection("cigarette_stick", 0.06, (1109, 330, 1142, 973))

    kept = drop_nested([full_b, half_b, sleeve_a, full_a, stick_in_b])

    assert kept == [full_b, full_a, stick_in_b]


def test_side_by_side_packs_are_both_kept() -> None:
    left = Detection("cigarette_pack", 0.5, (0, 0, 100, 150))
    right = Detection("cigarette_pack", 0.4, (90, 0, 190, 150))

    assert drop_nested([left, right]) == [left, right]


def test_strong_gallery_match_excuses_a_weak_detector_score() -> None:
    assert is_weak(0.16, 0.84)
    assert not is_weak(0.16, 0.89)
    assert is_weak(0.16, None)
    assert not is_weak(0.35, 0.60)


def test_same_sku_splits_by_category() -> None:
    items = [
        _item("cigarette_pack", (0, 0, 100, 100)),
        _item("cigarette_stick", (200, 0, 400, 20)),
    ]

    assert set(_counts(items)) == {
        ("brand_a", "cigarette_pack"),
        ("brand_a", "cigarette_stick"),
    }
