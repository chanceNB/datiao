import pytest
from pydantic import ValidationError

from datiao.r1.parser import parse_raw_points
from datiao.r1.stroke import StrokeBuildConfig, build_strokes


def record(point_id, timestamp_ms, *, session_id="s-1", page_id="p-1", sequence=0):
    return {
        "point_id": point_id,
        "session_id": session_id,
        "page_id": page_id,
        "x": timestamp_ms / 10 if timestamp_ms is not None else 1,
        "y": 2,
        "timestamp_ms": timestamp_ms,
        "sequence": sequence,
    }


def test_continuous_points_form_one_stroke():
    points = parse_raw_points(
        (record("p-1", 1000), record("p-2", 1050, sequence=1), record("p-3", 1100, sequence=2))
    )

    strokes = build_strokes(points)

    assert len(strokes) == 1
    assert strokes[0].raw_order == ("p-1", "p-2", "p-3")
    assert strokes[0].processed_order == ("p-1", "p-2", "p-3")


def test_context_or_time_gap_creates_multiple_strokes():
    points = parse_raw_points(
        (
            record("p-1", 1000),
            record("p-2", 1050, sequence=1),
            record("p-3", 3000, sequence=2),
            record("p-4", 3050, session_id="s-2", sequence=3),
        )
    )

    strokes = build_strokes(points, StrokeBuildConfig(max_time_gap_ms=500))

    assert len(strokes) == 3
    assert strokes[0].raw_order == ("p-1", "p-2")
    assert strokes[1].raw_order == ("p-3",)
    assert strokes[2].raw_order == ("p-4",)


def test_out_of_order_points_are_retained_and_processed_order_can_sort():
    points = parse_raw_points(
        (record("p-1", 1200, sequence=2), record("p-2", 1100, sequence=1))
    )

    strokes = build_strokes(points)

    assert len(strokes) == 1
    assert "OUT_OF_ORDER" in points[1].quality_flags
    assert strokes[0].raw_order == ("p-1", "p-2")
    assert strokes[0].processed_order == ("p-2", "p-1")
    assert "PARTIAL_DATA" in strokes[0].quality_flags


def test_duplicate_points_are_retained_and_mark_stroke_quality():
    points = parse_raw_points((record("same", 1000), record("same", 1050, sequence=1)))

    strokes = build_strokes(points)

    assert len(strokes) == 1
    assert strokes[0].raw_order == ("same", "same")
    assert "DUPLICATE_POINT" in points[0].quality_flags
    assert "PARTIAL_DATA" in strokes[0].quality_flags


def test_missing_timestamp_forms_incomplete_stroke_without_drop():
    points = parse_raw_points((record("p-1", None), record("p-2", 1050, sequence=1)))

    strokes = build_strokes(points)

    assert len(strokes) == 1
    assert strokes[0].raw_order == ("p-1", "p-2")
    assert "INCOMPLETE" in strokes[0].quality_flags


def test_empty_input_returns_empty_tuple():
    assert build_strokes(()) == ()


def test_stroke_raw_order_is_immutable():
    point = parse_raw_points((record("p-1", 1000),))[0]
    stroke = build_strokes((point,))[0]

    with pytest.raises(ValidationError):
        stroke.raw_order = ("changed",)
    with pytest.raises(TypeError):
        stroke.raw_order[0] = "changed"
