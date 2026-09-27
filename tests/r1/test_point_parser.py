import pytest
from pydantic import ValidationError

from datiao.r1.parser import parse_raw_points


def record(point_id, *, timestamp_ms=1000, sequence=0, x=10, y=20, session_id="s-1", page_id="p-1"):
    return {
        "point_id": point_id,
        "session_id": session_id,
        "page_id": page_id,
        "x": x,
        "y": y,
        "timestamp_ms": timestamp_ms,
        "sequence": sequence,
        "device": {"source": "fixture"},
    }


def test_parser_returns_immutable_points_and_preserves_payload():
    source = record("point-1")
    points = parse_raw_points((source,))

    assert len(points) == 1
    assert points[0].source_payload["device"]["source"] == "fixture"
    assert len(points[0].source_payload_hash) == 64
    with pytest.raises(TypeError):
        points[0].source_payload["x"] = 99


def test_parser_keeps_missing_coordinate_record_with_quality_flag():
    points = parse_raw_points((record("point-1", x=None),))

    assert len(points) == 1
    assert "INVALID_COORDINATE" in points[0].quality_flags
    assert points[0].normalized.x is None


def test_parser_marks_missing_timestamp_without_dropping_point():
    points = parse_raw_points((record("point-1", timestamp_ms=None),))

    assert len(points) == 1
    assert "MISSING_TIMESTAMP" in points[0].quality_flags
    assert points[0].normalized.timestamp_ms is None


def test_parser_marks_duplicate_ids_on_all_duplicate_records():
    points = parse_raw_points((record("same"), record("same", sequence=1)))

    assert len(points) == 2
    assert all("DUPLICATE_POINT" in point.quality_flags for point in points)


def test_parser_marks_sequence_out_of_order():
    points = parse_raw_points((record("point-1", sequence=2), record("point-2", sequence=1)))

    assert "OUT_OF_ORDER" not in points[0].quality_flags
    assert "OUT_OF_ORDER" in points[1].quality_flags


def test_empty_input_returns_empty_tuple():
    assert parse_raw_points(()) == ()
