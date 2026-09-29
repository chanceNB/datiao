import pytest
from pydantic import ValidationError
from datiao.r1.models import Point
from datiao.r1.parser import parse_raw_points


def test_canonical_synthetic_point_preserves_raw_and_distinguishes_pressure_zero():
    point = parse_raw_points(({
        "point_id": "point_001",
        "session_id": "sim_session_001",
        "participant_id": "sim_p_001",
        "task_segment_id": "sim_segment_practice_01",
        "device_id": "wifi_pen",
        "page_id": "page_01",
        "x": 12.5,
        "y": 8.0,
        "timestamp_ms": 10,
        "sequence": 0,
        "pressure": 0,
        "pen_state": "DOWN",
        "source_file": "fixture.jsonl",
        "source_index": 0,
    },))[0]
    assert point.x_raw == 12.5 and point.y_raw == 8.0
    assert point.x_mm is None and point.x_norm is None
    assert point.pressure_raw == 0
    assert point.pressure_norm is None
    assert point.participant_id == "sim_p_001"
    assert point.source_payload["x"] == 12.5
    assert point.source_index == 0


def test_invalid_norm_and_missing_required_field_are_rejected():
    with pytest.raises(ValidationError):
        Point(
            schema_version="1.0.0", point_id="p", session_id="s", participant_id=None,
            task_segment_id=None, device_id="d", page_id="page", sequence=0,
            timestamp_ms=0, x_raw=1.0, y_raw=2.0, x_mm=None, y_mm=None,
            x_norm=1.2, y_norm=0.5, pressure_raw=None, pressure_norm=None,
            pen_state_raw=None, pen_state="UNKNOWN", source_file=None,
            source_index=0, raw_order=0, processed_order=0,
            source_payload={}, source_payload_hash="" + "0" * 64,
        )

    with pytest.raises(ValidationError):
        Point(schema_version="1.0.0", point_id="p", session_id="s")


def test_unknown_pen_state_is_explicit():
    point = Point(
        schema_version="1.0.0", point_id="p", session_id="s", participant_id=None,
        task_segment_id=None, device_id="d", page_id=None, sequence=0,
        timestamp_ms=0, x_raw=1.0, y_raw=2.0, x_mm=None, y_mm=None,
        x_norm=None, y_norm=None, pressure_raw=None, pressure_norm=None,
        pen_state_raw="vendor_type", pen_state="UNKNOWN", source_file=None,
        source_index=0, raw_order=0, processed_order=0, source_payload={},
    )
    assert point.pen_state == "UNKNOWN"
from datiao.r1.parser import adapt_canonical_records


def test_canonical_adapter_accepts_explicit_v1_raw_coordinates():
    records = adapt_canonical_records(({
        "point_id": "p", "session_id": "s", "page_id": "page", "x_raw": 1.0,
        "y_raw": 2.0, "timestamp_ms": 0, "sequence": 0,
    },))
    assert records[0]["x_raw"] == 1.0
