import pytest
from pydantic import ValidationError

from datiao.r1.models import NormalizedPoint, Point, StudentProcessEvent


def make_point(payload=None):
    return Point(
        schema_version="1.0.0",
        point_id="point-001",
        session_id="session-001",
        participant_id=None,
        task_segment_id=None,
        device_id="test-device",
        page_id="page-01",
        sequence=0,
        timestamp_ms=0,
        x_raw=1.0,
        y_raw=2.0,
        x_mm=None,
        y_mm=None,
        x_norm=None,
        y_norm=None,
        pressure_raw=None,
        pressure_norm=None,
        pen_state_raw=None,
        pen_state="UNKNOWN",
        source_file=None,
        source_index=0,
        raw_order=0,
        processed_order=0,
        source_payload=payload or {"z": 2, "nested": {"value": 1}},
    )


def test_point_is_immutable_including_source_payload():
    point = make_point()

    with pytest.raises(ValidationError):
        point.raw_index = 1
    with pytest.raises(TypeError):
        point.source_payload["new"] = "value"
    with pytest.raises(TypeError):
        point.source_payload["nested"]["value"] = 2


def test_source_payload_hash_is_stable_for_mapping_order():
    first = make_point({"a": 1, "b": [2, 3]})
    second = make_point({"b": [2, 3], "a": 1})

    assert first.source_payload_hash == second.source_payload_hash


def test_source_payload_hash_mismatch_is_rejected():
    with pytest.raises(ValidationError):
        Point(
            point_id="point-001",
            session_id="session-001",
            raw_index=0,
            normalized={"x": 1.0, "y": 2.0},
            source_payload={"a": 1},
            source_payload_hash="0" * 64,
        )


def test_forbidden_metadata_fields_are_rejected():
    with pytest.raises(ValidationError, match="out-of-scope field"):
        StudentProcessEvent(
            event_id="event-001",
            schema_version="1.0.0",
            session_id="session-001",
            event_type="UNKNOWN",
            sequence=0,
            start_time_ms=None,
            end_time_ms=None,
            quality_status="DEGRADED",
            quality_flags=("TIME_UNAVAILABLE",),
            metadata={"emotion": "unknown"},
        )


def test_event_model_has_no_forbidden_output_fields():
    forbidden = {"emotion", "attention_score", "careless", "wrong_reason"}
    assert forbidden.isdisjoint(StudentProcessEvent.model_fields)


def test_event_default_metadata_is_immutable():
    event = StudentProcessEvent(
        event_id="event-002",
        schema_version="1.0.0",
        session_id="session-001",
        event_type="UNKNOWN",
        sequence=0,
        start_time_ms=None,
        end_time_ms=None,
        quality_status="DEGRADED",
        quality_flags=("TIME_UNAVAILABLE",),
    )

    with pytest.raises(TypeError):
        event.metadata["extra"] = "value"
