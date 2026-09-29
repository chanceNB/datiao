import pytest
from pydantic import ValidationError
from datiao.r1.models import StudentProcessEvent


def make_event(event_type="RETURN", **kwargs):
    values = dict(
        event_id="sim_evt_000001", event_type=event_type, session_id="sim_session_001",
        task_segment_id="sim_segment_practice_01", participant_id="sim_p_001", question_id="Q05",
        start_time_ms=18320, end_time_ms=18740, point_refs=("point_001231",), stroke_refs=("stroke_000087",),
        quality_status="VALID", algorithm_version="r1-event-rule-v0.1",
        provenance={"dataset_type": "synthetic", "generator_version": "pen-sim-v0.1", "seed": 20260929,
                    "scenario_id": "sim_return_q5_001", "ground_truth_source": "scenario_plan",
                    "manifest_hash": "sha256:" + "a" * 64},
    )
    values.update(kwargs)
    return StudentProcessEvent(**values)


def test_all_eight_event_types_are_accepted():
    for event_type in ("WRITING", "QUESTION_VISIT", "QUESTION_LEAVE", "RETURN", "REVISION_CANDIDATE", "PAGE_CHANGE", "PROCESS_END", "UNKNOWN"):
        assert make_event(event_type).event_type == event_type


def test_event_time_range_and_legacy_fields_are_not_standard_output():
    with pytest.raises(ValidationError):
        make_event(start_time_ms=10, end_time_ms=9)
    event = make_event()
    assert event.model_dump().keys() >= {"point_refs", "stroke_refs", "start_time_ms", "end_time_ms", "quality_status", "provenance"}
    assert "occurred_at_ms" not in event.model_dump()
    assert "source_point_ids" not in event.model_dump()


def test_synthetic_provenance_rejects_placeholder_hash_and_psychological_fields():
    with pytest.raises(ValidationError):
        make_event(provenance={"dataset_type": "synthetic", "generator_version": "v", "seed": 1,
                               "scenario_id": "s", "ground_truth_source": "scenario_plan",
                               "manifest_hash": "sha256:REPLACE_WITH_REAL_HASH"})
    with pytest.raises(ValidationError):
        make_event(metadata={"emotion": "anxious"})
from datiao.r1.event import detect_student_process_events
from datiao.r1.mapper import StrokeMapping


def test_detector_emits_contract_version_1_0_0():
    events = detect_student_process_events((StrokeMapping(
        stroke_id="s", session_id="session", page_id="page", question_id="Q",
        status="MAPPED", start_time_ms=0, end_time_ms=1, point_refs=("p",),
    ),))
    assert events[0].schema_version == "1.0.0"
