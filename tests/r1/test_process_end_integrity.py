import pytest

from datiao.r1.event import EventDetectionConfig, detect_student_process_events
from datiao.r1.mapper import StrokeMapping


def mapping(stroke_id, participant_id, start, end):
    return StrokeMapping(
        stroke_id=stroke_id,
        session_id="s1",
        participant_id=participant_id,
        page_id="p1",
        question_id="q1",
        status="MAPPED",
        start_time_ms=start,
        end_time_ms=end,
        point_refs=(f"{stroke_id}-p",),
    )


def test_legacy_eof_flag_is_removed_and_cannot_create_process_end():
    with pytest.raises(TypeError):
        EventDetectionConfig(emit_process_end=True)
    events = detect_student_process_events((mapping("s1", "A", 0, 10),))
    assert all(event.event_type != "PROCESS_END" for event in events)


def test_timeout_is_participant_aware():
    events = detect_student_process_events(
        (mapping("a", "A", 0, 1_000), mapping("b", "B", 0, 5_000)),
        config=EventDetectionConfig(process_end_timeout_ms=3_000),
        session_end_ms=7_000,
    )
    ends = [event for event in events if event.event_type == "PROCESS_END"]
    assert {event.participant_id for event in ends} == {"A"}
    assert ends[0].stroke_refs == ("a",)


def test_two_point_refs_without_stroke_do_not_create_writing():
    events = detect_student_process_events((mapping("s1", "A", 0, 10).model_copy(update={"point_refs": ("p1", "p2")}),), process_end_signal=False)
    assert [event.event_type for event in events] == ["QUESTION_VISIT"]
