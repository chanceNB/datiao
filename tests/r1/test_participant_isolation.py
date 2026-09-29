from datiao.r1.pipeline import run_r1_pipeline
from datiao.r1.models import QuestionRegion
from datiao.r1.stroke import StrokeBuildConfig


def region(region_id, page_id, question_id, left):
    return QuestionRegion(
        region_id=region_id,
        page_id=page_id,
        question_id=question_id,
        region_type="rectangle",
        polygon_norm=((left, 0.0), (left + 0.1, 0.0), (left + 0.1, 0.2), (left, 0.2)),
    )


REGIONS = (
    region("p1-q1", "p1", "q1", 0.0),
    region("p1-q2", "p1", "q2", 0.2),
    region("p2-q1", "p2", "q1", 0.0),
)


def point(index, participant, page, x, timestamp, *, sequence=None):
    return {
        "point_id": f"{participant or 'unknown'}-{index}",
        "session_id": "session",
        "participant_id": participant,
        "page_id": page,
        "x": x,
        "y": 5.0,
        "x_norm": x / 100.0,
        "y_norm": 0.05,
        "timestamp_ms": timestamp,
        "sequence": index if sequence is None else sequence,
    }


def action(index, participant, page, x, timestamp):
    return [point(index * 2, participant, page, x, timestamp), point(index * 2 + 1, participant, page, x + 1, timestamp + 1)]


def run(records, *, end=False, session_end_ms=None, timeout=None):
    from datiao.r1.event import EventDetectionConfig

    return run_r1_pipeline(
        records,
        REGIONS,
        "session",
        stroke_config=StrokeBuildConfig(max_spatial_jump_norm=1.0),
        process_end_signal=end,
        session_end_ms=session_end_ms,
        event_config=EventDetectionConfig(process_end_timeout_ms=timeout) if timeout is not None else None,
    )


def events_for(result, participant, event_type=None):
    return [event for event in result.student_process_events if event.participant_id == participant and (event_type is None or event.event_type == event_type)]


def test_interleaved_participants_have_independent_visit_leave_state():
    result = run(action(0, "A", "p1", 5, 0) + action(1, "B", "p1", 5, 10) + action(2, "A", "p1", 25, 20) + action(3, "B", "p1", 25, 30))
    assert [event.event_type for event in events_for(result, "A") if event.event_type in {"QUESTION_VISIT", "QUESTION_LEAVE", "RETURN"}] == ["QUESTION_VISIT", "QUESTION_LEAVE", "QUESTION_VISIT"]
    assert [event.event_type for event in events_for(result, "B") if event.event_type in {"QUESTION_VISIT", "QUESTION_LEAVE", "RETURN"}] == ["QUESTION_VISIT", "QUESTION_LEAVE", "QUESTION_VISIT"]
    assert all(event.participant_id in {"A", "B"} for event in result.student_process_events)


def test_return_and_revision_history_do_not_cross_participants():
    records = action(0, "A", "p1", 5, 0) + action(1, "B", "p1", 5, 10) + action(2, "A", "p1", 25, 20) + action(3, "B", "p1", 6, 30) + action(4, "A", "p1", 5, 5_000)
    result = run(records)
    assert any(event.event_type == "RETURN" and event.participant_id == "A" and event.question_id == "q1" for event in result.student_process_events)
    assert not any(event.event_type == "RETURN" and event.participant_id == "B" for event in result.student_process_events)
    assert any(event.event_type == "REVISION_CANDIDATE" and event.participant_id == "A" for event in result.student_process_events)
    assert not any(event.event_type == "REVISION_CANDIDATE" and event.participant_id == "B" for event in result.student_process_events)


def test_page_change_is_scoped_to_participant():
    result = run(action(0, "A", "p1", 5, 0) + action(1, "B", "p1", 5, 10) + action(2, "A", "p2", 5, 20) + action(3, "B", "p1", 25, 30))
    assert any(event.event_type == "PAGE_CHANGE" and event.participant_id == "A" for event in result.student_process_events)
    assert not any(event.event_type == "PAGE_CHANGE" and event.participant_id == "B" for event in result.student_process_events)


def test_explicit_end_is_emitted_for_each_observed_participant():
    result = run(action(0, "A", "p1", 5, 0) + action(1, "B", "p1", 5, 10), end=True)
    ends = events_for(result, "A", "PROCESS_END") + events_for(result, "B", "PROCESS_END")
    assert {event.participant_id for event in ends} == {"A", "B"}
    assert all(event.stroke_refs for event in ends)


def test_unknown_participant_bucket_does_not_mix_with_known_participant():
    result = run(action(0, None, "p1", 5, 0) + action(1, "A", "p1", 5, 10))
    assert len(events_for(result, None, "QUESTION_VISIT")) == 1
    assert len(events_for(result, "A", "QUESTION_VISIT")) == 1


def test_same_input_has_deterministic_merged_event_order():
    records = action(0, "A", "p1", 5, 0) + action(1, "B", "p1", 5, 10) + action(2, "A", "p1", 25, 20)
    first = run(records).student_process_events
    second = run(records).student_process_events
    assert first == second
