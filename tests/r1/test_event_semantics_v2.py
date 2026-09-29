from datiao.r1.event import EventDetectionConfig, detect_student_process_events
from datiao.r1.mapper import map_strokes_to_regions
from datiao.r1.models import QuestionRegion
from datiao.r1.parser import parse_raw_points
from datiao.r1.pipeline import run_r1_pipeline
from datiao.r1.stroke import StrokeBuildConfig, build_strokes


def records(actions):
    output = []
    for i, (x, y, timestamp) in enumerate(actions):
        output.extend(
            [
                {"point_id": f"p{i}a", "session_id": "s1", "page_id": "p1", "x": x, "y": y, "x_norm": x / 100, "y_norm": y / 100, "timestamp_ms": timestamp, "sequence": i * 2},
                {"point_id": f"p{i}b", "session_id": "s1", "page_id": "p1", "x": x + 1, "y": y + 1, "x_norm": (x + 1) / 100, "y_norm": (y + 1) / 100, "timestamp_ms": timestamp + 10, "sequence": i * 2 + 1},
            ]
        )
    return output


def q1_region():
    return QuestionRegion(region_id="q1", page_id="p1", question_id="q1", region_type="rectangle", polygon_norm=((0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)))


def event_types(raw, *, config=None, end=False, session_end_ms=None):
    result = run_r1_pipeline(raw, (q1_region(),), "s1", stroke_config=StrokeBuildConfig(max_spatial_jump_norm=1.0), event_config=config, process_end_signal=end, session_end_ms=session_end_ms)
    return result, [event.event_type for event in result.student_process_events]


def test_writing_and_question_visit_are_both_emitted():
    result, types = event_types(records([(5, 5, 0)]))
    assert types == ["WRITING", "QUESTION_VISIT"]
    assert result.student_process_events[0].algorithm_version == "r1-event-rule-v0.2"


def test_normal_same_question_multistroke_is_not_revision():
    _result, types = event_types(records([(5, 5, 0), (6, 6, 1_200)]))
    assert types == ["WRITING", "QUESTION_VISIT", "WRITING"]


def test_revision_requires_pause_and_overlap():
    _result, types = event_types(records([(5, 5, 0), (5, 5, 5_000)]), config=EventDetectionConfig(revision_pause_threshold_ms=3_000, revision_overlap_threshold=0.1))
    assert types == ["WRITING", "QUESTION_VISIT", "WRITING", "REVISION_CANDIDATE"]


def test_revision_negative_cases_do_not_overdetect():
    _result, types = event_types(records([(5, 5, 0), (15, 15, 5_000)]))
    assert "REVISION_CANDIDATE" not in types
    _result, types = event_types(records([(5, 5, 0), (5, 5, 100)]), config=EventDetectionConfig(revision_pause_threshold_ms=3_000))
    assert "REVISION_CANDIDATE" not in types


def test_process_end_requires_explicit_signal_or_external_timeout():
    _result, types = event_types(records([(5, 5, 0)]))
    assert "PROCESS_END" not in types
    _result, types = event_types(records([(5, 5, 0)]), end=True)
    assert types[-1] == "PROCESS_END"
    _result, types = event_types(records([(5, 5, 0)]), config=EventDetectionConfig(process_end_timeout_ms=1_000), session_end_ms=2_000)
    assert types[-1] == "PROCESS_END"


def test_unknown_mapping_can_still_emit_writing_but_keeps_unknown():
    raw = records([(80, 80, 0)])
    result = run_r1_pipeline(raw, (q1_region(),), "s1", stroke_config=StrokeBuildConfig(max_spatial_jump_norm=1.0))
    assert [event.event_type for event in result.student_process_events] == ["WRITING", "UNKNOWN"]


def test_page_transition_is_a_question_leave_even_when_question_id_repeats():
    raw = records([(5, 5, 0)]) + [
        {**item, "point_id": item["point_id"].replace("p0", "p1"), "page_id": "p2", "timestamp_ms": item["timestamp_ms"] + 1000, "sequence": item["sequence"] + 2}
        for item in records([(5, 5, 0)])
    ]
    regions = (
        q1_region(),
        QuestionRegion(region_id="q1-p2", page_id="p2", question_id="q1", region_type="rectangle", polygon_norm=((0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2))),
    )
    result = run_r1_pipeline(raw, regions, "s1", stroke_config=StrokeBuildConfig(max_spatial_jump_norm=1.0))
    assert [event.event_type for event in result.student_process_events] == ["WRITING", "QUESTION_VISIT", "PAGE_CHANGE", "WRITING", "QUESTION_LEAVE", "RETURN", "REVISION_CANDIDATE"]
