import pytest

from datiao.r1.event import EventDetectionConfig, detect_student_process_events
from datiao.r1.mapper import map_strokes_to_regions
from datiao.r1.models import QuestionRegion, legacy_rectangle, validate_canonical_point_for_export
from datiao.r1.parser import parse_raw_points
from datiao.r1.pipeline import run_r1_pipeline
from datiao.r1.stroke import StrokeBuildConfig, build_strokes
from datiao.r1.synthetic import Scenario, generate_synthetic_case


def norm_region(region_id, page_id, question_id, left):
    return QuestionRegion(region_id=region_id, page_id=page_id, question_id=question_id, region_type="rectangle", polygon_norm=((left, 0.0), (left + 0.1, 0.0), (left + 0.1, 0.2), (left, 0.2)))


def raw_record(index, participant, segment, page, x, timestamp, device="device-a"):
    return {
        "point_id": f"{participant}-{segment}-{index}",
        "session_id": "s1",
        "participant_id": participant,
        "task_segment_id": segment,
        "device_id": device,
        "page_id": page,
        "x": x,
        "y": 5.0,
        "timestamp_ms": timestamp,
        "sequence": index,
    }


def action(index, participant, segment, page, x, timestamp, device="device-a"):
    return [raw_record(index * 2, participant, segment, page, x, timestamp, device), raw_record(index * 2 + 1, participant, segment, page, x + 1, timestamp + 1, device)]


def run(records, regions, **kwargs):
    return run_r1_pipeline(records, regions, "s1", stroke_config=StrokeBuildConfig(max_spatial_jump_norm=1.0), **kwargs)


def test_different_devices_split_strokes_and_record_domain():
    points = parse_raw_points(action(0, "A", "S1", "p1", 5, 0, "device-a") + action(1, "A", "S1", "p1", 5, 10, "device-b"))
    strokes = build_strokes(points, StrokeBuildConfig(max_spatial_jump_norm=1.0))
    assert len(strokes) == 2
    assert [stroke.provenance["device_id"] for stroke in strokes] == ["device-a", "device-b"]
    assert [stroke.provenance["raw_coordinate_domain"] for stroke in strokes] == ["device:device-a", "device:device-b"]


def test_raw_bbox_overlap_across_devices_does_not_create_revision():
    records = action(0, "A", "S1", "p1", 5, 0, "device-a") + action(1, "A", "S1", "p1", 5, 5_000, "device-b")
    regions = (legacy_rectangle(region_id="q1", page_id="p1", question_id="q1", x=0, y=0, width=20, height=20),)
    result = run(records, regions)
    assert not any(event.event_type == "REVISION_CANDIDATE" for event in result.student_process_events)


def test_unknown_device_raw_bbox_cannot_drive_revision():
    records = action(0, "A", "S1", "p1", 5, 0, None) + action(1, "A", "S1", "p1", 5, 5_000, None)
    regions = (legacy_rectangle(region_id="q1", page_id="p1", question_id="q1", x=0, y=0, width=20, height=20),)
    result = run(records, regions)
    assert not any(event.event_type == "REVISION_CANDIDATE" for event in result.student_process_events)


def test_same_device_raw_overlap_can_still_be_revision():
    records = action(0, "A", "S1", "p1", 5, 0, "device-a") + action(1, "A", "S1", "p1", 5, 5_000, "device-a")
    regions = (legacy_rectangle(region_id="q1", page_id="p1", question_id="q1", x=0, y=0, width=20, height=20),)
    result = run(records, regions)
    assert any(event.event_type == "REVISION_CANDIDATE" for event in result.student_process_events)


def test_task_segments_have_independent_visit_return_page_and_revision_state():
    regions = (
        legacy_rectangle(region_id="q1", page_id="p1", question_id="q1", x=0, y=0, width=20, height=20),
        legacy_rectangle(region_id="q2", page_id="p1", question_id="q2", x=20, y=0, width=20, height=20),
        legacy_rectangle(region_id="q1p2", page_id="p2", question_id="q1", x=0, y=0, width=20, height=20),
    )
    result = run(action(0, "A", "S1", "p1", 5, 0) + action(1, "A", "S2", "p1", 5, 2_000), regions)
    visits = [event for event in result.student_process_events if event.event_type in {"QUESTION_VISIT", "RETURN"}]
    assert [event.event_type for event in visits] == ["QUESTION_VISIT", "QUESTION_VISIT"]
    assert [event.task_segment_id for event in visits] == ["S1", "S2"]

    same_segment = run(action(0, "A", "S1", "p1", 5, 0) + action(1, "A", "S1", "p1", 25, 2_000) + action(2, "A", "S1", "p1", 5, 4_000), regions)
    assert any(event.event_type == "RETURN" and event.task_segment_id == "S1" for event in same_segment.student_process_events)

    page_first = run(action(0, "A", "S1", "p1", 5, 0) + action(1, "A", "S2", "p2", 5, 2_000), regions)
    assert not any(event.event_type == "PAGE_CHANGE" and event.task_segment_id == "S2" for event in page_first.student_process_events)


def test_segment_prior_ink_does_not_create_cross_segment_revision():
    regions = (norm_region("q1", "p1", "q1", 0.0),)
    result = run(action(0, "A", "S1", "p1", 5, 0) + action(1, "A", "S2", "p1", 5, 5_000), regions)
    assert not any(event.event_type == "REVISION_CANDIDATE" for event in result.student_process_events)


def test_mapping_segment_is_event_source_and_conflict_is_rejected():
    regions = (norm_region("q1", "p1", "q1", 0.0),)
    result = run(action(0, "A", "S1", "p1", 5, 0), regions)
    assert all(event.task_segment_id == "S1" for event in result.student_process_events)
    with pytest.raises(ValueError, match="context mismatch"):
        run(action(0, "A", "S1", "p1", 5, 0), regions, task_segment_id="S2")


def test_explicit_end_and_timeout_are_segment_scoped():
    mappings = []
    for stroke_id, participant, segment, end in (("a", "A", "S1", 1_000), ("b", "A", "S2", 5_000), ("c", "B", "S1", 1_000)):
        from datiao.r1.mapper import StrokeMapping

        mappings.append(StrokeMapping(stroke_id=stroke_id, session_id="s1", participant_id=participant, task_segment_id=segment, page_id="p1", question_id="q1", status="MAPPED", start_time_ms=end - 1, end_time_ms=end, point_refs=(stroke_id,)))
    explicit = detect_student_process_events(tuple(mappings), process_end_signal=True)
    assert {(event.participant_id, event.task_segment_id) for event in explicit if event.event_type == "PROCESS_END"} == {("A", "S1"), ("A", "S2"), ("B", "S1")}
    timeout = detect_student_process_events(tuple(mappings), config=EventDetectionConfig(process_end_timeout_ms=3_000), session_end_ms=7_000)
    assert {(event.participant_id, event.task_segment_id) for event in timeout if event.event_type == "PROCESS_END"} == {("A", "S1"), ("B", "S1")}


def test_synthetic_device_id_passes_canonical_export_gate():
    case = generate_synthetic_case(Scenario("device", "sequential_visit", seed=3))
    assert {point.device_id for point in case.raw_points} == {"sim_pen_001"}
    assert case.manifest.device_id == "sim_pen_001"
    assert all(stroke.provenance["device_id"] == "sim_pen_001" for stroke in case.strokes)
    assert all(validate_canonical_point_for_export(point) == point for point in case.raw_points)
