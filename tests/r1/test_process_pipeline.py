import pytest
from pydantic import ValidationError

from datiao.r1.event import detect_student_process_events
from datiao.r1.mapper import StrokeMapping, map_strokes_to_regions
from datiao.r1.models import QuestionRegion
from datiao.r1.pipeline import run_r1_pipeline
from datiao.r1.parser import (
    CanonicalPointAdapterError,
    adapt_canonical_records,
    parse_raw_points,
)
from datiao.r1.stroke import StrokeBuildConfig, build_strokes
from datiao.r1.trace import TraceResolutionError, resolve_event_trace, resolve_event_traces
from datiao.r1.verification import (
    apply_verification_label,
    build_verification_template,
    summarize_verification,
)


def record(point_id, timestamp_ms, x, y, *, session_id="s-1", page_id="p-1", sequence=0):
    return {
        "point_id": point_id,
        "session_id": session_id,
        "page_id": page_id,
        "x": x,
        "y": y,
        "timestamp_ms": timestamp_ms,
        "sequence": sequence,
    }


def region(region_id, question_id, page_id="p-1", x=0.0, y=0.0, width=10.0, height=10.0, priority=0):
    return QuestionRegion(
        region_id=region_id,
        page_id=page_id,
        question_id=question_id,
        geometry_type="rectangle",
        coordinates=(x, y, width, height),
        priority=priority,
    )


def pipeline(actions):
    records = []
    for index, (page_id, x, y) in enumerate(actions):
        base = index * 2_000
        records.extend(
            [
                record(f"p-{index}-a", 1_000 + base, x, y, page_id=page_id, sequence=index * 2),
                record(f"p-{index}-b", 1_050 + base, x + 1, y + 1, page_id=page_id, sequence=index * 2 + 1),
            ]
        )
    points = parse_raw_points(records)
    strokes = build_strokes(points)
    regions = (
        region("r-q1", "q1"),
        region("r-q2", "q2", x=20.0),
        region("r-q3", "q3", page_id="p-2", x=0.0),
    )
    mappings = map_strokes_to_regions(strokes, points, regions)
    return points, strokes, mappings


def test_mapping_and_return_event_sequence_is_deterministic():
    points, strokes, mappings = pipeline(
        [("p-1", 2.0, 2.0), ("p-1", 22.0, 2.0), ("p-1", 2.0, 2.0)]
    )

    assert [mapping.question_id for mapping in mappings] == ["q1", "q2", "q1"]
    events = detect_student_process_events(mappings)

    assert [event.event_type for event in events] == [
        "QUESTION_VISIT",
        "QUESTION_LEAVE",
        "QUESTION_VISIT",
        "QUESTION_LEAVE",
        "RETURN",
        "PROCESS_END",
    ]
    assert events[4].question_id == "q1"
    assert events[4].source_stroke_ids == (strokes[2].stroke_id,)
    assert events[-1].source_point_ids == strokes[-1].raw_order

    traces = resolve_event_traces(events, strokes, points)
    assert all(trace.source_strokes for trace in traces)
    assert all(trace.source_points for trace in traces)


def test_page_change_and_unknown_mapping_are_explicit():
    points, strokes, mappings = pipeline(
        [("p-1", 2.0, 2.0), ("p-2", 2.0, 2.0), ("p-1", 100.0, 100.0)]
    )

    assert mappings[1].status == "MAPPED"
    assert mappings[1].question_id == "q3"
    assert mappings[2].status == "UNKNOWN"
    assert "NO_REGION_MATCH" in mappings[2].quality_flags
    events = detect_student_process_events(mappings)

    assert [event.event_type for event in events] == [
        "QUESTION_VISIT",
        "PAGE_CHANGE",
        "QUESTION_LEAVE",
        "QUESTION_VISIT",
        "PAGE_CHANGE",
        "QUESTION_LEAVE",
        "UNKNOWN",
        "PROCESS_END",
    ]
    assert events[1].metadata["from_page_id"] == "p-1"
    assert events[1].metadata["to_page_id"] == "p-2"


def test_same_question_new_stroke_is_only_a_revision_candidate():
    _points, _strokes, mappings = pipeline(
        [("p-1", 2.0, 2.0), ("p-1", 2.0, 2.0)]
    )

    events = detect_student_process_events(mappings)
    assert [event.event_type for event in events] == [
        "QUESTION_VISIT",
        "REVISION_CANDIDATE",
        "PROCESS_END",
    ]
    assert all("emotion" not in event.metadata for event in events)


def test_trace_rejects_missing_declared_source():
    points, strokes, mappings = pipeline([("p-1", 2.0, 2.0)])
    event = detect_student_process_events(mappings)[0].model_copy(
        update={"source_point_ids": ("missing-point",)}
    )

    with pytest.raises(TraceResolutionError, match="missing-point"):
        resolve_event_trace(event, strokes, points)


def test_canonical_adapter_rejects_unmapped_device_aliases():
    with pytest.raises(CanonicalPointAdapterError, match="point_id"):
        adapt_canonical_records(
            ({"id": "p1", "time": 1, "page": "p1", "x": 1, "y": 1},)
        )

    records = ({
        "point_id": "p1",
        "session_id": "s1",
        "page_id": "page1",
        "x": 1,
        "y": 1,
        "timestamp_ms": 1,
        "sequence": 0,
        "device_type": "example",
    },)
    adapted = adapt_canonical_records(records)
    assert adapted[0]["device_type"] == "example"


def test_timestamp_disorder_without_sequence_is_quality_flagged():
    points = parse_raw_points(
        (
            record("a", 1_200, 1, 1, sequence=0),
            record("b", 1_100, 2, 2, sequence=0),
        )
    )

    assert "OUT_OF_ORDER" in points[1].quality_flags


def test_stroke_gap_config_requires_integer():
    with pytest.raises(ValueError, match="integer"):
        StrokeBuildConfig(max_time_gap_ms=1.5)


def test_question_region_validates_geometry():
    with pytest.raises(ValidationError, match="rectangle coordinates"):
        QuestionRegion(
            region_id="r1",
            page_id="p1",
            question_id="q1",
            geometry_type="rectangle",
            coordinates=(0.0, 0.0, 1.0),
        )

    polygon = QuestionRegion(
        region_id="poly",
        page_id="p1",
        question_id="q1",
        geometry_type="polygon",
        coordinates=(0.0, 0.0, 10.0, 0.0, 5.0, 10.0),
    )
    assert polygon.contains(5.0, 5.0)
    assert not polygon.contains(0.0, 11.0)


def test_unified_pipeline_propagates_contract_fields_and_trace():
    records = (
        record("p-1", 1_000, 2.0, 2.0, sequence=0),
        record("p-2", 1_050, 3.0, 3.0, sequence=1),
    )
    result = run_r1_pipeline(
        records,
        (region("r-q1", "q1"),),
        "s-1",
        task_segment_id="segment-7",
        data_version="points-v2",
        source_provenance={"fixture": "pipeline"},
    )

    assert result.quality_status == "OK"
    assert result.points and result.strokes and result.stroke_mappings
    assert [event.event_type for event in result.student_process_events] == [
        "QUESTION_VISIT",
        "PROCESS_END",
    ]
    assert all(event.task_segment_id == "segment-7" for event in result.student_process_events)
    assert all(event.data_version == "points-v2" for event in result.student_process_events)
    assert result.student_process_events[0].source_provenance["fixture"] == "pipeline"
    assert len(result.event_traces) == len(result.student_process_events)
    assert result.page_replay.frames[0].stroke_id == result.strokes[0].stroke_id


def test_unified_pipeline_empty_input_is_invalid_but_replay_is_serializable():
    result = run_r1_pipeline((), (), "s-empty")

    assert result.quality_status == "INVALID"
    assert result.quality_flags == ("EMPTY_INPUT",)
    assert result.points == ()
    assert result.page_replay.session_id == "s-empty"
    assert result.page_replay.frames == ()


def test_unified_pipeline_unknown_mapping_is_degraded_and_keeps_sources():
    records = (
        record("p-1", 1_000, 100.0, 100.0, sequence=0),
        record("p-2", 1_050, 101.0, 101.0, sequence=1),
    )
    result = run_r1_pipeline(records, (region("r-q1", "q1"),), "s-1")

    assert result.quality_status == "DEGRADED"
    assert result.stroke_mappings[0].status == "UNKNOWN"
    assert [event.event_type for event in result.student_process_events] == [
        "UNKNOWN",
        "PROCESS_END",
    ]
    assert result.student_process_events[0].source_point_ids == ("p-1", "p-2")


def test_unified_pipeline_records_trace_failures_without_dropping_events():
    records = (
        record("p-1", 1_000, 2.0, 2.0, sequence=0),
        record("p-1", 1_050, 3.0, 3.0, sequence=1),
    )
    result = run_r1_pipeline(records, (region("r-q1", "q1"),), "s-1")

    assert result.quality_status == "DEGRADED"
    assert "TRACE_ERROR" in result.quality_flags
    assert result.trace_errors
    assert result.student_process_events
    assert not result.event_traces


def test_manual_verification_defaults_uncertain_and_summarizes_labels():
    result = run_r1_pipeline(
        (record("p-1", 1_000, 2.0, 2.0, sequence=0),),
        (region("r-q1", "q1"),),
        "s-1",
    )
    template = build_verification_template(result.student_process_events)
    assert template and all(row.label == "UNCERTAIN" for row in template)
    labeled = apply_verification_label(template, template[0].event_id, "YES", reviewer="r4")
    summary = summarize_verification(labeled)
    assert summary.yes == 1
    assert summary.uncertain == len(template) - 1
