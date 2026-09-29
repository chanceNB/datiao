import json
import math
import pytest
import jsonschema
from pydantic import ValidationError

from datiao.r1.mapper import map_strokes_to_regions
from datiao.r1.models import Point, QuestionRegion, StudentProcessEvent
from datiao.r1.models.legacy import legacy_rectangle
from datiao.r1.parser import parse_raw_points
from datiao.r1.stroke import build_strokes
from datiao.r1.synthetic import Scenario, default_regions, generate_synthetic_case
from datiao.r1.trace import TraceResolutionError, resolve_event_trace


def test_v1_models_reject_legacy_input_fields_and_versions():
    with pytest.raises(ValidationError):
        Point(point_id="p", session_id="s", raw_index=0, normalized={"x": 1.0, "y": 2.0}, source_payload={})
    with pytest.raises(ValidationError):
        StudentProcessEvent(event_id="e", event_type="UNKNOWN", session_id="s", schema_version="r1.v1", start_time_ms=None, end_time_ms=None)
    with pytest.raises(ValidationError):
        QuestionRegion(region_id="r", page_id="p", question_id="q", region_type="polygon", polygon_norm=((math.nan, 0.0), (0.5, 0.0), (0.0, 0.5)))


def test_normalized_mapping_uses_norm_coordinates_and_flags_missing_norm():
    records = ({"point_id": "p", "session_id": "s", "page_id": "page", "x": 100.0, "y": 100.0, "x_norm": 0.5, "y_norm": 0.5, "timestamp_ms": 0, "sequence": 0},)
    points = parse_raw_points(records)
    strokes = build_strokes(points)
    region = QuestionRegion(region_id="r", page_id="page", question_id="Q", region_type="polygon", polygon_norm=((0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)))
    assert map_strokes_to_regions(strokes, points, (region,))[0].status == "MAPPED"
    raw_only = parse_raw_points(({k: v for k, v in records[0].items() if k not in {"x_norm", "y_norm"}},))
    unknown = map_strokes_to_regions(build_strokes(raw_only), raw_only, (region,))[0]
    assert unknown.status == "UNKNOWN"
    assert "NO_NORMALIZED_COORDINATES" in unknown.quality_flags


def test_generated_regions_validate_as_v1_and_trace_checks_stroke_point_consistency():
    schema = json.loads(open("contracts/question_region.schema.json", encoding="utf-8").read())
    validator = jsonschema.Draft202012Validator(schema)
    for region in default_regions():
        validator.validate(region.model_dump(mode="json"))
    case = generate_synthetic_case(Scenario("trace_contract", "return_visit", seed=1))
    event = case.algorithm_output.events[0]
    bad = event.model_copy(update={"point_refs": ("missing-point",)})
    with pytest.raises(TraceResolutionError):
        resolve_event_trace(bad, case.strokes, case.raw_points)


def test_event_ref_legality_is_explicit():
    with pytest.raises(ValidationError):
        StudentProcessEvent(event_id="e", event_type="WRITING", session_id="s", start_time_ms=0, end_time_ms=0)
    assert StudentProcessEvent(event_id="e", event_type="UNKNOWN", session_id="s", start_time_ms=None, end_time_ms=None).point_refs == ()


def test_partial_norm_is_preserved_as_unknown_with_quality_flag():
    point = parse_raw_points(({"point_id": "p", "session_id": "s", "page_id": "p", "x": 1.0, "y": 2.0, "x_norm": 0.2, "timestamp_ms": 0, "sequence": 0},))[0]
    assert point.x_norm is None and point.y_norm is None
    assert "INVALID_NORM_COORDINATE" in point.quality_flags
