import pytest
from pydantic import ValidationError
from datiao.r1.models import QuestionRegion, Stroke, BoundingBox


def test_stroke_v1_refs_and_duration():
    stroke = Stroke(
        stroke_id="stroke_000001", session_id="sim_session_001", participant_id="sim_p_001",
        task_segment_id="sim_segment_practice_01", page_id="page_01",
        point_refs=("point_001", "point_002"), raw_order=("point_001", "point_002"),
        processed_order=("point_001", "point_002"), start_time_ms=10, end_time_ms=40,
        bbox=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
        algorithm_version="r1-stroke-v1", provenance={"source": "test"},
    )
    assert stroke.duration_ms == 30
    assert stroke.point_refs == stroke.raw_order
    assert stroke.start_time_ms <= stroke.end_time_ms


def test_question_region_v1_requires_normalized_polygon():
    region = QuestionRegion(
        region_id="region_q5", question_id="Q05", page_id="page_01",
        region_type="polygon", polygon_norm=((0.1, 0.1), (0.3, 0.1), (0.2, 0.3)),
    )
    assert region.contains(0.2, 0.2)
    with pytest.raises(ValidationError):
        QuestionRegion(
            region_id="bad", question_id="Q", page_id="p", region_type="polygon",
            polygon_norm=((0.0, 0.0), (1.2, 0.0), (0.0, 1.0)),
        )
