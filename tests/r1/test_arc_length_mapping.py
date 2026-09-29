from datiao.r1.mapper import QuestionMappingConfig, map_strokes_to_regions
from datiao.r1.models import QuestionRegion
from datiao.r1.parser import parse_raw_points
from datiao.r1.stroke import StrokeBuildConfig, build_strokes


def region(region_id, question_id, left, right, priority=0):
    return QuestionRegion(
        region_id=region_id,
        page_id="p1",
        question_id=question_id,
        region_type="rectangle",
        polygon_norm=((left, 0.0), (right, 0.0), (right, 1.0), (left, 1.0)),
        priority=priority,
    )


def points_for(xs):
    return parse_raw_points(
        tuple(
            {
                "point_id": f"p{i}",
                "session_id": "s1",
                "page_id": "p1",
                "x": x,
                "y": 0.5,
                "x_norm": x,
                "y_norm": 0.5,
                "timestamp_ms": i * 10,
                "sequence": i,
            }
            for i, x in enumerate(xs)
        )
    )


def map_points(points):
    strokes = build_strokes(points, StrokeBuildConfig(max_spatial_jump_norm=1.0))
    return map_strokes_to_regions(strokes, points, (region("a", "q-a", 0.0, 0.2), region("b", "q-b", 0.2, 1.0))), strokes


def test_arc_length_beats_point_count_for_uneven_sampling():
    points = points_for([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.90])
    mappings, _strokes = map_points(points)
    assert mappings[0].question_id == "q-b"
    assert mappings[0].mapping_method == "ARC_LENGTH"
    assert mappings[0].confidence > 0.5


def test_sampling_density_does_not_change_mapping():
    dense, _ = map_points(points_for([0.05, 0.10, 0.15, 0.25, 0.45, 0.65, 0.85]))
    sparse, _ = map_points(points_for([0.05, 0.25, 0.65, 0.85]))
    assert dense[0].question_id == sparse[0].question_id == "q-b"


def test_zero_length_uses_explicit_point_fallback():
    points = points_for([0.1, 0.1])
    mappings, _ = map_points(points)
    assert mappings[0].mapping_method == "POINT_FALLBACK"
    assert mappings[0].question_id == "q-a"


def test_equal_region_coverage_is_ambiguous():
    points = points_for([0.1, 0.9])
    strokes = build_strokes(points, StrokeBuildConfig(max_spatial_jump_norm=1.0))
    mappings = map_strokes_to_regions(
        strokes,
        points,
        (region("a", "q-a", 0.0, 0.5), region("b", "q-b", 0.5, 1.0)),
        config=QuestionMappingConfig(min_coverage=0.1, ambiguity_epsilon=0.01),
    )
    assert mappings[0].status == "AMBIGUOUS"
    assert mappings[0].mapping_method == "UNKNOWN"


def test_polygon_crossing_uses_partial_segment_length():
    points = parse_raw_points(
        (
            {"point_id": "p0", "session_id": "s1", "page_id": "p1", "x": 0.1, "y": 0.5, "x_norm": 0.1, "y_norm": 0.5, "timestamp_ms": 0, "sequence": 0},
            {"point_id": "p1", "session_id": "s1", "page_id": "p1", "x": 0.9, "y": 0.5, "x_norm": 0.9, "y_norm": 0.5, "timestamp_ms": 10, "sequence": 1},
        )
    )
    triangle = QuestionRegion(
        region_id="tri", page_id="p1", question_id="q-tri", region_type="polygon",
        polygon_norm=((0.2, 0.0), (0.8, 0.0), (0.5, 1.0)),
    )
    strokes = build_strokes(points, StrokeBuildConfig(max_spatial_jump_norm=1.0))
    mappings = map_strokes_to_regions(strokes, points, (triangle,), min_coverage=0.2)
    assert mappings[0].question_id == "q-tri"
    assert 0.2 < mappings[0].confidence < 0.8
