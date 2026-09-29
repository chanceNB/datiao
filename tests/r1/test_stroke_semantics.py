from datiao.r1.parser import parse_raw_points
from datiao.r1.stroke import StrokeBuildConfig, build_strokes


def point(point_id, timestamp, x, y, *, page="p1", participant="u1", pen_state="UNKNOWN", sequence=0):
    return {
        "point_id": point_id,
        "session_id": "s1",
        "participant_id": participant,
        "page_id": page,
        "x": x,
        "y": y,
        "x_norm": x,
        "y_norm": y,
        "timestamp_ms": timestamp,
        "sequence": sequence,
        "pen_state": pen_state,
    }


def test_pen_state_groups_down_move_up_and_starts_next_contact():
    points = parse_raw_points(
        (
            point("a", 0, 0.1, 0.1, pen_state="DOWN"),
            point("b", 10, 0.11, 0.1, pen_state="MOVE", sequence=1),
            point("c", 20, 0.12, 0.1, pen_state="UP", sequence=2),
            point("d", 30, 0.13, 0.1, pen_state="DOWN", sequence=3),
            point("e", 40, 0.14, 0.1, pen_state="UP", sequence=4),
        )
    )
    strokes = build_strokes(points, StrokeBuildConfig(max_spatial_jump_norm=1.0))
    assert [stroke.raw_order for stroke in strokes] == [("a", "b", "c"), ("d", "e")]
    assert strokes[0].algorithm_version == "r1-stroke-rule-v2"
    assert strokes[0].provenance["path_length_norm"] > 0


def test_spatial_jump_splits_when_time_is_continuous():
    points = parse_raw_points((point("a", 0, 0.1, 0.1), point("b", 10, 0.8, 0.8, sequence=1)))
    strokes = build_strokes(points, StrokeBuildConfig(max_spatial_jump_norm=0.2))
    assert len(strokes) == 2


def test_participant_and_task_context_do_not_mix():
    points = parse_raw_points(
        (
            point("a", 0, 0.1, 0.1),
            point("b", 10, 0.11, 0.1, participant="u2", sequence=1),
        )
    )
    assert len(build_strokes(points, StrokeBuildConfig(max_spatial_jump_norm=1.0))) == 2


def test_missing_timestamp_is_retained_and_does_not_create_fake_time():
    points = parse_raw_points((point("a", None, 0.1, 0.1), point("b", 10, 0.11, 0.1, sequence=1)))
    strokes = build_strokes(points, StrokeBuildConfig(max_spatial_jump_norm=1.0))
    assert len(strokes) == 2
    assert points[0].timestamp_ms is None
    assert "MISSING_TIMESTAMP" in points[0].quality_flags
