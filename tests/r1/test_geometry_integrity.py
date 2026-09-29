from datiao.r1.parser import parse_raw_points
from datiao.r1.stroke import StrokeBuildConfig, build_strokes


def record(point_id, *, x_raw, y_raw, x_norm=None, y_norm=None, timestamp=0, sequence=0, device_id=None):
    return {
        "point_id": point_id,
        "session_id": "s1",
        "participant_id": "p1",
        "device_id": device_id,
        "page_id": "page1",
        "x": x_raw,
        "y": y_raw,
        "x_raw": x_raw,
        "y_raw": y_raw,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "timestamp_ms": timestamp,
        "sequence": sequence,
    }


def test_bbox_chooses_one_complete_coordinate_space_without_mixing():
    points = parse_raw_points(
        (
            record("p1", x_raw=1000.0, y_raw=2000.0, x_norm=0.1, y_norm=0.2),
            record("p2", x_raw=1100.0, y_raw=2100.0, timestamp=10, sequence=1),
        )
    )
    stroke = build_strokes(points, StrokeBuildConfig(max_spatial_jump_norm=1.0))[0]
    assert stroke.provenance["bbox_coordinate_space"] == "raw"
    assert stroke.bbox.x == 1000.0
    assert stroke.bbox.width == 100.0


def test_bbox_is_unavailable_when_neither_space_is_complete():
    points = parse_raw_points(
        (
            record("p1", x_raw=None, y_raw=None, x_norm=0.1, y_norm=0.2),
            record("p2", x_raw=1100.0, y_raw=2100.0, timestamp=10, sequence=1),
        )
    )
    stroke = build_strokes(points, StrokeBuildConfig(max_spatial_jump_norm=1.0))[0]
    assert stroke.provenance["bbox_coordinate_space"] == "unavailable"


def test_raw_bbox_is_allowed_when_all_points_share_raw_fallback():
    points = parse_raw_points(
        (
            record("p1", x_raw=1000.0, y_raw=2000.0, timestamp=0),
            record("p2", x_raw=1100.0, y_raw=2100.0, timestamp=10, sequence=1),
        )
    )
    stroke = build_strokes(points, StrokeBuildConfig(max_spatial_jump_norm=1.0))[0]
    assert stroke.provenance["bbox_coordinate_space"] == "raw"
