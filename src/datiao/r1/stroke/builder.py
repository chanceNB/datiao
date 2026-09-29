"""Build strokes from Points while preserving raw input order."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import BoundingBox, Point, Stroke
from .geometry import coordinate_pair, path_length, point_distance
from .quality import point_sequence, point_timestamp, stroke_quality_flags


@dataclass(frozen=True, slots=True)
class StrokeBuildConfig:
    """Deterministic segmentation thresholds (development defaults)."""

    max_time_gap_ms: int = 1_000
    max_spatial_jump_norm: float = 0.15
    use_pen_state: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_time_gap_ms, bool)
            or not isinstance(self.max_time_gap_ms, int)
            or self.max_time_gap_ms < 0
        ):
            raise ValueError("max_time_gap_ms must be a non-negative integer")
        if isinstance(self.max_spatial_jump_norm, bool) or not isinstance(self.max_spatial_jump_norm, (int, float)) or self.max_spatial_jump_norm < 0:
            raise ValueError("max_spatial_jump_norm must be a non-negative number")


def _same_context(previous: Point, current: Point) -> bool:
    return (
        previous.session_id == current.session_id
        and previous.participant_id == current.participant_id
        and previous.task_segment_id == current.task_segment_id
        and previous.page_id == current.page_id
    )


def _time_continuous(previous: Point, current: Point, config: StrokeBuildConfig) -> bool:
    previous_time = point_timestamp(previous)
    current_time = point_timestamp(current)
    if previous_time is None or current_time is None:
        # Missing time makes this rule unavailable; context, pen-state and
        # spatial continuity still decide whether the points stay together.
        return True
    return abs(current_time - previous_time) <= config.max_time_gap_ms


def _spatial_continuous(previous: Point, current: Point, config: StrokeBuildConfig) -> bool:
    result = point_distance(previous, current, "norm")
    return result.distance is None or result.distance <= config.max_spatial_jump_norm


def _pen_boundary(previous: Point, current: Point, *, contact_open: bool, config: StrokeBuildConfig) -> tuple[bool, bool]:
    """Apply DOWN/MOVE/UP as a contact state machine.

    A DOWN starts a contact only after a closed contact.  MOVE continues it,
    and UP closes it after being included in the current Stroke.  UNKNOWN
    falls back to time/spatial/context rules.
    """

    if not config.use_pen_state or (previous.pen_state == "UNKNOWN" and current.pen_state == "UNKNOWN"):
        return False, contact_open
    if current.pen_state == "DOWN":
        if previous.pen_state == "UP":
            return True, True
        return False, True
    if current.pen_state == "MOVE":
        return False, contact_open
    if current.pen_state == "UP":
        return False, False
    return False, contact_open


def _processed_order(points: tuple[Point, ...]) -> tuple[str, ...]:
    if any(point_timestamp(point) is None for point in points):
        # With incomplete time, input order is the only deterministic order
        # that does not invent chronology across the missing sample.
        return tuple(point.point_id for point in points)
    return tuple(
        point.point_id
        for point in sorted(
            points,
            key=lambda point: (
                point_timestamp(point) is None,
                point_timestamp(point) if point_timestamp(point) is not None else 0,
                point_sequence(point) if point_sequence(point) is not None else float("inf"),
                point.raw_index,
            ),
        )
    )


def _bbox(points: tuple[Point, ...]) -> tuple[BoundingBox, str]:
    valid = [point for point in points if coordinate_pair(point, "norm") is not None or coordinate_pair(point, "raw") is not None]
    norm_coordinates = [coordinate_pair(point, "norm") for point in valid]
    raw_coordinates = [coordinate_pair(point, "raw") for point in valid]
    if valid and all(value is not None for value in norm_coordinates):
        coordinates = [value for value in norm_coordinates if value is not None]
        coordinate_space = "norm"
    elif valid and all(value is not None for value in raw_coordinates):
        coordinates = [value for value in raw_coordinates if value is not None]
        coordinate_space = "raw"
    else:
        return BoundingBox(x=0.0, y=0.0, width=0.0, height=0.0), "unavailable"
    xs = [value[0] for value in coordinates]
    ys = [value[1] for value in coordinates]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return BoundingBox(
        x=min_x,
        y=min_y,
        width=max_x - min_x,
        height=max_y - min_y,
    ), coordinate_space


def _make_stroke(stroke_index: int, points: tuple[Point, ...]) -> Stroke:
    timestamps = [point_timestamp(point) for point in points]
    known_timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    norm_path = path_length(points, "norm")
    bbox, bbox_coordinate_space = _bbox(points)
    return Stroke(
        stroke_id=f"stroke-{stroke_index:04d}",
        session_id=points[0].session_id,
        participant_id=points[0].participant_id,
        task_segment_id=points[0].task_segment_id,
        page_id=points[0].page_id,
        point_refs=tuple(point.point_id for point in points),
        raw_order=tuple(point.point_id for point in points),
        processed_order=_processed_order(points),
        start_time_ms=min(known_timestamps) if known_timestamps else None,
        end_time_ms=max(known_timestamps) if known_timestamps else None,
        bbox=bbox,
        quality_flags=stroke_quality_flags(points),
        algorithm_version="r1-stroke-rule-v2",
        provenance={
            "builder": "stroke_builder",
            "path_length_norm": norm_path,
            "bbox_coordinate_space": bbox_coordinate_space,
        },
    )


def build_strokes(points: tuple[Point, ...], config: StrokeBuildConfig | None = None) -> tuple[Stroke, ...]:
    """Group same-session/page points with timestamps within the configured gap."""

    config = config or StrokeBuildConfig()
    if not points:
        return ()

    groups: list[list[Point]] = [[points[0]]]
    contact_open = points[0].pen_state == "DOWN"
    for point in points[1:]:
        previous = groups[-1][-1]
        pen_split, next_contact_open = _pen_boundary(previous, point, contact_open=contact_open, config=config)
        should_split = (
            not _same_context(previous, point)
            or not _time_continuous(previous, point, config)
            or not _spatial_continuous(previous, point, config)
            or pen_split
        )
        if not should_split:
            groups[-1].append(point)
        else:
            groups.append([point])
        contact_open = next_contact_open if not should_split else point.pen_state == "DOWN"
    return tuple(
        _make_stroke(index, tuple(group))
        for index, group in enumerate(groups)
    )
