"""Build strokes from Points while preserving raw input order."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import BoundingBox, Point, Stroke
from .quality import point_sequence, point_timestamp, stroke_quality_flags


@dataclass(frozen=True, slots=True)
class StrokeBuildConfig:
    """First-version stroke segmentation configuration."""

    max_time_gap_ms: int = 1_000

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_time_gap_ms, bool)
            or not isinstance(self.max_time_gap_ms, int)
            or self.max_time_gap_ms < 0
        ):
            raise ValueError("max_time_gap_ms must be a non-negative integer")


def _same_context(previous: Point, current: Point) -> bool:
    return previous.session_id == current.session_id and previous.page_id == current.page_id


def _time_continuous(previous: Point, current: Point, config: StrokeBuildConfig) -> bool:
    previous_time = point_timestamp(previous)
    current_time = point_timestamp(current)
    if previous_time is None or current_time is None:
        return False
    return abs(current_time - previous_time) <= config.max_time_gap_ms


def _processed_order(points: tuple[Point, ...]) -> tuple[str, ...]:
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


def _bbox(points: tuple[Point, ...]) -> BoundingBox:
    valid = [
        point
        for point in points
        if (point.x_norm is not None and point.y_norm is not None)
        or (point.x_raw is not None and point.y_raw is not None)
    ]
    if not valid:
        return BoundingBox(x=0.0, y=0.0, width=0.0, height=0.0)
    xs = [point.x_norm if point.x_norm is not None else point.x_raw for point in valid]
    ys = [point.y_norm if point.y_norm is not None else point.y_raw for point in valid]
    assert all(value is not None for value in xs + ys)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return BoundingBox(
        x=min_x,
        y=min_y,
        width=max_x - min_x,
        height=max_y - min_y,
    )


def _make_stroke(stroke_index: int, points: tuple[Point, ...]) -> Stroke:
    timestamps = [point_timestamp(point) for point in points]
    known_timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
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
        bbox=_bbox(points),
        quality_flags=stroke_quality_flags(points),
        algorithm_version="r1-stroke-rule-v1",
        provenance={"builder": "stroke_builder"},
    )


def build_strokes(points: tuple[Point, ...], config: StrokeBuildConfig | None = None) -> tuple[Stroke, ...]:
    """Group same-session/page points with timestamps within the configured gap."""

    config = config or StrokeBuildConfig()
    if not points:
        return ()

    groups: list[list[Point]] = [[points[0]]]
    for point in points[1:]:
        previous = groups[-1][-1]
        if _same_context(previous, point) and _time_continuous(previous, point, config):
            groups[-1].append(point)
        else:
            groups.append([point])
    return tuple(
        _make_stroke(index, tuple(group))
        for index, group in enumerate(groups)
    )
