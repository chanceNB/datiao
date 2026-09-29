"""Arc-length weighted Stroke to QuestionRegion mapping."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import hypot

from ..models import Point, QuestionRegion, Stroke
from ..stroke.geometry import coordinate_pair
from .models import StrokeMapping

DEFAULT_MIN_COVERAGE = 0.5  # development default; freeze after Dev/Validation calibration
DEFAULT_AMBIGUITY_EPSILON = 0.02
MAPPING_ALGORITHM_VERSION = "r1-qmap-arc-v1"


@dataclass(frozen=True, slots=True)
class QuestionMappingConfig:
    min_coverage: float = DEFAULT_MIN_COVERAGE
    ambiguity_epsilon: float = DEFAULT_AMBIGUITY_EPSILON

    def __post_init__(self) -> None:
        if not 0 < self.min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1]")
        if isinstance(self.ambiguity_epsilon, bool) or not isinstance(self.ambiguity_epsilon, (int, float)) or self.ambiguity_epsilon < 0:
            raise ValueError("ambiguity_epsilon must be a non-negative number")


def map_strokes_to_regions(
    strokes: Iterable[Stroke],
    points: Iterable[Point],
    regions: Iterable[QuestionRegion],
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    config: QuestionMappingConfig | None = None,
) -> tuple[StrokeMapping, ...]:
    """Map each Stroke using trajectory length inside each region.

    For every candidate region, each segment is clipped against the polygon
    and only its portion inside contributes to coverage.  A zero-length Stroke
    uses explicit point-containment fallback and records that provenance.
    """

    active = config or QuestionMappingConfig(min_coverage=min_coverage)
    if config is not None and min_coverage != DEFAULT_MIN_COVERAGE:
        active = QuestionMappingConfig(min_coverage=min_coverage, ambiguity_epsilon=config.ambiguity_epsilon)
    point_index = {point.point_id: point for point in points}
    region_list = tuple(regions)
    results: list[StrokeMapping] = []
    for stroke in strokes:
        source_point_ids = tuple(stroke.raw_order)
        flags = list(stroke.quality_flags)
        referenced_points: list[Point] = []
        missing_reference = False
        for point_id in stroke.processed_order:
            point = point_index.get(point_id)
            if point is None:
                missing_reference = True
            else:
                referenced_points.append(point)
        if missing_reference:
            flags.append("MISSING_SOURCE_POINT")
        if stroke.page_id is None:
            flags.append("MISSING_PAGE")
        if "PARTIAL_DATA" in flags or "MISSING_SOURCE_POINT" in flags or "MISSING_PAGE" in flags:
            results.append(_unknown_mapping(stroke, source_point_ids, flags))
            continue

        candidates: list[tuple[float, int, str, QuestionRegion, str]] = []
        saw_coordinates = False
        for region in region_list:
            if region.page_id != stroke.page_id:
                continue
            coordinate_space = "norm" if region.coordinate_space == "norm" else "raw"
            coordinates = [coordinate_pair(point, coordinate_space) for point in referenced_points]
            valid_coordinates = [value for value in coordinates if value is not None]
            if not valid_coordinates:
                continue
            saw_coordinates = True
            coverage, method = _coverage(referenced_points, region, coordinate_space)
            if coverage >= active.min_coverage:
                candidates.append((coverage, region.priority, region.region_id, region, method))

        if not candidates:
            results.append(_unknown_mapping(stroke, source_point_ids, ["NO_REGION_MATCH" if saw_coordinates else "NO_NORMALIZED_COORDINATES"]))
            continue

        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        best = candidates[0]
        if len(candidates) > 1:
            second = candidates[1]
            if abs(best[0] - second[0]) <= active.ambiguity_epsilon and best[1] == second[1]:
                results.append(_unknown_mapping(stroke, source_point_ids, ["AMBIGUOUS_REGION"], status="AMBIGUOUS"))
                continue

        coverage, _priority, region_id, region, method = best
        results.append(
            StrokeMapping(
                stroke_id=stroke.stroke_id,
                session_id=stroke.session_id,
                participant_id=stroke.participant_id,
                task_segment_id=stroke.task_segment_id,
                page_id=stroke.page_id,
                question_id=region.question_id,
                region_id=region_id,
                status="MAPPED",
                confidence=coverage,
                start_time_ms=stroke.start_time_ms,
                end_time_ms=stroke.end_time_ms,
                point_refs=source_point_ids,
                quality_flags=tuple(dict.fromkeys(flags)),
                algorithm_version=MAPPING_ALGORITHM_VERSION,
                mapping_method=method,
            )
        )
    return tuple(results)


def _coverage(points: list[Point], region: QuestionRegion, coordinate_space: str) -> tuple[float, str]:
    coordinates = [coordinate_pair(point, coordinate_space) for point in points]
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for first, second in zip(coordinates, coordinates[1:]):
        if first is not None and second is not None:
            segments.append((first, second))
    total = sum(hypot(second[0] - first[0], second[1] - first[1]) for first, second in segments)
    if total > 1e-12:
        inside = sum(_segment_inside_length(first, second, region) for first, second in segments)
        return min(1.0, max(0.0, inside / total)), "ARC_LENGTH"

    valid_points = [value for value in coordinates if value is not None]
    if not valid_points:
        return 0.0, "UNKNOWN"
    inside_count = sum(region.contains(x, y) for x, y in valid_points)
    return inside_count / len(valid_points), "POINT_FALLBACK"


def _segment_inside_length(first: tuple[float, float], second: tuple[float, float], region: QuestionRegion) -> float:
    """Clip a segment at every polygon edge and sum intervals whose midpoint is inside."""

    dx, dy = second[0] - first[0], second[1] - first[1]
    segment_length = hypot(dx, dy)
    if segment_length <= 1e-12:
        return 0.0
    parameters = [0.0, 1.0]
    vertices = region.polygon_norm
    for index, edge_start in enumerate(vertices):
        edge_end = vertices[(index + 1) % len(vertices)]
        parameters.extend(_segment_intersection_parameters(first, second, edge_start, edge_end))
    ordered = sorted({max(0.0, min(1.0, value)) for value in parameters})
    inside_length = 0.0
    for left, right in zip(ordered, ordered[1:]):
        if right - left <= 1e-12:
            continue
        midpoint = (left + right) / 2
        if region.contains(first[0] + midpoint * dx, first[1] + midpoint * dy):
            inside_length += (right - left) * segment_length
    return inside_length


def _segment_intersection_parameters(
    first: tuple[float, float],
    second: tuple[float, float],
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
) -> tuple[float, ...]:
    r = (second[0] - first[0], second[1] - first[1])
    s = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])
    denominator = r[0] * s[1] - r[1] * s[0]
    q_minus_p = (edge_start[0] - first[0], edge_start[1] - first[1])
    if abs(denominator) <= 1e-12:
        if abs(q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) > 1e-12:
            return ()
        rr = r[0] * r[0] + r[1] * r[1]
        if rr <= 1e-12:
            return ()
        return (
            (q_minus_p[0] * r[0] + q_minus_p[1] * r[1]) / rr,
            ((edge_end[0] - first[0]) * r[0] + (edge_end[1] - first[1]) * r[1]) / rr,
        )
    t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / denominator
    u = (q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / denominator
    if -1e-12 <= t <= 1 + 1e-12 and -1e-12 <= u <= 1 + 1e-12:
        return (t,)
    return ()


def _unknown_mapping(
    stroke: Stroke,
    source_point_ids: tuple[str, ...],
    flags: list[str],
    *,
    status: str = "UNKNOWN",
) -> StrokeMapping:
    return StrokeMapping(
        stroke_id=stroke.stroke_id,
        session_id=stroke.session_id,
        participant_id=stroke.participant_id,
        task_segment_id=stroke.task_segment_id,
        page_id=stroke.page_id,
        status=status,
        start_time_ms=stroke.start_time_ms,
        end_time_ms=stroke.end_time_ms,
        point_refs=source_point_ids,
        quality_flags=tuple(dict.fromkeys(flags)),
        algorithm_version=MAPPING_ALGORITHM_VERSION,
        mapping_method="UNKNOWN",
    )
