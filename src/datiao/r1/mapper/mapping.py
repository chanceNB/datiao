"""Map strokes to configured question regions."""

from __future__ import annotations

from collections.abc import Iterable

from ..models import Point, QuestionRegion, Stroke
from .models import StrokeMapping

DEFAULT_MIN_COVERAGE = 0.5


def map_strokes_to_regions(
    strokes: Iterable[Stroke],
    points: Iterable[Point],
    regions: Iterable[QuestionRegion],
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> tuple[StrokeMapping, ...]:
    """Return one mapping result per stroke using valid-point coverage.

    A stroke with incomplete or partial quality is kept but marked unknown.
    Mapping never invents a question ID when no region is sufficiently clear.
    """

    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
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
        if flags:
            results.append(_unknown_mapping(stroke, source_point_ids, flags))
            continue

        candidates: list[tuple[float, int, str, QuestionRegion]] = []
        saw_norm_without_coords = False
        for region in region_list:
            if region.page_id != stroke.page_id:
                continue
            coordinates: list[tuple[float, float]] = []
            for point in referenced_points:
                if region.coordinate_space == "norm":
                    if point.x_norm is None or point.y_norm is None:
                        saw_norm_without_coords = True
                        continue
                    coordinates.append((point.x_norm, point.y_norm))
                elif point.x_raw is not None and point.y_raw is not None:
                    coordinates.append((point.x_raw, point.y_raw))
            if not coordinates:
                continue
            inside = sum(
                region.contains(x, y) for x, y in coordinates
            )
            coverage = inside / len(coordinates)
            if coverage >= min_coverage:
                candidates.append((coverage, region.priority, region.region_id, region))

        if not candidates:
            results.append(
                _unknown_mapping(stroke, source_point_ids, ["NO_NORMALIZED_COORDINATES" if saw_norm_without_coords else "NO_REGION_MATCH"])
            )
            continue

        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        best = candidates[0]
        tied = [
            candidate
            for candidate in candidates
            if candidate[0] == best[0] and candidate[1] == best[1]
        ]
        if len(tied) > 1:
            results.append(
                _unknown_mapping(stroke, source_point_ids, ["AMBIGUOUS_REGION"])
            )
            continue

        coverage, _priority, region_id, region = best
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
            )
        )
    return tuple(results)


def _unknown_mapping(
    stroke: Stroke,
    source_point_ids: tuple[str, ...],
    flags: list[str],
) -> StrokeMapping:
    return StrokeMapping(
        stroke_id=stroke.stroke_id,
        session_id=stroke.session_id,
        participant_id=stroke.participant_id,
        task_segment_id=stroke.task_segment_id,
        page_id=stroke.page_id,
        status="UNKNOWN",
        start_time_ms=stroke.start_time_ms,
        end_time_ms=stroke.end_time_ms,
        point_refs=source_point_ids,
        quality_flags=tuple(dict.fromkeys(flags)),
    )

