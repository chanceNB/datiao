"""Coordinate-aware geometry helpers used by Stroke and Mapping rules.

Distances are never computed across coordinate spaces.  The caller chooses
``norm`` or ``raw`` explicitly and a missing pair is reported as unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from collections.abc import Iterable

from ..models import Point


@dataclass(frozen=True, slots=True)
class DistanceResult:
    distance: float | None
    coordinate_space: str | None
    valid: bool


def coordinate_pair(point: Point, coordinate_space: str) -> tuple[float, float] | None:
    if coordinate_space == "norm":
        if point.x_norm is None or point.y_norm is None:
            return None
        return point.x_norm, point.y_norm
    if coordinate_space == "raw":
        if point.x_raw is None or point.y_raw is None:
            return None
        return point.x_raw, point.y_raw
    raise ValueError(f"unsupported coordinate space: {coordinate_space}")


def point_distance(previous: Point, current: Point, coordinate_space: str = "norm") -> DistanceResult:
    """Return a distance only when both points share the requested space."""

    first = coordinate_pair(previous, coordinate_space)
    second = coordinate_pair(current, coordinate_space)
    if first is None or second is None:
        return DistanceResult(None, coordinate_space, False)
    return DistanceResult(hypot(second[0] - first[0], second[1] - first[1]), coordinate_space, True)


def path_length(points: Iterable[Point], coordinate_space: str = "norm") -> float | None:
    """Sum available consecutive segment lengths in one coordinate space.

    Missing coordinates make individual segments unavailable; they do not
    invent a zero-length segment or mix raw/mm/norm values.
    """

    materialized = tuple(points)
    if len(materialized) < 2:
        return 0.0 if materialized and coordinate_pair(materialized[0], coordinate_space) is not None else None
    total = 0.0
    available = False
    for previous, current in zip(materialized, materialized[1:]):
        result = point_distance(previous, current, coordinate_space)
        if result.valid and result.distance is not None:
            total += result.distance
            available = True
    return total if available else None
