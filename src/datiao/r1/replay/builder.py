"""Build deterministic page replay frames from normalized points and strokes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..models import Point, Stroke
from ..stroke.quality import point_sequence
from .models import PageReplay, ReplayFrame


def build_page_replay(
    points: Iterable[Point],
    strokes: Iterable[Stroke],
    *,
    session_id: str | None = None,
) -> PageReplay:
    """Return one chronologically ordered replay while preserving bad points."""

    point_list = tuple(points)
    stroke_list = tuple(strokes)
    resolved_session = session_id or (point_list[0].session_id if point_list else None)
    if not resolved_session:
        raise ValueError("session_id is required for an empty page replay")
    if any(point.session_id != resolved_session for point in point_list):
        raise ValueError("all replay points must belong to one session")

    stroke_for_point: dict[str, str] = {}
    duplicate_point_refs: set[str] = set()
    for stroke in stroke_list:
        if stroke.session_id != resolved_session:
            raise ValueError("all replay strokes must belong to one session")
        for point_id in stroke.raw_order:
            if point_id in stroke_for_point:
                duplicate_point_refs.add(point_id)
            else:
                stroke_for_point[point_id] = stroke.stroke_id

    ordered = sorted(
        point_list,
        key=lambda point: (
            point.normalized.timestamp_ms is None,
            point.normalized.timestamp_ms
            if point.normalized.timestamp_ms is not None
            else 0,
            point_sequence(point) is None,
            point_sequence(point) if point_sequence(point) is not None else 0,
            point.raw_index,
        ),
    )
    frames: list[ReplayFrame] = []
    page_order: list[str] = []
    for frame_index, point in enumerate(ordered):
        page_id = point.page_id
        if page_id is not None and page_id not in page_order:
            page_order.append(page_id)
        flags = list(point.quality_flags)
        if point.point_id in duplicate_point_refs:
            flags.append("DUPLICATE_SOURCE_ID")
        frames.append(
            ReplayFrame(
                frame_index=frame_index,
                point_id=point.point_id,
                stroke_id=stroke_for_point.get(point.point_id),
                page_id=page_id,
                timestamp_ms=point.normalized.timestamp_ms,
                x=point.normalized.x,
                y=point.normalized.y,
                raw_index=point.raw_index,
                quality_flags=tuple(dict.fromkeys(flags)),
            )
        )
    return PageReplay(
        session_id=resolved_session,
        page_order=tuple(page_order),
        frames=tuple(frames),
    )
