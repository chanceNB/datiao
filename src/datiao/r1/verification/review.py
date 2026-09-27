"""Create and summarize manual event verification records."""

from __future__ import annotations

from collections.abc import Iterable

from ..models import StudentProcessEvent
from ..synthetic.models import SyntheticTruthEvent
from .models import (
    ManualVerificationRecord,
    ManualVerificationSummary,
    VerificationLabel,
)


def build_verification_template(
    events: Iterable[StudentProcessEvent],
    truth_events: Iterable[SyntheticTruthEvent] = (),
) -> tuple[ManualVerificationRecord, ...]:
    """Build an initially UNCERTAIN review row for each predicted event."""

    predicted = tuple(events)
    truth = tuple(truth_events)
    return tuple(
        ManualVerificationRecord(
            record_id=f"verification:{event.event_id}",
            event_id=event.event_id,
            predicted_event_type=event.event_type,
            expected_event_type=(truth[index].event_type if index < len(truth) else None),
            source_stroke_ids=event.source_stroke_ids,
            source_point_ids=event.source_point_ids,
        )
        for index, event in enumerate(predicted)
    )


def apply_verification_label(
    records: Iterable[ManualVerificationRecord],
    event_id: str,
    label: VerificationLabel,
    *,
    reviewer: str | None = None,
    notes: str | None = None,
) -> tuple[ManualVerificationRecord, ...]:
    """Return copied records with one event's manual label updated."""

    updated = False
    result: list[ManualVerificationRecord] = []
    for record in records:
        if record.event_id != event_id:
            result.append(record)
            continue
        updated = True
        result.append(
            record.model_copy(
                update={
                    "label": label,
                    "reviewer": reviewer,
                    "notes": notes,
                }
            )
        )
    if not updated:
        raise KeyError(f"event not found: {event_id}")
    return tuple(result)


def summarize_verification(
    records: Iterable[ManualVerificationRecord],
) -> ManualVerificationSummary:
    rows = tuple(records)
    counts = {"YES": 0, "NO": 0, "UNCERTAIN": 0}
    for record in rows:
        counts[record.label] += 1
    return ManualVerificationSummary(
        total=len(rows),
        yes=counts["YES"],
        no=counts["NO"],
        uncertain=counts["UNCERTAIN"],
    )
