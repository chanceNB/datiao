"""Independent leakage and reference-integrity audits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..models import Point, StudentProcessEvent, Stroke
from ..mapper import StrokeMapping
from ..synthetic.models import SyntheticTruthEvent
from .models import CaseManifestRecord, DatasetManifest


class LeakageAudit(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    status: Literal["PASS", "FAIL"]
    participant_overlap: tuple[str, ...] = ()
    session_overlap: tuple[str, ...] = ()
    group_overlap: tuple[str, ...] = ()
    case_overlap: tuple[str, ...] = ()
    duplicate_raw_source: tuple[str, ...] = ()
    train_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)
    test_count: int = Field(ge=0)


class IntegrityAudit(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    status: Literal["PASS", "FAIL"]
    refs: bool
    truth: bool
    prediction: bool
    hashes: bool
    split_completeness: bool
    dataset_type: bool
    errors: tuple[str, ...] = ()


def _overlap_by_split(records: Sequence[Mapping[str, Any]], field: str) -> tuple[str, ...]:
    values: dict[str, set[str]] = defaultdict(set)
    for record in records:
        value = record.get(field)
        split = record.get("split")
        if value is not None and split is not None:
            values[str(value)].add(str(split))
    return tuple(sorted(value for value, splits in values.items() if len(splits) > 1))


def run_leakage_audit(records: Iterable[CaseManifestRecord | Mapping[str, Any]]) -> LeakageAudit:
    """Audit participant/session/group/case and raw-source split leakage."""

    items = [record.model_dump(mode="json") if isinstance(record, CaseManifestRecord) else dict(record) for record in records]
    participant_overlap = _overlap_by_split(items, "participant_id")
    session_overlap = _overlap_by_split(items, "session_id")
    group_overlap = _overlap_by_split(items, "group_id")
    case_overlap = _overlap_by_split(items, "case_id")
    raw_overlap = _overlap_by_split(items, "raw_records_hash")
    counts = {split: sum(1 for item in items if item.get("split") == split) for split in ("train", "validation", "test")}
    failures = participant_overlap + session_overlap + group_overlap + case_overlap + raw_overlap
    return LeakageAudit(
        status="FAIL" if failures else "PASS",
        participant_overlap=participant_overlap,
        session_overlap=session_overlap,
        group_overlap=group_overlap,
        case_overlap=case_overlap,
        duplicate_raw_source=raw_overlap,
        train_count=counts["train"],
        validation_count=counts["validation"],
        test_count=counts["test"],
    )


def _record_id(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    return str(value) if value is not None else ""


def run_integrity_audit(
    cases: Iterable[CaseManifestRecord | Mapping[str, Any]],
    points: Iterable[Mapping[str, Any]],
    strokes: Iterable[Mapping[str, Any]],
    mappings: Iterable[Mapping[str, Any]],
    truth_events: Iterable[Mapping[str, Any]],
    predicted_events: Iterable[Mapping[str, Any]],
    *,
    manifest: DatasetManifest | Mapping[str, Any] | None = None,
    file_hashes_valid: bool = True,
) -> IntegrityAudit:
    """Check references, separation, split completeness, and synthetic typing."""

    case_items = [case.model_dump(mode="json") if isinstance(case, CaseManifestRecord) else dict(case) for case in cases]
    point_items = [dict(item) for item in points]
    stroke_items = [dict(item) for item in strokes]
    mapping_items = [dict(item) for item in mappings]
    truth_items = [dict(item) for item in truth_events]
    prediction_items = [dict(item) for item in predicted_events]
    errors: list[str] = []
    case_ids = [str(item.get("case_id")) for item in case_items]
    refs_ok = len(set(case_ids)) == len(case_ids)
    if not refs_ok:
        errors.append("duplicate case_id")
    points_by_case: dict[str, dict[str, Any]] = defaultdict(dict)
    strokes_by_case: dict[str, dict[str, Any]] = defaultdict(dict)
    mappings_by_case: dict[str, dict[str, Any]] = defaultdict(dict)
    for item in point_items:
        case_id = str(item.get("case_id"))
        point = item.get("point", item)
        point_id = _record_id(point, "point_id")
        if point_id in points_by_case[case_id]:
            # ``duplicate_point`` is an intentional degraded scenario. Keep
            # it in the development dataset while still rejecting an
            # unexplained duplicate reference in other inputs.
            prior = points_by_case[case_id][point_id]
            known_duplicate = "DUPLICATE_POINT" in prior.get("quality_flags", ()) or "DUPLICATE_POINT" in point.get("quality_flags", ())
            if not known_duplicate:
                refs_ok = False
                errors.append(f"duplicate point_id: {case_id}/{point_id}")
        points_by_case[case_id][point_id] = point
    for item in stroke_items:
        case_id = str(item.get("case_id"))
        stroke = item.get("stroke", item)
        stroke_id = _record_id(stroke, "stroke_id")
        if stroke_id in strokes_by_case[case_id]:
            refs_ok = False
            errors.append(f"duplicate stroke_id: {case_id}/{stroke_id}")
        strokes_by_case[case_id][stroke_id] = stroke
        if any(point_ref not in points_by_case[case_id] for point_ref in stroke.get("point_refs", ())):
            refs_ok = False
            errors.append(f"missing stroke point ref: {case_id}/{stroke_id}")
    for item in mapping_items:
        case_id = str(item.get("case_id"))
        mapping = item.get("mapping", item)
        stroke_id = _record_id(mapping, "stroke_id")
        mappings_by_case[case_id][stroke_id] = mapping
        if stroke_id not in strokes_by_case[case_id]:
            refs_ok = False
            errors.append(f"missing mapping stroke ref: {case_id}/{stroke_id}")
    for item in (*truth_items, *prediction_items):
        case_id = str(item.get("case_id"))
        event = item.get("event", item)
        if case_id not in points_by_case:
            refs_ok = False
            errors.append(f"event references missing case: {case_id}")
        if any(stroke_ref not in strokes_by_case[case_id] for stroke_ref in event.get("stroke_refs", ())):
            refs_ok = False
            errors.append(f"missing event stroke ref: {case_id}/{event.get('event_id', '')}")
        point_refs = event.get("point_refs", event.get("source_point_ids", ()))
        if any(point_ref not in points_by_case[case_id] for point_ref in point_refs):
            refs_ok = False
            errors.append(f"missing event point ref: {case_id}/{event.get('event_id', '')}")
    truth_ok = bool(truth_items) and all(item.get("event", item).get("event_id") for item in truth_items)
    prediction_ok = bool(prediction_items) and all(item.get("event", item).get("event_id") for item in prediction_items)
    split_case_ids = [str(item.get("case_id")) for item in case_items]
    split_completeness = all(item.get("split") in {"train", "validation", "test"} for item in case_items) and len(split_case_ids) == len(set(split_case_ids))
    dataset_type_ok = all(item.get("dataset_type", "synthetic") == "synthetic" for item in case_items)
    if manifest is not None:
        manifest_data = manifest.model_dump(mode="json") if isinstance(manifest, DatasetManifest) else dict(manifest)
        dataset_type_ok = dataset_type_ok and manifest_data.get("dataset_type") == "synthetic"
    if not truth_ok:
        errors.append("truth event separation or IDs invalid")
    if not prediction_ok:
        errors.append("prediction event separation or IDs invalid")
    if not hashes_valid(file_hashes_valid):
        errors.append("file hash verification failed")
    status = "PASS" if refs_ok and truth_ok and prediction_ok and file_hashes_valid and split_completeness and dataset_type_ok else "FAIL"
    return IntegrityAudit(
        status=status,
        refs=refs_ok,
        truth=truth_ok,
        prediction=prediction_ok,
        hashes=file_hashes_valid,
        split_completeness=split_completeness,
        dataset_type=dataset_type_ok,
        errors=tuple(dict.fromkeys(errors)),
    )


def hashes_valid(value: bool) -> bool:
    return bool(value)
