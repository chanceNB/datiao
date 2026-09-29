"""Independent leakage and reference-integrity audits."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import CaseManifestRecord, DatasetManifest
from .models import SplitManifest
from .manifest import validate_case_record_hash


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


def run_leakage_audit(
    records: Iterable[CaseManifestRecord | Mapping[str, Any]],
    *,
    split_manifests: Iterable[SplitManifest | Mapping[str, Any]] | None = None,
) -> LeakageAudit:
    """Audit participant/session/group/case and raw-source split leakage."""

    items = [record.model_dump(mode="json") if isinstance(record, CaseManifestRecord) else dict(record) for record in records]
    split_consistent = True
    membership_items = []
    if split_manifests is not None:
        case_index = {item["case_id"]: item for item in items}
        for split in split_manifests:
            split = split if isinstance(split, SplitManifest) else SplitManifest.model_validate(split)
            for case_id in split.case_ids:
                if case_id not in case_index:
                    split_consistent = False
                    continue
                case = dict(case_index[case_id])
                split_consistent = split_consistent and case["split"] == split.split_name
                case["split"] = split.split_name
                membership_items.append(case)
        split_consistent = split_consistent and Counter(item["case_id"] for item in membership_items) == Counter(item["case_id"] for item in items)
    overlap_items = items + membership_items
    participant_overlap = _overlap_by_split(overlap_items, "participant_id")
    session_overlap = _overlap_by_split(overlap_items, "session_id")
    group_overlap = _overlap_by_split(overlap_items, "group_id")
    case_overlap = _overlap_by_split(overlap_items, "case_id")
    raw_overlap = _overlap_by_split(overlap_items, "raw_records_hash")
    counts = {split: sum(1 for item in items if item.get("split") == split) for split in ("train", "validation", "test")}
    failures = participant_overlap + session_overlap + group_overlap + case_overlap + raw_overlap
    return LeakageAudit(
        status="FAIL" if failures or not split_consistent else "PASS",
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


REQUIRED_DATASET_FILES = frozenset(
    {
        "config_snapshot.json",
        "cases.jsonl",
        "raw_records.jsonl",
        "points.jsonl",
        "strokes.jsonl",
        "mappings.jsonl",
        "regions.jsonl",
        "truth_events.jsonl",
        "predicted_events.jsonl",
        "splits/train.json",
        "splits/validation.json",
        "splits/test.json",
    }
)


def verify_file_hashes(root: str | Path, expected_file_hashes: Mapping[str, str]) -> tuple[bool, tuple[str, ...]]:
    """Read every expected asset and compare its actual SHA-256."""

    dataset_root = Path(root)
    errors: list[str] = []
    if not REQUIRED_DATASET_FILES.issubset(expected_file_hashes):
        errors.append("manifest file_hashes missing required dataset assets")
    for relative_path, expected_hash in sorted(expected_file_hashes.items()):
        candidate = dataset_root / relative_path
        if Path(relative_path).is_absolute() or not candidate.resolve().is_relative_to(dataset_root.resolve()):
            errors.append(f"unsafe manifest path: {relative_path}")
            continue
        if not candidate.is_file():
            errors.append(f"missing file: {relative_path}")
            continue
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        actual = f"sha256:{digest}"
        if actual != expected_hash:
            errors.append(f"hash mismatch: {relative_path}")
    return not errors, tuple(errors)


def _validate_split_semantics(
    cases: Sequence[Mapping[str, Any]],
    split_manifests: Sequence[SplitManifest],
    *,
    expected_dataset_id: str | None = None,
    expected_dataset_version: str | None = None,
    expected_split_seed: int | None = None,
    expected_split_version: str = "r1-synth-split-v1",
) -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = []
    if {manifest.split_name for manifest in split_manifests} != {"train", "validation", "test"} or len(split_manifests) != 3:
        errors.append("split manifests must contain exactly train, validation and test")
        return False, tuple(errors)
    case_by_id = {str(case.get("case_id")): case for case in cases}
    if expected_dataset_id is None and cases:
        expected_dataset_id = str(cases[0].get("dataset_id"))
    if expected_dataset_version is None and cases:
        expected_dataset_version = str(cases[0].get("dataset_version"))
    if expected_split_seed is None and split_manifests:
        expected_split_seed = split_manifests[0].split_seed
    memberships: dict[str, list[str]] = defaultdict(list)
    for manifest in split_manifests:
        if expected_dataset_id is not None and manifest.dataset_id != expected_dataset_id:
            errors.append(f"split dataset_id mismatch: {manifest.split_name}")
        if expected_dataset_version is not None and manifest.dataset_version != expected_dataset_version:
            errors.append(f"split dataset_version mismatch: {manifest.split_name}")
        if expected_split_seed is not None and manifest.split_seed != expected_split_seed:
            errors.append(f"split split_seed mismatch: {manifest.split_name}")
        if manifest.split_version != expected_split_version:
            errors.append(f"split split_version mismatch: {manifest.split_name}")
        if manifest.count != len(manifest.case_ids):
            errors.append(f"split count mismatch: {manifest.split_name}")
        actual_cases = [case_by_id.get(case_id) for case_id in manifest.case_ids]
        if any(case is None for case in actual_cases):
            errors.append(f"split references unknown case: {manifest.split_name}")
            continue
        actual_cases = [case for case in actual_cases if case is not None]
        actual_participants = {str(case["participant_id"]) for case in actual_cases}
        actual_groups = {str(case["group_id"]) for case in actual_cases}
        if set(manifest.participant_ids) != actual_participants:
            errors.append(f"split participant_ids mismatch: {manifest.split_name}")
        if set(manifest.group_ids) != actual_groups:
            errors.append(f"split group_ids mismatch: {manifest.split_name}")
        if manifest.count != sum(case.get("split") == manifest.split_name for case in cases):
            errors.append(f"split labeled case count mismatch: {manifest.split_name}")
        for case in actual_cases:
            case_id = str(case["case_id"])
            memberships[case_id].append(manifest.split_name)
            if case.get("split") != manifest.split_name:
                errors.append(f"case split label mismatch: {case_id}")
    expected_case_ids = set(case_by_id)
    actual_case_ids = set(memberships)
    if actual_case_ids != expected_case_ids:
        errors.append("split union does not equal cases.jsonl")
    if any(len(splits) != 1 for splits in memberships.values()):
        errors.append("split sets are not disjoint")
    return not errors, tuple(dict.fromkeys(errors))


def run_integrity_audit(
    cases: Iterable[CaseManifestRecord | Mapping[str, Any]],
    points: Iterable[Mapping[str, Any]],
    strokes: Iterable[Mapping[str, Any]],
    mappings: Iterable[Mapping[str, Any]],
    truth_events: Iterable[Mapping[str, Any]],
    predicted_events: Iterable[Mapping[str, Any]],
    *,
    manifest: DatasetManifest | Mapping[str, Any] | None = None,
    dataset_root: str | Path | None = None,
    expected_file_hashes: Mapping[str, str] | None = None,
    split_manifests: Iterable[SplitManifest | Mapping[str, Any]] | None = None,
    expected_dataset_id: str | None = None,
    expected_dataset_version: str | None = None,
    expected_split_seed: int | None = None,
    expected_split_version: str = "r1-synth-split-v1",
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
    if any(not validate_case_record_hash(case) for case in case_items):
        refs_ok = False
        errors.append("case_manifest_hash mismatch")
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
    manifest_data: dict[str, Any] | None = None
    if manifest is not None:
        manifest_data = manifest.model_dump(mode="json") if isinstance(manifest, DatasetManifest) else dict(manifest)
        dataset_type_ok = dataset_type_ok and manifest_data.get("dataset_type") == "synthetic"
        expected_dataset_id = expected_dataset_id or manifest_data.get("dataset_id")
        expected_dataset_version = expected_dataset_version or manifest_data.get("dataset_version")
    if not truth_ok:
        errors.append("truth event separation or IDs invalid")
    if not prediction_ok:
        errors.append("prediction event separation or IDs invalid")
    if expected_file_hashes is None and manifest_data is not None:
        expected_file_hashes = manifest_data.get("file_hashes")
    if dataset_root is None or expected_file_hashes is None:
        hashes_ok = False
        hash_errors = ("dataset_root and expected file hashes are required",)
    else:
        hashes_ok, hash_errors = verify_file_hashes(dataset_root, expected_file_hashes)
    errors.extend(hash_errors)
    split_ok = False
    if dataset_root is not None:
        try:
            disk_splits = tuple(
                SplitManifest.model_validate(json.loads((Path(dataset_root) / "splits" / f"{name}.json").read_text(encoding="utf-8")))
                for name in ("train", "validation", "test")
            )
            if any(split.split_name != name for split, name in zip(disk_splits, ("train", "validation", "test"))):
                errors.append("split_name does not match split filename")
                disk_splits = ()
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"split manifest loading failed: {error}")
            disk_splits = ()
        if split_manifests is not None:
            supplied = tuple(item if isinstance(item, SplitManifest) else SplitManifest.model_validate(item) for item in split_manifests)
            if {item.split_name: item.model_dump(mode="json") for item in supplied} != {item.split_name: item.model_dump(mode="json") for item in disk_splits}:
                errors.append("supplied split manifests differ from disk")
                disk_splits = ()
        split_manifests = disk_splits
    if split_manifests is None:
        errors.append("split manifests are required")
    else:
        split_items = tuple(item if isinstance(item, SplitManifest) else SplitManifest.model_validate(item) for item in split_manifests)
        split_ok, split_errors = _validate_split_semantics(
            case_items,
            split_items,
            expected_dataset_id=expected_dataset_id,
            expected_dataset_version=expected_dataset_version,
            expected_split_seed=expected_split_seed,
            expected_split_version=expected_split_version,
        )
        errors.extend(split_errors)
    split_completeness = split_completeness and split_ok
    status = "PASS" if refs_ok and truth_ok and prediction_ok and hashes_ok and split_completeness and dataset_type_ok else "FAIL"
    return IntegrityAudit(
        status=status,
        refs=refs_ok,
        truth=truth_ok,
        prediction=prediction_ok,
        hashes=hashes_ok,
        split_completeness=split_completeness,
        dataset_type=dataset_type_ok,
        errors=tuple(dict.fromkeys(errors)),
    )
