"""Deterministic JSON/JSONL materialization and reload helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..mapper import StrokeMapping
from ..models import Point, QuestionRegion, StudentProcessEvent, Stroke
from ..synthetic.models import SyntheticTruthEvent
from .models import CaseManifestRecord, DatasetConfig, DatasetManifest
from .models import SplitManifest
from .manifest import compute_case_collection_hash, compute_config_hash, compute_dataset_manifest_hash, compute_split_manifest_hash, validate_case_record_hash
from .audit import run_integrity_audit, run_leakage_audit, verify_file_hashes


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dump(value) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json_dump(record) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        records.append(value)
    return tuple(records)


def _tuple_fields(value: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    payload = dict(value)
    for field in fields:
        if isinstance(payload.get(field), list):
            payload[field] = tuple(payload[field])
    return payload


class ReloadedDataset(dict[str, Any]):
    """Dictionary-like reload result with validated model collections."""


def reload_dataset(output: str | Path) -> ReloadedDataset:
    root = Path(output)
    manifest_data = read_json(root / "dataset_manifest.json")
    manifest = DatasetManifest.model_validate(manifest_data)
    if manifest.manifest_hash != compute_dataset_manifest_hash(manifest):
        raise ValueError("dataset manifest hash mismatch")
    hashes_ok, hash_errors = verify_file_hashes(root, manifest.file_hashes)
    if not hashes_ok:
        raise ValueError(f"manifest file hash verification failed: {hash_errors}")
    if set(manifest.files) != set(manifest.file_hashes):
        raise ValueError("manifest files and file_hashes coverage mismatch")
    config = DatasetConfig.model_validate(read_json(root / "config_snapshot.json"))
    cases = tuple(CaseManifestRecord.model_validate(item) for item in read_jsonl(root / "cases.jsonl"))
    if manifest.dataset_config_hash != compute_config_hash(config.model_dump(mode="json")):
        raise ValueError("dataset config hash mismatch")
    if (manifest.dataset_id, manifest.dataset_version) != (config.dataset_id, config.dataset_version):
        raise ValueError("dataset config identity mismatch")
    case_payloads = tuple(case.model_dump(mode="json") for case in cases)
    if any(not validate_case_record_hash(case) for case in case_payloads):
        raise ValueError("per-case manifest hash mismatch")
    if manifest.case_manifest_hash != compute_case_collection_hash(case_payloads):
        raise ValueError("case collection hash mismatch")
    split_manifests = tuple(
        SplitManifest.model_validate(read_json(root / "splits" / f"{name}.json"))
        for name in ("train", "validation", "test")
    )
    if manifest.split_manifest_hash != compute_split_manifest_hash(tuple(split.model_dump(mode="json") for split in split_manifests)):
        raise ValueError("split manifest hash mismatch")
    raw_records = read_jsonl(root / "raw_records.jsonl")
    point_records = read_jsonl(root / "points.jsonl")
    stroke_records = read_jsonl(root / "strokes.jsonl")
    mapping_records = read_jsonl(root / "mappings.jsonl")
    region_records = read_jsonl(root / "regions.jsonl")
    truth_records = read_jsonl(root / "truth_events.jsonl")
    prediction_records = read_jsonl(root / "predicted_events.jsonl")
    points = tuple(Point.model_validate(item["point"]) for item in point_records)
    strokes = tuple(Stroke.model_validate(item["stroke"]) for item in stroke_records)
    mappings = tuple(StrokeMapping.model_validate(_tuple_fields(item["mapping"], ("point_refs", "quality_flags"))) for item in mapping_records)
    regions = tuple(QuestionRegion.model_validate(item["region"]) for item in region_records)
    truth_events = tuple(SyntheticTruthEvent.model_validate(_tuple_fields(item["event"], ("source_point_ids",))) for item in truth_records)
    predicted_events = tuple(StudentProcessEvent.model_validate(item["event"]) for item in prediction_records)
    leakage = run_leakage_audit(cases, split_manifests=split_manifests)
    integrity = run_integrity_audit(
        cases, point_records, stroke_records, mapping_records, truth_records, prediction_records,
        manifest=manifest, dataset_root=root, split_manifests=split_manifests,
        expected_split_seed=config.split_seed,
    )
    if leakage.status != "PASS":
        raise ValueError(f"recomputed leakage audit failed: {leakage.model_dump(mode='json')}")
    if integrity.status != "PASS":
        raise ValueError(f"recomputed integrity audit failed: {integrity.model_dump(mode='json')}")
    stored_leakage = read_json(root / "audits" / "leakage_audit.json")
    stored_integrity = read_json(root / "audits" / "integrity_audit.json")
    if stored_leakage != leakage.model_dump(mode="json"):
        raise ValueError("stored leakage audit differs from recomputed audit")
    if stored_integrity != integrity.model_dump(mode="json"):
        raise ValueError("stored integrity audit differs from recomputed audit")
    summary = read_json(root / "dataset_summary.json")
    if summary.get("manifest_hash") != manifest.manifest_hash:
        raise ValueError("summary manifest hash mismatch")
    return ReloadedDataset(
        output_path=str(root),
        manifest=manifest,
        config=config,
        cases=cases,
        split_manifests=split_manifests,
        raw_records=raw_records,
        point_records=point_records,
        stroke_records=stroke_records,
        mapping_records=mapping_records,
        truth_records=truth_records,
        prediction_records=prediction_records,
        points=points,
        strokes=strokes,
        mappings=mappings,
        regions=regions,
        truth_events=truth_events,
        predicted_events=predicted_events,
        summary=summary,
        leakage_audit=leakage.model_dump(mode="json"),
        integrity_audit=integrity.model_dump(mode="json"),
    )


load_materialized_dataset = reload_dataset
