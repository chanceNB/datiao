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
from .manifest import compute_dataset_manifest_hash


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
    for relative_path, expected_hash in manifest.file_hashes.items():
        candidate = root / relative_path
        if not candidate.is_file():
            raise ValueError(f"manifest file missing: {relative_path}")
        import hashlib

        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if expected_hash != f"sha256:{digest}":
            raise ValueError(f"manifest file hash mismatch: {relative_path}")
    config = DatasetConfig.model_validate(read_json(root / "config_snapshot.json"))
    cases = tuple(CaseManifestRecord.model_validate(item) for item in read_jsonl(root / "cases.jsonl"))
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
    return ReloadedDataset(
        output_path=str(root),
        manifest=manifest,
        config=config,
        cases=cases,
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
        summary=read_json(root / "dataset_summary.json"),
        leakage_audit=read_json(root / "audits" / "leakage_audit.json"),
        integrity_audit=read_json(root / "audits" / "integrity_audit.json"),
    )


load_materialized_dataset = reload_dataset
