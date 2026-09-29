"""Build and materialize the deterministic R1 Synthetic Development Dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from ..models import is_canonical_v1_eligible
from ..models.immutable import stable_json_hash
from ..synthetic import SyntheticCaseSpec, generate_synthetic_case_from_spec
from .audit import run_integrity_audit, run_leakage_audit
from .io import write_json, write_jsonl
from .manifest import compute_case_manifest_hash, compute_config_hash, compute_dataset_manifest_hash, compute_split_manifest_hash
from .models import CaseManifestRecord, DatasetBuildResult, DatasetConfig, DatasetManifest, DatasetSummary, SplitManifest
from .split import build_split_manifests


DEFAULT_SCENARIO_TYPES = (
    "sequential_visit",
    "return_visit",
    "revision_candidate",
    "cross_question_jump",
    "page_change",
    "unknown_page",
    "pause_resume",
    "unknown_region",
    "missing_point",
    "duplicate_point",
    "out_of_order",
    "continuous_same_question_writing",
    "true_revision_overlap",
    "same_question_no_overlap",
    "explicit_process_end",
    "open_process_no_end",
    "spatial_jump_split",
    "arc_length_cross_region",
)


def load_dataset_config(path: str | Path) -> DatasetConfig:
    source = Path(path)
    return DatasetConfig.model_validate_json(source.read_text(encoding="utf-8"))


def _derive_seed(dataset_seed: int, participant_index: int, scenario_index: int) -> int:
    payload = f"{dataset_seed}:{participant_index}:{scenario_index}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _case_id(dataset_version: str, participant_id: str, scenario_type: str, seed: int) -> str:
    return f"{dataset_version}__{participant_id}__{scenario_type}__seed_{seed:010d}"


def _case_specs(config: DatasetConfig) -> tuple[SyntheticCaseSpec, ...]:
    specs: list[SyntheticCaseSpec] = []
    for participant_index in range(1, config.participant_count + 1):
        participant_id = f"sim_p_{participant_index:03d}"
        for scenario_index, scenario_type in enumerate(config.scenario_types):
            seed = _derive_seed(config.dataset_seed, participant_index, scenario_index)
            case_id = _case_id(config.dataset_version, participant_id, scenario_type, seed)
            specs.append(
                SyntheticCaseSpec(
                    case_id=case_id,
                    participant_id=participant_id,
                    session_id=f"sim_session_{participant_id}_{scenario_type}",
                    task_segment_id=f"sim_segment_{scenario_type}",
                    device_id=config.default_device_id,
                    scenario_id=case_id,
                    scenario_type=scenario_type,
                    seed=seed,
                    task_family=config.task_family_mapping[scenario_type],
                    # Stable source group identity; splitting is explicitly
                    # performed at participant level below.
                    group_id=f"{participant_id}::{scenario_type}",
                    generator_version=config.generator_version,
                )
            )
    return tuple(specs)


def _quality_status(case: Any) -> str:
    if any(point.quality_flags for point in case.raw_points) or any(stroke.quality_flags for stroke in case.strokes) or any(mapping.status != "MAPPED" for mapping in case.mappings) or any(event.quality_status != "VALID" for event in case.algorithm_output.events):
        return "DEGRADED"
    return "OK"


def _case_record(config: DatasetConfig, spec: SyntheticCaseSpec, case: Any, split: str = "train") -> CaseManifestRecord:
    raw_hash = case.manifest.raw_records_hash
    base = {
        "case_id": spec.case_id,
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "participant_id": spec.participant_id,
        "session_id": spec.session_id,
        "task_segment_id": spec.task_segment_id,
        "device_id": spec.device_id,
        "scenario_id": spec.scenario_id,
        "scenario_type": spec.scenario_type,
        "task_family": spec.task_family,
        "seed": spec.seed,
        "generator_version": config.generator_version,
        "group_id": spec.group_id,
        "raw_records_hash": raw_hash,
        "quality_status": _quality_status(case),
        "split": split,
    }
    base["case_manifest_hash"] = f"sha256:{stable_json_hash(base)}"
    return CaseManifestRecord.model_validate(base)


def _wrapped(case_id: str, key: str, model: Any) -> dict[str, Any]:
    return {"case_id": case_id, key: model.model_dump(mode="json") if hasattr(model, "model_dump") else model}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_cases(root: Path, cases: Iterable[tuple[SyntheticCaseSpec, Any]], records: tuple[CaseManifestRecord, ...]) -> dict[str, list[dict[str, Any]]]:
    record_by_id = {record.case_id: record for record in records}
    outputs = {name: [] for name in ("raw_records", "points", "strokes", "mappings", "regions", "truth_events", "predicted_events")}
    for spec, case in sorted(cases, key=lambda item: item[0].case_id):
        case_id = spec.case_id
        for point in case.raw_points:
            raw = dict(point.source_payload)
            raw["case_id"] = case_id
            outputs["raw_records"].append(raw)
            outputs["points"].append(_wrapped(case_id, "point", point))
        outputs["strokes"].extend(_wrapped(case_id, "stroke", stroke) for stroke in case.strokes)
        outputs["mappings"].extend(_wrapped(case_id, "mapping", mapping) for mapping in case.mappings)
        outputs["regions"].extend(_wrapped(case_id, "region", region) for region in case.regions)
        outputs["truth_events"].extend(_wrapped(case_id, "event", event) for event in case.truth.truth_events)
        outputs["predicted_events"].extend(_wrapped(case_id, "event", event) for event in case.algorithm_output.events)
    write_jsonl(root / "cases.jsonl", (record.model_dump(mode="json") for record in records))
    for name, rows in outputs.items():
        write_jsonl(root / f"{name}.jsonl", rows)
    return outputs


def _split_counts(records: Iterable[CaseManifestRecord]) -> dict[str, dict[str, int]]:
    items = tuple(records)
    result: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        subset = tuple(record for record in items if record.split == split)
        result[split] = {"participants": len({record.participant_id for record in subset}), "cases": len(subset)}
    return result


def _event_distribution(events: Iterable[Any]) -> dict[str, int]:
    counts = Counter(event.event_type for event in events)
    return {key: counts[key] for key in sorted(counts)}


def _quality_summary(cases: Iterable[Any]) -> dict[str, Any]:
    case_items = tuple(cases)
    points = tuple(point for case in case_items for point in case.raw_points)
    strokes = tuple(stroke for case in case_items for stroke in case.strokes)
    mappings = tuple(mapping for case in case_items for mapping in case.mappings)
    events = tuple(event for case in case_items for event in case.algorithm_output.events)
    point_flags = Counter(flag for point in points for flag in point.quality_flags)
    stroke_flags = Counter(flag for stroke in strokes for flag in stroke.quality_flags)
    mapping_status = Counter(mapping.status for mapping in mappings)
    event_quality = Counter(event.quality_status for event in events)
    return {
        "canonical_export_eligible_points": sum(is_canonical_v1_eligible(point) for point in points),
        "degraded_points": sum(bool(point.quality_flags) or not is_canonical_v1_eligible(point) for point in points),
        "point_quality_flags": {key: point_flags[key] for key in sorted(point_flags)},
        "stroke_quality_flags": {key: stroke_flags[key] for key in sorted(stroke_flags)},
        "mapping_status": {key: mapping_status[key] for key in sorted(mapping_status)},
        "event_quality": {key: event_quality[key] for key in sorted(event_quality)},
        "UNKNOWN": sum(event.event_type == "UNKNOWN" for event in events),
    }


def _scenario_counts(config: DatasetConfig, cases: Iterable[Any]) -> tuple[dict[str, int], dict[str, int]]:
    case_items = tuple(cases)
    scenario_counts = Counter(case.scenario.scenario_type for case in case_items)
    family_counts = Counter(config.task_family_mapping[case.scenario.scenario_type] for case in case_items)
    missing = [scenario for scenario in config.scenario_types if scenario_counts[scenario] == 0]
    if missing:
        raise ValueError(f"configured scenarios generated no cases: {missing}")
    return ({key: scenario_counts[key] for key in sorted(scenario_counts)}, {key: family_counts[key] for key in sorted(family_counts)})


def materialize_dataset(config: DatasetConfig, output: str | Path, *, overwrite: bool = False) -> DatasetBuildResult:
    """Generate, audit, materialize and reload-ready a complete development dataset."""

    root = Path(output)
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"dataset output already exists: {root}")
        if not root.is_dir() or root.resolve() == Path(root.anchor).resolve():
            raise ValueError("refusing to overwrite an unsafe dataset output")
        for child in root.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    specs = _case_specs(config)
    generated = tuple((spec, generate_synthetic_case_from_spec(spec)) for spec in specs)
    scenario_counts, family_counts = _scenario_counts(config, (case for _, case in generated))
    provisional = tuple(_case_record(config, spec, case) for spec, case in generated)
    records, split_manifests = build_split_manifests(
        provisional,
        dataset_id=config.dataset_id,
        dataset_version=config.dataset_version,
        split_seed=config.split_seed,
        split_ratio=config.split_ratio,
        group_field="participant_id",
    )
    write_json(root / "config_snapshot.json", config.model_dump(mode="json"))
    outputs = _write_cases(root, generated, records)
    split_dir = root / "splits"
    for split_manifest in split_manifests:
        write_json(split_dir / f"{split_manifest.split_name}.json", split_manifest.model_dump(mode="json"))
    split_hash = compute_split_manifest_hash(tuple(manifest.model_dump(mode="json") for manifest in split_manifests))
    case_hash = compute_case_manifest_hash(tuple(record.model_dump(mode="json") for record in records))
    core_files = [
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
    ]
    file_hashes = {relative: _sha256(root / relative) for relative in core_files}
    leakage = run_leakage_audit(records)
    if leakage.status != "PASS":
        raise ValueError(f"leakage audit failed: {leakage.model_dump(mode='json')}")
    write_json(root / "audits" / "leakage_audit.json", leakage.model_dump(mode="json"))
    integrity = run_integrity_audit(
        records,
        outputs["points"],
        outputs["strokes"],
        outputs["mappings"],
        outputs["truth_events"],
        outputs["predicted_events"],
        file_hashes_valid=True,
    )
    if integrity.status != "PASS":
        raise ValueError(f"integrity audit failed: {integrity.model_dump(mode='json')}")
    write_json(root / "audits" / "integrity_audit.json", integrity.model_dump(mode="json"))
    all_points = tuple(point for _, case in generated for point in case.raw_points)
    all_strokes = tuple(stroke for _, case in generated for stroke in case.strokes)
    all_mappings = tuple(mapping for _, case in generated for mapping in case.mappings)
    all_truth = tuple(event for _, case in generated for event in case.truth.truth_events)
    all_prediction = tuple(event for _, case in generated for event in case.algorithm_output.events)
    quality = _quality_summary((case for _, case in generated))
    split_summary = _split_counts(records)
    split_summary_with_targets = {
        split: {**values, "target_ratio": getattr(config.split_ratio, split), "actual_ratio": values["cases"] / len(records)}
        for split, values in split_summary.items()
    }
    manifest_base = DatasetManifest(
        dataset_id=config.dataset_id,
        dataset_version=config.dataset_version,
        dataset_type="synthetic",
        generator_version=config.generator_version,
        contract_version=config.contract_version,
        dataset_config_hash=compute_config_hash(config.model_dump(mode="json")),
        participant_count=len({record.participant_id for record in records}),
        session_count=len({record.session_id for record in records}),
        case_count=len(records),
        point_count=len(all_points),
        stroke_count=len(all_strokes),
        mapping_count=len(all_mappings),
        truth_event_count=len(all_truth),
        predicted_event_count=len(all_prediction),
        scenario_counts=scenario_counts,
        task_family_counts=family_counts,
        quality_counts={"OK": sum(record.quality_status == "OK" for record in records), "DEGRADED": sum(record.quality_status == "DEGRADED" for record in records)},
        split_summary=split_summary_with_targets,
        files=tuple(core_files),
        file_hashes=file_hashes,
        case_manifest_hash=case_hash,
        split_manifest_hash=split_hash,
    )
    manifest_hash = compute_dataset_manifest_hash(manifest_base)
    manifest = manifest_base.model_copy(update={"manifest_hash": manifest_hash})
    summary = DatasetSummary(
        dataset_id=config.dataset_id,
        dataset_version=config.dataset_version,
        dataset_type="synthetic",
        participants=manifest.participant_count,
        cases=manifest.case_count,
        points=manifest.point_count,
        strokes=manifest.stroke_count,
        mappings=manifest.mapping_count,
        truth_events=manifest.truth_event_count,
        predicted_events=manifest.predicted_event_count,
        scenario_counts=scenario_counts,
        task_family_counts=family_counts,
        truth_event_distribution=_event_distribution(all_truth),
        prediction_event_distribution=_event_distribution(all_prediction),
        quality_summary=quality,
        split_counts=split_summary_with_targets,
        audit_status={"leakage": leakage.status, "integrity": integrity.status},
        manifest_hash=manifest_hash,
    )
    write_json(root / "dataset_summary.json", summary.model_dump(mode="json"))
    write_json(root / "dataset_manifest.json", manifest.model_dump(mode="json"))
    return DatasetBuildResult(output_path=str(root), manifest=manifest, summary=summary, split_manifests=split_manifests)


build_dataset = materialize_dataset


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build the R1 Synthetic Development Dataset")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_dataset_config(args.config)
    result = materialize_dataset(config, args.output, overwrite=args.overwrite)
    print(json.dumps({"output": result.output_path, "manifest_hash": result.manifest.manifest_hash, "cases": result.manifest.case_count}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    _main()
