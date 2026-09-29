from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .models import FEATURE_ORDER, LABEL_ORDER, AlignmentRecord, FeatureRow, ObservationEpisode, SequenceSample


FORBIDDEN_FEATURE_KEYS = {
    "case_id", "participant_id", "session_id", "task_segment_id", "scenario_id", "scenario_type",
    "task_family", "device_id", "split", "episode_id", "stroke_id", "truth", "prediction", "target",
    "event_type", "event_id", "hash", "manifest_hash", "seed", "quality_flags", "page_id", "question_id",
}


def run_feature_leakage_audit(rows: Iterable[FeatureRow]) -> dict[str, Any]:
    rows = tuple(rows)
    observed = sorted({key for row in rows for key in row.features})
    outside = sorted(set(observed) - set(FEATURE_ORDER))
    forbidden = sorted(set(observed) & FORBIDDEN_FEATURE_KEYS)
    status = "PASS" if set(observed) == set(FEATURE_ORDER) and not outside and not forbidden else "FAIL"
    return {"status": status, "feature_order": list(FEATURE_ORDER), "observed_feature_keys": observed,
            "outside_whitelist": outside, "forbidden_keys_observed": forbidden,
            "denylist": sorted(FORBIDDEN_FEATURE_KEYS), "row_count": len(rows)}


def run_feature_range_audit(rows: Iterable[FeatureRow]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    bounded = {"question_occupancy", "previous_ink_iou"}
    nonnegative = {"duration_s", "path_length_norm", "mean_speed_norm_per_s", "pause_before_s", "pause_inside_s", "bbox_width_norm", "bbox_height_norm", "visit_index", "return_count"}
    for row in rows:
        for key, value in row.features.items():
            if value is None:
                continue
            if key in nonnegative and value < 0:
                violations.append({"episode_id": row.episode_id, "feature": key, "value": value})
            if key in bounded and not 0 <= value <= 1:
                violations.append({"episode_id": row.episode_id, "feature": key, "value": value})
            if key == "quality_valid" and value not in (0, 1):
                violations.append({"episode_id": row.episode_id, "feature": key, "value": value})
    return {"status": "PASS" if not violations else "FAIL", "violations": violations, "row_count": len(tuple(rows)) if not isinstance(rows, tuple) else len(rows)}


def run_alignment_audit(records: Iterable[AlignmentRecord]) -> dict[str, Any]:
    records = tuple(records)
    counts = Counter(record.alignment_status for record in records)
    return {"status": "PASS" if not any(counts[key] for key in ("UNMATCHED", "AMBIGUOUS", "ONE_TO_MANY")) else "FAIL",
            "record_count": len(records), "status_counts": dict(sorted(counts.items())),
            "unmatched": [r.truth_event_id for r in records if r.alignment_status == "UNMATCHED"],
            "ambiguous": [r.truth_event_id for r in records if r.alignment_status in {"AMBIGUOUS", "ONE_TO_MANY"}]}


def run_feature_split_audit(episodes: Iterable[ObservationEpisode], sequences: Iterable[SequenceSample], source_splits: dict[str, str]) -> dict[str, Any]:
    episodes = tuple(episodes); sequences = tuple(sequences)
    errors = []
    by_case = {}
    for ep in episodes:
        expected = source_splits.get(ep.case_id)
        by_case.setdefault(ep.case_id, set()).add(ep.split)
        if expected != ep.split:
            errors.append(f"episode split mismatch:{ep.episode_id}")
    for seq in sequences:
        if source_splits.get(seq.case_id) != seq.split:
            errors.append(f"sequence split mismatch:{seq.sequence_id}")
        if len(by_case.get(seq.case_id, set())) != 1:
            errors.append(f"case episode split inconsistent:{seq.case_id}")
    participants = {}
    for ep in episodes:
        participants.setdefault(ep.participant_id, set()).add(ep.split)
    for participant, splits in participants.items():
        if len(splits) != 1:
            errors.append(f"participant crosses split:{participant}")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors,
            "split_counts": {split: sum(ep.split == split for ep in episodes) for split in ("train", "validation", "test")},
            "sequence_split_counts": {split: sum(seq.split == split for seq in sequences) for split in ("train", "validation", "test")}}
