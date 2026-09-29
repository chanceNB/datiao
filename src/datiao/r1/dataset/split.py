"""Deterministic participant/group-level split assignment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import math

from .models import CaseManifestRecord, SplitConfig, SplitManifest


def _stable_order(groups: Iterable[str], seed: int) -> list[str]:
    return sorted(set(groups), key=lambda group: (hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(), group))


def _split_counts(total: int, ratios: SplitConfig) -> dict[str, int]:
    if total < 3:
        raise ValueError("at least three groups are required for non-empty train/validation/test splits")
    names = ("train", "validation", "test")
    raw = {name: total * getattr(ratios, name) for name in names}
    counts = {name: max(1, math.floor(raw[name])) for name in names}
    while sum(counts.values()) > total:
        candidates = [name for name in names if counts[name] > 1]
        if not candidates:
            raise ValueError("split ratios cannot produce non-empty splits")
        name = max(candidates, key=lambda item: (counts[item] - raw[item], item))
        counts[name] -= 1
    while sum(counts.values()) < total:
        name = max(names, key=lambda item: (raw[item] - math.floor(raw[item]), -names.index(item)))
        counts[name] += 1
    return counts


def assign_group_split(groups: Iterable[str], *, split_seed: int, split_ratio: SplitConfig) -> dict[str, str]:
    """Assign each stable group to exactly one deterministic split."""

    ordered = _stable_order(groups, split_seed)
    counts = _split_counts(len(ordered), split_ratio)
    result: dict[str, str] = {}
    cursor = 0
    for split_name in ("train", "validation", "test"):
        for group in ordered[cursor : cursor + counts[split_name]]:
            result[group] = split_name
        cursor += counts[split_name]
    if len(result) != len(ordered):
        raise AssertionError("group split assignment did not cover every group")
    return result


def build_split_manifests(
    cases: Iterable[CaseManifestRecord | Mapping[str, object]],
    *,
    dataset_id: str,
    dataset_version: str,
    split_seed: int,
    split_ratio: SplitConfig,
    split_version: str = "r1-synth-split-v1",
    group_field: str = "group_id",
) -> tuple[tuple[CaseManifestRecord, ...], tuple[SplitManifest, ...]]:
    """Return case records with split labels and deterministic split manifests."""

    case_items = [case if isinstance(case, CaseManifestRecord) else CaseManifestRecord.model_validate(case) for case in cases]
    if len({case.case_id for case in case_items}) != len(case_items):
        raise ValueError("duplicate case_id")
    if group_field not in {"group_id", "participant_id", "session_id"}:
        raise ValueError(f"unsupported split group field: {group_field}")
    group_values = {case.case_id: getattr(case, group_field) for case in case_items}
    group_to_split = assign_group_split(group_values.values(), split_seed=split_seed, split_ratio=split_ratio)
    assigned = tuple(sorted((case.model_copy(update={"split": group_to_split[group_values[case.case_id]]}) for case in case_items), key=lambda case: case.case_id))
    manifests: list[SplitManifest] = []
    for split_name in ("train", "validation", "test"):
        subset = tuple(case for case in assigned if case.split == split_name)
        if not subset:
            raise ValueError(f"split {split_name} is empty")
        participants = tuple(sorted({case.participant_id for case in subset}))
        groups = tuple(sorted({group_values[case.case_id] for case in subset}))
        manifests.append(
            SplitManifest(
                split_version=split_version,
                split_name=split_name,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                split_seed=split_seed,
                participant_ids=participants,
                group_ids=groups,
                case_ids=tuple(case.case_id for case in subset),
                count=len(subset),
            )
        )
    return assigned, tuple(manifests)
