import json
from pathlib import Path

import numpy as np
import torch

from datiao.r1.features.models import LABEL_ORDER, FEATURE_ORDER, SequenceSample
from datiao.r1.tcn.data import (
    build_sequence_length_audit,
    collate_sequences,
    fit_normalizer,
    load_tcn_dataset,
    prepare_model_input,
    target_causality_audit,
)


def _sample(sequence_id: str, split: str, length: int, value: float = 0.0) -> SequenceSample:
    return SequenceSample(
        sequence_id=sequence_id,
        case_id=f"case-{sequence_id}",
        participant_id=f"participant-{sequence_id}",
        task_segment_id="segment-1",
        split=split,
        episode_ids=tuple(f"episode-{sequence_id}-{i}" for i in range(length)),
        values=tuple(tuple(value + i + j for j in range(12)) for i in range(length)),
        feature_mask=tuple(tuple(1 for _ in FEATURE_ORDER) for _ in range(length)),
        targets=tuple(tuple(0 for _ in LABEL_ORDER) for _ in range(length)),
    )


def test_frozen_feature_dataset_lineage_and_counts():
    dataset = load_tcn_dataset("artifacts/r1_synthetic_features_v1")
    assert dataset.manifest.manifest_hash == "sha256:40b0326baf0f71206da1fd14b05c8ab5bdf1473c708735c57c019ea820cf5848"
    assert len(dataset.sequences) == 360
    assert len(dataset.episodes) == 700
    assert {split: len(dataset.by_split[split]) for split in ("train", "validation", "test")} == {
        "train": 252,
        "validation": 54,
        "test": 54,
    }


def test_collator_right_pads_raw_values_and_masks():
    batch = collate_sequences([_sample("short", "train", 1, 1.0), _sample("long", "train", 3, 2.0)])
    assert batch["values"].shape == (2, 3, 12)
    assert batch["feature_mask"].shape == (2, 3, 12)
    assert batch["targets"].shape == (2, 3, 8)
    assert batch["padding_mask"].tolist() == [[1, 0, 0], [1, 1, 1]]
    assert torch.all(batch["values"][0, 1:] == 0)
    assert torch.all(batch["feature_mask"][0, 1:] == 0)
    assert torch.all(batch["targets"][0, 1:] == 0)


def test_missing_zero_differs_from_observed_zero_in_model_input():
    sample = _sample("mask", "train", 1)
    values = torch.zeros((1, 1, 12), dtype=torch.float32)
    observed = torch.ones((1, 1, 12), dtype=torch.float32)
    missing = torch.zeros((1, 1, 12), dtype=torch.float32)
    normalizer = fit_normalizer([sample])
    observed_input = prepare_model_input(values, observed, normalizer)
    missing_input = prepare_model_input(values, missing, normalizer)
    assert not torch.equal(observed_input, missing_input)
    assert torch.all(observed_input[..., 12:] == 1)
    assert torch.all(missing_input[..., 12:] == 0)


def test_train_only_normalizer_is_unchanged_by_validation_and_test_values():
    train = _sample("train", "train", 2, 1.0)
    normalizer = fit_normalizer([train])
    before = normalizer.to_dict()
    fit_normalizer([train, _sample("validation", "validation", 2, 1000.0), _sample("test", "test", 2, -1000.0)])
    assert normalizer.to_dict() == before
    assert normalizer.fit_split == "train"


def test_sequence_length_audit_reports_rf_coverage_and_warning():
    audit = build_sequence_length_audit(
        [_sample("a", "train", 1), _sample("b", "validation", 3), _sample("c", "test", 7)],
        receptive_field=7,
    )
    assert audit["overall"]["sequence_count"] == 3
    assert audit["overall"]["real_timestep_count"] == 11
    assert audit["overall"]["count_T_eq_1"] == 1
    assert audit["overall"]["count_T_ge_2"] == 2
    assert audit["overall"]["ratio_T_ge_2"] == 2 / 3
    assert audit["overall"]["sequence_count_length_ge_RF"] == 1
    assert audit["overall"]["effective_temporal_context_warning"] == "SHORT_SEQUENCE_LIMITED_CONTEXT"


def test_target_causality_audit_is_frozen_and_explanatory():
    audit = target_causality_audit()
    assert audit["status"] == "PASS_WITH_NON_CAUSAL_LABELS"
    assert audit["causal_observable_label_names"] == ["WRITING", "QUESTION_VISIT", "RETURN", "REVISION_CANDIDATE", "PAGE_CHANGE", "UNKNOWN"]
    assert audit["retrospective_labels"] == ["QUESTION_LEAVE"]
    assert audit["external_signal_labels"] == ["PROCESS_END"]
