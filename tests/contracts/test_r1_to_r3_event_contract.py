import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, RefResolver
from pydantic import ValidationError

from datiao.r1.integration.r3 import (
    R1R3EventBatchV01,
    R1R3EventV01,
    export_batch_to_r3,
    export_event_to_r3,
    export_process_result_to_r3,
)
from datiao.r1.models import StudentProcessEvent
from datiao.r1.models.legacy import legacy_rectangle
from datiao.r1.pipeline import run_r1_pipeline
from datiao.r1.synthetic import Scenario, generate_synthetic_case
from datiao.r1.synthetic import compute_manifest_hash


def make_event(event_type="RETURN", **kwargs):
    values = dict(
        event_id="sim_session_001:event:0004",
        event_type=event_type,
        session_id="sim_session_001",
        task_segment_id="sim_segment_01",
        participant_id="sim_p_001",
        question_id="q-05",
        start_time_ms=100,
        end_time_ms=120,
        point_refs=("p-1",),
        stroke_refs=("s-1",),
        quality_status="VALID",
        algorithm_version="r1-event-rule-v0.2.2",
        provenance={
            "dataset_type": "synthetic",
            "generator_version": "r1.synthetic.v1",
            "seed": 20260929,
            "scenario_id": "sim_return_01",
            "ground_truth_source": "scenario_plan",
            "manifest_hash": "sha256:" + "a" * 64,
        },
    )
    values.update(kwargs)
    return StudentProcessEvent(**values)


def export(event):
    return export_event_to_r3(event, point_ids={"p-1"}, stroke_ids={"s-1"})


def test_all_eight_event_types_project_without_enum_drift():
    for event_type in (
        "WRITING",
        "QUESTION_VISIT",
        "QUESTION_LEAVE",
        "RETURN",
        "REVISION_CANDIDATE",
        "PAGE_CHANGE",
        "PROCESS_END",
        "UNKNOWN",
    ):
        event = make_event(event_type, question_id=None if event_type in {"WRITING", "PAGE_CHANGE", "PROCESS_END"} else "q-05")
        payload = export(event)
        assert payload.event_type == event_type


def test_unknown_question_projects_to_wire_unknown_without_mutating_core():
    event = make_event("UNKNOWN", question_id="UNKNOWN")
    before = event.model_dump(mode="json")
    payload = export(event)
    assert payload.question_id == "UNKNOWN"
    assert event.model_dump(mode="json") == before


def test_transport_preserves_semantic_fields_and_versions():
    event = make_event("REVISION_CANDIDATE", quality_status="DEGRADED")
    payload = export(event)
    assert payload.event_id == event.event_id
    assert payload.algorithm_version == "r1-event-rule-v0.2.2"
    assert payload.quality_status == "DEGRADED"
    assert payload.provenance["generator_version"] == "r1.synthetic.v1"


@pytest.mark.parametrize("field,value", [
    ("session_id", "session_001"),
    ("participant_id", "participant_001"),
])
def test_synthetic_ids_require_sim_prefix(field, value):
    with pytest.raises(ValueError, match="sim_"):
        export(make_event(**{field: value}))


def test_synthetic_manifest_hash_must_be_real():
    with pytest.raises((ValueError, ValidationError), match="manifest"):
        export(make_event(provenance={**make_event().provenance, "manifest_hash": "sha256:REPLACE_WITH_REAL_HASH"}))


@pytest.mark.parametrize("kind", ["point", "stroke"])
def test_unresolved_reference_rejected(kind):
    event = make_event(point_refs=("missing",)) if kind == "point" else make_event(stroke_refs=("missing",))
    with pytest.raises(ValueError, match="ref"):
        export_event_to_r3(event, point_ids={"p-1"}, stroke_ids={"s-1"})


def test_extra_wire_fields_rejected():
    with pytest.raises(ValidationError):
        R1R3EventV01(**{**export(make_event()).model_dump(), "page_id": "page-1"})


def test_batch_consistency_rejects_mismatched_children():
    first = make_event(event_id="e-1")
    second = make_event(event_id="e-2", algorithm_version="r1-event-rule-v0.3.0")
    with pytest.raises(ValueError, match="algorithm_version"):
        export_batch_to_r3(
            (first, second),
            batch_id="batch-1",
            dataset_version="r1-synthetic-penprocess-v1@1.0.0",
            point_ids={"p-1"},
            stroke_ids={"s-1"},
        )


def test_batch_uses_explicit_dataset_version():
    batch = export_batch_to_r3(
        (make_event(),),
        batch_id="batch-1",
        dataset_version="r1-synthetic-penprocess-v1@1.0.0",
        point_ids={"p-1"},
        stroke_ids={"s-1"},
    )
    assert batch.contract_version == "r1-event-batch-v0.1"
    assert batch.dataset_version == "r1-synthetic-penprocess-v1@1.0.0"
    assert batch.events[0].event_id == "sim_session_001:event:0004"


def test_pipeline_return_unknown_revision_and_process_end_are_transportable():
    for scenario_type in ("return_visit", "unknown_region", "revision_candidate", "explicit_process_end"):
        case = generate_synthetic_case(Scenario(f"sim_{scenario_type}", scenario_type, seed=20260929))
        exported = [
            export_event_to_r3(event, point_ids={p.point_id for p in case.raw_points}, stroke_ids={s.stroke_id for s in case.strokes})
            for event in case.algorithm_output.events
        ]
        assert exported
        if scenario_type == "return_visit":
            assert any(event.event_type == "RETURN" for event in exported)
        if scenario_type == "unknown_region":
            assert any(event.event_type == "UNKNOWN" and event.question_id == "UNKNOWN" for event in exported)
        if scenario_type == "revision_candidate":
            assert any(event.event_type == "REVISION_CANDIDATE" for event in exported)
        if scenario_type == "explicit_process_end":
            assert any(event.event_type == "PROCESS_END" for event in exported)


def test_golden_schema_is_strict_draft_2020_12():
    schema_path = Path("contracts/r1_to_r3_event_v0_1.schema.json")
    golden_path = Path("contracts/golden/r1_to_r3_event_v0_1.json")
    if not schema_path.exists() or not golden_path.exists():
        pytest.fail("golden transport schema and payload must exist")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(golden), key=str)
    assert not errors, [error.message for error in errors]


def test_event_dto_enforces_time_and_valid_time_gates():
    with pytest.raises(ValidationError):
        R1R3EventV01(**{**export(make_event()).model_dump(), "start_time_ms": None})
    with pytest.raises(ValidationError):
        R1R3EventV01(**{**export(make_event()).model_dump(), "start_time_ms": 200, "end_time_ms": 100})


def test_event_dto_rejects_empty_refs_and_unversioned_algorithm():
    with pytest.raises(ValidationError):
        R1R3EventV01(**{**export(make_event()).model_dump(), "point_refs": ("",)})
    with pytest.raises(ValidationError):
        R1R3EventV01(**{**export(make_event()).model_dump(), "algorithm_version": "latest"})


def test_event_dto_owns_synthetic_provenance_and_identity_gates():
    provenance = dict(make_event().provenance)
    provenance.pop("generator_version")
    with pytest.raises(ValidationError):
        R1R3EventV01(**{**export(make_event()).model_dump(), "provenance": provenance})
    with pytest.raises(ValidationError):
        R1R3EventV01(**{**export(make_event()).model_dump(), "session_id": "session_001"})


def test_batch_dto_self_validates_consistency_and_versions():
    first = export(make_event(event_id="e-1"))
    second = export(make_event(event_id="e-2"))
    base = dict(
        batch_id="batch-1",
        session_id=first.session_id,
        task_segment_id=first.task_segment_id,
        dataset_version="r1-synthetic-penprocess-v1@1.0.0",
        algorithm_version=first.algorithm_version,
        events=(first, second),
    )
    assert R1R3EventBatchV01(**base).events
    with pytest.raises(ValidationError):
        R1R3EventBatchV01(**{**base, "events": (first.model_copy(update={"session_id": "sim_other"}), second)})
    with pytest.raises(ValidationError):
        R1R3EventBatchV01(**{**base, "dataset_version": "current"})
    with pytest.raises(ValidationError):
        R1R3EventBatchV01(**{**base, "algorithm_version": "current"})
    with pytest.raises(ValidationError):
        R1R3EventBatchV01(**{**base, "events": ()})
    with pytest.raises(ValidationError):
        R1R3EventBatchV01(**{**base, "wire_extra": 1})


def test_public_pipeline_export_uses_result_context_and_preserves_result():
    records = (
        {"point_id": "p-1", "session_id": "s-1", "task_segment_id": "seg-1", "participant_id": "sim-p", "page_id": "p-1", "x": 2.0, "y": 2.0, "timestamp_ms": 1000, "sequence": 0},
        {"point_id": "p-2", "session_id": "s-1", "task_segment_id": "seg-1", "participant_id": "sim-p", "page_id": "p-1", "x": 3.0, "y": 3.0, "timestamp_ms": 1050, "sequence": 1},
    )
    region = legacy_rectangle(region_id="r-q1", page_id="p-1", question_id="q1", x=0.0, y=0.0, width=20.0, height=20.0)
    result = run_r1_pipeline(records, (region,), "s-1", task_segment_id="seg-1", data_version="points-v1")
    before = result.model_dump(mode="json")
    batch = export_process_result_to_r3(result, batch_id="batch-1", dataset_version="points-v1")
    assert batch.session_id == result.session_id
    assert batch.task_segment_id == result.task_segment_id
    assert result.model_dump(mode="json") == before
    with pytest.raises(ValueError, match="data_version"):
        export_process_result_to_r3(result, batch_id="batch-1", dataset_version="other")


def test_golden_batch_validates_as_schema_and_dto():
    schema_path = Path("contracts/r1_to_r3_event_batch_v0_1.schema.json").resolve()
    golden_path = Path("contracts/golden/r1_to_r3_event_batch_v0_1.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    event_schema = json.loads(Path("contracts/r1_to_r3_event_v0_1.schema.json").read_text(encoding="utf-8"))
    resolver = RefResolver(
        schema_path.as_uri(),
        schema,
        store={event_schema["$id"]: event_schema},
    )
    errors = sorted(Draft202012Validator(schema, resolver=resolver).iter_errors(golden), key=str)
    assert not errors, [error.message for error in errors]
    assert R1R3EventBatchV01.model_validate_json(golden_path.read_text(encoding="utf-8"))


def _validate_event_schema(payload):
    schema = json.loads(Path("contracts/r1_to_r3_event_v0_1.schema.json").read_text(encoding="utf-8"))
    return sorted(Draft202012Validator(schema).iter_errors(payload), key=str)


def test_schema_and_dto_parity_for_synthetic_real_and_forbidden_provenance():
    valid = export(make_event()).model_dump(mode="json")
    assert not _validate_event_schema(valid)

    missing_generator = {**valid, "provenance": {**valid["provenance"], "generator_version": None}}
    assert _validate_event_schema(missing_generator)
    with pytest.raises(ValidationError):
        R1R3EventV01.model_validate(missing_generator)

    bad_hash = {**valid, "provenance": {**valid["provenance"], "manifest_hash": "sha256:REPLACE_WITH_REAL_HASH"}}
    assert _validate_event_schema(bad_hash)
    with pytest.raises(ValidationError):
        R1R3EventV01.model_validate(bad_hash)

    forbidden = {**valid, "provenance": {**valid["provenance"], "emotion": "anxious"}}
    assert _validate_event_schema(forbidden)
    with pytest.raises(ValidationError):
        R1R3EventV01.model_validate(forbidden)

    real = {**valid, "session_id": "real-session", "participant_id": None, "task_segment_id": None, "provenance": {}}
    assert not _validate_event_schema(real)
    assert R1R3EventV01.model_validate(real).provenance.model_dump(exclude_none=True) == {}

    real_tagged = {**real, "provenance": {"dataset_type": "real"}}
    assert not _validate_event_schema(real_tagged)
    assert R1R3EventV01.model_validate(real_tagged).provenance.dataset_type == "real"


def test_public_pipeline_covers_required_four_synthetic_scenarios():
    for scenario_type, expected in (
        ("return_visit", "RETURN"),
        ("unknown_region", "UNKNOWN"),
        ("revision_candidate", "REVISION_CANDIDATE"),
        ("explicit_process_end", "PROCESS_END"),
    ):
        case = generate_synthetic_case(Scenario(f"sim_{scenario_type}_public", scenario_type, 20260929))
        records = tuple(
            {
                "point_id": point.point_id,
                "session_id": point.session_id,
                "participant_id": point.participant_id,
                "task_segment_id": point.task_segment_id,
                "page_id": point.page_id,
                "x": point.x_raw,
                "y": point.y_raw,
                "x_norm": point.x_norm,
                "y_norm": point.y_norm,
                "timestamp_ms": point.timestamp_ms,
                "sequence": point.sequence,
            }
            for point in case.raw_points
        )
        provenance = {
            "dataset_type": "synthetic",
            "generator_version": case.scenario.generator_version,
            "seed": case.scenario.seed,
            "scenario_id": case.scenario.scenario_id,
            "ground_truth_source": "scenario_plan",
            "manifest_hash": compute_manifest_hash(case.manifest),
        }
        result = run_r1_pipeline(
            records,
            case.regions,
            case.raw_points[0].session_id,
            task_segment_id=case.raw_points[0].task_segment_id,
            source_provenance=provenance,
            process_end_signal=scenario_type == "explicit_process_end",
        )
        before = result.model_dump(mode="json")
        batch = export_process_result_to_r3(
            result,
            batch_id=f"batch_{scenario_type}",
            dataset_version="r1-synthetic-penprocess-v1@1.0.0",
        )
        assert any(event.event_type == expected for event in batch.events)
        assert result.model_dump(mode="json") == before
        assert all(
            event.provenance["manifest_hash"] == compute_manifest_hash(case.manifest)
            for event in batch.events
        )
