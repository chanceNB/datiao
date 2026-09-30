import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from datiao.r1.integration.r3 import (
    R1R3EventBatchV01,
    R1R3EventV01,
    export_batch_to_r3,
    export_event_to_r3,
)
from datiao.r1.models import StudentProcessEvent
from datiao.r1.synthetic import Scenario, generate_synthetic_case


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
