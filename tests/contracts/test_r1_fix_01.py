import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from datiao.r1.models import (
    NormalizedPoint,
    Point,
    QuestionRegion,
    StudentProcessEvent,
    is_canonical_v1_eligible,
    validate_canonical_point_for_export,
)
from datiao.r1.models.legacy import LegacyQuestionRegion, legacy_event, legacy_point, legacy_rectangle
from datiao.r1.synthetic import (
    Scenario,
    SyntheticManifest,
    compute_manifest_hash,
    generate_synthetic_case,
)

ROOT = Path(__file__).parents[2]


def _event(event_type="UNKNOWN", **overrides):
    values = dict(
        event_id="event-1",
        event_type=event_type,
        session_id="session-1",
        start_time_ms=10,
        end_time_ms=20,
        point_refs=("point-1",),
        stroke_refs=("stroke-1",),
        quality_status="VALID",
    )
    values.update(overrides)
    return StudentProcessEvent(**values)


def test_synthetic_manifest_hash_is_separate_from_raw_records_hash_and_reproducible():
    first = generate_synthetic_case(Scenario("manifest-case", "return_visit", seed=7))
    second = generate_synthetic_case(Scenario("manifest-case", "return_visit", seed=7))

    assert isinstance(first.manifest, SyntheticManifest)
    assert first.manifest == second.manifest
    assert compute_manifest_hash(first.manifest) == compute_manifest_hash(second.manifest)
    assert first.manifest.raw_records_hash != compute_manifest_hash(first.manifest)
    assert first.algorithm_output.events[0].provenance["manifest_hash"] == compute_manifest_hash(first.manifest)


def test_different_scenario_or_seed_changes_manifest_hash():
    base = generate_synthetic_case(Scenario("manifest-case", "return_visit", seed=7))
    other_seed = generate_synthetic_case(Scenario("manifest-case", "return_visit", seed=8))
    other_scenario = generate_synthetic_case(Scenario("other-case", "return_visit", seed=7))

    assert compute_manifest_hash(base.manifest) != compute_manifest_hash(other_seed.manifest)
    assert compute_manifest_hash(base.manifest) != compute_manifest_hash(other_scenario.manifest)


def test_degraded_point_is_retained_but_not_eligible_for_standard_export():
    point = Point(
        point_id="p",
        session_id="s",
        sequence=0,
        timestamp_ms=None,
        x_raw=None,
        y_raw=None,
        source_index=0,
        raw_order=0,
        processed_order=0,
        source_payload={"point_id": "p"},
        quality_flags=("MISSING_TIMESTAMP", "INVALID_COORDINATE"),
    )

    assert not is_canonical_v1_eligible(point)
    with pytest.raises(ValueError, match="standard Canonical Point V1"):
        validate_canonical_point_for_export(point)


def test_valid_event_requires_time_and_question_events_require_question_id():
    with pytest.raises(ValidationError):
        _event(start_time_ms=None)
    with pytest.raises(ValidationError):
        _event(event_type="QUESTION_VISIT", question_id=None)
    with pytest.raises(ValidationError):
        _event(event_type="RETURN", question_id=None)
    with pytest.raises(ValidationError):
        _event(event_type="REVISION_CANDIDATE", question_id=None)


def test_degraded_event_can_have_missing_time_only_with_explanatory_flag():
    with pytest.raises(ValidationError):
        _event(start_time_ms=None, end_time_ms=None, quality_status="DEGRADED")

    event = _event(
        start_time_ms=None,
        end_time_ms=None,
        quality_status="DEGRADED",
        quality_flags=("TIME_UNAVAILABLE",),
    )
    assert event.quality_status == "DEGRADED"


def test_legacy_rectangle_is_separate_from_standard_v1_region():
    legacy = legacy_rectangle(
        region_id="legacy",
        page_id="page",
        question_id="Q1",
        x=5,
        y=5,
        width=10,
        height=10,
    )
    assert isinstance(legacy, LegacyQuestionRegion)
    assert legacy.coordinate_space == "legacy"
    with pytest.raises(ValidationError):
        QuestionRegion.model_validate(legacy.model_dump())

    schema = json.loads((ROOT / "contracts" / "question_region.schema.json").read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(legacy.model_dump(mode="json"))


def test_every_standard_v1_region_serialization_is_schema_valid():
    schema = json.loads((ROOT / "contracts" / "question_region.schema.json").read_text(encoding="utf-8"))
    region = QuestionRegion(
        region_id="r",
        page_id="page",
        question_id="Q1",
        region_type="polygon",
        polygon_norm=((0.1, 0.1), (0.4, 0.1), (0.2, 0.4)),
    )
    jsonschema.Draft202012Validator(schema).validate(region.model_dump(mode="json"))


def test_point_gate_output_and_event_cross_fields_match_json_schemas():
    point = Point(
        point_id="p",
        session_id="s",
        device_id="device",
        page_id="page",
        sequence=0,
        timestamp_ms=1,
        x_raw=1.0,
        y_raw=2.0,
        source_index=0,
        raw_order=0,
        processed_order=0,
        source_payload={"point_id": "p"},
    )
    validate_canonical_point_for_export(point)
    point_schema = json.loads((ROOT / "contracts" / "point.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(point_schema).validate(point.model_dump(mode="json"))

    event_schema = json.loads((ROOT / "contracts" / "student_event.schema.json").read_text(encoding="utf-8"))
    valid = _event(event_type="QUESTION_VISIT", question_id="Q1")
    jsonschema.Draft202012Validator(event_schema).validate(valid.model_dump(mode="json"))
    invalid = valid.model_dump(mode="json")
    invalid["start_time_ms"] = None
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(event_schema).validate(invalid)


def test_legacy_point_normalizes_pydantic_nested_payload_before_hashing():
    point = legacy_point(
        {
            "point_id": "p",
            "session_id": "s",
            "raw_index": 0,
            "normalized": NormalizedPoint(x=1.0, y=2.0, timestamp_ms=3),
        }
    )

    assert point.x_raw == 1.0
    assert point.source_payload["normalized"]["timestamp_ms"] == 3


def test_legacy_event_without_time_is_explicitly_degraded():
    event = legacy_event({"event_id": "e", "event_type": "UNKNOWN", "session_id": "s"})

    assert event.quality_status == "DEGRADED"
    assert "TIME_UNAVAILABLE" in event.quality_flags


def test_point_success_gate_rejects_empty_required_identifiers():
    with pytest.raises(ValidationError):
        Point(
            point_id="p",
            session_id="s",
            device_id="",
            page_id="page",
            sequence=0,
            timestamp_ms=1,
            x_raw=1.0,
            y_raw=2.0,
            source_index=0,
            raw_order=0,
            processed_order=0,
            source_payload={"point_id": "p"},
        )


def test_event_ids_and_reference_ids_cannot_serialize_as_empty_strings():
    with pytest.raises(ValidationError):
        _event(event_type="QUESTION_VISIT", question_id="")
    with pytest.raises(ValidationError):
        _event(point_refs=("",))
