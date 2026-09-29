import json
from pathlib import Path
import jsonschema
from datiao.r1.models import Point, Stroke, QuestionRegion, StudentProcessEvent

ROOT = Path(__file__).parents[2]


def test_all_v1_golden_examples_validate_against_draft_2020_12():
    pairs = (("point", Point), ("stroke", Stroke), ("question_region", QuestionRegion), ("student_event", StudentProcessEvent))
    for name, model in pairs:
        schema_path = ROOT / "contracts" / f"{name}.schema.json"
        golden_name = "event" if name == "student_event" else name
        golden_path = ROOT / "contracts" / "golden" / f"r1_{golden_name}_example.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        instance = json.loads(golden_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(instance)
        model.model_validate_json(golden_path.read_text(encoding="utf-8"))


def test_serialized_v1_models_round_trip_without_legacy_fields():
    event = StudentProcessEvent(
        event_id="sim_evt_000001", event_type="RETURN", session_id="sim_session_001", question_id="Q05",
        start_time_ms=1, end_time_ms=2, point_refs=("p",), stroke_refs=("s",),
        provenance={"dataset_type": "synthetic", "generator_version": "v", "seed": 1,
                    "scenario_id": "scenario", "ground_truth_source": "scenario_plan",
                    "manifest_hash": "sha256:" + "b" * 64},
    )
    dumped = event.model_dump(mode="json")
    assert "occurred_at_ms" not in dumped
    assert "source_point_ids" not in dumped
    assert StudentProcessEvent.model_validate_json(json.dumps(dumped)) == event

