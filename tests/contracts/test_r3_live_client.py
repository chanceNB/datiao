import hashlib
import json

import pytest

from datiao.r1.integration.r3_live import (
    R3LiveTransportError,
    parse_r3_response,
    post_r1_batch_to_r3,
    serialize_batch_request,
    validate_r3_response,
)
from datiao.r1.integration.r3 import export_batch_to_r3
from datiao.r1.models import StudentProcessEvent


def make_batch():
    event = StudentProcessEvent(
        event_id="sim_session_live:event:0001",
        event_type="RETURN",
        session_id="sim_session_live",
        task_segment_id="sim_segment_live",
        participant_id="sim_p_001",
        question_id="q-01",
        start_time_ms=100,
        end_time_ms=120,
        point_refs=("p-1",),
        stroke_refs=("s-1",),
        algorithm_version="r1-event-rule-v0.2.2",
        provenance={
            "dataset_type": "synthetic",
            "generator_version": "r1.synthetic.v1",
            "seed": 20260929,
            "scenario_id": "sim_live",
            "ground_truth_source": "scenario_plan",
            "manifest_hash": "sha256:" + "a" * 64,
        },
    )
    return export_batch_to_r3(
        (event,),
        batch_id="sim_live01_return_001",
        dataset_version="r1-synthetic-penprocess-v1@1.0.0",
        point_ids={"p-1"},
        stroke_ids={"s-1"},
    )


def response_body(**overrides):
    body = {
        "status": "ACCEPTED",
        "accepted": True,
        "ingestion_id": "ing-001",
        "batch_id": "sim_live01_return_001",
        "contract_version": "r1-event-batch-v0.1",
        "event_contract_version": "R1R3EventV01",
        "session_id": "sim_session_live",
        "task_segment_id": "sim_segment_live",
        "dataset_version": "r1-synthetic-penprocess-v1@1.0.0",
        "algorithm_version": "r1-event-rule-v0.2.2",
        "accepted_event_count": 1,
        "rejected_event_count": 0,
        "accepted_event_ids": ["sim_session_live:event:0001"],
        "payload_sha256": "a" * 64,
        "errors": [],
    }
    body.update(overrides)
    return body


def test_request_serialization_is_stable_and_hashable():
    batch = make_batch()
    first = serialize_batch_request(batch)
    second = serialize_batch_request(batch)
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
    assert json.loads(first) == batch.model_dump(mode="json")


def test_201_response_validates_atomicity_and_identity():
    batch = make_batch()
    response = parse_r3_response(201, response_body())
    validate_r3_response(batch, response, expected_status=201)
    assert response.status == "ACCEPTED"


def test_200_duplicate_requires_stable_ingestion_identity():
    batch = make_batch()
    response = parse_r3_response(
        200,
        {"status": "ALREADY_PRESENT_IDENTICAL", "batch_id": batch.batch_id, "ingestion_id": "ing-001"},
    )
    validate_r3_response(batch, response, expected_status=200, original_ingestion_id="ing-001")


@pytest.mark.parametrize(
    "status,code",
    [(422, "INVALID_SCHEMA"), (409, "BATCH_ID_CONFLICT"), (415, "UNSUPPORTED_MEDIA_TYPE"), (500, "INTERNAL_ERROR")],
)
def test_error_response_parser_preserves_http_and_code(status, code):
    response = parse_r3_response(
        status,
        {
            "status": "REJECTED",
            "accepted": False,
            "batch_id": "sim_live01_return_001",
            "accepted_event_count": 0,
            "rejected_event_count": 1,
            "accepted_event_ids": [],
            "code": code,
            "message": "rejected",
            "errors": [{"event_id": None, "field": "session_id", "code": code, "message": "rejected"}],
        },
    )
    assert response.http_status == status
    assert response.code == code
    assert response.accepted is False
    assert response.accepted_event_count == 0


def test_client_sends_same_body_and_required_headers():
    batch = make_batch()
    calls = []

    def transport(url, headers, body, timeout_s):
        calls.append((url, headers, body, timeout_s))
        return 201, json.dumps(response_body()).encode("utf-8")

    response = post_r1_batch_to_r3(batch, transport=transport)
    assert response.http_status == 201
    assert calls[0][0].endswith("/api/v1/r3/integrations/r1/event-batches")
    assert calls[0][1]["Content-Type"] == "application/json"
    assert calls[0][1]["Accept"] == "application/json"
    assert calls[0][2] == serialize_batch_request(batch)


def test_client_maps_transport_failure_without_retry():
    calls = []

    def transport(*_args):
        calls.append(1)
        raise TimeoutError("timed out")

    with pytest.raises(R3LiveTransportError, match="timed out"):
        post_r1_batch_to_r3(make_batch(), transport=transport)
    assert len(calls) == 1
