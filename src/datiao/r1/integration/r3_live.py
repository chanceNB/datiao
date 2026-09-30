"""HTTP transport for the frozen R1 to R3 Batch Wire V0.1 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .r3 import R1R3EventBatchV01

R3_BATCH_PATH = "/api/v1/r3/integrations/r1/event-batches"
Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]


class R3LiveErrorItem(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="allow")

    event_id: str | None = None
    field: str | None = None
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class R3LiveResponse(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="allow")

    http_status: int = Field(ge=100, le=599)
    status: str = Field(min_length=1)
    accepted: bool | None = None
    ingestion_id: str | None = None
    batch_id: str | None = None
    contract_version: str | None = None
    event_contract_version: str | None = None
    session_id: str | None = None
    task_segment_id: str | None = None
    dataset_version: str | None = None
    algorithm_version: str | None = None
    accepted_event_count: int = Field(default=0, ge=0)
    rejected_event_count: int = Field(default=0, ge=0)
    accepted_event_ids: tuple[str, ...] = ()
    payload_sha256: str | None = None
    code: str | None = None
    message: str | None = None
    errors: tuple[R3LiveErrorItem, ...] = ()

    @field_validator("accepted_event_ids", "errors", mode="before")
    @classmethod
    def normalize_json_arrays(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("payload_sha256")
    @classmethod
    def validate_payload_hash(cls, value: str | None) -> str | None:
        if value is not None:
            digest = value.removeprefix("sha256:")
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("payload_sha256 must be 64 lowercase hex characters, optionally prefixed sha256:")
        return value


class R3LiveTransportError(RuntimeError):
    """Raised when the R3 endpoint cannot be reached."""


class R3LiveResponseError(ValueError):
    """Raised when an R3 response is not valid JSON or its contract is invalid."""


def serialize_batch_request(batch: R1R3EventBatchV01) -> bytes:
    """Serialize one deterministic request body for retries and evidence."""

    return json.dumps(
        batch.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def request_body_sha256(request_bytes: bytes) -> str:
    return hashlib.sha256(request_bytes).hexdigest()


def parse_r3_response(http_status: int, body: bytes | str | Mapping[str, Any]) -> R3LiveResponse:
    if isinstance(body, Mapping):
        payload = dict(body)
    else:
        try:
            payload = json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise R3LiveResponseError(f"R3 response is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise R3LiveResponseError("R3 response JSON must be an object")
    try:
        return R3LiveResponse.model_validate({**payload, "http_status": http_status})
    except ValueError as error:
        raise R3LiveResponseError(str(error)) from error


def validate_r3_response(
    batch: R1R3EventBatchV01,
    response: R3LiveResponse,
    *,
    expected_status: int,
    original_ingestion_id: str | None = None,
    original_payload_sha256: str | None = None,
) -> R3LiveResponse:
    """Apply LIVE-01 response identity and atomicity gates."""

    if response.http_status != expected_status:
        raise R3LiveResponseError(f"expected HTTP {expected_status}, got {response.http_status}")
    if expected_status == 201:
        required = (
            response.status == "ACCEPTED",
            response.accepted is True,
            bool(response.ingestion_id),
            response.batch_id == batch.batch_id,
            response.contract_version == batch.contract_version,
            response.event_contract_version == "R1R3EventV01",
            response.session_id == batch.session_id,
            response.task_segment_id == batch.task_segment_id,
            response.dataset_version == batch.dataset_version,
            response.algorithm_version == batch.algorithm_version,
            response.accepted_event_count == len(batch.events),
            response.rejected_event_count == 0,
            response.accepted_event_ids == tuple(event.event_id for event in batch.events),
            bool(response.payload_sha256),
            response.errors == (),
        )
        if not all(required):
            raise R3LiveResponseError("201 response failed acceptance, identity, or atomicity validation")
    elif expected_status == 200:
        if response.status != "ALREADY_PRESENT_IDENTICAL" or response.batch_id != batch.batch_id or not response.ingestion_id:
            raise R3LiveResponseError("200 response failed identical retry validation")
        if original_ingestion_id is not None and response.ingestion_id != original_ingestion_id:
            raise R3LiveResponseError("duplicate response changed ingestion_id")
        if original_payload_sha256 is not None and response.payload_sha256 != original_payload_sha256:
            raise R3LiveResponseError("duplicate response changed payload_sha256")
    elif expected_status in {409, 422}:
        if response.accepted is not False or response.accepted_event_count != 0 or response.accepted_event_ids != ():
            raise R3LiveResponseError("error response was not atomic")
    return response


def post_r1_batch_to_r3(
    batch: R1R3EventBatchV01,
    *,
    base_url: str = "http://127.0.0.1:8000",
    timeout_s: float = 5.0,
    transport: Transport | None = None,
) -> R3LiveResponse:
    """POST one already validated Batch. No automatic business retries."""

    request_bytes = serialize_batch_request(batch)
    url = base_url.rstrip("/") + R3_BATCH_PATH
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    sender = transport or _urllib_transport
    try:
        http_status, response_body = sender(url, headers, request_bytes, timeout_s)
    except (TimeoutError, URLError, OSError) as error:
        raise R3LiveTransportError(str(error)) from error
    response = parse_r3_response(http_status, response_body)
    if http_status == 201:
        return validate_r3_response(batch, response, expected_status=201)
    if http_status == 200:
        return validate_r3_response(batch, response, expected_status=200)
    if http_status in {409, 422}:
        return validate_r3_response(batch, response, expected_status=http_status)
    return response


def _urllib_transport(url: str, headers: Mapping[str, str], body: bytes, timeout_s: float) -> tuple[int, bytes]:
    request = Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="POST an R1 R3 Batch Wire V0.1 payload")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    batch = R1R3EventBatchV01.model_validate_json(args.batch.read_text(encoding="utf-8"))
    try:
        response = post_r1_batch_to_r3(batch, base_url=args.base_url, timeout_s=args.timeout)
        payload = response.model_dump(mode="json")
        exit_code = 0 if response.http_status in {200, 201} else 1
    except R3LiveTransportError as error:
        payload = {"status": "BLOCKED_R3_NOT_READY", "message": str(error), "batch_id": batch.batch_id}
        exit_code = 2
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
