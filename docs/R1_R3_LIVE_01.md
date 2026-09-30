# R1 ↔ R3 LIVE-01

## Frozen endpoint and wire

- Method: `POST`
- URL: `http://127.0.0.1:8000/api/v1/r3/integrations/r1/event-batches`
- Content-Type: `application/json`
- Auth: none for LIVE-01
- Body: direct `r1-event-batch-v0.1`, with no wrapper

The client in `src/datiao/r1/integration/r3_live.py` uses only Python standard-library HTTP APIs. It serializes one deterministic request body, sends `Content-Type` and `Accept` JSON headers, and performs no automatic business retry.

## Response handling

The client parses success, duplicate, and error responses into `R3LiveResponse` and `R3LiveErrorItem`. A 201 response is checked for `ACCEPTED`, atomic counts, event IDs, contract/version identity, ingestion ID, and optional payload hash. A 200 response is checked for `ALREADY_PRESENT_IDENTICAL`, the original batch ID, and stable ingestion ID. 409 and 422 responses must be atomic with zero accepted events.

`payload_sha256` accepts either 64 lowercase hex characters or the `sha256:` prefixed form. The local request hash is calculated separately; no equality with an R3 payload hash is assumed.

## Current readiness result

The TCP readiness probe for `127.0.0.1:8000` failed with connection refused. Therefore no live POST was sent and LIVE-01 is currently `BLOCKED_R3_NOT_READY`. This is an R3 environment blocker, not an R1 contract failure.

The exact IDs prepared for R3 registration are in `artifacts/r1_r3_live_v0_1/required_r3_ids.json`. The local evidence summary is in `artifacts/r1_r3_live_v0_1/live_validation_report.json`.

## Offline usage

```powershell
python -m datiao.r1.integration.r3_live `
  --batch contracts/golden/r1_to_r3_event_batch_v0_1.json `
  --base-url http://127.0.0.1:8000 `
  --timeout 5 `
  --output artifacts/r1_r3_live_v0_1/response_return.json
```

The command returns exit code `2` with `BLOCKED_R3_NOT_READY` when the endpoint cannot be reached. Live evidence files must come from actual request bytes and responses; they are not fabricated while R3 is unavailable.
