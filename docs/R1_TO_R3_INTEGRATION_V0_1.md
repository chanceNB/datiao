# R1 → R3 Integration V0.1

## Contract boundary

`StudentProcessEvent V1` remains the R1 Core Event contract. The transport DTO in `src/datiao/r1/integration/r3.py` is a projection for the R3 wire boundary. It has exactly the R3 fields and never changes the source event.

The existing R1 core contract and `contracts/golden/r1_event_example.json` remain separate. The R3 batch envelope is a transport proposal named `R1R3EventBatchV01`; it is not an R1 Core Event contract.

## Version alignment

- Current detector algorithm: `r1-event-rule-v0.2.2`, copied from `event.algorithm_version`.
- Current formal synthetic generator: `r1.synthetic.v1`, copied from provenance.
- R3 document examples such as `r1-event-rule-v0.1` and `pen-sim-v0.1` are examples only and are not substituted.
- Batch `dataset_version` is required from the caller. This repository's golden payload uses `r1-synthetic-penprocess-v1@1.0.0`.

## Projection policy

- `event_id` is the stable R1 ID and is preserved verbatim.
- `question_id` is copied when mapped; a missing core value projects to `"UNKNOWN"`. No nearest, previous, or next question is inferred.
- `task_segment_id` is copied, including `null`; the Adapter never constructs a value.
- `quality_status`, refs, timestamps, and algorithm version are copied.
- The projection rejects unresolved declared point or stroke refs. Empty refs remain empty for event types that permit them.
- Synthetic events must have `sim_` session and participant IDs (and `sim_` task segments when present) plus a real `sha256:<64 lowercase hex>` manifest hash.
- Synthetic provenance must include `dataset_type`, `generator_version`, `seed`, `scenario_id`, `ground_truth_source`, and `manifest_hash`.

## Semantic boundaries

- `RETURN` means a previously visited question was left and then entered again.
- `REVISION_CANDIDATE` means prior ink plus a pause or revisit plus overlap. It does not claim that a correction succeeded.
- `PROCESS_END` is propagated only when R1 emitted it from an explicit end signal or configured timeout. The Adapter never creates one from EOF, idle time, or missing points.
- `UNKNOWN` preserves an untrusted mapping and `"UNKNOWN"` question ID.

## Structural and semantic gates

The Event JSON Schema is the structural gate. `R1R3EventV01` is the semantic gate for time ordering, VALID timestamps, non-empty refs, synthetic provenance and IDs, and concrete algorithm versions. External point and stroke existence remains an exporter gate.

The Batch JSON Schema is `contracts/r1_to_r3_event_batch_v0_1.schema.json`. `R1R3EventBatchV01` repeats the semantic gate for non-empty batches, session, task segment and algorithm consistency, and concrete dataset and algorithm versions.

## Batch consistency

`export_batch_to_r3` requires a caller supplied `batch_id` and `dataset_version`. Every event in the batch must share `session_id`, `task_segment_id`, and `algorithm_version`; a mismatch fails export. Child event values are never overwritten.

## Public pipeline handoff

`export_process_result_to_r3(result, batch_id=..., dataset_version=...)` is the public offline handoff. It consumes `R1ProcessResult.student_process_events`, `points`, and `strokes`, checks the Result session, task segment, and optional `data_version`, then delegates to the existing batch exporter. It does not implement HTTP or network I/O.

## Golden payloads

- Single RETURN event: `contracts/golden/r1_to_r3_event_v0_1.json`
- Handoff copy: `artifacts/r1_r3_integration_v0_1/single_event_golden.json`
- Batch handoff copy: `artifacts/r1_r3_integration_v0_1/batch_golden.json`
- Schema: `contracts/r1_to_r3_event_v0_1.schema.json`
- Batch schema: `contracts/r1_to_r3_event_batch_v0_1.schema.json`
- Version-controlled batch golden: `contracts/golden/r1_to_r3_event_batch_v0_1.json`
