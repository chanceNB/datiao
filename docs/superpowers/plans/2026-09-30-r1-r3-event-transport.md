# R1 to R3 Event Transport Adapter and Golden Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a strict, trace-validated R1 to R3 event projection and batch envelope without modifying frozen R1 Event semantics or baselines.

**Architecture:** A new `datiao.r1.integration.r3` module owns immutable Pydantic DTOs, projection validation, and batch consistency checks. JSON Schema, golden payloads, integration tests, and handoff docs are kept separate from the R1 core contract. Golden data is generated from the existing synthetic `return_visit` pipeline.

**Tech Stack:** Python 3.11+, Pydantic 2, JSON Schema Draft 2020-12, pytest.

**Spec:** User-provided TASK-R1-R3-INT-01 pasted request.

## Global Constraints

- Do not modify `src/datiao/r1/models/event.py`, `contracts/student_event.schema.json`, frozen datasets/features/targets/splits, or baseline implementations.
- Preserve event types, `event.algorithm_version`, `event.provenance`, `event_id`, refs, and quality statuses.
- Synthetic transport requires `sim_` session and participant IDs and a real `sha256:<64 lowercase hex>` manifest hash.
- Unmapped question IDs project to `"UNKNOWN"` without guessing.
- PROCESS_END is propagated only when present on the Core Event.
- Trace refs must resolve; unknown refs fail export.
- Batch metadata is caller supplied and all events must be consistent.
- No training or dataset rebuild.

## Review Focus

- Projection must reject synthetic IDs or manifest placeholders: covered by adapter contract tests.
- Projection must reject unresolved point/stroke refs: covered by trace gate tests.
- Event immutability and semantic labels must survive projection: covered by no-mutation and semantic tests.
- Batch must reject inconsistent metadata instead of overwriting child values: covered by batch tests.
- Golden payload must validate against Draft 2020-12 strict schema: covered by golden validation test.

### Task 1: Define failing adapter and batch contract tests

**Files:**
- Create: `tests/contracts/test_r1_to_r3_event_contract.py`

**Interfaces:**
- Tests target `export_event_to_r3`, `R1R3EventV01`, `R1R3EventBatchV01`, and `export_batch_to_r3`.

- [ ] Write tests for all eight event types, UNKNOWN question projection, semantic preservation, synthetic gates, trace resolution, extra-field rejection, batch consistency, and no mutation.
- [ ] Run the new test file with the bundled Python and confirm it fails because the integration module is absent.

### Task 2: Implement strict R1 to R3 DTOs and projection

**Files:**
- Create: `src/datiao/r1/integration/__init__.py`
- Create: `src/datiao/r1/integration/r3.py`

**Interfaces:**
- `class R1R3EventV01(BaseModel)` with exactly the twelve wire fields and `extra="forbid"`.
- `class R1R3EventBatchV01(BaseModel)` with `contract_version`, `batch_id`, `session_id`, `task_segment_id`, `dataset_version`, `algorithm_version`, `events`.
- `export_event_to_r3(event, *, point_ids, stroke_ids) -> R1R3EventV01`.
- `export_batch_to_r3(events, *, batch_id, dataset_version) -> R1R3EventBatchV01`.

- [ ] Implement minimal projection using `event.event_id`, enum, IDs, times, refs, quality, algorithm version, and controlled provenance projection.
- [ ] Validate trace refs against supplied ID collections and synthetic `sim_`/manifest gates.
- [ ] Implement batch consistency validation without mutating child events.
- [ ] Run adapter tests and confirm green.

### Task 3: Add schema, golden payloads, docs, and pipeline integration coverage

**Files:**
- Create: `contracts/r1_to_r3_event_v0_1.schema.json`
- Create: `contracts/golden/r1_to_r3_event_v0_1.json`
- Create: `artifacts/r1_r3_integration_v0_1/single_event_golden.json`
- Create: `artifacts/r1_r3_integration_v0_1/batch_golden.json`
- Create: `docs/R1_TO_R3_INTEGRATION_V0_1.md`
- Create: `docs/R1_R3_INTEGRATION_CHECKLIST.md`
- Modify: `tests/contracts/test_r1_to_r3_event_contract.py`

- [ ] Generate a real `return_visit` case with seed `20260929`, select its RETURN event, and export it with actual refs.
- [ ] Export a batch with explicit dataset version and validate both payloads using Draft 2020-12.
- [ ] Add raw point → stroke → mapping → event → transport assertions for RETURN, UNKNOWN, REVISION_CANDIDATE, and PROCESS_END.
- [ ] Document Core vs wire semantics, version alignment, provenance, trace, and batch proposal status.

### Task 4: Verify freeze integrity and full suite

**Files:**
- No frozen files may change.

- [ ] Verify the freeze manifest hash is unchanged.
- [ ] Run `python -m pytest tests/r1 -q`, `python -m pytest tests/contracts -q`, `python -m pytest`, and `git diff --check` using the bundled Python executable where `python` is unavailable.
- [ ] Review `git diff` and report exact baseline and final commit.
