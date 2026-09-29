# R1 Contract Alignment V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate R1 Point, Stroke, QuestionRegion, Student Event, Synthetic provenance, schemas, trace, tests and docs to Contract V1 without implementing R1-05 semantics or downstream systems.

**Architecture:** V1 Pydantic models are the sole standard cross-module objects. Explicit legacy adapters isolate old field names during migration. JSON Schema and Golden fixtures validate serialized V1 objects; the existing parser, stroke builder, detector, replay, trace and pipeline remain the behavioral backbone.

**Tech Stack:** Python >=3.11, Pydantic 2.x, JSON Schema Draft 2020-12, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-r1-contract-alignment-design.md`

## Global Constraints

- Contract version is `1.0.0`.
- Event types are exactly `WRITING`, `QUESTION_VISIT`, `QUESTION_LEAVE`, `RETURN`, `REVISION_CANDIDATE`, `PAGE_CHANGE`, `PROCESS_END`, `UNKNOWN`.
- Raw payload is immutable and hashable; unknown device semantics remain `null`/`UNKNOWN`/quality flags.
- Do not implement R1-05 algorithms, R2/R3/R4, Vision, Fusion, models, agents or frontend.
- Truth is independent from algorithm output; provenance/truth fields are never automatic features.

## Review Focus

- Raw coordinates and source payload remain unchanged when canonical fields are serialized — test in Task 1.
- Null calibration and pressure zero remain distinct from invalid values — test in Task 1.
- Legacy event fields cannot leak into the V1 serialized object — test in Task 4.
- Empty refs are allowed only where the event boundary is explicitly legal; no fake IDs are generated — test in Task 4/5.
- Synthetic manifest hashes are computed values, and truth/prediction objects remain separate — test in Task 5.

### Task 1: Point V1 model and adapter

**Files:**
- Modify: `src/datiao/r1/models/point.py`, `src/datiao/r1/parser/adapter.py`, `src/datiao/r1/parser/raw_point_parser.py`, `src/datiao/r1/models/__init__.py`
- Test: `tests/contracts/test_r1_contract_semantics.py`, `tests/contracts/test_r1_golden_examples.py`

**Interfaces:** `parse_raw_points()` returns immutable V1 `Point`; `CanonicalPointAdapter` requires explicit canonical fields and preserves extra source payload; optional `LegacyPointAdapter` maps only old internal shapes.

- [ ] Write failing tests for valid synthetic point, raw preservation/hash, participant null/known, pressure 0 vs null, invalid norm, missing required fields and unknown pen state.
- [ ] Run the focused contract tests and verify they fail on missing V1 fields.
- [ ] Implement V1 fields/validators and parser derivation with `x_mm/y_mm/x_norm/y_norm` null unless explicitly supplied and source index/order provenance preserved.
- [ ] Run focused tests and then existing point/parser tests; update old tests to assert V1 names through the explicit compatibility path.

### Task 2: Stroke V1 and QuestionRegion polygon contract

**Files:**
- Modify: `src/datiao/r1/models/stroke.py`, `src/datiao/r1/stroke/builder.py`, `src/datiao/r1/models/question_region.py`, `src/datiao/r1/mapper/mapping.py`
- Test: `tests/contracts/test_r1_contract_semantics.py`, `tests/contracts/test_r1_golden_examples.py`, `tests/r1/test_stroke_builder.py`, `tests/r1/test_process_pipeline.py`

**Interfaces:** `Stroke` exposes `point_refs`, `start_time_ms`, `end_time_ms`, `duration_ms`, `algorithm_version`, `provenance`; `QuestionRegion` exposes `region_type` and `polygon_norm`, while a named legacy constructor converts old rectangles.

- [ ] Write failing tests for resolvable refs, time ordering, raw/processed order, normalized polygon bounds, rectangle conversion and invalid polygon shape.
- [ ] Run focused tests and verify failures identify old field names/coordinate semantics.
- [ ] Implement minimal V1 stroke fields and polygon validation/conversion without adding advanced features or coordinate guesses.
- [ ] Update mapper and stroke builder to emit V1 fields while retaining current segmentation and mapping behavior.
- [ ] Run stroke, mapping and contract tests.

### Task 3: Student Event V1 and detector/pipeline migration

**Files:**
- Modify: `src/datiao/r1/models/event.py`, `src/datiao/r1/event/detector.py`, `src/datiao/r1/pipeline.py`, `src/datiao/r1/models/policy.py`
- Test: `tests/contracts/test_r1_contract_semantics.py`, `tests/r1/test_models.py`, `tests/r1/test_process_pipeline.py`

**Interfaces:** `StudentProcessEvent` exposes V1 event fields and `quality_status`; detector keeps current state-machine semantics but emits `start_time_ms/end_time_ms`, `point_refs/stroke_refs`, `algorithm_version`, and provenance. `WRITING` is accepted but not inferred by the old detector.

- [ ] Write failing tests for all 8 enum values, rejected values, time ordering, V1 refs, quality status/flags distinction, psychological-field rejection and no legacy fields in `model_dump()`.
- [ ] Run focused tests and verify expected failures.
- [ ] Implement V1 event model, strict provenance validation, and explicit legacy conversion helper.
- [ ] Update detector, trace call sites and pipeline quality handling; preserve current event ordering and UNKNOWN behavior.
- [ ] Run event, trace and pipeline tests.

### Task 4: JSON Schemas, Golden fixtures and serialization validation

**Files:**
- Create: `contracts/R1_POINT_DATA_CONTRACT_V1.md`, `contracts/point.schema.json`, `contracts/stroke.schema.json`, `contracts/question_region.schema.json`, `contracts/student_event.schema.json`, `contracts/golden/r1_point_example.json`, `contracts/golden/r1_stroke_example.json`, `contracts/golden/r1_event_example.json`
- Create: `tests/contracts/test_r1_schema_validation.py`, `tests/contracts/test_r1_golden_examples.py`
- Modify: `pyproject.toml` only if a JSON Schema test dependency is required

**Interfaces:** Schemas validate `model_dump(mode="json")` output under Draft 2020-12; Golden fixtures use Contract Version `1.0.0` and a real fixture SHA-256 manifest hash.

- [ ] Add schema/golden tests first, including placeholder hash rejection and Pydantic round-trip.
- [ ] Run them to verify failure before schema/model completion.
- [ ] Write strict schemas with required fields, enums, bounds, time/ref constraints and forbidden psychological fields.
- [ ] Add contract document describing status vs flags, breaking changes and compatibility boundaries.
- [ ] Run schema and golden tests.

### Task 5: Synthetic provenance, IDs, truth separation and trace integrity

**Files:**
- Modify: `src/datiao/r1/synthetic/models.py`, `src/datiao/r1/synthetic/generator.py`, `src/datiao/r1/trace/resolver.py`, `src/datiao/r1/synthetic/__init__.py`
- Test: `tests/contracts/test_r1_contract_semantics.py`, `tests/r1/test_synthetic.py`, `tests/r1/test_process_pipeline.py`

**Interfaces:** `generate_synthetic_case()` returns deterministic V1 points/events with `sim_session_`, `sim_p_`, `sim_segment_` identifiers and provenance containing six required fields; `resolve_event_trace()` resolves V1 refs to Stroke, Point and immutable Raw.

- [ ] Write failing tests for deterministic same-seed output, six provenance fields, actual manifest hash, session-relative time, truth/prediction separation and V1 trace resolution.
- [ ] Run focused tests and verify failure.
- [ ] Implement deterministic fixture manifest hashing and independent truth generation without flattening metadata into canonical features.
- [ ] Update trace resolver and synthetic IDs/records while keeping current scenario semantics.
- [ ] Run synthetic and trace tests.

### Task 6: Documentation, runbook and full verification

**Files:**
- Create: `docs/R1_TASK_R1_04.md`
- Modify: `docs/R1_RUNBOOK.md`, affected `tests/r1/*`

- [ ] Document old-to-new fields, eight event types, provenance, breaking changes, compatibility, limitations and R1-05 next step.
- [ ] Update runbook examples and test counts to match actual output.
- [ ] Install only required project test dependencies if absent, then run `python -m pytest tests/contracts -q`, `python -m pytest tests/r1 -q`, and `python -m pytest`.
- [ ] Record Python version, commit SHA, passed/failed/skipped counts and known limitations in the final report.

