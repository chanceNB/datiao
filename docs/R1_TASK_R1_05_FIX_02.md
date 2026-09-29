# TASK-R1-05-FIX-02 FINAL REPORT

## 1. STATUS

PASS

## 2. BASELINE

- commit: `4ae1149f719c937c5a41d286d00b2b783bcf48cb`
- Python: `3.12.14` (bundled runtime)
- tests before: `tests/r1` 68，`tests/contracts` 31，full 99

## 3. MODIFIED FILES

- `src/datiao/r1/stroke/builder.py`
- `src/datiao/r1/event/detector.py`
- `src/datiao/r1/pipeline.py`
- `src/datiao/r1/models/legacy.py`
- `src/datiao/r1/synthetic/generator.py`
- `src/datiao/r1/synthetic/manifest.py`
- `tests/r1/test_context_boundaries.py`
- `tests/r1/test_event_semantics_v2.py`
- `tests/r1/test_process_pipeline.py`
- `tests/r1/test_stroke_semantics.py`
- `docs/R1_TASK_R1_05_FIX_02.md`
- `docs/R1_TASK_R1_05.md`
- `docs/R1_RUNBOOK.md`

## 4. DEVICE BOUNDARY

- stroke context: `_same_context()` compares session, participant, task segment, page and `device_id`.
- raw coordinate domain: Stroke provenance records `device_id`, `raw_coordinate_domain` and bbox coordinate space.
- unknown device: `device_id=None` remains an internal degraded/unknown domain and is never promoted to a real device identity.
- revision comparison: raw bbox history includes the device domain; different domains and `unknown` raw domains provide no cross-Stroke IoU evidence.

## 5. TASK SEGMENT ISOLATION

- state key: `(participant_bucket, task_segment_bucket)` with explicit unknown buckets.
- VISIT/RETURN: visited questions and active question state do not cross segments.
- PAGE_CHANGE: previous page state is segment-scoped.
- Revision history: participant, task segment, question, page, coordinate space and coordinate domain are all part of the history key.

## 6. EVENT SEGMENT PROPAGATION

- mapping source: the event uses the source mapping's `task_segment_id`; QUESTION_LEAVE keeps the previous active mapping's segment.
- caller fallback: the detector function argument is used only when the source and current mapping have no segment.
- conflict behavior: caller/mapping conflicts raise `ValueError` instead of being silently overwritten; pipeline point context is checked as well.

## 7. PROCESS_END

- explicit: `process_end_signal=True` emits one end event for every observed participant/segment state.
- timeout: configured inactivity timeout is evaluated independently for each participant/segment's last mapping.
- participant+segment behavior: each event references that state’s own final mapping and segment.

## 8. SYNTHETIC DEVICE

- device_id: synthetic raw records, Points, Stroke provenance and optional Manifest field use `sim_pen_001`.
- Canonical export gate: synthetic Points pass `validate_canonical_point_for_export`.

## 9. TRACE / REPLAY

Mixed participant, segment and device inputs retain source Stroke/Point references; existing trace and replay regression tests pass.

## 10. TEST RESULTS

- tests/r1: 77 passed
- tests/contracts: 31 passed
- full: 108 passed

## 11. REGRESSIONS FIXED

- Prevented Stroke joining across devices with identical session/participant/page/segment context.
- Prevented raw-coordinate revision comparisons across unknown or different device domains.
- Prevented segment state, page state and prior ink from leaking between task segments.
- Preserved source mapping segment IDs on emitted events and rejected conflicting caller context.
- Made explicit and timeout process-end emission segment-aware.

## 12. KNOWN LIMITATIONS

Raw-coordinate overlap remains unavailable when the device domain is unknown. Norm-coordinate overlap remains comparable after normalization; geometry thresholds still require later Dev/Validation calibration.

## 13. OUT OF SCOPE CONFIRMATION

Dataset: not changed  
Split: not changed  
Feature Builder: not changed  
LightGBM: not changed  
Pen TCN: not changed  
R3: not changed  
Vision: not changed  
Fusion: not changed  
Agent: not changed  
Frontend: not changed

## 14. FINAL COMMIT

SHA: recorded after final commit and remote `main` synchronization.

## 15. NEXT RECOMMENDED TASK

TASK-R1-06 — Synthetic Dataset / Manifest / Group Split  
只建议，不执行。
