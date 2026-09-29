# TASK-R1-06-FIX-01 FINAL REPORT

## 1. STATUS

PASS

## 2. BASELINE

- commit: `690f996ea4f4dac8612d900476c6e8bd08cd4ab3`
- python: `3.12.14` bundled runtime (the system `python` command is unavailable)
- tests collected before: R1 82，Contract 31，Full 113
- tests passed before: R1 82，Contract 31，Full 113

## 3. MODIFIED FILES

- `src/datiao/r1/dataset/manifest.py`
- `src/datiao/r1/dataset/models.py`
- `src/datiao/r1/dataset/audit.py`
- `src/datiao/r1/dataset/io.py`
- `src/datiao/r1/dataset/split.py`
- `src/datiao/r1/dataset/builder.py`
- `tests/r1/test_dataset_builder.py`
- `tests/r1/test_dataset_integrity.py`
- `docs/R1_TASK_R1_06.md`
- `docs/R1_RUNBOOK.md`
- `docs/R1_TO_R4_SYNTHETIC_DATA_HANDOFF.md`

## 4. PER-CASE HASH

- definition: immutable case manifest content identity
- split excluded: yes; `split` and `case_manifest_hash` are removed before hashing
- validation: every case in train, validation and test is independently checked on build and reload
- split-seed invariance: verified; changing `split_seed` preserves every `case_manifest_hash` and `raw_records_hash`

## 5. DATASET CASE HASH

- definition: stable ordered collection hash of immutable case payloads, excluding split assignment and per-case hash fields
- hash: `sha256:06365a88845b3032b9a036f97d572934713b86243b790df5cbd8f322d9df56a6`

## 6. FILE HASH INTEGRITY

- actual verification: `verify_file_hashes()` reads every manifest-listed asset and computes SHA-256; Integrity no longer accepts a caller boolean
- tamper test: appending a byte to `points.jsonl` fails reload and Integrity; removing a case from `splits/test.json` also fails
- audit and summary JSON files are derived verification artifacts and are excluded from the content file hash list

## 7. SPLIT SEMANTIC INTEGRITY

- union: `train ∪ validation ∪ test` equals all `cases.jsonl` IDs
- disjoint: all three case ID sets are pairwise disjoint
- case labels: each `cases.jsonl.split` equals its split manifest membership
- participants: manifest participant IDs equal the actual split case participant set
- groups: manifest group IDs equal the actual split case source-group set; participant is the assignment unit
- counts: `count`, case ID list length and labeled case count must agree
- dataset identity: dataset ID/version, split seed/version and split filename/name are checked

## 8. RELOAD VALIDATION

- split manifests: loaded and validated as `SplitManifest` models
- recomputed leakage: PASS
- recomputed integrity: PASS
- stored audit comparison: both stored reports are compared with fresh recomputation; stale reports fail reload

## 9. MATERIALIZED DATASET

- cases: 360
- points: 1,640
- strokes: 700
- mappings: 700
- truth: 1,640
- predictions: 1,565

## 10. UPDATED HASHES

- dataset manifest: `sha256:c493f792f9add5547427d002eb74fa0cc0bb663f7f0bb9b19c2bd8c97674af82`
- case collection: `sha256:06365a88845b3032b9a036f97d572934713b86243b790df5cbd8f322d9df56a6`
- split manifest: `sha256:434580886db58540ad7b16e5dc777ad99063a7d16a7886237af258e9b1a85499`

## 11. LEAKAGE AUDIT

status: PASS

## 12. INTEGRITY AUDIT

status: PASS

## 13. TEST RESULTS

- R1 collected: 100
- R1 passed: 100
- Contract collected: 31
- Contract passed: 31
- Full collected: 131
- Full passed: 131

## 14. DOC CORRECTIONS

Updated R1-06, Runbook and R4 handoff with the corrected test counts, hash semantics, new materialized hashes and derived-audit status.

## 15. KNOWN LIMITATIONS

The split remains an R1 deterministic development candidate. R4 formal split approval and release freezing are pending. Synthetic truth is scenario-plan truth and is not real classroom evidence.

## 16. OUT OF SCOPE

Feature Builder: not changed
LightGBM: not changed
Pen TCN: not changed
R3: not changed
Vision: not changed
Fusion: not changed
Agent: not changed
Frontend: not changed

## 17. FINAL COMMIT

Recorded after final verification and remote synchronization.

## 18. NEXT RECOMMENDED TASK

TASK-R1-07 — Episode / Feature Builder
只建议，不执行。
