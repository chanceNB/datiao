# TASK-R1-06 FINAL REPORT

## 1. STATUS

PASS

## 2. DATASET IDENTITY

- dataset_id: `r1-synthetic-penprocess-v1`
- dataset_version: `1.0.0`
- dataset_type: `synthetic`
- generator_version: `r1.synthetic.v1`
- contract_version: `1.0.0`
- split status: R1 deterministic candidate split; R4 formal approval is pending.

This is a development dataset for engineering and reproducibility. It is not real student data or formal research evidence.

## 3. DATASET CONFIG

- config: `configs/r1_synthetic_dataset_v1.json`
- participants: 20 (`sim_p_001` through `sim_p_020`)
- scenario types: all 18 configured R1 scenarios
- task families: `navigation`, `revision`, `quality`, `ending`, `geometry`
- dataset seed: `20260929`
- split seed: `20260930`
- development split ratios: train `0.70`, validation `0.15`, test `0.15`
- default device: `sim_pen_001`

Case seeds are derived with SHA-256 from dataset seed, participant index and scenario index. No Python builtin hash or random UUID is used.

## 4. MATERIALIZED DATASET

- output: `artifacts/r1_synthetic_penprocess_v1/`
- cases: 360
- points: 1,640
- strokes: 700
- mappings: 700
- truth events: 1,640
- predicted events: 1,565

The generated directory uses deterministic UTF-8 JSON/JSONL. Truth and R1 rule predictions are separate files: `truth_events.jsonl` and `predicted_events.jsonl`.

## 5. DIRECTORY AND MANIFEST

The output contains `dataset_manifest.json`, `dataset_summary.json`, `config_snapshot.json`, aggregated JSONL files for raw records, Points, Strokes, mappings, regions, truth and predictions, three split files, and `audits/leakage_audit.json` plus `audits/integrity_audit.json`.

- dataset manifest hash: `sha256:c493f792f9add5547427d002eb74fa0cc0bb663f7f0bb9b19c2bd8c97674af82`
- config hash: `sha256:ee0bb06a5d21c0472d52fe0a707784a670e6a4895eefaca52390b9f159a3c385`
- case manifest hash: `sha256:06365a88845b3032b9a036f97d572934713b86243b790df5cbd8f322d9df56a6`
- split manifest hash: `sha256:434580886db58540ad7b16e5dc777ad99063a7d16a7886237af258e9b1a85499`
- core file SHA-256 values are recorded in `dataset_manifest.json`.

The manifest hash excludes its own `manifest_hash` field. Per-case hashes and the aggregate case collection hash exclude split assignment and the hash field itself. Paths stored in the manifest are relative only. Summary and audit JSON files are derived verification artifacts and are excluded from the content file hash list.

## 6. GROUP SPLIT

- group unit: `participant_id`; all cases from one participant stay in one split.
- train: 14 participants, 252 cases
- validation: 3 participants, 54 cases
- test: 3 participants, 54 cases
- actual ratios: `0.70 / 0.15 / 0.15`

`group_id` identifies the participant plus source scenario plan for future variant tracking; split assignment uses the participant group. This is a deterministic R1 candidate split and future R4 review may replace it.

## 7. LEAKAGE AUDIT

- participant overlap: empty
- session overlap: empty
- group overlap: empty
- case overlap: empty
- duplicate raw source across splits: empty
- status: `PASS`

The audit also has negative tests that intentionally put one participant and one raw hash in two splits and require `FAIL`.

## 8. INTEGRITY AUDIT

- refs: PASS
- truth: PASS
- prediction: PASS
- hashes: PASS
- split completeness: PASS
- dataset type: all `synthetic`
- status: `PASS`

Planned degraded scenarios such as missing timestamps, unknown regions and duplicate points remain present and are counted; they do not cause Dataset Build failure.

## 9. QUALITY AND COVERAGE

- canonical export eligible points: 1,620
- degraded points: 111
- mappings: 620 `MAPPED`, 80 `UNKNOWN`
- event quality: 1,385 `VALID`, 180 `DEGRADED`
- prediction `UNKNOWN` events: 80
- scenario counts: every configured scenario has 20 cases
- family counts: navigation 80, revision 100, quality 100, ending 40, geometry 40

Truth and prediction distributions remain separate in `dataset_summary.json`; no formal Precision/Recall/F1 is calculated.

## 10. RELOAD AND DETERMINISM

`reload_dataset()` validates the Dataset Manifest hash, every recorded core file hash, split manifests against `cases.jsonl`, per-case and aggregate case hashes, Point/Stroke/QuestionRegion/Mapping/Event models, then recomputes Leakage and Integrity audits and compares them with the stored reports. Same-config builds produce identical file hashes and manifest hashes. Changing only `split_seed` leaves generated content and case hashes unchanged while changing `cases.jsonl`, split files and the split/overall manifest hashes; changing `dataset_seed` changes raw-record and case content hashes.

## 11. R4 HANDOFF

See `docs/R1_TO_R4_SYNTHETIC_DATA_HANDOFF.md`. R4 formal approval and release freezing are pending; no R4 approval is claimed.

## 12. KNOWN LIMITATIONS

- The split is a development candidate and is not a formal research protocol.
- Synthetic scenario truth is scenario-plan truth, not human annotation.
- Planned degraded samples exercise data-quality paths and should not be treated as real classroom prevalence.
- No model features, training, metrics or scientific conclusions are produced here.

## 13. OUT OF SCOPE CONFIRMATION

Feature Builder, LightGBM, Pen TCN, model training, metrics, R3, Vision, Fusion, Agent and Frontend were not developed.

## 15. NEXT RECOMMENDED TASK

TASK-R1-07 — Episode / Feature Builder
只建议，不执行。

## 14. TEST RESULTS

- tests/r1: 100 passed
- tests/contracts: 31 passed
- full: 131 passed

## 15. TASK-R1-06-FIX-01

PASS. The follow-up fixed case hash semantics, actual file SHA-256 verification, split manifest cross-checks, reload recomputation and tamper detection. See `docs/R1_TASK_R1_06_FIX_01.md` for the detailed report.
