# TASK-R1-09 — Pen TCN Baseline

TASK-R1-09 adds the frozen CPU Pen TCN baseline `r1-pen-tcn-v1@1.0.0` for R1 point-process sequences. It consumes only `artifacts/r1_synthetic_features_v1`, verifies Feature Manifest `sha256:40b0326baf0f71206da1fd14b05c8ab5bdf1473c708735c57c019ea820cf5848`, and preserves the frozen 12-feature/8-label order.

The model receives 12 train-only normalized values plus their 12 binary feature masks (24 channels). It uses hidden widths `[32, 32]`, kernel `2`, dilations `[1, 2]`, two causal convolutions per block, ReLU, dropout `0.10`, and an 8-logit timestep head. Left-only padding gives receptive field 7. The fixed run uses CPU deterministic algorithms, seed `20260929`, AdamW, learning rate `1e-3`, weight decay `1e-4`, batch 32, max 200 epochs, patience 20, minimum improvement `1e-5`, gradient clipping 1.0, and threshold 0.5.

```powershell
$env:PYTHONPATH = "src"
python -m datiao.r1.tcn.train `
  --features artifacts/r1_synthetic_features_v1 `
  --output artifacts/r1_tcn_v1 `
  --config configs/r1_tcn_v1.json `
  --lightgbm-run artifacts/r1_lightgbm_v1 `
  --overwrite
```

Install all dependencies used by the R1 and TCN suites before running the command:

```powershell
python -m pip install -e ".[test,ml,dl]"
```

The runner performs two fresh deterministic executions internally and writes canonical metrics, causal diagnostics, predictions, normalization, sequence-length and causality audits, the model checkpoint, LightGBM comparison, model reload audit, and reproducibility audit under `artifacts/r1_tcn_v1/`. The artifact directory is ignored by Git. `reload_tcn_run(output, features)` explicitly reloads the frozen feature dataset and verifies source lineage, hashes, state identity, and per-episode validation/test probability equality.

Canonical metrics retain all eight labels for direct LightGBM comparison. Causal-observable diagnostics cover `WRITING`, `QUESTION_VISIT`, `RETURN`, `REVISION_CANDIDATE`, `PAGE_CHANGE`, and `UNKNOWN`; `QUESTION_LEAVE` remains retrospective and `PROCESS_END` remains external-signal dependent. Frozen sequence lengths are shorter than the receptive field, so the audit records `SHORT_SEQUENCE_LIMITED_CONTEXT` descriptively and does not alter the architecture.

TASK-R1-09-FIX-01 corrected semantic identity so checkpoint and environment artifact bytes do not alter `semantic_run_hash`, while `manifest_integrity_hash` and artifact hashes continue to protect files. It also changed validation and training-history BCE reporting to global valid-label-position weighting. See [R1_TASK_R1_09_FIX_01.md](R1_TASK_R1_09_FIX_01.md).
