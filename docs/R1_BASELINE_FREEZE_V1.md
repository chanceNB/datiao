# R1 Baseline Freeze V1

`r1-baseline-freeze-v1@1.0.0` is the read-only handoff container for the two completed R1 baselines. It records the exact synthetic dataset, feature contract, LightGBM OVR run, causal Pen TCN run, metrics, support counts, target causality audit, temporal-context audit, limitations, and report hashes.

The canonical manifest is [`manifests/r1_baseline_freeze_v1.json`](../manifests/r1_baseline_freeze_v1.json). Its current `freeze_manifest_hash` is `sha256:9a6705af7d918d5d22cac718adda8efe3341786223042e89bdc6b5dfb9bea5ef`. Materialized reports are under `artifacts/r1_baseline_freeze_v1/` and are intentionally ignored by Git because they contain model and prediction artifacts. The freeze was built from source commit `2846c7f7346ebe36f12fe63fb5813e2d230c162a`.

| Item | Frozen identity |
| --- | --- |
| Dataset | `r1-synthetic-penprocess-v1@1.0.0`, manifest `sha256:c493f792f9add5547427d002eb74fa0cc0bb663f7f0bb9b19c2bd8c97674af82` |
| Feature dataset | `r1-synthetic-features-v1@1.0.0`, manifest `sha256:40b0326baf0f71206da1fd14b05c8ab5bdf1473c708735c57c019ea820cf5848` |
| LightGBM | `r1-lightgbm-ovr-v1@1.0.0`, run `sha256:3b6235b64974c244fe3dc80f81798d815404dc48786323ca1fbb94ead33118e0` |
| Pen TCN | `r1-pen-tcn-v1@1.0.0`, semantic run `sha256:1ff84b184c64240cfe4b0f97b0490ba025879b2ac30aa537f445e7ab86a11f62` |

LightGBM and Pen TCN remain complementary baselines. The comparison report deliberately has no winner or ranking field. Canonical eight-label metrics are retained alongside causal diagnostics; `QUESTION_LEAVE` is retrospective and `PROCESS_END` needs an external signal.

The version-controlled `metrics` object is explicitly namespaced as `metrics.lightgbm` and `metrics.pen_tcn`, each with `validation` and `synthetic_test` aggregates. `comparison_summary` stores exact TCN-minus-LightGBM deltas, and `causal_diagnostics` stores the six causal-observable labels plus both model aggregates for validation and synthetic test. `TASK-R1-10-FIX-01` records this canonical metrics completeness correction; the freeze id and version remain V1.

Validate the materialized freeze with:

```powershell
$env:PYTHONPATH = "src"
python -m datiao.r1.freeze.validator
```

Rebuild it only from the already-frozen inputs with:

```powershell
$env:PYTHONPATH = "src"
python -m datiao.r1.freeze.builder --overwrite
```

The builder performs reload and integrity checks only. It has no training, optimizer, checkpoint-selection, or tuning entrypoint. The synthetic-only shortcut risk, all-positive `WRITING` label, short sequences (`max T=3`, TCN receptive field `7`), epoch-cap boundary, retrospective label, and external-signal dependency are recorded in `limitations.json` and must remain visible in downstream reports.
