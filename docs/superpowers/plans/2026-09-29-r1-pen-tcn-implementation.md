# TASK-R1-09 Pen TCN Baseline Implementation Plan

## Scope and constraints

Implement only the approved Pen TCN baseline. The frozen Feature Manifest, Feature/Target/Sequence contracts, split membership, LightGBM parameters, threshold, and LightGBM artifact hashes remain unchanged. The official run consumes `reload_feature_dataset("artifacts/r1_synthetic_features_v1")`, uses CPU and one Torch thread, and compares LightGBM only after TCN training and evaluation are complete.

The bundled Python is 3.12.14 with NumPy 2.3.5 and scikit-learn 1.9.1; Torch is currently unavailable. Task 0 must make the deterministic DL runtime available before any Torch-dependent RED test. Add an optional `dl` dependency without changing existing `test` or `ml` pins, install a compatible CPU Torch runtime for the run, and record the exact version in `environment.json`.

## Task 0: Bootstrap deterministic DL environment

**Files:**
- Modify: `pyproject.toml`

- [ ] Add `[project.optional-dependencies].dl` with `torch>=2.2,<3.0`, the existing compatible NumPy constraint, and `scikit-learn>=1.4,<2.0`; do not alter `test` or `ml` dependency behavior.
- [ ] Install the CPU Torch package into the bundled Python 3.12.14 runtime using the project optional dependency path or the compatible CPU wheel source.
- [ ] Verify before creating or running TCN tests: `python -c "import torch; print(torch.__version__); print(torch.device('cpu'))"`.
- [ ] Record the exact Torch version for `environment.json`; do not create TCN production source, train a model, or materialize TCN artifacts in this task.
- [ ] Proceed to Task 1 only when the verification imports Torch successfully; otherwise stop with an environment BLOCKED status.

## Task 1: Add failing contract and audit tests

**Files:**
- Create: `tests/r1/test_tcn_data.py`
- Create: `tests/r1/test_tcn_model.py`
- Create: `tests/r1/test_tcn_training.py`
- Create: `tests/r1/test_tcn_reproducibility.py`
- Create: `tests/r1/test_tcn_reload.py`

**Interfaces:**
- Tests consume the public functions and classes that later tasks define in `datiao.r1.tcn.data`, `model`, `train`, and `io`.
- Tests assert the frozen Feature Manifest, 360/700 counts, 252/54/54 sequence split, 490/105/105 real timestep split, feature/label order, and no test loader in `fit`.

- [ ] Write RED tests for `sequence_length_audit`, target-causality audit labels/classes, right-padding collator shapes, binary padding mask, and episode ownership.
- [ ] Write RED tests for missing-vs-real-zero channels, Train-only normalization, normalization serialization, zero-variance handling, and validation/test mutation invariance.
- [ ] Write RED tests for causal future perturbation, padding invariance, receptive field calculation, `[B,T,8]` forward shape, real optimizer parameter update, finite logits/loss/gradients, and masked loss invariance.
- [ ] Write RED tests for best-checkpoint training metadata, complete episode predictions, canonical metric compatibility, causal diagnostics, save/reload probability identity by episode ID, validation/test LightGBM ID equality, two-run semantic hashes, and all required tamper cases.
- [ ] Run each new test file after Task 0 and confirm failures are caused by missing TCN interfaces or expected behavioral failures, never `ModuleNotFoundError: torch`.

## Task 2: Add configuration and shared evaluator compatibility

**Files:**
- Create: `configs/r1_tcn_v1.json`
- Create: `src/datiao/r1/tcn/__init__.py`
- Create: `src/datiao/r1/tcn/config.py`
- Create: `src/datiao/r1/evaluation/__init__.py`
- Create: `src/datiao/r1/evaluation/multilabel.py`
- Modify: `src/datiao/r1/lightgbm/metrics.py`

**Interfaces:**
- `load_tcn_config(path: str | Path) -> TCNConfig`.
- `TCNConfig` exposes the fixed values: baseline id/version, seed, device, batch size, max epochs, learning rate, weight decay, patience, min delta, gradient clip, threshold, hidden channels, kernel, dilations, convolutions per block, and dropout.
- `calculate_metrics` remains importable from `datiao.r1.lightgbm.metrics` and is the same implementation re-exported by `datiao.r1.evaluation.multilabel`.

- [ ] Run RED config and LightGBM hash regression tests.
- [ ] Implement strict frozen config validation and JSON loading with no tuning fields.
- [ ] Extract or re-export the existing evaluator without changing output field names, numeric behavior, or LightGBM metrics hash.
- [ ] Run the focused shared-evaluator and existing LightGBM tests; confirm the frozen LightGBM metrics and run hashes remain unchanged.

## Task 3: Implement frozen sequence input, audit, and normalization

**Files:**
- Create: `src/datiao/r1/tcn/data.py`
- Create: `src/datiao/r1/tcn/manifest.py`

**Interfaces:**
- `load_tcn_dataset(features: str | Path) -> TCNDataset` calls `reload_feature_dataset`, validates the frozen Feature Manifest and input integrity, and exposes sequences grouped by split.
- `fit_normalizer(train_sequences: Sequence[SequenceSample]) -> Normalizer` uses only supplied Train sequences and real timesteps where feature mask is 1.
- `Normalizer.transform(values, feature_mask) -> normalized_values` preserves missing zero values and masks.
- `collate_sequences(batch) -> dict[str, Tensor | list[str]]` returns raw values `[B,T,12]`, feature masks `[B,T,12]`, targets `[B,T,8]`, padding mask `[B,T]`, episode IDs, and sequence IDs; it never normalizes values.
- `prepare_model_input(values, feature_mask, normalizer) -> Tensor` is the sole shared raw-to-model path: Train-fitted normalization, missing-zero preservation, feature-mask concatenation, and `[B,T,24]` output.
- `build_sequence_length_audit(sequences, receptive_field) -> dict` produces overall/train/validation/test counts, real timestep counts, min/mean/median/p90/max, T thresholds, RF coverage, and warning.

- [ ] Implement lineage validation against manifest `sha256:40b0326b...cf5848`, exact frozen order, binary masks/targets, unique episode ownership, and no metadata/prediction channels.
- [ ] Implement Train-only means/stds/observed counts, scale 1.0 for near-zero std, `normalization_hash`, and JSON serialization with `fit_split="train"`.
- [ ] Implement right padding with zero placeholders and a separate padding mask; never concatenate padding mask to model channels.
- [ ] Implement target causality audit artifact with the six `CAUSAL_OBSERVABLE`, one `RETROSPECTIVE_TRANSITION`, and one `EXTERNAL_SIGNAL_DEPENDENT` labels and status `PASS_WITH_NON_CAUSAL_LABELS`.
- [ ] Freeze the order `reload Feature Dataset -> select Train sequences -> fit_normalizer(Train only) -> create Train/Validation loaders`; Test lengths and split membership may be read for integrity/length audit only.
- [ ] Run `test_tcn_data.py` RED-to-GREEN and verify that mutating validation/test values leaves normalization semantics unchanged.

## Task 4: Implement fixed causal TCN and masked loss

**Files:**
- Create: `src/datiao/r1/tcn/model.py`

**Interfaces:**
- `CausalConv1d(in_channels: int, out_channels: int, kernel_size: int, dilation: int)` applies left-only padding `(kernel_size - 1) * dilation` and `Conv1d(padding=0)`.
- `PenTCN(input_channels=24, hidden_channels=(32,32), kernel_size=2, dilations=(1,2), dropout=0.1, output_channels=8) -> nn.Module` returns logits `[B,T,8]`.
- `compute_receptive_field(kernel_size, dilations, convs_per_block=2) -> int` derives RF from actual causal convolution definitions and returns the expected fixed result at runtime.
- `masked_bce_with_logits(logits, targets, padding_mask) -> Tensor` masks only padding and divides by valid label positions.
- `model_state_hash(model) -> str` hashes sorted state-dict key, dtype, shape, and CPU-contiguous raw tensor bytes.

- [ ] Implement the fixed two-block residual architecture, 1x1 projections where needed, ReLU, dropout, and Linear(32,8) head with no softmax.
- [ ] Implement finite checks and explicit errors for non-finite logits, loss, or gradients.
- [ ] Run model RED-to-GREEN tests for causality, padding invariance, RF, output shape, real parameter update, finite values, and padding loss invariance.

## Task 5: Implement isolated single-run training, inference, and artifacts

**Files:**
- Create: `src/datiao/r1/tcn/train.py`

**Interfaces:**
- `fit(model, train_loader, validation_loader, normalizer, config) -> TrainingResult` accepts no test loader and returns best epoch, last epoch, best validation loss, history, and an immutable best state snapshot.
- `evaluate(model, loader, normalizer, threshold) -> EvaluationResult` filters only real timesteps and emits stable episode/timestep prediction records.
- `evaluate(model, loader, normalizer, threshold) -> EvaluationResult` uses `prepare_model_input` for every split and runs `model.eval()` under `torch.no_grad()`.
- `execute_training_once(data: TCNDataset, output: str | Path, config: TCNConfig) -> SingleRunResult` performs one fresh Train/Validation fit and final post-checkpoint Validation/Test inference; it does not compare LightGBM, reload, compute final run hash, or launch Run2.
- Task 5 exposes only `execute_training_once(...)`; final `run_baseline(...)`, comparison, reload, semantic run hash, and Run2 orchestration belong to Task 6.

- [ ] Add deterministic setup resetting Python, NumPy, Torch, deterministic algorithms, one thread, and a fresh seeded DataLoader generator for every run.
- [ ] Fit only Train and Validation; use AdamW, unweighted masked BCE, gradient clipping, validation BCE early stopping, and best checkpoint restoration.
- [ ] When validation BCE improves, snapshot `{key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}`; never retain the live `state_dict()` mapping. Set `model.eval()` and `torch.no_grad()` for validation, final inference, padding, causality, and reload paths; use `model.train()` only for optimization.
- [ ] Write `training_history.json`, `sequence_length_audit.json`, `target_causality_audit.json`, `normalization.json`, `metrics.json`, stable `predictions.jsonl`, config, environment, and input/normalizer/causality/padding audits.
- [ ] Enforce 700 unique predictions total with 490/105/105 split counts and no padded timestep prediction.
- [ ] Generate canonical eight-label metrics and local causal-observable diagnostics; compute the matching LightGBM diagnostics only in Task 6 after training.
- [ ] Keep Test tensors and Test DataLoader out of fit, normalization, optimizer, early stopping, architecture/config selection, threshold selection, and checkpoint selection. Test split IDs, lengths, and real timestep counts may be read for the pre-training frozen integrity/length audit; create Test inference tensors only after restoring the best Validation checkpoint.
- [ ] Run Task 5 tests for best-state aliasing by taking one optimizer step after snapshot and proving the saved tensors remain unchanged.

## Task 6: Implement comparison, reload, hashes, and tamper detection

**Files:**
- Create: `src/datiao/r1/tcn/io.py`
- Modify: `src/datiao/r1/lightgbm/io.py` only if a compatibility helper is needed; preserve its existing behavior.

**Interfaces:**
- `run_baseline(features: str | Path, output: str | Path, config: TCNConfig, lightgbm_run: str | Path, overwrite=False) -> dict` performs final orchestration.
- `reload_tcn_run(output: str | Path, features: str | Path) -> dict` requires the frozen Feature Dataset path, reloads normalization and a fresh model, recomputes Validation/Test probabilities, and validates all artifact hashes.
- `inspect_tcn_run_artifacts(output: str | Path) -> dict` is the separate artifact-only hash inspection API, if needed; it never claims probability reload.
- `compare_with_lightgbm(tcn_result, lightgbm_run) -> dict` requires exact Validation/Test Episode ID sets, strict target-vector equality, and exact `LABEL_ORDER` equality before comparing canonical plus diagnostic metrics.
- `semantic_run_hash(manifest_without_hash) -> str` excludes model file bytes and volatile environment fields.

- [ ] Save `best_model.pt`, model file hash, model state hash, normalization semantic hash, predictions/metrics/diagnostic/comparison hashes, and semantic run hash.
- [ ] Execute in dependency order: input integrity -> fresh `execute_training_once` Run1 -> post-training LightGBM comparison -> comparison hash -> semantic manifest -> semantic run hash -> fresh `reload_tcn_run` -> fresh `execute_training_once` Run2 -> recompute comparison/hash values -> semantic reproducibility comparison.
- [ ] Ensure `comparison_lightgbm.json` contains baseline IDs, source Feature Manifest, model state hash, LightGBM run hash, exact validation/test ID and target match fields, canonical metrics, diagnostics, and deltas, but never the final TCN semantic run hash; this prevents a comparison-hash/run-hash cycle.
- [ ] Reload a fresh model and frozen dataset, recompute validation/test probabilities, and align by episode ID with max diff `<=1e-12`.
- [ ] Run two entirely fresh training executions, each resetting Python/NumPy/Torch/DataLoader RNG and creating fresh model, optimizer, loaders, and early stopping; compare prediction, metrics, diagnostics, model state, and semantic run hashes, and report model file hash differences separately.
- [ ] Add tamper failures for checkpoint, normalization, predictions, metrics, diagnostics, comparison, and run manifest.
- [ ] Run `test_tcn_reproducibility.py` and `test_tcn_reload.py` RED-to-GREEN.

## Task 7: Documentation and official run

**Files:**
- Create: `docs/R1_TASK_R1_09.md`
- Create: `docs/R1_TO_R4_TCN_HANDOFF.md`
- Modify: `docs/R1_RUNBOOK.md`

- [ ] Document actual target-causality classes, temporal length distribution, effective context warning, canonical/diagnostic metric distinction, and all limitations.
- [ ] Add install and CLI instructions for `.[test,dl]` and `.[test,ml,dl]`.
- [ ] Document and run `python -m datiao.r1.tcn.train --features ... --output ... --config ... --lightgbm-run ... --overwrite`; the CLI delegates to Task 6 `run_baseline`.
- [ ] Run the official command against the frozen feature dataset and LightGBM run with `--overwrite`.
- [ ] Run it a second independent time in a temporary output, then call `reload_tcn_run` on the formal output.
- [ ] Confirm no Feature Dataset files changed and no TCN artifact is tracked by Git.

## Task 8: Final verification, commit, and sync

- [ ] Run focused TCN tests, existing R1 tests, contract tests, and full pytest.
- [ ] Re-run `git diff --check`, inspect `git status`, and verify the final artifact manifest and hashes.
- [ ] Report actual R1/Contracts/Full collected and passed counts, Torch version, RF/length audit, canonical and diagnostic metrics, comparison, reload, reproducibility, limitations, and final commit.
- [ ] Commit the implementation and documentation, then push to `origin/main`.

## Self-review

- Task 0 precedes every Torch-dependent RED test and records the exact runtime; frozen input, target, split, feature order, label order, and LightGBM regression are covered by Tasks 2 and 3.
- The raw collator, Train-only normalizer, and shared `prepare_model_input` path remove normalization ambiguity; Task 5 is single-run only, while Task 6 owns comparison, reload, semantic hash, and Run2.
- Target causality, RF, short-sequence warning, shared metrics, semantic hashes, independent runs, episode-ID reload, strict target/label-order comparison, and Validation/Test isolation are covered by Tasks 3, 5, and 6.
- All required tests and artifacts are assigned to concrete tasks; no task changes the approved architecture or frozen target.
- The only expected external setup is installing the optional CPU Torch dependency because the current bundled runtime has no Torch.
