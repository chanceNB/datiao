# TASK-R1-09 Pen TCN Baseline Design

**Design status:** `TASK-R1-09 DESIGN READY`; implementation has not started.

## 1. Goal and fixed scope

Implement one reproducible Pen TCN baseline over the frozen R1 Feature Dataset `r1-synthetic-features-v1@1.0.0`. The model consumes only `SequenceSample.values`, `feature_mask`, `targets`, `episode_ids`, and `split`; it does not rebuild features, labels, or splits and never consumes rule or LightGBM predictions as input. The run compares TCN outputs with the frozen LightGBM baseline only after TCN training completes.

The baseline uses CPU, one Torch thread, seed `20260929`, batch size 32, AdamW (`1e-3`, `1e-4`), unweighted masked BCEWithLogitsLoss, gradient clipping at 1.0, maximum 200 epochs, validation-BCE patience 20, min delta `1e-5`, and threshold 0.5. There is no architecture or threshold search.

## 2. Data flow and lineage

`reload_feature_dataset("artifacts/r1_synthetic_features_v1")` is the only formal input boundary. The loader validates the frozen source Feature Manifest `sha256:40b0326baf0f71206da1fd14b05c8ab5bdf1473c708735c57c019ea820cf5848`, source dataset and split manifests, feature/label order, and existing feature audits. The TCN input adapter then validates 360 sequences, 700 real timesteps, sequence splits 252/54/54, episode counts 490/105/105, twelve feature columns, eight label columns, binary feature masks and targets, unique episode ownership, and absence of metadata or prediction channels.

Variable-length samples are right-padded per batch. Padded values, feature masks, and target placeholders are zero; `padding_mask` is one for real timesteps and zero for padding. `padding_mask` is used only for loss, metric flattening, and validity checks. The model receives 24 channels formed by normalized values concatenated with the twelve binary feature-mask channels, transposed to `[B,24,T]`.

Train-only normalization computes one mean, standard deviation, and observed count per frozen feature using Train real timesteps where `feature_mask=1`. Near-zero standard deviation uses scale 1.0. Missing values are normalized to zero and retain mask zero. Validation/Test values never affect the saved normalization asset.

Before training, `sequence_length_audit.json` records overall and per-split sequence count, real timestep count, min/mean/median/p90/max length, counts and ratios for `T=1`, `T>=2`, `T>=3`, `T>=4`, and counts/ratio reaching the model receptive field.

## 3. Fixed causal model

The model is a two-block residual TCN. Each block contains causal Conv1d, ReLU, Dropout(0.10), a second causal Conv1d, ReLU, Dropout(0.10), and residual addition. Block widths are `[32, 32]`; input projection is a 1x1 Conv1d from 24 to 32 channels; the head is Linear(32, 8) applied per real timestep. Kernel size is 2 and dilations are `[1, 2]`.

Each causal convolution uses left padding `(kernel_size - 1) * dilation` followed by `Conv1d(padding=0)`, preserving sequence length while preventing access to future timesteps. The implementation computes receptive field from the actual layer definitions and stores the computed value and parameter count in metadata; it does not hard-code the audit result. For all stride-1 causal convolutions, the design formula is `RF = 1 + Σ((kernel_i - 1) * dilation_i)`. The fixed two-block, two-convolution-per-block architecture has a theoretical expected RF of 7, but 7 is not treated as a run result until the implementation computes it. The causality test changes only future input and requires earlier logits to remain equal within tolerance. Padding invariance is checked in evaluation mode by comparing a sequence alone with the same sequence batched beside a longer sample.

The output is `[B,T,8]` logits. Loss uses `BCEWithLogitsLoss(reduction="none")`, multiplied by `padding_mask[..., None]`, and divided by the number of valid label positions. WRITING remains an all-positive label and participates in BCE, but its ROC-AUC and Average Precision are serialized as null with `SINGLE_CLASS_UNDEFINED`, matching LightGBM.

### Target Causality Audit

The frozen target contract remains unchanged. Before training, the implementation writes a semantic audit that describes how each target is generated and whether it is strictly available at the anchored timestep. The audit does not modify `y` or shift labels.

| label | truth generation rule and anchor | information required | current timestep sufficient? | past context required? | future timestep required? | external signal required? | causality class | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `WRITING` | Independent truth anchors the current action's point refs. | Current observable pen geometry/contact. | Yes | No | No | No | `CAUSAL_OBSERVABLE` | All frozen Train labels are positive; this is not a discriminative positive/negative task. |
| `QUESTION_VISIT` | Truth anchors the current action when the active question is absent or a new question is entered. | Current question/page mapping plus visited/active state. | Yes with state | Yes | No | No | `CAUSAL_OBSERVABLE` | The event is online-definable from current and historical observable state. |
| `QUESTION_LEAVE` | Truth is added when the next question differs, but is anchored to the previous episode (`action_index - 1`). | Previous active question plus the next question/page mapping. | No | Yes | Yes | No | `RETROSPECTIVE_TRANSITION` | This target is retrospectively attributed and is not strictly observable at the anchored timestep for an online causal model. |
| `RETURN` | Truth anchors the current action when the question has already been visited. | Current mapping plus visited-question history. | Yes with state | Yes | No | No | `CAUSAL_OBSERVABLE` | No future action is needed. |
| `REVISION_CANDIDATE` | Truth anchors the current action when prior ink, pause/revisit, and overlap criteria hold. | Current geometry plus prior ink and temporal history. | Yes with state | Yes | No | No | `CAUSAL_OBSERVABLE` | The candidate rule is causal, although it relies on historical context. |
| `PAGE_CHANGE` | Truth anchors the current action when its page differs from the active page. | Current page mapping plus previous page state. | Yes with state | Yes | No | No | `CAUSAL_OBSERVABLE` | The detector emits it before the current writing event. |
| `PROCESS_END` | Truth anchors the final action only for explicit process-end scenarios. | Explicit end signal or session-end reference plus timeout. | No | No | No | Yes | `EXTERNAL_SIGNAL_DEPENDENT` | The external signal is not present in `SequenceSample.values` or `feature_mask`; features cannot prove process end. |
| `UNKNOWN` | Truth anchors the current action when the independent scenario marks the mapping as unknown. | Current mapping/quality observables. | Yes | No | No | No | `CAUSAL_OBSERVABLE` | The event is contemporaneous; the frozen 12-feature projection may not expose every mapping detail. |

`CAUSAL_OBSERVABLE` means the label is definable from current observations and finite past state without future input or an external signal. `RETROSPECTIVE_TRANSITION` means its frozen anchor precedes the observation that establishes the transition. `EXTERNAL_SIGNAL_DEPENDENT` means the truth source contains a signal absent from the TCN input. These classes are interpretive metadata only; the frozen eight-label target is still trained and evaluated unchanged.

The full audit is saved as `target_causality_audit.json` with per-label `truth_generation_rule`, `truth_anchor_episode`, `information_required`, `current_timestep_sufficient`, `past_context_required`, `future_timestep_required`, `external_signal_required`, `causality_class`, and `notes`, plus `causal_observable_label_names = [WRITING, QUESTION_VISIT, RETURN, REVISION_CANDIDATE, PAGE_CHANGE, UNKNOWN]`, `retrospective_labels = [QUESTION_LEAVE]`, `external_signal_labels = [PROCESS_END]`, and overall `status = "PASS_WITH_NON_CAUSAL_LABELS"`. This status means the target semantics are documented and the frozen task may proceed, while non-causal labels remain visible in interpretation.

### Canonical and causal-observable diagnostic metrics

`metrics.json` remains the canonical eight-label evaluator used by LightGBM, including all aggregate and per-label fields. It must include the explicit interpretation that its micro-F1 and macro-F1 include the degenerate all-positive `WRITING` label. The run also writes `causal_diagnostics.json` (or the equivalent comparison section) for the subset of labels classified as `CAUSAL_OBSERVABLE`; this diagnostic does not replace or hide canonical metrics. LightGBM comparison computes the same diagnostic subset from reloaded LightGBM predictions, so TCN and LightGBM use identical label membership and metric semantics.

## 4. Shared evaluation and artifacts

The frozen LightGBM evaluator is moved or re-exported through `src/datiao/r1/evaluation/multilabel.py`; existing LightGBM imports remain compatible and its metrics JSON/hash must remain unchanged. TCN evaluation uses the same evaluator after flattening only real timesteps. Every valid timestep produces one stable prediction record containing `episode_id`, `probability_vector`, `prediction_vector`, `target`, `split`, `sequence_id`, and `timestep_index`; predictions are sorted by `(split, episode_id, sequence_id, timestep_index)`.

The formal output is `artifacts/r1_tcn_v1/` with:

```text
run_manifest.json
metrics.json
predictions.jsonl
training_history.json
sequence_length_audit.json
target_causality_audit.json
causal_diagnostics.json
normalization.json
config_snapshot.json
environment.json
comparison_lightgbm.json
model/best_model.pt
audits/input_integrity.json
audits/normalizer.json
audits/causality.json
audits/padding.json
audits/model_reload.json
audits/reproducibility.json
```

The manifest records source lineage, feature/label order, counts, normalization semantic hash, architecture, receptive field, parameter count, seed/device/threshold, best epoch and validation loss, model file hash, semantic model state hash, predictions/metrics/comparison hashes, and a run hash that excludes itself, wall-clock time, absolute paths, host identity, hardware identifiers, and the PyTorch checkpoint container byte hash. `model_file_hash` is artifact integrity only. `model_state_hash` is semantic model identity: state-dict keys are sorted and each tensor is detached, moved to CPU, made contiguous, then encoded with its name, dtype, shape, and raw bytes. The semantic run hash includes source lineage, config hash, normalization semantic hash, architecture, model state hash, predictions hash, metrics hash, and comparison hash. A different checkpoint file hash alone is reported as `artifact_byte_reproducibility` and does not fail semantic reproducibility when those semantic values agree.

`reload_tcn_run()` validates all manifest and artifact hashes, source lineage, normalization, and model metadata. It reloads the frozen Feature Dataset, loads saved normalization and a new model instance, reruns validation/test inference, and joins predictions by `episode_id` before comparing probabilities; array position alone is not sufficient. It requires validation and test probability max absolute differences at most `1e-12`. Tamper tests cover model file, normalization, predictions, metrics, comparison, and run manifest hash changes.

## 5. Training and comparison isolation

The training API accepts only Train and Validation loaders. Train uses a seeded `torch.Generator` with shuffle enabled; Validation and Test use deterministic non-shuffled loaders with `num_workers=0`. Test is loaded only by the final evaluation path after the best validation checkpoint is restored. No per-epoch Test metric is written. Every independent run resets Python `random`, NumPy, Torch, Torch deterministic mode, Torch thread count, and the DataLoader generator, then creates a fresh model, optimizer, loaders, and early-stopping state.

The `--lightgbm-run` argument is passed only to the post-training comparison stage. That stage calls `reload_lightgbm_run()` and verifies identical source Feature Manifest, label/feature order, and exact Validation and Test episode sets before writing `comparison_lightgbm.json`. Any validation or test ID mismatch fails comparison. The argument is never read by the input adapter and LightGBM probabilities never enter TCN features. The comparison includes canonical eight-label metrics and the same causal-observable diagnostic subset, plus per-label F1 and ROC-AUC deltas, without making a direction or generalization claim.

### Short-sequence and effective-context policy

`sequence_length_audit.json` is descriptive evidence, not a tuning signal. The fixed architecture remains unchanged even if most sequences are shorter than the computed RF. The audit records `effective_temporal_context_warning = "SHORT_SEQUENCE_LIMITED_CONTEXT"` when the fraction reaching RF is low; this is a passing audit with an explicit limitation, not a build failure. The final report must state the length distribution, computed RF, fraction reaching RF, and that the available temporal context is limited for short synthetic sequences.

## 6. Tests and acceptance

Tests are written before implementation for: sequence length audit, target causality audit schema and frozen eight-label membership, collator shapes and masks, missing-vs-real-zero distinction, Train-only normalization and serialization, causal receptive behavior, padding invariance, forward shape, real optimizer parameter updates, finite loss/gradients/logits, masked loss ignoring padding, save/reload identity by episode ID, Test isolation, validation/test episode-set equality, episode prediction completeness, canonical/diagnostic metric compatibility, deterministic two-run semantic hashes, and all required tamper cases. Existing R1, contract, and LightGBM tests must remain green.

Acceptance requires the frozen Feature Manifest and counts, all input, target-causality, and length audits, causal and mask tests, fixed architecture/configuration, best-checkpoint early stopping, complete 700-episode predictions with 105 validation and 105 test rows, canonical eight-label metrics plus causal-observable diagnostics, model reload tolerance, two independent runs with matching semantic prediction/metrics/model-state/run hashes, successful validation/test-aligned LightGBM comparison, and full pytest success. `model_file_hash` is separately reported as artifact-byte integrity; a file-byte difference alone does not fail semantic reproducibility when model state, predictions, metrics, and run hash agree. Short sequences are reported as a limitation: the effective temporal context is limited when sequence length is below the computed receptive field, and results do not support claims about long-range classroom behavior or real-student generalization.

## 7. Deliberate non-goals

No Transformer, LSTM/GRU, hyperparameter tuning, Optuna, Grid Search, feature redesign, dataset redesign, real-data training, R3, vision, fusion, agents, or frontend work is part of TASK-R1-09.
