# TASK-R1-09 Pen TCN Baseline Design

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

Each causal convolution uses left padding `(kernel_size - 1) * dilation` followed by `Conv1d(padding=0)`, preserving sequence length while preventing access to future timesteps. The implementation computes receptive field from the actual layer definitions and stores the computed value and parameter count in metadata; it does not hard-code the audit result. The causality test changes only future input and requires earlier logits to remain equal within tolerance. Padding invariance is checked in evaluation mode by comparing a sequence alone with the same sequence batched beside a longer sample.

The output is `[B,T,8]` logits. Loss uses `BCEWithLogitsLoss(reduction="none")`, multiplied by `padding_mask[..., None]`, and divided by the number of valid label positions. WRITING remains an all-positive label and participates in BCE, but its ROC-AUC and Average Precision are serialized as null with `SINGLE_CLASS_UNDEFINED`, matching LightGBM.

## 4. Shared evaluation and artifacts

The frozen LightGBM evaluator is moved or re-exported through `src/datiao/r1/evaluation/multilabel.py`; existing LightGBM imports remain compatible and its metrics JSON/hash must remain unchanged. TCN evaluation uses the same evaluator after flattening only real timesteps. Every valid timestep produces one stable prediction record containing `episode_id`, `probability_vector`, `prediction_vector`, `target`, `split`, `sequence_id`, and `timestep_index`; predictions are sorted by `(split, episode_id, sequence_id, timestep_index)`.

The formal output is `artifacts/r1_tcn_v1/` with:

```text
run_manifest.json
metrics.json
predictions.jsonl
training_history.json
sequence_length_audit.json
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

The manifest records source lineage, feature/label order, counts, normalization hash, architecture, receptive field, parameter count, seed/device/threshold, best epoch and validation loss, model file hash, semantic model state hash, predictions/metrics/comparison hashes, and a run hash that excludes itself, wall-clock time, absolute paths, host identity, and hardware identifiers. `model_state_hash` hashes sorted state-dict key, dtype, shape, and raw tensor bytes; model file hash separately detects container tampering.

`reload_tcn_run()` validates all manifest and artifact hashes, source lineage, normalization, and model metadata. It loads a new model instance and requires validation and test probability max absolute differences at most `1e-12`. Tamper tests cover model file, normalization, predictions, metrics, comparison, and run manifest hash changes.

## 5. Training and comparison isolation

The training API accepts only Train and Validation loaders. Train uses a seeded `torch.Generator` with shuffle enabled; Validation and Test use deterministic non-shuffled loaders with `num_workers=0`. Test is loaded only by the final evaluation path after the best validation checkpoint is restored. No per-epoch Test metric is written.

The `--lightgbm-run` argument is passed only to the post-training comparison stage. That stage calls `reload_lightgbm_run()` and verifies identical source Feature Manifest, label/feature order, and exact Test episode set before writing `comparison_lightgbm.json`. It compares validation and Synthetic Test aggregate metrics plus per-label F1 and ROC-AUC deltas without making a direction or generalization claim.

## 6. Tests and acceptance

Tests are written before implementation for: sequence length audit, collator shapes and masks, missing-vs-real-zero distinction, Train-only normalization and serialization, causal receptive behavior, padding invariance, forward shape, real optimizer parameter updates, finite loss/gradients/logits, masked loss ignoring padding, save/reload identity, Test isolation, episode prediction completeness, metric compatibility, deterministic two-run hashes, and all required tamper cases. Existing R1, contract, and LightGBM tests must remain green.

Acceptance requires the frozen Feature Manifest and counts, all input and length audits, causal and mask tests, fixed architecture/configuration, best-checkpoint early stopping, complete 700-episode predictions with 105 validation and 105 test rows, metric compatibility, model reload tolerance, two independent runs with matching prediction/metrics/model-state/run hashes, successful LightGBM comparison, and full pytest success. Short sequences are reported as a limitation: the effective temporal context is limited when sequence length is below the computed receptive field, and results do not support claims about long-range classroom behavior or real-student generalization.

## 7. Deliberate non-goals

No Transformer, LSTM/GRU, hyperparameter tuning, Optuna, Grid Search, feature redesign, dataset redesign, real-data training, R3, vision, fusion, agents, or frontend work is part of TASK-R1-09.
