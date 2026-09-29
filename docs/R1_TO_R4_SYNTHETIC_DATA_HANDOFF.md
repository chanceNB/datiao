# R1 → R4 Synthetic Data Handoff

## Dataset

- dataset_id: `r1-synthetic-penprocess-v1`
- dataset_version: `1.0.0`
- dataset_type: `synthetic`
- output: `artifacts/r1_synthetic_penprocess_v1/`
- config: `configs/r1_synthetic_dataset_v1.json`
- dataset manifest hash: `sha256:53b9213d92311955edd02a9df415b338798c2fd63155fb191148ccc438d29d20`

## Versions

- generator: `r1.synthetic.v1`
- contract: `1.0.0`
- Stroke: `r1-stroke-rule-v2.2`
- Mapping: `r1-qmap-arc-v1`
- Event: `r1-event-rule-v0.2.2`
- split: `r1-synth-split-v1`

## Candidate split and audits

R1 generated a deterministic participant-level candidate split: 14 train participants, 3 validation participants and 3 test participants. Leakage and integrity audits both report `PASS`; their JSON reports are under `audits/` in the materialized directory.

## Truth and prediction

Truth is generated from the scenario plan and is stored independently in `truth_events.jsonl`. R1 rule output is stored independently in `predicted_events.jsonl`. Prediction is a baseline asset and must not be flattened into future model input features by default.

## R4 decision boundary

R4 formal approval, experiment protocol, split freezing and release gating are pending. This handoff does not claim R4 approval or scientific validity. The dataset is synthetic development data and does not represent real student or classroom evidence.
