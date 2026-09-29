# R1 → R4 Experiment Handoff

R4 should begin from [R1 Baseline Freeze V1](R1_BASELINE_FREEZE_V1.md), not from a newly trained copy. Load and validate the canonical manifest and local artifacts before an experiment:

```powershell
$env:PYTHONPATH = "src"
python -m datiao.r1.freeze.validator
```

R4 may add experiments that consume the frozen feature matrix and split assignments, but every result must record the freeze manifest hash, feature order, label order, source split hash, and whether it uses the canonical eight-label task or a declared causal subset. Keep LightGBM and Pen TCN as named reference points; do not silently replace either baseline or claim a winner from the synthetic test.

R4 must not mutate the frozen dataset, feature artifacts, model files, predictions, metrics, normalization, target causality audit, or temporal-context audit. It must not use the synthetic test split for fitting, normalization, architecture selection, threshold selection, early stopping, or checkpoint selection. `QUESTION_LEAVE` remains retrospective and `PROCESS_END` remains external-signal dependent. The current sequences are short (`max T=3`) relative to the TCN receptive field (`RF=7`), so temporal gains are not established by this freeze.

Real data, new labels, TCN V2, Transformer, fusion, vision, agent, and frontend work require a new task and a new manifest version. They are not edits to this frozen handoff.
