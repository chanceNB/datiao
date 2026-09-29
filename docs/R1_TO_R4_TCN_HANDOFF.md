# R1 → R4 Pen TCN handoff

R1 delivered a reproducible reference implementation and artifact for `r1-pen-tcn-v1@1.0.0`. R4 should treat the Feature Manifest, split manifest, label order, target specification, normalization semantics, and run manifest as immutable inputs when extending the pipeline.

The model is strictly causal: every temporal convolution pads on the left only, with no bidirectional operation, future pooling, or attention. The input is `[normalized feature values, feature masks]`; sequence padding is represented separately by the padding mask. Normalization is fit only from observed train timesteps and is serialized with a semantic hash.

The eight-label target is deliberately retained. `QUESTION_LEAVE` is retrospective and `PROCESS_END` requires an external signal, so causal-subset diagnostics must not replace canonical eight-label evaluation. Use `metrics.json` for the canonical task and `causal_diagnostics.json` only for interpretation.

Before downstream changes, reload `artifacts/r1_tcn_v1` and verify `audits/input_integrity.json`, `audits/normalizer.json`, `audits/causality.json`, `audits/padding.json`, `audits/model_reload.json`, and `audits/reproducibility.json`. Any source manifest or label-order drift is a compatibility failure. Hyperparameter tuning, TCN V2, Transformer, real-data training, fusion, vision, agent, and frontend work are outside this handoff.
