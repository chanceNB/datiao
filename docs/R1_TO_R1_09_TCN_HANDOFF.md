# R1 → R1-09 TCN Handoff

建议后续时序任务：读取 SequenceSample 的 values、feature_mask 和 targets，使用 mask 感知的 8-logit TCN。保留变长序列边界与 source split，不把 metadata 或未来标签送入模型。
