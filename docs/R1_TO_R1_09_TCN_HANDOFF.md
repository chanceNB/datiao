# R1 → R1-09 TCN Handoff

建议后续时序任务：读取 SequenceSample 的 values、feature_mask 和 targets，使用 mask 感知的 8-logit TCN。保留变长序列边界与 source split，不把 metadata 或未来标签送入模型。


当前 Feature Manifest hash：sha256:40b0326baf0f71206da1fd14b05c8ab5bdf1473c708735c57c019ea820cf5848；Sequence 为 360（252/54/54 split），values 为 [T,12]，targets 为 [T,8]。
