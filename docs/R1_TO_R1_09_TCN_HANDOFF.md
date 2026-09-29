# R1 → R1-09 TCN Handoff

建议后续时序任务：读取 SequenceSample 的 values、feature_mask 和 targets，使用 mask 感知的 8-logit TCN。保留变长序列边界与 source split，不把 metadata 或未来标签送入模型。


当前 Feature Manifest hash：sha256:40b0326baf0f71206da1fd14b05c8ab5bdf1473c708735c57c019ea820cf5848；Sequence 为 360（252/54/54 split），values 为 [T,12]，targets 为 [T,8]。

R1-08 baseline reference：`r1-lightgbm-ovr-v1@1.0.0`。TCN 不得使用 LightGBM predictions 作为输入；仍只读取冻结 SequenceSample。

R1-08-FIX-01 已复核 baseline reproducibility：run hash `sha256:3b6235b64974c244fe3dc80f81798d815404dc48786323ca1fbb94ead33118e0`，两次独立执行一致，模型重载 validation/test max diff 为 `0.0`。建议下一任务为 TASK-R1-09 Pen TCN Baseline；本任务没有执行 TCN。
