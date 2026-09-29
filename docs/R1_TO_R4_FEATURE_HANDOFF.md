# R1 → R4 Feature Handoff

R4 接收 `artifacts/r1_synthetic_features_v1/`，先调用 `reload_feature_dataset` 并验证 `feature_manifest.json`。FeatureRow 是逐 Episode 平面输入，SequenceSample 是按 case 的变长时序输入。X 只允许 12 个 frozen feature，缺失由 null / feature_mask 传递；target 使用固定 8-label multi-hot 顺序。

训练/评估必须使用 manifest 中继承的 train/validation/test split，不能重新随机切分，也不能读取预测事件生成标签。


当前 Feature Manifest hash：sha256:40b0326baf0f71206da1fd14b05c8ab5bdf1473c708735c57c019ea820cf5848；alignment_records.jsonl 已纳入核心文件 hash。
