# TASK-R1-07-FIX-01 — Feature Contract & Training Input Integrity

状态：PASS（2026-09-29）。本修复将 `contracts/r1_feature_schema_v1.json` 与 `contracts/r1_target_spec_v1.json` 设为唯一真源；Builder 加载并验证 canonical contract 后原样写入 runtime artifacts。

本轮收紧了 FeatureRow extra/missing key 校验、Draft 2020-12 JSON Schema、完整 norm path 语义、累计 return_count、UNKNOWN mapping 历史行为、FeatureSplitManifest reload 与 union/disjoint/lineage 校验，并将逐 Truth Event 的 `alignment_records.jsonl` 纳入 manifest 核心 hash。

完整重建命令：

```text
python -m datiao.r1.features.builder --dataset artifacts/r1_synthetic_penprocess_v1 --output artifacts/r1_synthetic_features_v1 --config configs/r1_feature_builder_v1.json --overwrite
```

最终结果：700 episodes、360 sequences；train/validation/test 为 490/105/105 episodes、252/54/54 sequences；truth alignment 1640/1640，unmatched=0，ambiguous=0；Feature leakage、range、split、reload 均 PASS。

最终 Feature Manifest hash：`sha256:40b0326baf0f71206da1fd14b05c8ab5bdf1473c708735c57c019ea820cf5848`。

本轮没有训练 LightGBM、Pen TCN 或任何模型。

Updated missingness (overall): duration_s 0/700; path_length_norm 19/700; mean_speed_norm_per_s 39/700; pause_before_s 360/700; pause_inside_s 20/700; bbox_width_norm 19/700; bbox_height_norm 19/700; question_occupancy 80/700; visit_index 80/700; return_count 80/700; previous_ink_iou 580/700; quality_valid 0/700.
