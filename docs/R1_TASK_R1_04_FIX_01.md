# TASK-R1-04-FIX-01 — Contract Integrity & Reproducibility Fix

本修复保持 R1-04 的数据契约边界，解决换人复现、Manifest 语义和跨层一致性问题；没有进入 R1-05 的算法升级。

## 修复内容

- `jsonschema` 已加入 `.[test]`，标准方式是 `python -m pip install -e ".[test]"`。
- 新增 `SyntheticManifest` V1。`raw_records_hash` 指向原始记录内容，`compute_manifest_hash()` 指向不含自身 hash 的冻结 Manifest payload；相同 scenario、seed、generator version 会得到相同 hash。
- Synthetic generator 将真实 Manifest hash 写入 Event provenance，并把 Manifest 挂在 `SyntheticCase` 上。
- Point 内部模型继续保留缺失字段的 degraded/raw 数据；标准成功导出必须通过 `is_canonical_v1_eligible()`。
- `VALID` Event 必须有开始和结束时间。缺失时间只允许在 degraded/invalid 状态下，并且必须有时间不可用质量标志。题目相关事件必须有 `question_id`，映射失败使用 `UNKNOWN`。
- 标准 `QuestionRegion` 只接受 `coordinate_space="norm"`；Legacy rectangle 使用独立 `LegacyQuestionRegion`，不能伪装成 V1 JSON。
- Pydantic 模型和 JSON Schema 同步表达上述边界。

## 测试

本任务完成后运行 `tests/contracts`、`tests/r1` 和全量 pytest，并进行 `python -m pip install -e ".[test]"` 的依赖安装检查。具体结果写入最终报告。

## 范围确认

未实现 R1-05 的 arc-length Mapping、IoU Revision、timeout Process End、spatial-jump Stroke、WRITING detector，也未训练或接入 LightGBM、Pen TCN、R3、Vision、Fusion、Agent 或 Frontend。
