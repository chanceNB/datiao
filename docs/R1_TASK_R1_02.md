# R1 TASK-R1-02

## 实现内容

- 新增 `parse_raw_points()`，将原始 record 转换为不可变 `Point` tuple。
- 保留每条输入 record 为不可变 `source_payload`，并由 Point 生成稳定 hash。
- 缺坐标、缺时间、重复 `point_id`、sequence 逆序均保留原始数据并写入质量标记。
- 新增 `build_strokes(points, config)`，按同 session、同 page 和时间连续规则构建 Stroke tuple。
- Stroke 的 `raw_order` 保持输入顺序，`processed_order` 按时间、sequence 和 raw index 排序。
- 增加 Point/Stroke 质量常量和质量汇总逻辑。

## 输入输出

```text
raw records
    -> parse_raw_points()
Point tuple
    -> build_strokes(points, StrokeBuildConfig)
Stroke tuple
```

默认 `max_time_gap_ms` 为 1000。缺失 timestamp 的点不被删除，并形成独立的 `INCOMPLETE` stroke；同一 stroke 中的轻微时间逆序仍可保留，处理顺序会单独排序。

## 测试结果

```text
pytest
```

覆盖连续点、多 stroke、乱序、重复 point、缺失 timestamp、缺失坐标、空输入、source payload 不可变和 Stroke raw order 不可变。

## 已知限制

- 仅实现 Raw Point 到 Stroke，不包含 Question Mapping。
- 不包含 Event Detection 或任何 StudentProcessEvent 生成逻辑。
- Stroke 分组使用相邻输入点和固定时间间隔，尚未实现设备级 pen-up/pen-down 信号。
- 不提供外部协议适配器；parser 当前读取 mapping records。

本任务不包含 TASK-R1-03。
