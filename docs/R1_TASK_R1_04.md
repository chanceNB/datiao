# TASK-R1-04 — R1 Contract Alignment V1

## 任务目的

将 2026-09-27 的 R1 规则型离线链迁移到 2026-09-29 冻结的 Contract V1，使 Point、Stroke、QuestionRegion、Student Process Event、Synthetic provenance 和 Trace 可稳定交给 R3/R4 使用。

## 旧 Contract → 新 Contract

- Point：`normalized`/`raw_index` 迁移到 raw/mm/norm 坐标、source index/order、pressure 和 pen state；Raw payload 与 hash 继续保留。
- Stroke：`start_timestamp_ms`/`end_timestamp_ms`/隐含点序迁移到 `point_refs`、`start_time_ms`、`end_time_ms`、`duration_ms` 和 algorithm provenance。
- QuestionRegion：旧 rectangle/polygon 坐标通过兼容构造保留；V1 标准字段是 `region_type` 与 `polygon_norm`。旧 synthetic 0~50 坐标标为 legacy coordinate space。
- Event：旧 `occurred_at_ms`、`source_stroke_ids`、`source_point_ids`、`source_provenance` 不再进入标准序列化；V1 使用 `start_time_ms`/`end_time_ms`、`stroke_refs`/`point_refs`、`quality_status`、`algorithm_version`、`provenance`。

## Point V1

Contract version 为 `1.0.0`。Point 区分 `x_raw/y_raw`、`x_mm/y_mm` 和 `x_norm/y_norm`；没有标定或可靠页面映射时保持 `null`。`participant_id`、`task_segment_id`、`device_id` 可为 `null`。`pressure_raw=0` 与 `null` 保持区别，`pressure_norm` 未校准时为 `null`。标准 pen state 为 `DOWN/MOVE/UP/UNKNOWN`。`source_payload` 不被标准化覆盖。

## Stroke V1

Stroke 保留 `raw_order`、`processed_order`、bbox 和 quality flags，并提供 `point_refs`、时间区间、duration、algorithm version 和 provenance。路径长度、速度、加速度、pause 和压力统计不在本任务伪造。

## QuestionRegion V1

跨模板标准表达为 0~1 的 `polygon_norm`。矩形通过四顶点表达。旧坐标只能通过显式兼容输入使用，并带 `coordinate_space=legacy`。

## Event V1

事件枚举固定为：

`WRITING`、`QUESTION_VISIT`、`QUESTION_LEAVE`、`RETURN`、`REVISION_CANDIDATE`、`PAGE_CHANGE`、`PROCESS_END`、`UNKNOWN`。

时间满足 `end_time_ms >= start_time_ms`。`quality_status` 表示总状态（`VALID`、`DEGRADED`、`INVALID`），`quality_flags` 表示原因；UNKNOWN 事件不等于 INVALID 数据。事件通过 `point_refs` 和 `stroke_refs` 维持 Event → Stroke → Point → Raw 回查。

## Synthetic provenance

Synthetic 使用 `sim_session_...`、`sim_p_...`、`sim_segment_...` 标识，Session 统一从 0ms 起算。事件 provenance 包含 `dataset_type`、`generator_version`、整数 `seed`、`scenario_id`、`ground_truth_source`、实际 `manifest_hash`。`SyntheticTruth` 由 scenario plan 生成，`SyntheticAlgorithmOutput` 由 R1 detector 生成，二者不合并。

## Breaking Changes 与兼容策略

标准对象只序列化 V1 字段。旧 payload 必须通过 `datiao.r1.models.legacy` 的显式适配器；V1 模型拒绝旧构造字段。内部仍保留少量只读访问别名以保护 Trace；设备字段别名仍必须由显式 Adapter 映射，未确认的单位、标定和页面语义保持 null/UNKNOWN。下游应读取 V1 字段，不应依赖旧属性。

## 当前明确未完成事项

- R1-05 的 arc-length Question Mapping、IoU Revision、timeout Process End 和 spatial jump Stroke splitting。
- 完整 Dataset Manifest/Split 管理和生产数据流水线。
- Pen Features、Pen Model、R3 聚合、Vision、Fusion、Agent 和前端。

## 下一步

`TASK-R1-05 — Algorithm Semantics Upgrade`，只给建议，不在本任务执行。

