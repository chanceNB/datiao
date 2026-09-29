# TASK-R1-04 R1 Contract Alignment V1 Design

## Goal

将现有 R1 离线过程链迁移到稳定、版本化、可验证的 Contract V1，并保持 Raw provenance、PageReplay、Event → Stroke → Point → Raw trace、Synthetic Truth/Prediction 隔离和 `run_r1_pipeline()` 可用。

## Scope and boundaries

本次只做 Contract migration。R1-05 的新事件语义、arc-length mapping、IoU revision、timeout process-end、spatial-jump stroke splitting、R2/R3/R4、Vision、Fusion、模型训练和前端均不实现。

标准对象使用 JSON Schema Draft 2020-12，Contract Version 为 `1.0.0`。旧字段只允许存在于显式 Legacy Adapter 或迁移辅助中，不与 V1 标准对象混用。

## Canonical Point V1

`Point` 的标准字段包括：`schema_version`、`point_id`、`session_id`、`participant_id`、`task_segment_id`、`device_id`、`page_id`、`sequence`、`timestamp_ms`、`x_raw`、`y_raw`、`x_mm`、`y_mm`、`x_norm`、`y_norm`、`pressure_raw`、`pressure_norm`、`pen_state_raw`、`pen_state`、`source_file`、`source_index`、`raw_order`、`processed_order`、`quality_flags`。

Raw x/y、原始类型和值继续保存在不可变 `source_payload` 并计算 hash。无物理标定时 mm 坐标为 `null`；无可靠页面映射时 norm 坐标为 `null`。`participant_id`、`task_segment_id` 可为 `null`；Synthetic 使用 `sim_` 前缀。`sequence` 使用 ingest order 时显式标记来源扩展。`timestamp_ms` 为整数毫秒；压力 raw 的 `0` 与 `null` 保持区别；标准 pen state 为 `DOWN/MOVE/UP/UNKNOWN`。

## Stroke V1

`Stroke` 至少包含 `stroke_id`、`session_id`、`participant_id`、`task_segment_id`、`page_id`、`point_refs`、`raw_order`、`processed_order`、`start_time_ms`、`end_time_ms`、`duration_ms`、`bbox`、`quality_flags` 和 `algorithm_version`/`provenance`。高级笔画特征留给后续任务，不填充伪造数值。

## QuestionRegion V1

标准表达为 `region_id`、`question_id`、`page_id`、`region_type`、`polygon_norm`。`polygon_norm` 必须是至少三个点的闭合语义多边形，所有坐标均在 0~1。旧矩形构造通过显式兼容转换为四个顶点；不得把旧 synthetic 的 0~50 坐标直接标为 norm。

## Student Event V1

事件枚举冻结为：`WRITING`、`QUESTION_VISIT`、`QUESTION_LEAVE`、`RETURN`、`REVISION_CANDIDATE`、`PAGE_CHANGE`、`PROCESS_END`、`UNKNOWN`。

标准事件字段为 `event_id`、`event_type`、`session_id`、`task_segment_id`、`participant_id`、`question_id`、`start_time_ms`、`end_time_ms`、`point_refs`、`stroke_refs`、`quality_status`、`algorithm_version`、`provenance`，并保留必要的 page/sequence/quality flags 扩展。事件时间满足 `end_time_ms >= start_time_ms`；瞬时事件允许相等。`quality_status` 统一为 `VALID/DEGRADED/INVALID`，表示总状态；`quality_flags` 表示具体原因。`UNKNOWN` 是可观测事件类别，不等价于 INVALID 数据。

Synthetic event provenance 必须包含 `dataset_type`、`generator_version`、`seed`、`scenario_id`、`ground_truth_source`、`manifest_hash`，且 seed 为整数、manifest hash 为真实计算值或测试 fixture 的实际 SHA-256。

## Adapters and trace

保留 `PointRecordAdapter`/`CanonicalPointAdapter` 抽象；设备别名必须通过显式设备 Adapter 映射，未确认语义保持 `null`/`UNKNOWN`/quality flag。Legacy Event/Point 适配器仅用于内部旧调用迁移。Trace resolver 使用 V1 `point_refs`/`stroke_refs`，继续解析到不可变 Point 的 Raw payload。

## Synthetic and leakage boundary

Synthetic session、participant、task segment 使用 `sim_` 前缀，session-relative 时间以 0ms 为原点。同 seed 结果保持确定性。`SyntheticTruth` 从 scenario plan 独立生成，`SyntheticAlgorithmOutput` 来自 R1 算法，二者不互相构造。scenario/question/generator/provenance 字段不能自动 flatten 为训练 feature；文档要求未来 Feature Builder 使用 whitelist。

## Verification

新增 `contracts/` schemas、Golden JSON 和 `tests/contracts/`，覆盖 schema validation、Pydantic round-trip、enum/time/ref/provenance/manifest hash、null vs UNKNOWN、polygon bounds、raw preservation、trace integrity、determinism 与 truth/prediction separation。旧 `tests/r1` 迁移到 V1 API 后与全量 pytest 一起运行。

