# R1 离线复现说明

## 安装和测试

在仓库根目录执行标准安装和测试：

```powershell
python -m pip install -e ".[test]"
python -m pytest
```

`.[test]` 会安装项目和全部 Contract/R1 测试依赖，包括 `pytest` 与 `jsonschema`。运行手册不依赖某个开发者的本机 Python 路径。

当前仓库全量测试以实际 pytest 输出为准；TASK-R1-06-FIX-01 完成时为 131 个（R1 100，Contract 31）。真实设备字段尚未冻结时，只能传入以下规范化字段：

```text
point_id, session_id, page_id, timestamp_ms, x, y, sequence
```

`id`、`page`、`time` 等别名不会自动转换。确认设备协议后，应实现独立的 `PointRecordAdapter`，将设备记录明确转换为上述字段，并保留原始记录。

## 统一离线入口

```python
from datiao.r1 import run_r1_pipeline

result = run_r1_pipeline(
    canonical_records,
    question_regions,
    session_id="session-001",
    task_segment_id="task-01",
    data_version="canonical-v1",
    source_provenance={"input_kind": "synthetic"},
)
```

返回对象包含：

```text
points
strokes
stroke_mappings
student_process_events
page_replay
event_traces
trace_errors
quality_status
quality_flags
source_provenance
```

正常样例的事件序列可能是：

```text
WRITING, QUESTION_VISIT
```

输入 EOF 不代表过程结束。只有显式传入 `process_end_signal=True`，或同时提供 `session_end_ms` 与配置的 inactivity timeout，才会产生 `PROCESS_END`。同一题的普通多 Stroke 续写只产生 `WRITING`；`REVISION_CANDIDATE` 需要 prior ink、暂停/回访和空间重叠证据。

Question Mapping 使用 `r1-qmap-arc-v1` 的轨迹弧长覆盖；零长度 Stroke 会显式记录 `POINT_FALLBACK`。当前所有阈值均为 development default，正式实验必须在 Dev/Validation 校准后冻结。

Student Process state 按 `(participant_id, task_segment_id)` 隔离；缺失值使用独立的内部 bucket。Stroke context 还比较 `device_id`，并在 provenance 中记录 `raw_coordinate_domain`。Raw bbox revision 只在同 participant、同 segment、同 page、同 coordinate space 和同可信 raw domain 下比较；unknown device domain 不产生跨 Stroke raw overlap。缺失 timestamp 不再自动切 Stroke，但会保留 degraded quality。PROCESS_END 只接受显式 end 或带外部 session end reference 的 participant/segment-aware timeout。Event 的 `task_segment_id` 来自来源 mapping，调用方参数只作为一致性校验和缺失映射时的 fallback。

无法映射或来源不完整时，结果仍保留原始点和 Stroke，并将状态设为 `DEGRADED`；空输入返回 `INVALID`。这些状态不表示学生心理、能力或作答正确性。

## Synthetic 复现

```powershell
$env:PYTHONPATH = "src"
python -c "from datiao.r1.synthetic import Scenario, generate_synthetic_case; c=generate_synthetic_case(Scenario('demo-return','return_visit',7)); print([e.event_type for e in c.algorithm_output.events]); print([e.event_type for e in c.truth.truth_events])"
```

算法输出和独立真值分别存放在 `algorithm_output` 与 `truth`，不能用算法输出生成真值。

## Contract V1（TASK-R1-04）

标准输出版本为 `1.0.0`，schema 位于 `contracts/`。下游读取 Point 的 raw/mm/norm 坐标、Stroke 的 `point_refs`、标准 QuestionRegion 的 `polygon_norm` 和 Event 的 `start_time_ms/end_time_ms`、`point_refs/stroke_refs`。内部 degraded Point 会保留原始数据，但只有通过 Canonical Point V1 Gate 才能作为成功标准输出。旧 `occurred_at_ms` 与 `source_*_ids` 只用于迁移兼容，不会出现在标准 `model_dump()`。

Synthetic 使用 `sim_session_...`、`sim_p_...`、`sim_segment_...`，每个 Session 从 0ms 开始；Synthetic provenance 必须带六个字段和来自 Synthetic Manifest 的实际 SHA-256 manifest hash，Manifest 同时保留 raw records hash。`quality_status` 是总状态，`quality_flags` 是原因；UNKNOWN 事件不表示数据 INVALID。8 类 Event Contract 已冻结；部分事件检测语义将在 R1-05 完成。

Contract 专项测试：

```powershell
python -m pytest tests/contracts -q
python -m pytest tests/r1 -q
python -m pytest
```

## R1 Synthetic Development Dataset（TASK-R1-06）

配置文件为 `configs/r1_synthetic_dataset_v1.json`，默认生成 `r1-synthetic-penprocess-v1`，并明确标记 `dataset_type=synthetic`。它是 R1 的确定性开发候选数据与 participant-level Group Split，未来由 R4 重新审核和冻结，不代表真实学生数据或正式科研评测。

```powershell
$env:PYTHONPATH = "src"
python -m datiao.r1.dataset.builder `
  --config configs/r1_synthetic_dataset_v1.json `
  --output artifacts/r1_synthetic_penprocess_v1
```

已有输出默认不会被覆盖；需要重建时显式增加 `--overwrite`。构建会生成 Dataset Manifest、Summary、JSONL 数据文件、`splits/`、`audits/`，并在写出前执行 participant/session/group/case 与文件引用完整性审计。

```powershell
$env:PYTHONPATH = "src"
python -c "from datiao.r1.dataset import reload_dataset; d=reload_dataset('artifacts/r1_synthetic_penprocess_v1'); print(d['manifest'].manifest_hash)"
```

Truth 与 R1 rule prediction 分别写入 `truth_events.jsonl` 和 `predicted_events.jsonl`。`scenario_type`、`scenario_id`、seed、truth、prediction、participant、split、case_id 和 manifest 字段属于审计/分组信息，未来 Feature Builder 默认不得直接展开为模型输入特征。





## R1-07 Feature Builder

Use python -m datiao.r1.features.builder --dataset artifacts/r1_synthetic_penprocess_v1 --output artifacts/r1_synthetic_features_v1 --config configs/r1_feature_builder_v1.json --overwrite; reload with datiao.r1.features.io.reload_feature_dataset. Expected materialization: 700 episodes and 360 sequences; source splits remain 490/105/105 episodes.

R1-07-FIX-01 final feature manifest: sha256:40b0326baf0f71206da1fd14b05c8ab5bdf1473c708735c57c019ea820cf5848; alignment records are core hashed assets and runtime contracts load from contracts/r1_feature_schema_v1.json and r1_target_spec_v1.json.
# R1 LightGBM optional dependencies

Install the fixed baseline dependencies with:

```powershell
python -m pip install -e ".[test,ml]"
```
