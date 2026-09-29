# R1 离线复现说明

## 安装和测试

在仓库根目录执行标准安装和测试：

```powershell
python -m pip install -e ".[test]"
python -m pytest
```

`.[test]` 会安装项目和全部 Contract/R1 测试依赖，包括 `pytest` 与 `jsonschema`。运行手册不依赖某个开发者的本机 Python 路径。

当前仓库全量测试以实际 pytest 输出为准；本任务完成时为 68 个。真实设备字段尚未冻结时，只能传入以下规范化字段：

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
QUESTION_VISIT, PROCESS_END
```

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




