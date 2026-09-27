# R1 TASK R1 03

## 目标

把 Stroke 转换为题目级过程事件，并保留从事件到原始笔画和点的可追溯关系。

## 已实现内容

- `QuestionRegion` 支持矩形和多边形，校验几何坐标并提供边界包含判断。
- `map_strokes_to_regions()` 按有效点覆盖率映射 Stroke；默认阈值为 50%。
- 无题区匹配、区域歧义、缺页或 Stroke 质量异常时保留来源并返回 `UNKNOWN` 映射。
- `detect_student_process_events()` 实现 `QUESTION_VISIT`、`QUESTION_LEAVE`、`RETURN`、`REVISION_CANDIDATE`、`PAGE_CHANGE`、`PROCESS_END` 和 `UNKNOWN`。
- `resolve_event_trace()` 将事件中的 Stroke ID 和 Point ID 解析为实际对象，并拒绝缺失来源或跨场次来源。
- `adapt_canonical_records()` 固定规范化字段，不猜测设备字段别名；设备字段需要单独适配器明确转换。
- Synthetic 场景保存独立真值，同时运行实际 Mapping 和 Event Detection 输出。

## 事件边界

- `REVISION_CANDIDATE` 只表示同一题出现新的独立书写段或重叠候选，不表示改错、不会、焦虑或粗心。
- 无法可靠映射的输入返回 `UNKNOWN`，不填充虚假的题目或完成结论。
- 每个事件保存场次、时间、页面、题目和来源 Stroke/Point 引用。

## 仿真场景

包含按序作答、跨题跳转、返回、重复书写、页面切换、暂停继续、未知题区、缺时间、重复点和乱序点。

## 验收命令

```powershell
python -m pytest
```

当前验收包括 31 个测试：既有 Point/Stroke 基线，也包括映射、事件顺序、独立真值、质量降级和 Evidence Trace。

## 复现示例

```powershell
python -c "import sys; sys.path.insert(0, 'src'); from datiao.r1.synthetic import Scenario, generate_synthetic_case; c=generate_synthetic_case(Scenario('demo-return','return_visit',7)); print([e.event_type for e in c.algorithm_output.events])"
```

输出应为：

```text
['QUESTION_VISIT', 'QUESTION_LEAVE', 'QUESTION_VISIT', 'QUESTION_LEAVE', 'RETURN', 'PROCESS_END']
```

## 未包含内容

- 真实设备协议适配器；
- 班级态势、后端服务和前端页面；
- 视觉识别和心理状态推断。
