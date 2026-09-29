# TASK-R1-05-FIX-01 FINAL REPORT

## 1. STATUS

PASS

## 2. BASELINE

- commit: `86529e06c77f8e35dce9b6f0b6899b53c07a96e1`
- Python: 3.12.14
- tests before: R1 55，Contract 31，全量 86

## 3. MODIFIED FILES

- `src/datiao/r1/event/detector.py`
- `src/datiao/r1/stroke/builder.py`
- `src/datiao/r1/synthetic/generator.py`
- `tests/r1/test_participant_isolation.py`
- `tests/r1/test_geometry_integrity.py`
- `tests/r1/test_process_end_integrity.py`
- 更新的 R1 回归测试、`docs/R1_TASK_R1_05.md`、`docs/R1_RUNBOOK.md`

## 4. PARTICIPANT STATE ISOLATION

- state key: `participant_id`；`None` 使用明确的 `UNKNOWN_PARTICIPANT` bucket，不与已知 participant 混用。
- VISIT/RETURN: 每个 participant 独立维护 visited/active 状态。
- PAGE_CHANGE: 每个 participant 独立维护 page 状态。
- UNKNOWN participant: 多个真实未知参与者暂时共享 unknown bucket，无法区分真实身份；可靠多人过程需要上游 participant identity mapping。
- 事件整体按时间、participant bucket、Stroke、语义顺序稳定合并。

## 5. REVISION HISTORY

- history key: `(participant_bucket, question_id, page_id, coordinate_space)`。
- page isolation: 跨页不复用 prior ink。
- coordinate-space isolation: 只有同 page、同 participant、同 coordinate space 才计算 IoU。
- cross-page behavior: `PAGE_CHANGE`、`QUESTION_LEAVE`、`RETURN` 可产生，但不会仅因 bbox 数值相同产生 revision。

## 6. BBOX GEOMETRY

- norm: 所有有效点都有 norm 时使用 norm bbox。
- raw fallback: norm 不完整但所有点都有 raw 时使用 raw bbox。
- mixed-space prevention: 两种空间都无法完整覆盖时标记 `unavailable`，不混合坐标。
- provenance: Stroke provenance 增加 `bbox_coordinate_space`。

## 7. PROCESS_END

- EOF: 不产生 PROCESS_END。
- legacy `emit_process_end`: 已从 `EventDetectionConfig` 删除。
- explicit end: `process_end_signal=True`，对每个已观察 participant 产生结束事件。
- timeout: 需要 `session_end_ms`、`process_end_timeout_ms` 和该 participant 的最后活动时间。
- multiple participants: 每个 participant 使用自己的最后 Stroke 和 refs。

## 8. MISSING TIME

- stroke fallback: timestamp 任一缺失时不自动切分；若 context、pen-state、空间和输入顺序连续，则保留在同一 Stroke。
- quality propagation: `MISSING_TIMESTAMP` 继续传播为 Stroke `INCOMPLETE` 及事件 degraded 质量。

## 9. WRITING

- geometry requirement: 必须存在 Stroke，并且至少两个有效点在同一坐标空间形成可测运动长度。
- no-stroke behavior: 仅有两个 `point_refs` 不产生 WRITING。

## 10. SYNTHETIC FIX

`arc_length_cross_region` 现在生成同一 session、participant、page、连续时间的一条 Stroke：前段在 A 区域密集采样，后段在 B 区域拥有更长轨迹。Arc-length mapping 选择 B，Truth 由场景计划独立生成。

## 11. TEST RESULTS

- tests/r1: 68 passed
- tests/contracts: 31 passed
- full pytest: 99 passed

## 12. REGRESSIONS FIXED

- participant 之间不再共享 VISIT/RETURN/PAGE_CHANGE/revision 状态。
- 跨页 bbox 不再直接做 revision IoU。
- bbox 不再混用 norm/raw。
- EOF 与 legacy config 无法伪造 PROCESS_END。
- 缺失时间戳不再机械拆 Stroke。
- 无 Stroke 几何证据不再产生 WRITING。
- Synthetic crossing 场景与名称一致，确实为单 Stroke 跨 Region。

## 13. KNOWN LIMITATIONS

- `participant_id=None` 的多个真实参与者仍无法被内部区分。
- raw 坐标 fallback 依赖上游保证同一坐标系统；设备身份缺失时不应跨来源比较。
- revision 仍使用可解释的 BBox IoU，阈值需要真实 Dev/Validation 数据校准。

## 14. OUT OF SCOPE CONFIRMATION

- Dataset Split: 未开发
- Feature Builder: 未开发
- LightGBM: 未开发
- Pen TCN: 未开发
- R3: 未开发
- Vision: 未开发
- Fusion: 未开发
- Agent: 未开发
- Frontend: 未开发

## 15. FINAL COMMIT

以本轮最终同步到远程 `main` 的 commit 为准。

## 16. NEXT RECOMMENDED TASK

TASK-R1-06 — Synthetic Dataset / Manifest / Group Split（只建议，不执行）。
