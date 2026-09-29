# TASK-R1-05 FINAL REPORT

## 1. STATUS

PASS

## 2. BASELINE

- commit: `cc78ac2170ba18f16179d75008e9e1d92a516dfc`
- Python: 3.12.14
- tests before: `tests/r1` 39，`tests/contracts` 31，full 70

## 3. MODIFIED FILES

- `src/datiao/r1/stroke/geometry.py`
- `src/datiao/r1/stroke/builder.py`
- `src/datiao/r1/stroke/__init__.py`
- `src/datiao/r1/mapper/models.py`
- `src/datiao/r1/mapper/mapping.py`
- `src/datiao/r1/mapper/__init__.py`
- `src/datiao/r1/event/detector.py`
- `src/datiao/r1/event/__init__.py`
- `src/datiao/r1/pipeline.py`
- `src/datiao/r1/models/legacy.py`
- `src/datiao/r1/synthetic/models.py`
- `src/datiao/r1/synthetic/generator.py`
- `tests/r1/test_stroke_semantics.py`
- `tests/r1/test_arc_length_mapping.py`
- `tests/r1/test_event_semantics_v2.py`
- 更新的既有 R1 测试、`docs/R1_RUNBOOK.md`

## 4. STROKE SEMANTICS

- pen state: `DOWN` 开始接触，`MOVE` 延续，`UP` 关闭；下一次可靠 `DOWN` 开启新 Stroke。`UNKNOWN` 使用降级规则。
- time gap: `StrokeBuildConfig.max_time_gap_ms`，development default 为 1000ms。
- spatial jump: 只使用同一 `norm` 坐标空间，`max_spatial_jump_norm` development default 为 0.15；不混合 raw/mm/norm。
- page/context: session、participant、task segment、page 发生变化时切分。
- 缺失时间戳保留 `MISSING_TIMESTAMP`，不伪造时间。
- algorithm version: `r1-stroke-rule-v2`。
- Stroke provenance 保存可用的 `path_length_norm`；无完整 norm 坐标时为 `null`。

## 5. QUESTION MAPPING

- old method: 有效点数量 coverage。
- new method: 对每个相邻点段做 polygon clipping，使用 `Σ inside_length / Σ total_length` 的 arc-length coverage。
- zero-length fallback: 点包含 fallback，映射 provenance 为 `POINT_FALLBACK`；否则为 `ARC_LENGTH`。
- ambiguity: `QuestionMappingConfig.ambiguity_epsilon` development default 为 0.02；同优先级近似并列返回 `AMBIGUOUS` 和 `UNKNOWN` 语义。
- threshold: `min_coverage` development default 为 0.5，正式实验前必须在 Dev/Validation 校准并冻结。
- algorithm version: `r1-qmap-arc-v1`。

## 6. WRITING

- trigger: Stroke 至少有两个有效坐标点且存在可观察运动长度；无坐标或单个静止点不产生 WRITING。
- invalid cases: 纯无效点、缺失几何证据不会伪造 WRITING；UNKNOWN mapping 仍可保留可观察 WRITING。
- traceability: WRITING 通过 `stroke_refs`、`point_refs` 回溯到 Stroke 和 Raw Point。

## 7. REVISION_CANDIDATE

- prior ink: 按 participant/question 维护历史 Stroke bbox。
- pause/revisit: 需要回访，或相邻同题 Stroke 间隔达到 `revision_pause_threshold_ms`（development default 3000ms）。
- overlap: BBox IoU 达到 `revision_overlap_threshold`（development default 0.10）。
- negative cases: 普通同题续写、短间隔续写、无重叠续写均只产生 WRITING，不判 revision。

## 8. PROCESS_END

- EOF behavior: 输入列表结束不产生 PROCESS_END。
- explicit end: `process_end_signal=True` 才产生 PROCESS_END。
- timeout behavior: 只有 caller 提供 `session_end_ms` 且配置 `process_end_timeout_ms`，并满足 inactivity threshold，才产生 PROCESS_END。

## 9. OTHER EVENTS

- QUESTION_VISIT: 当前过程第一次对 question 产生有效活动。
- QUESTION_LEAVE: 映射转到其它 question/page 或 mapping 不可用；pen UP 本身不触发。
- RETURN: 已访问过的 question 在离开后再次进入。
- PAGE_CHANGE: 两个明确 page_id 之间发生转移；UNKNOWN page 不伪造变化。
- UNKNOWN: 证据不足、映射冲突或不可可靠归题。

## 10. SYNTHETIC SCENARIOS

- 新增场景：`continuous_same_question_writing`、`true_revision_overlap`、`same_question_no_overlap`、`explicit_process_end`、`open_process_no_end`、`spatial_jump_split`、`arc_length_cross_region`。
- truth separation：Truth 继续由 scenario plan 独立生成，未读取 detector 输出；显式结束只在 `explicit_process_end` 计划中出现。

## 11. TEST RESULTS

- tests/r1: 55 passed
- tests/contracts: 31 passed
- full pytest: 86 passed

## 12. BEHAVIOR CHANGES

- same-question second stroke: `REVISION_CANDIDATE` → 只有 prior ink + pause/revisit + overlap 才判 revision，否则只有 `WRITING`。
- input EOF: `PROCESS_END` → 不产生 PROCESS_END。
- point-count coverage: 点数比例 → polygon segment 的 arc-length coverage。
- Stroke segmentation: 时间/页面 → pen-state + time gap + spatial jump + context。
- active Stroke: pen UP → 只关闭 Stroke，不直接产生 QUESTION_LEAVE。

## 13. KNOWN LIMITATIONS

- BBox IoU 对跨行答案的空间关系仍是保守近似，阈值需要真实 Dev/Validation 数据校准。
- 原始坐标没有明确同一物理空间时不参与 norm spatial jump 或 norm mapping。
- timeout 需要调用方提供 session end reference，不能从 EOF 推断。

## 14. OUT OF SCOPE CONFIRMATION

确认没有开发 Dataset Split、Feature Builder、LightGBM、Pen TCN、R3、Vision、Fusion、Agent、Frontend。

## 15. FINAL COMMIT

最终 commit SHA 以仓库 `HEAD` 与远程 `main` 的同步结果为准。

## 16. NEXT RECOMMENDED TASK

TASK-R1-06 — Synthetic Dataset / Manifest / Group Split（只建议，不执行）。

## Post-validation fix

TASK-R1-05-FIX-01 PASS
