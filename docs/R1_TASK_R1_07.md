# TASK-R1-07 — Episode / Feature Builder

状态：PASS（2026-09-29）。R1-06-FIX-01 的 materialized dataset 作为唯一输入，正式入口为：

```text
python -m datiao.r1.features.builder --dataset artifacts/r1_synthetic_penprocess_v1 --output artifacts/r1_synthetic_features_v1 --config configs/r1_feature_builder_v1.json --overwrite
```

本实现没有调用 synthetic generator，也没有训练模型。每个 Stroke 生成一个 ObservationEpisode，稳定 ID 为 `case_id::stroke_id`。输出包含 700 Episode、360 个按 case/participant/segment 分组的 SequenceSample，继承源 participant split：490/105/105 Episode，252/54/54 Sequence。

12 维 X 严格按 `contracts/r1_feature_schema_v1.json` 顺序输出。平面缺失值保持 `null`；Sequence 中对应值为 0，并由 feature_mask=0 标识。路径长度优先使用 Stroke provenance 的归一化值；时间、停顿、bbox、occupancy、访问/返回计数和 previous ink IoU 均由已回载的点/笔画/映射重算。

8 维 target 顺序为 `WRITING, QUESTION_VISIT, QUESTION_LEAVE, RETURN, REVISION_CANDIDATE, PAGE_CHANGE, PROCESS_END, UNKNOWN`。target 只来自独立 `truth_events.jsonl`，以 source point overlap 对齐，预测事件从未进入 X 或 y。对本数据集 alignment audit 为 PASS，未匹配、歧义和一对多均为 0。

Feature leakage、range、split inheritance 和 round-trip reload 均 PASS。主 manifest：`sha256:777ba8f0ba65b27a1f58db08983b17a1650753f1324887a010aec38a033faa78`；source manifest：`sha256:c493f792f9add5547427d002eb74fa0cc0bb663f7f0bb9b19c2bd8c97674af82`。

后续建模必须直接读取 FeatureRow/SequenceSample 与 schema/target contract，禁止把 case/participant/scenario/task metadata、truth/prediction、hash 或 split 字段加入 X。
