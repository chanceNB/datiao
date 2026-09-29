# TASK-R1-08-FIX-01 — Reproducibility & Evaluation Integrity

状态：PASS。FIX-01 只修复 LightGBM baseline 的复现审计、模型重载和评估序列化独立性；Feature/Target/Split、模型参数、阈值和 learner 选择保持冻结。

## 输入与执行

- source Feature Manifest：`sha256:40b0326baf0f71206da1fd14b05c8ab5bdf1473c708735c57c019ea820cf5848`
- full baseline：Train/Validation/Test = `490/105/105`
- 两次独立执行均在独立输出目录完成；第二次使用临时目录，不复用第一次模型或指标文件。
- Test 仅用于最终 inference；没有进入 fit、early stopping、调参或阈值选择。

## 审计结果

`audits/reproducibility.json` 同时记录两次执行的 prediction、metrics 和 run hash。两次执行三类 hash 全部相等，状态为 `PASS`；validation/test 的每个 label 模型重载概率最大绝对差均为 `0.0`。

`reload_lightgbm_run` 校验 run manifest 自哈希、冻结 feature/label order、source Feature Manifest lineage、全部模型文件 hash、predictions hash 和 metrics hash。测试覆盖 manifest、metrics 和模型文件篡改检测。

指标在写出前直接从按 source split 顺序的 `Y` 与概率矩阵计算，序列化 predictions 后再按 episode ID 排序不会改变指标；FIX-01 没有改变任何最终指标或模型输出。

## 最终结果

- Validation：micro-F1 `0.9087136929460581`；macro-F1 `0.6659054487179488`；macro ROC-AUC `0.9442766273542061`
- Synthetic Test：micro-F1 `0.9006085192697769`；macro-F1 `0.6612445525489004`；macro ROC-AUC `0.9287554772067418`
- predictions hash：`sha256:42972dcff9a63bd16d505919fa109683ce3a87286b06537e205556e594b7b553`
- metrics hash：`sha256:440ebc9995e70b5035b88c2513d5284301ba3026413999ae688cb1cb5c0e0fdc`
- run hash：`sha256:3b6235b64974c244fe3dc80f81798d815404dc48786323ca1fbb94ead33118e0`

## 依赖与测试

Bundled Python `3.12.14`、LightGBM `4.7.0`、scikit-learn `1.9.1`、NumPy `2.3.5`、SciPy `1.18.1`。FIX-01 后 `tests/r1` 为 `122 passed`、`tests/contracts` 为 `31 passed`、全量为 `153 passed`；仅有既有 LightGBM/sklearn warnings。

Synthetic 数据由规则生成，存在 shortcut risk，不能代表真实课堂或真实学生泛化性能。后续只建议进入 TASK-R1-09 TCN baseline，不在本任务执行。
