# TASK-R1-08 — LightGBM Multi-label Baseline

状态：实现并完成一次完整 Synthetic baseline 运行。输入只来自已 reload 的 `artifacts/r1_synthetic_features_v1`，source Feature Manifest 必须为 `sha256:40b0326baf0f71206da1fd14b05c8ab5bdf1473c708735c57c019ea820cf5848`。

CLI：

```text
python -m datiao.r1.lightgbm.train --features artifacts/r1_synthetic_features_v1 --output artifacts/r1_lightgbm_v1 --config configs/r1_lightgbm_v1.json --overwrite
```

模型采用固定参数的 8-label One-vs-Rest。WRITING 在 train 中为单类，使用 `CONSTANT_POSITIVE`，不伪造 LightGBM 模型；其 ROC-AUC/AP 在 validation/test 中为 null。其余 7 个标签实际执行 LightGBM binary fit，并使用 validation binary_logloss early stopping。阈值固定 0.5，Test 仅做最终 inference。

模型、predictions、metrics、feature importance、input integrity、reload/reproducibility audit 和 run manifest 写入 `artifacts/r1_lightgbm_v1/`。该目录属于本地 artifact，不提交大型模型文件。Synthetic Test 结果只用于开发基线，不代表真实课堂或真实学生泛化性能。

实测 baseline run：`r1-lightgbm-ovr-v1@1.0.0`。

- Train/Validation/Test：490/105/105
- Learners：WRITING=`CONSTANT_POSITIVE`；其余 7 labels=`LIGHTGBM`
- Validation micro-F1：0.9087136929460581；macro-F1：0.6659054487179488；macro ROC-AUC（7 evaluable）：0.9442766273542061
- Synthetic Test micro-F1：0.9006085192697769；macro-F1：0.6612445525489004；macro ROC-AUC（7 evaluable）：0.9287554772067418
- predictions hash：`sha256:42972dcff9a63bd16d505919fa109683ce3a87286b06537e205556e594b7b553`
- metrics hash：`sha256:440ebc9995e70b5035b88c2513d5284301ba3026413999ae688cb1cb5c0e0fdc`
- run hash：`sha256:3b6235b64974c244fe3dc80f81798d815404dc48786323ca1fbb94ead33118e0`
- reproducibility audit：PASS，reloaded test probability max absolute diff=0.0

这些指标仅描述 Synthetic Development Test，不能代表真实课堂或真实学生泛化性能；规则生成存在 shortcut risk。
