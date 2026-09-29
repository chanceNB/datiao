# R1 → R4 LightGBM Handoff

Baseline run：`r1-lightgbm-ovr-v1@1.0.0`。输入 Feature Manifest：`sha256:40b0326baf0f71206da1fd14b05c8ab5bdf1473c708735c57c019ea820cf5848`。训练矩阵严格为 frozen 12 features，目标为 frozen 8-label multi-hot，null 转换为 `np.nan` 交给 LightGBM 原生 missing handling。

WRITING 为单类 train support，使用 constant predictor；其他标签使用固定参数 LightGBM，并由 validation binary_logloss early stopping。Test 未参与拟合、调参或阈值选择。所有结果属于 Synthetic Development Test，存在规则生成 shortcut risk。

Baseline run hash：`sha256:3b6235b64974c244fe3dc80f81798d815404dc48786323ca1fbb94ead33118e0`。Synthetic Test 指标仅作开发基线，不能解释为真实课堂效果。
