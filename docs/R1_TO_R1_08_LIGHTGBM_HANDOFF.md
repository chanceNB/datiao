# R1 → R1-08 LightGBM Handoff

建议下一任务：实现并验证 LightGBM baseline。读取 FeatureRow，按 source split 训练 8 个 one-vs-rest 分类器，输出逐 Episode 概率和固定 8-label 指标。不得修改 feature whitelist、target order 或 split。


当前 Feature Manifest hash：sha256:40b0326baf0f71206da1fd14b05c8ab5bdf1473c708735c57c019ea820cf5848；Flat rows 为 700（490/105/105 split），target spec 与 canonical contract 一致。
