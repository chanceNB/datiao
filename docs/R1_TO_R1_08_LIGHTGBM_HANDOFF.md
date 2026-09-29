# R1 → R1-08 LightGBM Handoff

建议下一任务：实现并验证 LightGBM baseline。读取 FeatureRow，按 source split 训练 8 个 one-vs-rest 分类器，输出逐 Episode 概率和固定 8-label 指标。不得修改 feature whitelist、target order 或 split。
