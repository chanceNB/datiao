# R1 TASK-R1-01

## 完成内容

- 初始化 Python 3.11+ 工程、Pydantic v2 和 pytest 配置。
- 建立不可变的 `Point`、`Stroke`、`QuestionRegion` 和 `StudentProcessEvent` 模型。
- `Point` 保存 normalized fields、原始 `source_payload`、稳定 SHA-256 hash 和不可变质量标记。
- `Stroke` 同时保留 `raw_order` 与 `processed_order` 字段，为原始数据和处理数据分离提供边界。
- 建立可复现的 Synthetic 基础框架，支持正常、返回、重复书写、换页和质量异常场景。
- Synthetic case 同时保存 `raw_points`、独立 `truth` 和算法输出；真值与算法结果分开生成。
- 增加模型不可变性、hash 稳定性、seed 可复现性、truth/output 分离和禁止字段测试。

## 文件结构

```text
src/datiao/r1/
├── models/
├── parser/
├── stroke/
├── mapper/
├── event/
├── synthetic/
└── trace/

tests/r1/
docs/R1_TASK_R1_01.md
```

## 如何运行测试

安装开发依赖：

```powershell
python -m pip install -e ".[test]"
```

运行全部测试：

```powershell
pytest
```

## 当前边界

- R1-03 已实现 Question mapper、Event detector/state machine 和 Trace resolver。
- Synthetic algorithm output 会调用真实的 Mapping 和 Event Detection 链路；truth 仍由场景计划独立生成。
- 真实设备协议适配、班级态势、后端服务和前端页面仍属于后续任务。

本任务不包含 TASK-R1-02 及之后的功能开发。
