# MetaGate-HippoRAG 2 认知科学结课研究

本项目复现 ICML 2025 的 HippoRAG 2，并实现一个受元认知“监控—控制”框架启发的自适应检索门控器。研究比较稠密外部记忆、图结构联想记忆和受控二次检索在事实问答与多跳问答中的准确性、证据完整性、校准性和成本。

## 已锁定的研究范围

- 数据集：`nq_rear`、`musique`、`2wikimultihopqa`。
- 样本：每个数据集 100 条开发集、300 条测试集；固定随机种子 `20260711`。
- 方法：LLM-only、Dense RAG、HippoRAG 2、Always-Expand、MetaGate-HippoRAG 2。
- 生成模型：`gpt-4o-mini-2024-07-18`；温度 0。
- 嵌入模型：`text-embedding-3-large`；所有方法共享并缓存同一批向量。
- 上游代码：`OSU-NLP-Group/HippoRAG` 提交 `ad30fc3e2062202d9e975e32cd28212424a56ccb`。
- 主要指标：Recall@5、EM、Token-F1；同时报告门控校准、覆盖率、调用次数、token、时间和估算费用。

## 项目布局

```text
configs/                    冻结的实验、模型、数据和价格配置
data/                       数据来源说明、校验和与固定划分
docs/superpowers/specs/     已批准的研究与系统设计规范
docs/superpowers/plans/     可逐项执行的实现计划
patches/                    对固定上游提交的最小兼容补丁
scripts/                    环境、上游、数据及实验入口
src/metagate_hipporag/      研究代码
tests/                      单元、契约、集成和烟雾测试
artifacts/                  可恢复但不进 Git 的缓存、索引和逐题运行记录
results/                    可进论文的冻结表格、图和统计摘要
paper/                      中文论文源文件及引用库
slides/                     汇报 PPT、旁白稿和音频
third_party/                按提交下载的 HippoRAG 上游源码
```

详细设计见 [`docs/superpowers/specs/2026-07-11-metagate-hipporag-design.md`](docs/superpowers/specs/2026-07-11-metagate-hipporag-design.md)，实施步骤见 [`docs/superpowers/plans/2026-07-11-metagate-hipporag-implementation.md`](docs/superpowers/plans/2026-07-11-metagate-hipporag-implementation.md)。

## 当前状态

目前只建立研究骨架、设计规范与实施计划；尚未发起付费 API 调用。实施时先运行离线单元测试和 3 题烟雾实验，再提交完整 Batch 作业。
