# MetaGate-HippoRAG 2 研究与系统设计

**日期：** 2026-07-11

**状态：** 已获用户确认，可进入实施计划

**截止日期：** 2026-07-31

**课程材料：** 工作区根目录 `认知科学导论结课材料2026.5.pdf`

## 1. 目标与成功标准

本项目完成一项可复现的认知科学课程研究：以 ICML 2025 的 HippoRAG 2 为 SOTA 骨干，在事实问答和多跳问答上复现其“知识图谱 + Personalized PageRank + LLM”检索流程，并加入受元认知监控—控制框架启发的检索门控 MetaGate。研究关注的不是单纯搭建 RAG 系统，而是检验外部图式语义记忆与元认知控制在不同推理负荷下何时有效、为何失败以及付出多少资源。

项目成功必须同时满足以下条件：

1. 在 `nq_rear`、`musique`、`2wikimultihopqa` 上生成固定且无重叠的开发/测试划分，每个数据集分别为 100/300 条。
2. 使用同一生成模型、嵌入模型、语料、提示和答案评价规则完成五种方法的逐题实验。
3. 至少成功复现 Dense RAG 与 HippoRAG 2，并完成 Always-Expand 和 MetaGate 两个受控实验条件。
4. 每条结果能够追溯到数据样本、代码提交、上游提交、配置哈希、模型、提示哈希、缓存命中和 token 使用。
5. 报告 Recall@5、EM、Token-F1、门控校准、假停止、覆盖率、调用次数、耗时和费用，并给出置信区间与配对统计检验。
6. 形成 5000 字以上中文论文、规范参考文献、清晰图表，以及约 10–12 页、含旁白且不是视频的 PPT。
7. 不伪造结果，不把未运行的实验写成已完成，不把 LLM 输出直接当作个人研究结论。

## 2. 研究问题、假设与理论边界

### 2.1 研究问题

- **RQ1：** 相同语料和嵌入条件下，图结构外部记忆是否比稠密向量记忆更有利于多跳联想检索？
- **RQ2：** 基于证据充分性监控的条件式二次检索，能否比固定一次检索提高证据完整性和问答表现？
- **RQ3：** 与每题都扩展的策略相比，MetaGate 能否以更少调用达到相近或更好的准确性与可靠性？
- **RQ4：** 门控在何种题型上产生假停止、无效扩展或低置信强制作答？

### 2.2 预注册式假设

- **H1：** HippoRAG 2 相对 Dense RAG 的 Recall@5 和 Token-F1 增益，在 MuSiQue 与 2Wiki 上大于在 NQ 上的增益；以“两个多跳数据集平均增益减去 NQ 增益”的预设 bootstrap contrast 检验。
- **H2：** MetaGate 相对原始 HippoRAG 2 能提高两个多跳数据集的 Recall@5；若原始准确率没有显著提高，也应降低低证据回答的选择性风险。
- **H3：** MetaGate 的平均第二轮检索率低于 Always-Expand 的 100%，且 `MetaGate − Always-Expand` 测试集 Token-F1 配对差值的单侧 95% bootstrap 置信下界高于预设非劣界值 `−0.02`。

“假停止率随支持段落数量或复合问题类型增加而上升，成功扩展主要发生在首轮缺失桥接事实的样本中”作为探索性预测，不作为第四个验证性假设，也不据此新增隐藏超参数。

### 2.3 认知科学映射

- 知识图谱是外部语义记忆和认知卸载的计算实现；PPR 是联想扩散的工程近似。
- HippoRAG 2 的概念节点与段落节点对应概念信息和情境信息的联合索引。
- LLM 的 OpenIE、识别过滤和阅读问答分别承担编码、线索识别和解释生成。
- MetaGate 的“估计证据充分性—决定停止或继续—在低置信时标记弃答”对应元认知监控与控制。

这些对应关系只用于提出功能层面的可检验假设。论文不得声称 LLM、知识图谱或 PPR 真实复现了人类海马、新皮层、意识或主观认知过程。

## 3. 范围与明确排除项

### 3.1 纳入范围

- 英文开放域事实和多跳问答。
- HippoRAG 2 的离线 OpenIE 建图、OpenAI 嵌入、三元组过滤、PPR 和前 5 段问答。
- 单次扩展的 MetaGate；最多两轮证据监控。
- 固定模型快照、低温、缓存和逐题可恢复运行。
- 方法复现、消融、统计分析、论文和汇报材料。

### 3.2 排除范围

- 不训练或微调 7B/70B 模型，不复现论文全部七个数据集或全部原始表格。
- 不部署 Neo4j、Milvus、Qdrant 或外部向量数据库；默认使用上游 Parquet 与 igraph。
- 不建立通用 Agent 框架，不加入多智能体辩论、长期对话记忆或时序遗忘。
- 不把 `text-embedding-3-large` 的结果冒充 NV-Embed-v2 的精确复现。
- 不在测试集上选择门控阈值、提示示例、PPR 参数或其他超参数。
- 不将模型生成的“解释”直接视为真实思维链或神经机制证据。

## 4. 固定依赖与运行环境

- 主机：Windows，RTX 4080 Laptop 12GB，约 32GB RAM。当前未安装 WSL，因此正式实验采用原生 Windows 的 OpenAI-online 路径；不引入需要 Linux 的 vLLM。
- Python：3.10；使用 `uv` 创建和锁定环境，不使用当前系统 Python 3.14。
- 上游：`OSU-NLP-Group/HippoRAG` 固定提交 `ad30fc3e2062202d9e975e32cd28212424a56ccb`，其包版本为 `2.0.0-alpha.4`。
- 生成模型：`gpt-4o-mini-2024-07-18`，温度 0，固定 seed `20260711`。
- 嵌入模型：`text-embedding-3-large`；所有向量落盘并以内容哈希去重。
- 图算法：上游 `python-igraph` 的 Personalized PageRank，阻尼 0.5。
- 凭据：仅从 `OPENAI_API_KEY` 环境变量读取；`.env`、请求头和密钥不得提交或写入日志。

使用 OpenAI 嵌入而不是 NV-Embed-v2 的原因是 12GB 显存无法稳妥复现论文的 7B 嵌入设置。所有对照条件共享同一嵌入和缓存，因此方法间差异仍可归因于检索结构与控制策略；论文将明确这一外部有效性限制。

## 5. 系统架构与模块边界

### 5.1 上游管理

`scripts/bootstrap_upstream.py` 负责：

1. 克隆官方仓库到忽略目录 `third_party/HippoRAG/`。
2. 检出并验证固定 SHA；不允许静默使用其他提交。
3. 应用 `patches/hipporag-openai-only.patch`，将未使用的本地模型导入改为惰性导入，并移除 Windows 在线模式不需要的 `vllm` 依赖路径。
4. 以 `--no-deps` 可编辑安装上游；依赖完全由本项目锁文件控制。
5. 执行上游契约烟雾测试，验证 OpenAI embedding 类、`HippoRAG.retrieve`、`HippoRAG.qa` 和 PPR 可导入。

补丁只能解决安装和惰性导入，不改变检索公式、提示、过滤逻辑、PPR 或评价指标。

上游固定提交存在一个必须保留并披露的实现特征：OpenAI embedding 类接收 query-to-fact/query-to-passage instruction 参数，但未把 instruction 拼接到实际 API 输入。主实验按固定提交原样运行，以免把未合并修复冒充复现结果；如后续验证 instruction 修复，只能作为带独立配置哈希的附加消融。

### 5.2 项目代码

- `config.py`：读取 YAML，使用 Pydantic 校验所有枚举、阈值、路径和模型 ID，生成基础配置哈希；再把 gate/QA/OpenIE 模板内容、兼容补丁、数据 manifest 和固定上游提交的 SHA 合并生成 `effective_config_hash`。
- `data.py`：下载并校验官方数据，提取标准化 `Example`，生成一次性固定划分。
- `openai_client.py`：结构化输出、SQLite 缓存、重试、token/延迟记录和密钥脱敏。
- `embedding.py`：在上游 OpenAI embedding 类外增加永久请求缓存和 usage ledger，并在首次 index/retrieve 前注入引擎及三个 embedding store；不改变向量内容或相似度算法。
- `batch_openie.py`：按 NER、三元组抽取两个阶段生成和恢复 OpenAI Batch 作业，最终写出上游兼容 OpenIE JSON。
- `hipporag_adapter.py`：组合固定上游对象，公开单题 Dense/HippoRAG 检索、QA 和完整 `RetrievalTrace`。
- `metagate.py`：门控 schema、提示、阈值选择、单次扩展控制和最终低置信标记。
- `fusion.py`：实现确定性的 Reciprocal Rank Fusion；上游 chunk ID 作为稳定键，`k=60`。
- `methods.py`：五种实验条件的统一接口，保证相同数据与 QA 评测。
- `evaluation.py`：Recall、EM、Token-F1、校准、覆盖率、风险、费用和耗时汇总。
- `statistics.py`：配对 bootstrap、精确 McNemar、Holm 校正和效应差值表。
- `provenance.py`：运行清单、文件 SHA-256、代码/上游提交、提示哈希、环境快照和原子 JSONL 续跑。
- `cli.py`：`prepare-data`、`prepare-openie`、`build-index`、`tune-gate`、`run`、`analyze`、`verify-run` 子命令。

每个模块只通过显式 dataclass/Pydantic 类型交换数据，不共享隐式全局状态。

每个图索引使用不可变指纹目录：`artifacts/indexes/<dataset>/<corpus_sha12>/<upstream_sha12>/<llm_slug>/<embedding_slug>/<openie_prompt_sha12>/<index_config_sha12>/`。其中 `index_config_sha` 覆盖上游补丁、预处理版本、嵌入维度与 instruction mode、OpenIE 模板及所有建图/链接参数，但不包含 gate 阈值等不影响索引的参数。不能依赖上游 `force_index_from_scratch` 覆盖旧目录，因为该参数不会清理已有 Parquet 和 OpenIE 文件。查询向量在单次运行中由同一引擎跨方法共享，退出前导出为带有效配置哈希的 NPZ；恢复时只有相关哈希完全一致才可加载。

### 5.3 核心数据类型

```python
class Example(BaseModel):
    dataset: Literal["nq_rear", "musique", "2wikimultihopqa"]
    example_id: str
    question: str
    gold_answers: list[str]
    gold_docs: list[str]
    stratum: str

class RetrievedPassage(BaseModel):
    chunk_id: str
    text: str
    score: float
    rank: int

class RetrievalTrace(BaseModel):
    retrieval_query: str
    passages: list[RetrievedPassage]
    facts_before_filter: list[tuple[str, str, str]]
    facts_after_filter: list[tuple[str, str, str]]
    used_dense_fallback: bool
    filter_error: str | None
    prompt_tokens: int
    completion_tokens: int
    observed_latency_seconds: float
    method_equivalent_latency_seconds: float

class GateDecision(BaseModel):
    evidence_sufficient_probability: float
    missing_information: str
    retrieval_rewrite: str
    rationale_summary: str

class MethodResult(BaseModel):
    run_id: str
    method: str
    example: Example
    first_retrieval: RetrievalTrace | None
    second_retrieval: RetrievalTrace | None
    fused_passages: list[RetrievedPassage]
    answer: str
    gate_decisions: list[GateDecision]
    expanded: bool
    abstain_flag: bool
    usage: dict[str, int | float]
    errors: list[str]
```

`rationale_summary` 只能记录一句可审计的证据摘要，不保存或要求模型私有思维链。

## 6. 数据与采样协议

### 6.1 数据来源

从官方 `osunlp/HippoRAG_2` 的固定提交 `5ec05b38deecc3318bb432c69865959c56058990` 下载以下文件及对应 corpus：

- `nq_rear.json` / `nq_rear_corpus.json`
- `musique.json` / `musique_corpus.json`
- `2wikimultihopqa.json` / `2wikimultihopqa_corpus.json`

下载时记录最终 URL、revision、ETag、字节数、SHA-256 和时间。处理后文档格式严格为 `title + "\n" + text`，黄金文档按固定上游 `get_gold_docs()` 的 `supporting_facts → contexts → paragraphs` 分派规则生成，并以完整字符串精确匹配检索结果。

三个问答文件各含 1000 条复现样本；对应 corpus 分别含 NQ 9,633、MuSiQue 11,656、2Wiki 6,119 段。本项目的 100/300 是从这 1000 条中自行建立的开发集与留出测试集，不得写成数据集官方 dev/test 划分。

### 6.2 固定划分

- NQ：以 `example_id` 排序后使用固定 PRNG 无分层抽取 100 条开发样本、300 条留出测试样本。
- MuSiQue：按支持段落数 2/3/4 分层，使用最大余数法维持原比例，再分别抽取 100/300。
- 2Wiki：按 `compositional`、`comparison`、`bridge_comparison`、`inference` 分层并维持原比例。
- 同一数据集的 dev/test ID 不得重叠；划分文件含源文件哈希和每个 stratum 的计数。
- 若数据少于要求或存在重复 ID，命令直接失败，不自动降低样本量。

开发样本只用于门控提示定稿、阈值选择和烟雾验证；留出测试样本在配置、提示和阈值冻结后一次性运行。

## 7. 五种实验条件

### 7.1 LLM-only

只向同一 QA 模型提供原始问题，不提供检索证据。它衡量参数记忆，并作为检索增益的下界对照。

### 7.2 Dense RAG

调用同一 HippoRAG 索引中的 `dense_passage_retrieval`，取前 5 段并用相同 QA 提示作答。它与图方法共享 corpus、embedding 和 QA。

### 7.3 HippoRAG 2

使用上游 query-to-triple、识别过滤、段落/短语联合 PPR 和前 5 段 QA。适配层只暴露原本未返回的过滤日志，不改变排序。

固定上游没有 NQ 专用 QA 模板，会回退到 MuSiQue 的短答案模板。主实验保留该行为，并让五种方法全部使用同一模板；不得只为某一方法修改 NQ 提示。LLM-only 也调用同一 QA 渲染器，但证据列表为空并明确标记“无检索证据”；其 Recall 指标记为 `N/A`，不能写成 0。

### 7.4 Always-Expand

首轮 HippoRAG 2 后调用一次门控模型生成补充检索查询，但忽略其充分性概率，所有样本都运行第二轮 HippoRAG 2。两轮各取前 5 段，以 RRF 合并后取前 5 段；再运行与 MetaGate 扩展分支完全相同的第二次充分性评估并作答，但不据此跳过强制作答。它控制“更多调用和更多检索本身”的效果，使两种扩展分支只在是否条件停止上不同。

### 7.5 MetaGate-HippoRAG 2

首轮检索后，门控读取原问题、过滤前后事实和前 5 段，输出 `GateDecision`：

1. 若充分性概率大于等于阈值，直接停止并作答。
2. 若低于阈值，使用 `retrieval_rewrite` 执行且仅执行一次第二轮 HippoRAG 2。
3. 两轮结果按与 Always-Expand 完全相同的 RRF 合并，使用原始问题作答。
4. 对合并证据再评估一次充分性；若仍低于阈值，设置 `abstain_flag=true`，同时保留强制作答结果供公平 EM/F1 比较。

测试时门控不能看到答案、黄金文档、数据集标签或是否多跳。

## 8. 门控阈值与提示冻结

门控置信度的目标事件是“前 5 段包含全部黄金支持文档”，即逐题 Recall@5 等于 1。候选阈值为 `0.50, 0.55, ..., 0.95`。在三个开发集合并的首轮结果上计算平衡准确率，选择最高者；若并列，依次选择扩展率更低、阈值更高者。

门控使用零样本固定提示和严格 JSON schema，不加入人工挑选的开发集示例，以避免示例选择成为隐藏超参数。提示全文和 SHA-256 必须提交。提示冻结后不得根据测试结果修改；任何提示内容变化都会改变 `effective_config_hash` 和 run ID。

## 9. 指标与统计分析

### 9.1 主要结果

- 检索：逐题 Recall@2、Recall@5，正文以 Recall@5 为主。
- 问答：标准化 EM、Token-F1，正文以 Token-F1 为主。
- 主要比较：HippoRAG 2 vs Dense RAG；MetaGate vs HippoRAG 2；MetaGate vs Always-Expand。

### 9.2 元认知与效率

- 首轮假停止率：概率达到阈值但首轮 Recall@5 小于 1 的比例。
- 不必要扩展率：概率低于阈值但首轮 Recall@5 等于 1 的比例。
- Brier score、10 等频箱 ECE、可靠性图只基于所有门控方法都具备的首轮 gate；概率并列不拆开，并记录实际箱数。二轮 gate 仅作诊断。
- 最终 coverage、selective risk 和 risk-coverage 曲线以 `1 − EM` 为主风险，`1 − Token-F1` 作为补充；强制作答 EM/F1 始终同时报告。
- 每题 LLM 调用数、扩展率、输入/输出 token、缓存命中、墙钟时间和按冻结价格表估算的直接 API 费用。
- 共享 gate/rewrite/二轮检索时分别记录项目实际付费成本与“若各方法独立运行”的方法等价成本；论文的方法成本比较使用等价成本，预算保护使用实际成本。

首轮假停止率的主分母为所有首轮证据不充分题（`Recall@5 < 1`），不必要扩展率的主分母为所有首轮证据充分题（`Recall@5 = 1`）；同时附上占全体题目的事件比例与原始计数。实际缓存命中不计新增费用，方法等价成本则按该方法独立运行时重计共享调用；Batch 折扣只用于实际由 Batch 执行的 OpenIE 阶段。

### 9.3 统计规则

- 对 Recall@5、Token-F1、费用和调用次数的配对差值用种子 `20260711` 做 10,000 次按题 bootstrap，报告均值差和 95% percentile CI；H1 另报告预设跨数据集 contrast，H3 另报告相对 `−0.02` 的单侧非劣下界。
- 对逐题 EM 使用双侧精确 McNemar 检验。
- 对三个数据集 × 三个主要比较形成的 9 个 EM 精确 McNemar p 值作为一个检验族使用 Holm 校正，`alpha=0.05`；bootstrap CI 为效应量区间，不伪装成经 Holm 校正的显著性检验。
- 除总体均值外，MuSiQue 按 2/3/4 支持段落、2Wiki 按四种 `type` 报告探索性分层结果；这些分层不宣称独立验证性结论。
- 不以“p > .05”证明两种方法等价；MetaGate 与 Always-Expand 只能表述为差异未被当前样本检出，并结合 CI 与费用判断。

## 10. 失败恢复与数据完整性

- 所有 API 结果先写临时文件，完成 JSON/schema 校验后原子替换目标文件。
- 每条请求使用由阶段、数据集、样本 ID、模型、提示哈希和配置哈希组成的稳定 `custom_id`。
- Batch 作业状态、输入文件 ID、batch ID、输出文件 ID 和解析计数写入状态文件；重复命令只恢复未完成阶段。
- 网络错误使用指数退避；认证错误、余额不足、schema 连续三次失败或源文件哈希变化立即停止。
- 3 题 smoke 的实际费用硬上限为 1 美元，整个项目的记录费用硬上限为 18 美元；提交新 Batch 前若按 token 估算会越界则直接拒绝。
- 逐题运行以 JSONL 追加并在启动时建立完成索引；已完成且配置哈希一致的结果跳过，错误记录可单独重试。
- 任何批次若缺失、重复或无法匹配 `custom_id`，不得进入图索引或最终统计。
- OpenIE Batch 必须分为有先后依赖的 NER 和 triple 两阶段；每阶段再按账户 batch token 上限切成可恢复分片，前一阶段全部通过数量与 schema 校验后才创建后一阶段。
- 测试集运行完成后生成只读 manifest；后续分析只读取 manifest 中列出的文件。
- `artifacts/runs/` 保存可恢复的原始缓存与完整响应，不进 Git；完成后导出无提示正文、无密钥、无私有缓存的精简逐题记录到 `results/records/<run_id>.parquet`，并把文件哈希、有效配置哈希和环境信息写到可提交的 `results/manifests/<run_id>.json`。所有表格必须能仅依赖这两类冻结产物重建。

## 11. 测试策略与验收场景

### 11.1 离线单元测试

- 配置拒绝非法阈值、未知方法、错误模型和 dev/test 重叠。
- 数据适配器正确解析三种源 schema，并保持黄金文档字符串与上游一致。
- RRF 在并列、重复文档、空列表和不同长度输入上确定性排序。
- 门控 JSON 缺字段、越界概率或空 rewrite 时明确失败并可重试。
- 阈值选择不读取 test，且并列规则稳定。
- Recall、EM、F1、Brier、ECE、Holm 和 McNemar 与手算小样例一致。
- 运行记录重启后不重复执行已成功样本；API key 不出现在日志和 manifest。

### 11.2 上游契约与烟雾测试

- 固定 SHA 和补丁哈希匹配，否则 bootstrap 失败。
- 使用伪 embedding/LLM 检查 `retrieve_with_trace` 返回过滤前后事实和前 5 段。
- 对同一索引和查询比较项目 bridge 与官方 `retrieve()`：chunk ID 排名必须完全相同，分数必须 `allclose`，否则不得运行正式基线。
- 使用官方 sample corpus 完成 3 题在线烟雾实验，费用上限 1 美元。
- smoke 通过后依次执行一个数据集 10 条、每数据集完整 dev、最终 test；不直接启动全部测试。

### 11.3 最终验收

- 每个数据集、方法恰有 300 条唯一测试结果，无缺失、无重复、无 dev ID。
- 所有表格能从冻结 manifest 一条命令重建。
- 论文数字与 CSV 自动交叉检查；PPT 数字只引用论文冻结表。
- Word/PDF 和 PPT 均需渲染检查，确认图表可读、引用规范、旁白已嵌入且跨设备可播放。
- PPT 默认用系统已安装的 `Microsoft Huihui Desktop` 中文语音逐页生成 WAV 并嵌入；旁白稿同时提交，用户可在最终提交前用本人录音替换，但交付物本身必须已经可播放。

## 12. 交付结构

```text
README.md
pyproject.toml
configs/experiment.yaml
data/manifest.json
data/splits/*.json
docs/superpowers/specs/2026-07-11-metagate-hipporag-design.md
docs/superpowers/plans/2026-07-11-metagate-hipporag-implementation.md
patches/hipporag-openai-only.patch
scripts/bootstrap_upstream.py
src/metagate_hipporag/*.py
tests/*.py
results/tables/*.csv
results/figures/*.png
results/records/*.parquet
results/manifests/*.json
paper/*.docx
paper/references.bib
slides/*.pptx
slides/narration-script.md
```

`artifacts/`、完整数据和上游 checkout 默认不进 Git；其哈希和生成命令进入 `results/manifests/`，精简的逐题分析字段进入 `results/records/`，二者必须提交。

## 13. 时间安排

- **7 月 11–12 日：** 项目环境、固定上游、数据契约、离线测试和 3 题 smoke。
- **7 月 13–15 日：** OpenAI Batch OpenIE、三个图索引、Dense/HippoRAG 2 开发集复现。
- **7 月 16–18 日：** MetaGate、Always-Expand、阈值冻结和消融测试。
- **7 月 19–21 日：** 三数据集 300 条测试集正式运行与恢复检查。
- **7 月 22–24 日：** 统计检验、图表、误差案例和结果冻结。
- **7 月 25–28 日：** 论文初稿、引用核验、个人理解与讨论补写、格式检查。
- **7 月 29–30 日：** PPT、旁白、渲染检查和论文/PPT 数字一致性检查。
- **7 月 31 日：** 最终缓冲、上传并核验提交文件。

若任何完整图索引在 7 月 16 日仍未成功，只能在存在与本项目语料、模板和模型完全匹配且哈希可验证的官方预计算结果时替换；否则继续恢复或缩小为明确标注的探索性附录。任何缺少三数据集各 300 条测试结果的版本都不满足第 1 节主项目成功标准，不得通过改写措辞视作完成。

## 14. 主要风险与缓解

- **上游 alpha 代码变化：** 固定 SHA、补丁哈希与契约测试。
- **Windows 可选依赖失败：** 只启用 OpenAI 在线路径，重依赖惰性导入，不安装 vLLM。
- **Batch 两阶段依赖：** NER 完整校验后才创建 triple batch，状态机支持跨日恢复。
- **嵌入漂移：** 首次成功生成后缓存向量和 manifest；最终实验不重新嵌入。
- **API 非完全确定：** 固定快照、温度和 seed，保存原始响应；统计单位是题目而不是重复模型调用。
- **改进只因更多调用：** Always-Expand 使用相同 rewrite 与 RRF，隔离条件式控制的价值。
- **弃答抬高表面准确率：** 同时报告强制作答和选择性指标，绝不只报 answered-only 分数。
- **认知主张过度：** 全文使用“受启发”“功能类比”“计算近似”，并讨论差异与局限。

## 15. 关键来源

- HippoRAG 2：<https://proceedings.mlr.press/v267/gutierrez25a.html>
- 官方代码与数据说明：<https://github.com/OSU-NLP-Group/HippoRAG>
- CCF 人工智能目录：<https://www.ccf.org.cn/Academic_Evaluation/AI/>
- OpenAI GPT-4o mini 模型页：<https://developers.openai.com/api/docs/models/gpt-4o-mini>
- OpenAI text-embedding-3-large 模型页：<https://developers.openai.com/api/docs/models/text-embedding-3-large>
- Nelson–Narens 元认知框架：<https://doi.org/10.1016/S0079-7421(08)60053-5>
- Cognitive Offloading：<https://doi.org/10.1016/j.tics.2016.07.002>
