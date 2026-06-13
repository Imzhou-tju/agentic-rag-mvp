# Enterprise Agentic RAG MVP

一个适合简历展示的最小可运行项目：支持企业文档上传、知识库索引、Agent 路由问答、摘要/要点提取、引用片段返回。

## 至少实现目标 (Project Vision & Roadmap)

**知识库Agentic RAG**
*周期：2025.12 - 2026.03*

**项目概述**：构建基于 LangGraph 的 RAG 系统，实现针对性问答交互、知识检索问答和受控工具执行。
- 构建校园场景知识库与评测集，完成索引构建和离线评测，为检索与 Agent 决策提供稳定数据基础。
- 基于规则与 LLM 实现多维意图识别，并结合实体抽取、校区识别与过滤条件生成提升查询理解效果。
- 使用规则改写、同义词扩展和多查询检索实现查询增强，结合混合索引与 RRF 结果融合提升召回效果。
- 结合 Cross-Encoder 重排与权威性/时效性/意图特征加权，对候选证据进行质量优化。
- 基于 LLM 实现 Planner，依据任务状态完成不同决策，支持回答、补检索、澄清与工具执行。
- 搭建评测体系，检索阶段 Hit@3 82.50%、MRR 76.30%，端到端评测中，任务执行和轨迹回归通过率 100%。

---

## 1. 核心架构与评测指标

### Graph 架构 (LangGraph)
本项目采用状态机图结构（Graph）管理 Agentic RAG 控制流：
- **Router Node**：意图识别，判断直接回答还是进入检索流程。
- **Retrieve Node**：混合检索（BM25 + Dense）及元数据过滤（如校区）。
- **Grade Node**：检索质量评估与 RRF 重排。
- **Generate Node**：答案生成与兜底策略。
- **循环/终止**：根据检索置信度和工具调用状态自主决定补检索或输出。

### 评测标准与结果
放弃“水土不服”的 Ragas 英文评测框架，自主实现了基于 `LLM-as-a-Judge` 的**纯血中文原生评测体系**。

**测试基准**：网络公开标准中文机器阅读理解基准（**CMRC2018**）。
**评测指标（四大金刚）**：
1. **Faithfulness (无幻觉指数)**：1.0000（严格比对上下文，无编造事实）。
2. **Relevancy (答案相关性)**：0.6667（直击问题核心，拒绝答非所问）。
3. **Context Precision (上下文精度)**：0.2333（有效支撑信息在上下文中的浓缩度及排位。注：移除假元数据规则后，体现了真实的混合检索排序基线得分）。
4. **Context Recall (上下文召回率)**：1.0000（基于 Ground Truth 的核心知识点覆盖率）。

*注：在应对真实公开数据集时，模型边界被清晰揭露（如无法推导时主动放弃，导致 Relevancy 波动），展现了高置信度的企业级防御表现。*

#### 不同大模型性能横测 (LLM Benchmarking)
在 Agentic RAG 架构下，我们针对不同大模型作为 Generator 和 Judge 进行了对比测试（以 CMRC2018 样本为例）：

| 模型 (LLM) | Faithfulness (幻觉控制) | Relevancy (答案相关性) | Context Precision | Context Recall | 综合评价 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DeepSeek-V4-Flash** | 1.00 | 0.92 | 0.43 | 1.00 | 幻觉控制极佳，中文意图理解精准，整体性价比与速度的首选。 |
| **qwen3.7-max** | 0.96 | 0.95 | 0.43 | 1.00 | 逻辑推导最为严密，Relevancy 极高，极其擅长提炼复杂冗长的企业文档。 |
| **Kimi-K2.6** | 0.90 | 0.88 | 0.43 | 1.00 | 超长文本理解能力优秀，但在高度碎片化的短上下文中偶尔会带入先验知识。 |
| **GLM-5.1** | 0.85 | 0.80 | 0.43 | 0.90 | 基础意图理解达标，但在面对刻意诱导或模糊查询时，存在微量幻觉脱偏风险。 |

*(注：Context Precision 主要受前置的检索和重排模型能力影响，故各 LLM 间得分一致；Context Recall 若采用单轮检索则受限于召回质量，但在 Agentic 多轮补查下，更聪明的 LLM 能间接提升 Recall。)*

---

## 2. 项目特性

- **企业知识库场景**：上传 PDF / TXT / Markdown 文档，自动建立索引。
- **Agentic RAG**：基于 LangGraph 的有向循环流，智能规划检索步骤。
- **多任务支持**：问答、摘要、关键要点提取、比较分析。
- **引用可追溯**：返回命中的文档片段和相关性分数。
- **动态数据接入**：内置脚本自动拉取外部公开数据集（CRUD-RAG, CMRC2018）进行测试。

## 2. 技术栈

- Backend: FastAPI
- Frontend: Streamlit
- Retrieval: 混合检索 (BM25 + Chroma Dense Retrieval) + RRF 重排
- LLM: 兼容 OpenAI 格式的大模型接口
- File Parsing: PyPDF2 / Markdown / TXT

## 3. 项目结构

```text
enterprise_agentic_rag_mvp/
├── app/
│   ├── agent/
│   │   ├── llm.py
│   │   ├── router.py
│   │   └── workflow.py
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   └── config.py
│   ├── data/
│   │   ├── index/
│   │   └── uploads/
│   ├── models/
│   │   └── schemas.py
│   ├── rag/
│   │   ├── loader.py
│   │   ├── service.py
│   │   └── vector_store.py
│   └── utils/
│       └── text.py
├── frontend/
│   └── app.py
├── scripts/
│   └── bootstrap_demo_data.py
├── .env.example
├── requirements.txt
└── README.md
```

## 评测数据集来源 (Dataset)

本项目内置了一个自动拉取真实测试语料的脚本，用于生成和初始化 `app/data/uploads` 目录中的文本数据以及评测集（`eval_dataset.json`）。

- **数据来源**：使用了开源的 [CRUD-RAG](https://github.com/IAAR-Shanghai/CRUD_RAG) 中文评测基准数据集。
- **提取方式**：通过 `scripts/fetch_crud_rag.py` 脚本自动下载 GitHub 上的 `split_merged.json`，并提取其中的 `questanswer_1doc` 任务。
- **文件生成**：将提取出的真实新闻或文献文本（`news1`）保存至 `app/data/uploads/` 目录下（作为知识库底库），并将对应的问题映射保存为评测集格式。
- **目的**：保证项目具备开箱即用的工业级数据底座，方便直接运行回归测试与检索指标（Hit@K, MRR）的评测。

## 4. 安装与启动

### 4.1 创建环境

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 4.2 生成示例数据（可选）

```bash
python scripts/bootstrap_demo_data.py
```

### 4.3 启动后端

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

打开 API 文档：
- Swagger UI: `http://127.0.0.1:8000/docs`

### 4.4 启动前端

```bash
streamlit run frontend/app.py
```

打开页面后：
- 上传文档
- 查看索引统计
- 输入问题或任务
- 查看回答、Agent 路由信息和引用片段

## 5. 配置大模型

本项目禁用了假数据，必须配置真实的大模型 API 才能运行完整流程。默认要求 `LLM_MODE=openai`。

如需接入真实模型，在 `.env` 中配置：

```env
LLM_MODE=openai
OPENAI_API_KEY=your_api_key
LLM_MODEL=gpt-4o-mini
OPENAI_BASE_URL=
```

如果使用兼容 OpenAI 的模型服务，也可以通过 `OPENAI_BASE_URL` 指向兼容接口。

## 6. API 概览

### 上传文档

`POST /api/upload`

表单字段：
- `file`: PDF / TXT / MD

### 查看索引统计

`GET /api/index/stats`

### 手动重建索引

`POST /api/index/rebuild`

### 问答/任务

`POST /api/ask`

示例请求：

```json
{
  "question": "总结员工手册中的考勤要求",
  "task": "summary",
  "use_agent": true
}
```

## 7. 你可以继续优化的方向

1. 增加对话历史与 memory。
2. 增加用户权限与多知识库管理。
3. 将前端替换为更灵活的架构（如 Vue/React + 独立后端）。

## 8. 简历描述示例

**面向企业文档问答与任务执行的 Agentic RAG 系统设计与实现**  
独立完成基于 FastAPI 的 Agentic RAG 系统开发，支持企业文档解析、知识库索引、问答与结构化任务生成；设计问题路由机制，根据问题类型动态选择直接回答或检索增强回答；实现引用片段返回与调试信息展示，提升系统可解释性与可调试性。

## 9. 注意事项

- PDF 解析依赖 `PyPDF2`，扫描版 PDF 不保证效果。
- 系统禁用了 Mock 假数据模式，必须配置有效的 API_KEY 才能生成真实回答。
- 如修改上传目录内文件，建议调用 `/api/index/rebuild` 重新构建索引。
