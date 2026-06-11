# Enterprise Agentic RAG MVP

一个适合简历展示的最小可运行项目：支持企业文档上传、知识库索引、Agent 路由问答、摘要/要点提取、引用片段返回。

## 1. 项目亮点

- **企业知识库场景**：上传 PDF / TXT / Markdown 文档，自动建立索引。
- **Agentic RAG**：根据问题类型判断“直接回答”还是“检索后回答”。
- **多任务支持**：问答、摘要、关键要点提取、比较分析。
- **引用可追溯**：返回命中的文档片段和相关性分数。
- **可扩展架构**：可继续替换向量库、接入 reranker、增加权限控制与评测模块。

## 2. 技术栈

- Backend: FastAPI
- Frontend: Streamlit
- Retrieval: TF-IDF + cosine similarity（MVP 版本，便于本地快速运行）
- LLM: OpenAI 兼容接口 / Mock 抽取式模式
- File Parsing: PyPDF2 / Markdown / TXT

> 说明：为了让项目开箱即用、依赖更轻，当前版本使用 `scikit-learn` 检索实现。后续可替换为 `FAISS + embedding model`，作为项目优化点。

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

## 5. 配置真实大模型

默认 `LLM_MODE=mock`，无需 API 也能演示完整流程。

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

1. 将 TF-IDF 检索升级为 `embedding + FAISS`。
2. 增加 hybrid retrieval（BM25 + dense retrieval）。
3. 接入 reranker。
4. 增加评测集与 hit@k 指标。
5. 增加对话历史与 memory。
6. 增加用户权限与多知识库管理。

## 8. 简历描述示例

**面向企业文档问答与任务执行的 Agentic RAG 系统设计与实现**  
独立完成基于 FastAPI 的 Agentic RAG 系统开发，支持企业文档解析、知识库索引、问答与结构化任务生成；设计问题路由机制，根据问题类型动态选择直接回答或检索增强回答；实现引用片段返回与调试信息展示，提升系统可解释性与可调试性。

## 9. 注意事项

- PDF 解析依赖 `PyPDF2`，扫描版 PDF 不保证效果。
- 当前 mock 模式仅用于本地演示，真实回答建议配置 LLM API。
- 如修改上传目录内文件，建议调用 `/api/index/rebuild` 重新构建索引。
