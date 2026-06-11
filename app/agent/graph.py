from __future__ import annotations

import json
from typing import List, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.agent.llm import SYSTEM_PROMPT as BASE_SYSTEM_PROMPT
from app.agent.router import QueryRouter
from app.core.config import get_settings
from app.rag.service import KnowledgeBaseService

# Initialize global services
settings = get_settings()
kb_service = KnowledgeBaseService()
router = QueryRouter()

# Initialize LangChain ChatOpenAI for the LLM nodes
llm = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
    temperature=0.2,
)

class GraphState(TypedDict):
    question: str
    task: str
    rewritten_question: str
    documents: List[dict]
    generation: str
    route: str

# ---------------------------------------------------------
# Pydantic Output Parsers for LLM calls
# ---------------------------------------------------------
class GradeResult(BaseModel):
    """Boolean score for relevance check."""
    score: str = Field(description="相关性得分，'yes' 表示相关，'no' 表示不相关")

# ---------------------------------------------------------
# Nodes
# ---------------------------------------------------------
def retrieve_node(state: GraphState) -> dict:
    """Retrieve documents from vector store."""
    query = state.get("rewritten_question") or state["question"]
    docs = kb_service.search(query)
    return {"documents": docs}

def grade_documents_node(state: GraphState) -> dict:
    """Evaluate if retrieved documents are relevant to the question."""
    question = state["question"]
    documents = state["documents"]
    
    # We use a structured prompt to ask LLM if the document is relevant
    system = "你是一个文档评分员，评估给定的文档片段是否与用户的问题相关。如果你认为它包含了回答问题所需的关键词或语义，请回复 'yes'，否则回复 'no'。只需输出 yes 或 no。"
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "检索到的文档片段: \n\n {context} \n\n 用户问题: {question}")
    ])
    
    # Enable structured output if supported, or just use regular generation
    # Some models like DeepSeek support structured output well, but let's use standard generation and parse manually for maximum compatibility
    eval_chain = grade_prompt | llm
    
    filtered_docs = []
    for doc in documents:
        res = eval_chain.invoke({"context": doc["text"], "question": question})
        content = res.content.strip().lower()
        if "yes" in content:
            filtered_docs.append(doc)
            
    return {"documents": filtered_docs}

def generate_node(state: GraphState) -> dict:
    """Generate answer using relevant documents."""
    question = state["question"]
    task = state["task"]
    docs = state["documents"]
    
    # Build context similar to old workflow
    context = '\n\n'.join(
        [f"[来源: {r['document_name']} | 分数: {r.get('score', 0):.4f}]\n{r['text']}" for r in docs]
    )
    
    if task == 'summary':
        instruction = '请基于以下文档片段生成摘要，并列出关键结论。'
    elif task == 'key_points':
        instruction = '请基于以下文档片段提取关键要点，使用项目符号输出。'
    elif task == 'compare':
        instruction = '请基于以下文档片段完成比较分析，输出相同点、不同点和建议。'
    else:
        instruction = '请基于以下文档片段回答问题，并在回答中体现依据。'

    user_prompt = f"问题：{question}\n任务类型：{task}\n\n{instruction}\n\n文档片段：\n{context}"
    
    sys_prompt = (
        '你是一个企业知识库智能助手。回答时优先基于检索到的文档内容，'
        '保持准确、简洁，并尽量给出结构化结论。'
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "{user_prompt}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"user_prompt": user_prompt})
    
    return {"generation": response.content}

def direct_generate_node(state: GraphState) -> dict:
    """Generate answer without vector store context."""
    question = state["question"]
    sys_prompt = "你是一个智能助手，请直接回答用户的问题。"
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "{question}")
    ])
    chain = prompt | llm
    response = chain.invoke({"question": question})
    return {"generation": response.content}

def rewrite_question_node(state: GraphState) -> dict:
    """Rewrite the question to get better retrieval results."""
    question = state["question"]
    sys_prompt = (
        "你是一个查询重写专家。用户的问题可能表述不清或难以在普通知识库中检索到相关结果。"
        "请对用户的问题进行同义词扩展和重组，输出一个更适合向量检索的新问题。"
        "只输出重写后的问题，不要包含任何其他说明文字。"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "原始问题: {question}")
    ])
    chain = prompt | llm
    response = chain.invoke({"question": question})
    
    return {"rewritten_question": response.content.strip()}

# ---------------------------------------------------------
# Conditional Edges
# ---------------------------------------------------------
def route_question(state: GraphState) -> Literal["direct_generate_node", "retrieve_node"]:
    """Route question to direct generation or retrieval."""
    route, _ = router.route(state["question"], state["task"])
    if route == "direct_answer":
        return "direct_generate_node"
    return "retrieve_node"

def check_relevance(state: GraphState) -> Literal["generate_node", "rewrite_question_node"]:
    """Determine whether to generate answer or rewrite question based on filtered docs."""
    filtered_docs = state["documents"]
    # If no relevant documents found, rewrite and search again
    if not filtered_docs:
        # Prevent infinite loops: if we already rewrote once, just generate with empty context to admit failure
        if state.get("rewritten_question"):
            return "generate_node"
        return "rewrite_question_node"
    return "generate_node"

# ---------------------------------------------------------
# Compile Graph
# ---------------------------------------------------------
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("grade_documents_node", grade_documents_node)
workflow.add_node("generate_node", generate_node)
workflow.add_node("direct_generate_node", direct_generate_node)
workflow.add_node("rewrite_question_node", rewrite_question_node)

# Add edges
workflow.add_conditional_edges(START, route_question)
workflow.add_edge("direct_generate_node", END)
workflow.add_edge("retrieve_node", "grade_documents_node")
workflow.add_conditional_edges("grade_documents_node", check_relevance)
workflow.add_edge("rewrite_question_node", "retrieve_node")
workflow.add_edge("generate_node", END)

# Compile
app_graph = workflow.compile()
