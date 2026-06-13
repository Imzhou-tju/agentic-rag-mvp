from __future__ import annotations

import json
from typing import Annotated, Literal, Optional, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from app.core.config import get_settings
from app.rag.service import KnowledgeBaseService

# Initialize global services
settings = get_settings()
kb_service = KnowledgeBaseService()

# Initialize LangChain ChatOpenAI for the LLM nodes
llm = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
    temperature=0.2,
)

# ---------------------------------------------------------
# Tools
# ---------------------------------------------------------
@tool
def search_knowledge_base(query: str, campus: Optional[Literal["卫津路校区", "北洋园校区"]] = None) -> str:
    """在企业知识库中检索与用户问题相关的官方文档。
    如果用户问及任何校园、公司规定、指南、报销、选课、假期等业务问题，必须调用此工具。
    
    参数:
    query (str): 用于检索的关键词或搜索语句。
    campus (str, 可选): 如果用户问题中明确提到了特定校区（包括老校区、新校区等别称），必须提取并标准化为 "卫津路校区" 或 "北洋园校区" 传入此参数，以进行精准的元数据过滤。
    
    返回:
    包含多个相关文档片段的合并字符串，带有权重和出处。
    """
    filter_dict = {"campus": campus} if campus else None
    
    # 1. Retrieve initial documents
    docs = kb_service.search(query, top_k=settings.initial_top_k, filter_dict=filter_dict)
    
    if not docs:
        return "没有在知识库中找到相关文档。"
        
    # 2. Rerank
    reranked_docs = kb_service.rerank(query, docs)
    
    # 3. Apply Feature Weighting (Semantic + Timeliness + Authority)
    for doc in reranked_docs:
        sem_score = doc.get("rerank_score", 0.0)
        time_score = float(doc.get("timeliness_score", 0.5))
        auth_score = float(doc.get("authoritative_score", 0.5))
        
        final_score = 0.6 * sem_score + 0.2 * time_score + 0.2 * auth_score
        doc["final_score"] = final_score
        
    # 4. Sort and select top_k
    sorted_docs = sorted(reranked_docs, key=lambda x: x.get("final_score", 0.0), reverse=True)
    top_docs = sorted_docs[:settings.top_k]
    
    # 5. Format results
    if not top_docs:
        return "没有在知识库中找到相关文档。"
        
    context = '\n\n'.join(
        [f"[来源: {r.get('document_name', '未知')} | 相关性得分: {r.get('final_score', 0):.4f}]\n{r.get('text', '')}" for r in top_docs]
    )
    return context

@tool
def web_search_tool(query: str) -> str:
    """使用 DuckDuckGo 搜索引擎在互联网上查找外部信息。
    当用户询问近期新闻、通用常识、天气、或是超出企业内部规定范围的内容时，必须调用此工具。
    
    参数:
    query (str): 搜索关键词。
    
    返回:
    互联网搜索结果片段。
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        return search.invoke(query)
    except Exception as e:
        return f"网络搜索失败: {e}"

tools = [search_knowledge_base, web_search_tool]
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------
def agent_node(state: MessagesState):
    """The main LLM agent node."""
    messages = state["messages"]
    
    # Inject system prompt if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        sys_msg = SystemMessage(
            content=(
                "你是一个强大且专业的企业智能体助理。你的任务是回答用户问题。\n"
                "遇到需要专业知识、内部文档或政策规定的问题，必须优先使用 search_knowledge_base 工具。\n"
                "遇到询问时事新闻、通用百科或确认本地知识库中可能没有的外部信息时，必须优先使用 web_search_tool 工具。\n"
                "如果搜索到的结果不充分，你可以换个关键词多次搜索，甚至结合两个工具交叉验证。\n"
                "如果你发现用户的问题不明确，可以直接用文字回复向用户澄清。\n"
                "在最终回答中，请说明信息来源是内部知识库还是互联网。\n"
            )
        )
        messages = [sys_msg] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# We use prebuilt ToolNode
tools_node = ToolNode(tools)

# ---------------------------------------------------------
# Compile Graph
# ---------------------------------------------------------
workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)

# Entry point
workflow.add_edge(START, "agent")

# Agent makes decision: Tool or END
workflow.add_conditional_edges("agent", tools_condition)

# Tool always returns to Agent
workflow.add_edge("tools", "agent")

from langgraph.checkpoint.memory import MemorySaver

# Compile
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)
