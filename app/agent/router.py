from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.models.schemas import QueryAnalysis


class QueryAnalyzer:
    def __init__(self) -> None:
        settings = get_settings()
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.1,
        )
        # Use LangChain's structured output capability
        self.analyzer = self.llm.with_structured_output(QueryAnalysis)
        
        self.system_prompt = (
            "你是一个专业的企业知识库查询意图分析器。你的任务是分析用户的输入，提取关键信息以供下游检索系统使用。\n"
            "1. intent: 判断用户意图。日常打招呼选 chitchat；询问规定、流程、事实选 knowledge_qa；总结任务选 summary；比较分析选 compare。\n"
            "2. entities: 提取查询中的核心名词或动作作为关键词列表，用于辅助检索。\n"
            "3. campus: 如果用户明确提到了某个校区（如'卫津路', '北洋园'），提取出来并规范化为'卫津路校区'或'北洋园校区'。如果没有，返回 null。\n"
            "4. need_retrieval: 判断该问题是否需要查阅企业内部文档。纯闲聊(chitchat)设为 False，业务问题设为 True。"
        )

    def analyze(self, question: str, task: str = "qa") -> QueryAnalysis:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "当前UI指定任务类型: {task}\n用户问题: {question}")
        ])
        chain = prompt | self.analyzer
        
        try:
            result = chain.invoke({"task": task, "question": question})
            return result
        except Exception as e:
            # Fallback if structured output fails (e.g. model incompatibility)
            print(f"Structured output failed: {e}. Falling back to default.")
            return QueryAnalysis(
                intent="knowledge_qa",
                entities=[question],
                campus=None,
                need_retrieval=True
            )
