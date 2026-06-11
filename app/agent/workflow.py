from __future__ import annotations

from app.agent.llm import LLMService
from app.agent.router import QueryRouter
from app.models.schemas import AskRequest, AskResponse, SourceChunk
from app.rag.service import KnowledgeBaseService


SYSTEM_PROMPT = (
    '你是一个企业知识库智能助手。回答时优先基于检索到的文档内容，'
    '保持准确、简洁，并尽量给出结构化结论。'
)


class AgenticRAGWorkflow:
    def __init__(self) -> None:
        self.router = QueryRouter()
        self.kb = KnowledgeBaseService()
        self.llm = LLMService()

    def run(self, req: AskRequest) -> AskResponse:
        route, rewritten = self.router.route(req.question, req.task) if req.use_agent else ('retrieve_then_generate', req.question)
        if route == 'direct_answer':
            answer = self.llm.generate(SYSTEM_PROMPT, req.question, context_chunks=[])
            return AskResponse(
                question=req.question,
                task=req.task,
                route=route,
                answer=answer,
                rewritten_question=rewritten,
                sources=[],
                debug={'reason': 'agent judged that retrieval was unnecessary'},
            )

        retrieval_query = rewritten or req.question
        results = self.kb.search(retrieval_query)
        user_prompt = self._build_user_prompt(req.question, req.task, results)
        answer = self.llm.generate(SYSTEM_PROMPT, user_prompt, context_chunks=results)
        sources = [SourceChunk(**item) for item in results]
        return AskResponse(
            question=req.question,
            task=req.task,
            route=route,
            answer=answer,
            rewritten_question=rewritten,
            sources=sources,
            debug={
                'retrieval_query': retrieval_query,
                'retrieved_count': len(results),
            },
        )

    def _build_user_prompt(self, question: str, task: str, results: list[dict]) -> str:
        context = '\n\n'.join(
            [f"[来源: {r['document_name']} | 分数: {r['score']:.4f}]\n{r['text']}" for r in results]
        )
        if task == 'summary':
            instruction = '请基于以下文档片段生成摘要，并列出关键结论。'
        elif task == 'key_points':
            instruction = '请基于以下文档片段提取关键要点，使用项目符号输出。'
        elif task == 'compare':
            instruction = '请基于以下文档片段完成比较分析，输出相同点、不同点和建议。'
        else:
            instruction = '请基于以下文档片段回答问题，并在回答中体现依据。'

        return f"问题：{question}\n任务类型：{task}\n\n{instruction}\n\n文档片段：\n{context}"
