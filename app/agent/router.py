from __future__ import annotations

import re


RETRIEVAL_HINTS = [
    '根据文档', '制度', '流程', '报销', '请假', '员工手册', '知识库', '上传', 'pdf',
    '文档', '手册', '规定', '要求', '总结', '比较', '对比', '哪些', '什么是', '如何',
]


class QueryRouter:
    def route(self, question: str, task: str) -> tuple[str, str | None]:
        q = question.strip()
        if task in {'summary', 'key_points', 'compare'}:
            return 'retrieve_then_generate', self.rewrite(q, task)
        if len(q) > 18:
            return 'retrieve_then_generate', self.rewrite(q, task)
        if any(hint in q.lower() for hint in [h.lower() for h in RETRIEVAL_HINTS]):
            return 'retrieve_then_generate', self.rewrite(q, task)
        if re.search(r'\d|第[一二三四五六七八九十]+', q):
            return 'retrieve_then_generate', self.rewrite(q, task)
        return 'direct_answer', None

    def rewrite(self, question: str, task: str) -> str:
        if task == 'summary':
            return f'请检索与以下主题最相关的文档并总结：{question}'
        if task == 'key_points':
            return f'请检索与以下问题相关的关键条款和要点：{question}'
        if task == 'compare':
            return f'请检索与以下比较任务相关的内容并进行对比：{question}'
        return question
