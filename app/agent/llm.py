from __future__ import annotations

from typing import List

from app.core.config import get_settings


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.mode = self.settings.llm_mode.lower()

    def generate(self, system_prompt: str, user_prompt: str, context_chunks: List[dict] | None = None) -> str:
        if self.mode == 'openai' and self.settings.openai_api_key:
            return self._generate_openai(system_prompt, user_prompt)
        return self._generate_mock(user_prompt, context_chunks or [])

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        kwargs = {'api_key': self.settings.openai_api_key}
        if self.settings.openai_base_url:
            kwargs['base_url'] = self.settings.openai_base_url
        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ''

    def _generate_mock(self, user_prompt: str, context_chunks: List[dict]) -> str:
        if not context_chunks:
            return (
                '当前处于 mock 模式，且未提供足够的知识库上下文。\n\n'
                '建议：上传文档后再提问，或者配置 OPENAI_API_KEY 使用真实大模型回答。'
            )
        snippets = []
        for item in context_chunks[:3]:
            snippets.append(f"- 来源：{item['document_name']}\n  内容摘要：{item['text'][:180].replace(chr(10), ' ')}")
        return (
            '以下回答基于检索到的文档片段生成（mock 模式为抽取式总结）：\n\n'
            + '\n'.join(snippets)
            + '\n\n综合来看，这些片段与问题高度相关。你可以切换到 openai 模式获得更自然、更完整的生成式回答。'
        )
