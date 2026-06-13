from __future__ import annotations

from typing import List

from app.core.config import get_settings


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.mode = self.settings.llm_mode.lower()

    def generate(self, system_prompt: str, user_prompt: str, context_chunks: List[dict] | None = None) -> str:
        if self.mode != 'openai' or not self.settings.openai_api_key:
            raise ValueError("系统严禁使用假数据和 Mock 模式。请在 .env 中配置有效的 OPENAI_API_KEY 以调用真实大模型。")
        return self._generate_openai(system_prompt, user_prompt)

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
