from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str = 'Enterprise Agentic RAG MVP'
    app_host: str = '0.0.0.0'
    app_port: int = 8000
    allowed_origins: str = '*'

    data_dir: str = 'app/data'
    upload_dir: str = 'app/data/uploads'
    index_dir: str = 'app/data/index'

    llm_mode: str = 'mock'  # mock or openai
    llm_model: str = 'gpt-4o-mini'
    test_llm_models: str = '' # comma separated list of models for multi-model testing
    openai_api_key: str = ''
    openai_base_url: str = ''
    
    embedding_model: str = 'BAAI/bge-large-zh-v1.5'
    embedding_api_key: str = ''
    embedding_base_url: str = 'https://api.siliconflow.cn/v1'
    
    reranker_model: str = 'BAAI/bge-reranker-v2-m3'
    reranker_api_key: str = ''
    reranker_base_url: str = 'https://api.siliconflow.cn/v1/rerank'

    initial_top_k: int = 15
    top_k: int = 4
    chunk_size: int = 250
    chunk_overlap: int = 50

    @property
    def cors_origins(self) -> List[str]:
        if self.allowed_origins.strip() == '*':
            return ['*']
        return [item.strip() for item in self.allowed_origins.split(',') if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
