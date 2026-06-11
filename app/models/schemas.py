from typing import List, Optional

from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    document_name: str
    chunk_id: str
    score: float
    text: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    task: str = Field(default='qa', description='qa | summary | key_points | compare')
    use_agent: bool = True


class AskResponse(BaseModel):
    question: str
    task: str
    route: str
    answer: str
    sources: List[SourceChunk]
    rewritten_question: Optional[str] = None
    debug: dict = Field(default_factory=dict)


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


class IndexStatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: List[str]
