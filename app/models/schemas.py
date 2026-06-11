from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    intent: Literal["knowledge_qa", "chitchat", "summary", "compare", "task_execution"] = Field(
        description="The intent of the user query."
    )
    entities: List[str] = Field(
        description="Core entities or keywords extracted from the query.", default_factory=list
    )
    campus: Optional[str] = Field(
        description="Specific campus mentioned (e.g. '卫津路校区', '北洋园校区'). If none, null.", default=None
    )
    need_retrieval: bool = Field(
        description="Whether this query needs to retrieve documents from the knowledge base. True for qa/summary, False for chitchat.", default=True
    )

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
