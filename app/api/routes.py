from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.models.schemas import AskRequest, AskResponse, IndexStatsResponse, UploadResponse
from app.rag.loader import SUPPORTED_EXTENSIONS
from app.rag.service import KnowledgeBaseService
from app.agent.graph import app_graph
from app.models.schemas import SourceChunk

router = APIRouter()
settings = get_settings()
kb_service = KnowledgeBaseService()

@router.get('/health')
def health() -> dict:
    return {'status': 'ok', 'app': settings.app_name}


@router.get('/index/stats', response_model=IndexStatsResponse)
def index_stats() -> dict:
    return kb_service.stats()


@router.post('/index/rebuild')
def rebuild_index() -> dict:
    count = kb_service.rebuild_index()
    return {'message': 'Index rebuilt successfully', 'chunks_indexed': count}


@router.post('/upload', response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f'Unsupported file type: {suffix}')

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / file.filename
    with target.open('wb') as f:
        shutil.copyfileobj(file.file, f)

    count = kb_service.rebuild_index()
    return UploadResponse(
        filename=file.filename,
        chunks_indexed=count,
        message='File uploaded and knowledge index rebuilt.',
    )


@router.post('/ask', response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    stats = kb_service.stats()
    if stats['total_chunks'] == 0 and req.task != 'qa':
        raise HTTPException(status_code=400, detail='No documents indexed yet. Please upload files first.')
    initial_state = {
        "question": req.question,
        "task": req.task,
        "rewritten_question": "",
        "documents": [],
        "generation": "",
        "route": ""
    }
    
    final_state = app_graph.invoke(initial_state)
    
    sources = [
        SourceChunk(
            document_name=doc["document_name"],
            chunk_id=doc["chunk_id"],
            score=doc.get("score", 0.0),
            text=doc["text"]
        ) for doc in final_state.get("documents", [])
    ]
    
    return AskResponse(
        question=final_state["question"],
        task=final_state["task"],
        route=final_state.get("route", "langgraph"),
        answer=final_state.get("generation", ""),
        sources=sources,
        rewritten_question=final_state.get("rewritten_question"),
        debug={"final_docs_count": len(sources)}
    )
