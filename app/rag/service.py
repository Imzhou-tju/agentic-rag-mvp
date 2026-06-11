from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings
from app.rag.vector_store import SimpleVectorStore


class KnowledgeBaseService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.upload_dir = Path(self.settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.store = SimpleVectorStore()

    def rebuild_index(self) -> int:
        return self.store.add_documents_from_folder(str(self.upload_dir))

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        return self.store.search(query, top_k=top_k)

    def stats(self) -> dict:
        return self.store.stats()
