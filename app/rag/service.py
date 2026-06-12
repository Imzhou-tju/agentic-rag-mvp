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

    def search(self, query: str, top_k: int | None = None, filter_dict: dict | None = None) -> list[dict]:
        return self.store.search(query, top_k=top_k, filter_dict=filter_dict)

    def rerank(self, query: str, documents: list[dict]) -> list[dict]:
        if not documents:
            return []
            
        import requests
        
        url = self.settings.reranker_base_url
        headers = {
            "Authorization": f"Bearer {self.settings.reranker_api_key}",
            "Content-Type": "application/json"
        }
        
        texts = [doc["text"] for doc in documents]
        payload = {
            "model": self.settings.reranker_model,
            "query": query,
            "documents": texts,
            "return_documents": False
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Match scores back to documents
            # SiliconFlow API typically returns results in `results` array, with `index` and `relevance_score`
            results = data.get("results", [])
            for res in results:
                idx = res["index"]
                score = res["relevance_score"]
                if 0 <= idx < len(documents):
                    documents[idx]["rerank_score"] = float(score)
                    
            # Fallback for documents that didn't get a score (shouldn't happen)
            for doc in documents:
                if "rerank_score" not in doc:
                    doc["rerank_score"] = 0.0
                    
            return documents
        except Exception as e:
            print(f"Reranking failed: {e}")
            # If reranking fails, just give them a 0.0 score so the pipeline doesn't break
            for doc in documents:
                doc["rerank_score"] = doc.get("score", 0.0) # fallback to vector score
            return documents

    def stats(self) -> dict:
        return self.store.stats()
