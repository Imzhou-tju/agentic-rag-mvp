from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.core.config import get_settings
from app.utils.text import chunk_text

import jieba
from rank_bm25 import BM25Okapi

class SimpleVectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.index_dir = Path(self.settings.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SiliconFlow API based Embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model,
            api_key=self.settings.embedding_api_key,
            base_url=self.settings.embedding_base_url,
            chunk_size=1,
            check_embedding_ctx_length=False
        )
        
        # Initialize Chroma vector store
        self.vector_store = Chroma(
            collection_name="enterprise_knowledge",
            embedding_function=self.embeddings,
            persist_directory=str(self.index_dir)
        )
        
        self.bm25 = None
        self.bm25_docs = []

    def rebuild_from_documents(self, documents: list[tuple[str, str]]) -> int:
        """Clear existing index and rebuild from new documents."""
        # Clean existing collection
        self.vector_store.delete_collection()
        
        # Recreate an empty collection
        self.vector_store = Chroma(
            collection_name="enterprise_knowledge",
            embedding_function=self.embeddings,
            persist_directory=str(self.index_dir)
        )
        
        docs_to_add: List[Document] = []
        for doc_name, content in documents:
            chunks = chunk_text(
                content,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            for idx, chunk in enumerate(chunks):
                # Mock metadata injection for demonstration of filter and scoring
                campus = None
                if "卫津路" in doc_name or "卫津路" in chunk:
                    campus = "卫津路校区"
                elif "北洋园" in doc_name or "北洋园" in chunk:
                    campus = "北洋园校区"
                    
                # Simulated feature scores
                timeliness_score = 0.5
                if "2025" in doc_name or "2026" in doc_name or "最新" in doc_name:
                    timeliness_score = 1.0
                elif "2020" in doc_name or "2021" in doc_name or "旧版" in doc_name:
                    timeliness_score = 0.2
                    
                authoritative_score = 0.5
                if "规定" in doc_name or "教务处" in doc_name or "官方" in doc_name or "章程" in doc_name:
                    authoritative_score = 1.0
                elif "论坛" in doc_name or "讨论" in doc_name or "经验" in doc_name:
                    authoritative_score = 0.3

                metadata = {
                    "document_name": doc_name,
                    "chunk_index": idx,
                    "chunk_id": f"{doc_name}::chunk_{idx}",
                    "timeliness_score": float(timeliness_score),
                    "authoritative_score": float(authoritative_score)
                }
                if campus:
                    metadata["campus"] = campus

                doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                docs_to_add.append(doc)
                
        if not docs_to_add:
            return 0
            
        # Debug and filter
        valid_docs = []
        for doc in docs_to_add:
            if not doc.page_content.strip():
                print(f"Skipping empty chunk in {doc.metadata.get('document_name')}")
                continue
            if len(doc.page_content) > 512:
                print(f"Warning: chunk too long! Length: {len(doc.page_content)}")
            valid_docs.append(doc)
            
        if not valid_docs:
            return 0
            
        self.vector_store.add_documents(valid_docs)
        
        self.bm25_docs = valid_docs
        tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in valid_docs]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            
        return len(valid_docs)

    def add_documents_from_folder(self, folder: str) -> int:
        from app.rag.loader import DocumentLoader, SUPPORTED_EXTENSIONS

        loader = DocumentLoader()
        docs: list[tuple[str, str]] = []
        for path in sorted(Path(folder).glob('*')):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
                docs.append((path.name, loader.load(str(path))))
        return self.rebuild_from_documents(docs)

    def search(self, query: str, top_k: int | None = None, filter_dict: dict | None = None) -> list[dict]:
        top_k = top_k or self.settings.top_k
        
        try:
            # similarity_search_with_relevance_scores returns score between 0 and 1
            # where 1 is highly similar, 0 is dissimilar
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k, filter=filter_dict)
        except Exception:
            # Fallback to standard distance search if relevance score is not supported by the metric
            raw_results = self.vector_store.similarity_search_with_score(query, k=top_k, filter=filter_dict)
            # Chroma default L2 distance: lower is better. We invert it for a mock "score".
            results = [(doc, max(0.0, 1.0 - score)) for doc, score in raw_results]

        output = []
        for doc, score in results:
            output.append({
                'chunk_id': doc.metadata.get('chunk_id', 'unknown'),
                'document_name': doc.metadata.get('document_name', 'unknown'),
                'text': doc.page_content,
                'score': float(score),
                'timeliness_score': float(doc.metadata.get('timeliness_score', 0.5)),
                'authoritative_score': float(doc.metadata.get('authoritative_score', 0.5))
            })
        return output

    def bm25_search(self, query: str, top_k: int | None = None, filter_dict: dict | None = None) -> list[dict]:
        top_k = top_k or self.settings.top_k
        if not self.bm25 or not self.bm25_docs:
            return []
            
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        output = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                continue
            doc = self.bm25_docs[idx]
            if filter_dict and "campus" in filter_dict:
                if doc.metadata.get("campus") != filter_dict["campus"]:
                    continue
                    
            output.append({
                'chunk_id': doc.metadata.get('chunk_id', 'unknown'),
                'document_name': doc.metadata.get('document_name', 'unknown'),
                'text': doc.page_content,
                'score': float(score),
                'timeliness_score': float(doc.metadata.get('timeliness_score', 0.5)),
                'authoritative_score': float(doc.metadata.get('authoritative_score', 0.5))
            })
        return output

    def stats(self) -> dict:
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            # Extract unique documents if possible
            results = collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            docs = sorted({m.get("document_name") for m in metadatas if m and "document_name" in m})
            
            return {
                'total_documents': len(docs),
                'total_chunks': count,
                'documents': docs,
            }
        except Exception:
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'documents': [],
            }
