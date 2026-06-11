from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import get_settings
from app.utils.text import chunk_text


@dataclass
class ChunkRecord:
    chunk_id: str
    document_name: str
    text: str


class SimpleVectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.index_dir = Path(self.settings.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer_path = self.index_dir / 'vectorizer.pkl'
        self.records_path = self.index_dir / 'records.json'
        self.matrix_path = self.index_dir / 'matrix.npy'
        self.vectorizer: TfidfVectorizer | None = None
        self.records: List[ChunkRecord] = []
        self.matrix: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if self.records_path.exists():
            import pickle
            self.records = [ChunkRecord(**item) for item in json.loads(self.records_path.read_text(encoding='utf-8'))]
            self.matrix = np.load(self.matrix_path)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)

    def _save(self) -> None:
        import pickle
            
        self.records_path.write_text(
            json.dumps([asdict(r) for r in self.records], ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        if self.matrix is not None:
            np.save(self.matrix_path, self.matrix)
        if self.vectorizer is not None:
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)

    def rebuild_from_documents(self, documents: list[tuple[str, str]]) -> int:
        records: List[ChunkRecord] = []
        for doc_name, content in documents:
            chunks = chunk_text(
                content,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            for idx, chunk in enumerate(chunks):
                records.append(
                    ChunkRecord(
                        chunk_id=f'{doc_name}::chunk_{idx}',
                        document_name=doc_name,
                        text=chunk,
                    )
                )
        self.records = records
        if not self.records:
            self.vectorizer = TfidfVectorizer()
            self.matrix = np.empty((0, 0))
            self._save()
            return 0

        self.vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 2))
        texts = [r.text for r in self.records]
        sparse_matrix = self.vectorizer.fit_transform(texts)
        self.matrix = sparse_matrix.toarray().astype(np.float32)
        self._save()
        return len(self.records)

    def add_documents_from_folder(self, folder: str) -> int:
        from app.rag.loader import DocumentLoader, SUPPORTED_EXTENSIONS

        loader = DocumentLoader()
        docs: list[tuple[str, str]] = []
        for path in sorted(Path(folder).glob('*')):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
                docs.append((path.name, loader.load(str(path))))
        return self.rebuild_from_documents(docs)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        if self.vectorizer is None or self.matrix is None or not self.records:
            return []
        top_k = top_k or self.settings.top_k
        q = self.vectorizer.transform([query]).toarray().astype(np.float32)
        sims = cosine_similarity(q, self.matrix)[0]
        idxs = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in idxs:
            record = self.records[int(idx)]
            results.append(
                {
                    'chunk_id': record.chunk_id,
                    'document_name': record.document_name,
                    'text': record.text,
                    'score': float(sims[int(idx)]),
                }
            )
        return results

    def stats(self) -> dict:
        docs = sorted({r.document_name for r in self.records})
        return {
            'total_documents': len(docs),
            'total_chunks': len(self.records),
            'documents': docs,
        }
