from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings
from app.rag.vector_store import SimpleVectorStore

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class KnowledgeBaseService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.upload_dir = Path(self.settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.store = SimpleVectorStore()
        self.llm = ChatOpenAI(
            model=self.settings.llm_model,
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            temperature=0.7,
        )
        self.synonyms_path = Path("app/data/synonyms.json")
        self.synonyms = {}
        if self.synonyms_path.exists():
            import json
            try:
                with open(self.synonyms_path, "r", encoding="utf-8") as f:
                    self.synonyms = json.load(f)
            except Exception as e:
                print(f"加载同义词词典失败: {e}")

    def _apply_synonym_expansion(self, query: str) -> list[str]:
        """基于本地硬编码规则进行同义词替换扩写"""
        expanded_queries = []
        for key, values in self.synonyms.items():
            if key in query:
                for val in values:
                    expanded_queries.append(query.replace(key, val))
        return expanded_queries

    def generate_multi_queries(self, query: str, n: int = 3) -> list[str]:
        prompt = ChatPromptTemplate.from_template(
            "作为一位资深的知识管理专家，你的任务是基于用户的原始提问，生成 {n} 个意思相近但表述不同、侧重点不同的搜索查询语句。\n"
            "这有助于我们在知识库中进行多角度的检索，克服单次检索的局限性。\n\n"
            "原始问题: {query}\n\n"
            "请直接输出 {n} 个变体查询，每行一个，不要包含任何序号、前缀或额外的解释说明。"
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            response = chain.invoke({"query": query, "n": n})
            variations = [line.strip() for line in response.strip().split('\n') if line.strip()]
            
            clean_variations = []
            import re
            for v in variations:
                v = re.sub(r'^(\d+\.|-|\*)\s*', '', v).strip()
                if v:
                    clean_variations.append(v)
            return clean_variations[:n]
        except Exception as e:
            print(f"多查询生成失败: {e}")
            return []

    def rebuild_index(self) -> int:
        return self.store.add_documents_from_folder(str(self.upload_dir))

    def search(self, query: str, top_k: int | None = None, filter_dict: dict | None = None) -> list[dict]:
        variations = self.generate_multi_queries(query, n=3)
        rule_variations = self._apply_synonym_expansion(query)
        
        all_queries = [query] + variations + rule_variations
        # 去重
        all_queries = list(dict.fromkeys(all_queries))
        print(f"[Multi-Query] 并行检索查询列表: {all_queries}")
        
        merged_results = {}
        search_top_k = top_k if top_k else self.settings.top_k
        
        for q in all_queries:
            results = self.store.search(q, top_k=search_top_k, filter_dict=filter_dict)
            for res in results:
                doc_id = res.get('id')
                if doc_id not in merged_results:
                    merged_results[doc_id] = res
                else:
                    if res.get('score', 0) > merged_results[doc_id].get('score', 0):
                        merged_results[doc_id]['score'] = res['score']
                        
        final_list = list(merged_results.values())
        final_list = sorted(final_list, key=lambda x: x.get('score', 0), reverse=True)
        return final_list[:search_top_k * 2]

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
