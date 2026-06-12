import argparse
import json
from pathlib import Path
from typing import List

# To run this script correctly, it should be able to import app modules
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.rag.service import KnowledgeBaseService
from app.agent.router import QueryAnalyzer

def evaluate(disable_rerank: bool):
    print(f"Starting Evaluation... (Rerank Enabled: {not disable_rerank})")
    
    kb_service = KnowledgeBaseService()
    analyzer = QueryAnalyzer()
    
    # Rebuild index first to ensure we have the mock documents
    print("Rebuilding index with current documents in app/data/uploads/...")
    indexed_count = kb_service.rebuild_index()
    print(f"Indexed {indexed_count} chunks.")
    
    dataset_path = Path("app/data/eval_dataset.json")
    if not dataset_path.exists():
        print("Evaluation dataset not found. Please run generate_mock_data.py first.")
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    total_queries = len(eval_data)
    hits_at_1 = 0
    hits_at_3 = 0
    mrr_sum = 0.0
    
    print("-" * 60)
    print(f"{'Question':<30} | {'Rank':<5} | {'Expected Document':<20}")
    print("-" * 60)
    
    for item in eval_data:
        question = item["question"]
        expected_doc = item["expected_document_name"]
        
        # 1. Analyze query to get campus filter
        analysis = analyzer.analyze(question)
        filter_dict = None
        if analysis and analysis.campus:
            filter_dict = {"campus": analysis.campus}
            
        # 2. Retrieve initial top_k (e.g., 15)
        # If no_rerank is true, we just retrieve the final top_k (4) directly using Embedding
        from app.core.config import get_settings
        settings = get_settings()
        
        if disable_rerank:
            docs = kb_service.search(question, top_k=settings.top_k, filter_dict=filter_dict)
            final_docs = sorted(docs, key=lambda x: x.get("score", 0.0), reverse=True)
        else:
            docs = kb_service.search(question, top_k=settings.initial_top_k, filter_dict=filter_dict)
            reranked_docs = kb_service.rerank(question, docs)
            
            for doc in reranked_docs:
                sem_score = doc.get("rerank_score", 0.0)
                time_score = doc.get("timeliness_score", 0.5)
                auth_score = doc.get("authoritative_score", 0.5)
                doc["final_score"] = 0.6 * sem_score + 0.2 * time_score + 0.2 * auth_score
                
            final_docs = sorted(reranked_docs, key=lambda x: x.get("final_score", 0.0), reverse=True)[:settings.top_k]
            
        # 3. Calculate rank
        # Find the rank of the first chunk that comes from the expected_document
        rank = 0
        for idx, doc in enumerate(final_docs):
            if doc["document_name"] == expected_doc:
                rank = idx + 1
                break
                
        if rank > 0:
            if rank == 1:
                hits_at_1 += 1
            if rank <= 3:
                hits_at_3 += 1
            mrr_sum += 1.0 / rank
            rank_str = str(rank)
        else:
            rank_str = "Miss"
            
        q_disp = question[:27] + "..." if len(question) > 30 else question
        doc_disp = expected_doc[:17] + "..." if len(expected_doc) > 20 else expected_doc
        print(f"{q_disp:<30} | {rank_str:<5} | {doc_disp:<20}")

    print("-" * 60)
    hit_at_3_rate = (hits_at_3 / total_queries) * 100
    mrr = (mrr_sum / total_queries) * 100
    
    print(f"Total Queries: {total_queries}")
    print(f"Hit@1: {(hits_at_1 / total_queries) * 100:.2f}%")
    print(f"Hit@3: {hit_at_3_rate:.2f}%")
    print(f"MRR:   {mrr:.2f}%")
    print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Retrieval Pipeline")
    parser.add_argument("--no-rerank", action="store_true", help="Disable Cross-Encoder reranking")
    args = parser.parse_args()
    
    evaluate(args.no_rerank)
