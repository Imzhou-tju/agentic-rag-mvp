import os
import sys
import json
import requests
from pathlib import Path

DATA_DIR = Path("app/data")
UPLOAD_DIR = DATA_DIR / "uploads"

# 使用代理拉取 Github 上的 JSON (由于国内网络原因)
CRUD_URL = "https://raw.gitmirror.com/IAAR-Shanghai/CRUD_RAG/main/data/crud_split/split_merged.json"
FALLBACK_URL = "https://raw.githubusercontent.com/IAAR-Shanghai/CRUD_RAG/main/data/crud_split/split_merged.json"

def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    print("⏳ 开始从 Github 镜像 (jsDelivr CDN) 拉取 CRUD-RAG 数据集...")
    
    try:
        try:
            response = requests.get(CRUD_URL, timeout=30)
            response.raise_for_status()
        except Exception:
            print("代理源拉取失败，尝试直连 Github Raw...")
            response = requests.get(FALLBACK_URL, timeout=30)
            response.raise_for_status()
            
        dataset = response.json()
        
        qa_data = []
        doc_registry = set()
        
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # 清空原有测试数据
        for f in UPLOAD_DIR.glob("*.txt"):
            f.unlink()
            
        print("✅ 数据集下载完毕，开始清洗并生成本地评测集...")
        
        count = 0
        read_tasks = dataset.get("questanswer_1doc", [])
        for item in read_tasks:
            # CRUD-RAG Read tasks use 'questions' (list) and 'news1' (string)
            questions = item.get("questions", [])
            doc_content = item.get("news1", "")
            
            if not questions or not doc_content:
                continue
                
            question = questions[0]
            if len(doc_content) < 50: # skip very short context
                continue
                
            doc_id = item.get("id", f"crud_doc_{count}")
            filename = f"crud_{doc_id}.txt"
            
            # Save the document to uploads
            if filename not in doc_registry:
                filepath = UPLOAD_DIR / filename
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(doc_content)
                doc_registry.add(filename)
            
            # Add to eval dataset
            qa_data.append({
                "question": question,
                "expected_document_name": filename
            })
            
            count += 1
            if count >= 30: # Limit to 30 valid QAs for quick local evaluation
                break
                
        dataset_path = DATA_DIR / "eval_dataset.json"
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, ensure_ascii=False, indent=2)
            
        print(f"🎉 成功生成 {len(qa_data)} 条评测数据！")
        print(f"📄 提取了 {len(doc_registry)} 篇真实文献作为测试语料，保存在 {UPLOAD_DIR} 目录。")
        print("下一步请运行: python scripts/evaluate.py")

    except Exception as e:
        print(f"❌ 下载或处理失败: {e}")

if __name__ == "__main__":
    main()
