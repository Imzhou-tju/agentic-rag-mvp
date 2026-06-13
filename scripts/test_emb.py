import os
from langchain_openai import OpenAIEmbeddings

def test():
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-large-zh-v1.5",
        api_key="sk-sobefsxnyndmukpwysaywbvdamfyfhgubhdiqtiqlkwyclnt",
        base_url="https://api.siliconflow.cn/v1",
        check_embedding_ctx_length=False
    )
    res = embeddings.embed_documents(["测试内容"])
    print("Success, len:", len(res[0]))

if __name__ == "__main__":
    test()
