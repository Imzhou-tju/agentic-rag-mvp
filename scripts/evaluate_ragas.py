import sys
import os

# --- Monkey Patch for Ragas compatibility with latest langchain ---
class DummyVertexAI:
    pass
class DummyModule:
    ChatVertexAI = DummyVertexAI

sys.modules['langchain_community.chat_models.vertexai'] = DummyModule()
# ------------------------------------------------------------------

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.agent.graph import app_graph, kb_service, llm
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# For Ragas v0.1.x / v0.2.x wrapper compatibility
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError:
    LangchainLLMWrapper = lambda x: x
    LangchainEmbeddingsWrapper = lambda x: x

def evaluate_with_ragas():
    print("="*80)
    print(" Ragas 多维自动评测 (Faithfulness, Relevancy, Precision, Recall)")
    print("="*80)
    
    print("\n[系统] 正在初始化知识库并重建索引...")
    kb_service.rebuild_index()
    
    # 我们使用预设的标准答案作为 Ground Truth
    test_cases = [
        {
            "question": "员工离职需要提前几天通知？",
            "ground_truth": "员工离职通常需要提前30天书面通知。"
        },
        {
            "question": "这几天关于学生证补办有什么规定吗？另外今天北京天气怎么样？",
            "ground_truth": "学生证遗失需前往辅导员处开具证明，再到综合服务大厅12号窗口办理。关于北京天气，根据实时查询应有对应温度和天气状况。"
        },
        {
            "question": "卫津路校区的图书馆几点开门？",
            "ground_truth": "卫津路校区图书馆每天早上8:00开门。"
        }
    ]
    
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for i, tc in enumerate(test_cases):
        print(f"\n[运行用例 {i+1}/{len(test_cases)}]: {tc['question']}")
        
        initial_state = {
            "messages": [HumanMessage(content=tc["question"])]
        }
        config = {"configurable": {"thread_id": f"ragas_eval_{i}"}}
        
        final_answer = ""
        current_contexts = []
        
        try:
            for event in app_graph.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_state in event.items():
                    messages = node_state.get("messages", [])
                    if not messages:
                        continue
                        
                    last_msg = messages[-1]
                    
                    if node_name == "agent" and isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                        final_answer = last_msg.content
                        
                    if node_name == "tools" and isinstance(last_msg, ToolMessage):
                        # 记录被查到的 Context
                        current_contexts.append(last_msg.content)
        except Exception as e:
            print(f"Error during graph execution: {e}")
            final_answer = "Error generating answer"
            
        # Ragas 要求 contexts 不能为空，如果没调工具，就放个空字符串
        if not current_contexts:
            current_contexts = [""]
            
        data["question"].append(tc["question"])
        data["answer"].append(final_answer)
        data["contexts"].append(current_contexts)
        data["ground_truth"].append(tc["ground_truth"])
        
    print("\n[系统] 正在启动 Ragas LLM-as-a-Judge 裁判评分环节 (这可能需要花费几分钟)...")
    
    dataset = Dataset.from_dict(data)
    
    embeddings = kb_service.store.embeddings
    
    # 包装为 Ragas 能识别的对象
    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
        
        print("\n" + "="*80)
        print(" Ragas 评测结果报表")
        print("="*80)
        print(result)
        print("\n具体每条题目的打分明细：")
        df = result.to_pandas()
        import pandas as pd
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df[['question', 'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']])
    except Exception as e:
        print(f"\n[Ragas 评测异常]: {e}")

if __name__ == "__main__":
    evaluate_with_ragas()
