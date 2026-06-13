import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.agent.graph import app_graph, kb_service, llm

def get_faithfulness_score(llm, question, answer, contexts):
    """验证幻觉：判断 answer 是否完全基于 contexts 里的信息"""
    context_str = "\n".join(contexts)
    prompt = f"""你是一个苛刻的裁判。请评估下述回答是否出现“幻觉”。
回答必须仅仅基于提供的上下文信息。
如果回答中的所有事实都能在上下文中找到，请输出 1。
如果回答包含了上下文中没有提及的事实（幻觉），或者编造了数据，请输出 0。

上下文：
{context_str}

回答：
{answer}

请只输出一个数字（1或0），不要解释。"""
    try:
        res = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return float(res)
    except:
        return 0.0

def get_answer_relevancy_score(llm, question, answer):
    """验证相关性：判断 answer 是否直接回答了 question"""
    prompt = f"""你是一个裁判。请评估给定的“回答”是否有效解决了“问题”。
如果回答完美解答了问题，或者合理地解释了为什么无法回答，请输出 1。
如果回答答非所问、不知所云，请输出 0。

问题：{question}
回答：{answer}

请只输出一个数字（1或0），不要解释。"""
    try:
        res = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return float(res)
    except:
        return 0.0

def evaluate_llm_judge():
    print("="*80)
    print(" LLM-as-a-Judge 多维自动评测 (原生实现替代 Ragas)")
    print("="*80)
    
    print("\n[系统] 正在初始化知识库并重建索引...")
    kb_service.rebuild_index()
    
    test_cases = [
        {"question": "员工离职需要提前几天通知？"},
        {"question": "这几天关于学生证补办有什么规定吗？另外今天北京天气怎么样？"},
        {"question": "卫津路校区的图书馆几点开门？"},
        {"question": "学校对作弊的处理严格吗？"}
    ]
    
    results = []
    
    for i, tc in enumerate(test_cases):
        question = tc['question']
        print(f"\n[运行用例 {i+1}/{len(test_cases)}]: {question}")
        
        initial_state = {"messages": [HumanMessage(content=question)]}
        config = {"configurable": {"thread_id": f"llm_judge_{i}"}}
        
        final_answer = ""
        current_contexts = []
        
        for event in app_graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_state in event.items():
                messages = node_state.get("messages", [])
                if not messages:
                    continue
                last_msg = messages[-1]
                if node_name == "agent" and isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                    final_answer = last_msg.content
                if node_name == "tools" and isinstance(last_msg, ToolMessage):
                    current_contexts.append(last_msg.content)
                    
        if not current_contexts:
            current_contexts = ["未调用检索，无上下文"]
            
        print("  -> 生成回答完毕，正在调用裁判 LLM 打分...")
        faith_score = get_faithfulness_score(llm, question, final_answer, current_contexts)
        rel_score = get_answer_relevancy_score(llm, question, final_answer)
        
        results.append({
            "question": question,
            "faithfulness": faith_score,
            "relevancy": rel_score
        })
        print(f"  -> Faithfulness (无幻觉): {faith_score} | Relevancy (相关性): {rel_score}")
        
    print("\n" + "="*80)
    print(" LLM-as-a-Judge 评测结果报表")
    print("="*80)
    avg_faith = sum(r['faithfulness'] for r in results) / len(results)
    avg_rel = sum(r['relevancy'] for r in results) / len(results)
    
    print(f"{'Question':<30} | {'Faithfulness':<15} | {'Relevancy':<15}")
    print("-" * 80)
    for r in results:
        q_disp = r['question'][:25] + "..." if len(r['question']) > 28 else r['question']
        # simple alignment handling
        q_pad = q_disp.ljust(28 - len(q_disp.encode('gbk')) + len(q_disp))
        print(f"{q_pad} | {r['faithfulness']:<15} | {r['relevancy']:<15}")
    print("-" * 80)
    print(f"Average Faithfulness: {avg_faith:.2f}")
    print(f"Average Relevancy:    {avg_rel:.2f}")

if __name__ == "__main__":
    evaluate_llm_judge()
