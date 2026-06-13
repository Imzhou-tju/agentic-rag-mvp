import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.agent.graph import app_graph, kb_service

def safe_print(text):
    try:
        print(text)
    except Exception:
        print(text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore'))

def main():
    print("初始化知识库中...")
    kb_service.rebuild_index()
    
    question = "卫津路校区的图书馆几点开门？"
    print(f"\n[用户]: {question}\n")
    
    initial_state = {
        "messages": [HumanMessage(content=question)]
    }
    
    config = {"configurable": {"thread_id": "demo_1"}}
    
    for event in app_graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_state in event.items():
            messages = node_state.get("messages", [])
            if not messages:
                continue
                
            last_msg = messages[-1]
            
            if node_name == "agent" and isinstance(last_msg, AIMessage):
                if last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        safe_print(f"[Agent 决定行动] -> 准备调用工具: 【{tc['name']}】")
                        safe_print(f"   输入参数: {tc['args']}\n")
                elif last_msg.content:
                    safe_print(f"\n[Agent 最终回答] -> \n{last_msg.content}\n")
                    
            elif node_name == "tools" and isinstance(last_msg, ToolMessage):
                safe_print(f"[工具执行完毕] <- 成功获取背景资料并返回给 Agent 重新评估。\n")

if __name__ == "__main__":
    main()
