import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.agent.graph import app_graph, kb_service

def main():
    print("="*50)
    print(" Agentic RAG 命令行对话助手 (Agent Mode)")
    print("="*50)
    
    print("\n[系统] 正在初始化知识库并重建索引...")
    count = kb_service.rebuild_index()
    print(f"[系统] 索引重建完毕，共加载了 {count} 个文档切片。")
    print("\n[系统] 输入 'exit' 或 'quit' 退出，输入 'clear' 清屏。")
    print("-"*50)
    
    # Initialize thread state for memory (optional, but good for multi-turn)
    config = {"configurable": {"thread_id": "1"}}
    
    while True:
        try:
            question = input("\n 你: ")
            if not question.strip():
                continue
            
            if question.strip().lower() in ['exit', 'quit']:
                print(" 再见！")
                break
            elif question.strip().lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
                
            print("\n [Agent思考中...]")
            
            initial_state = {
                "messages": [HumanMessage(content=question)]
            }
            
            # Stream the events
            for event in app_graph.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_state in event.items():
                    messages = node_state.get("messages", [])
                    if not messages:
                        continue
                        
                    last_msg = messages[-1]
                    
                    if node_name == "agent" and isinstance(last_msg, AIMessage):
                        if last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                print(f"    调用工具: {tc['name']} (参数: {tc['args']})")
                        elif last_msg.content:
                            print(f"\n 回答: {last_msg.content}")
                            
                    elif node_name == "tools" and isinstance(last_msg, ToolMessage):
                        print(f"   工具返回: 获取到相关背景资料。")
                        
        except KeyboardInterrupt:
            print("\n 再见！")
            break
        except Exception as e:
            print(f"\n [错误]: {e}")

if __name__ == "__main__":
    main()
