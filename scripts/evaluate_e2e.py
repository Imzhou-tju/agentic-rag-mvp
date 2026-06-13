import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage, AIMessage
from app.agent.graph import app_graph, kb_service

def run_e2e_evaluation(custom_graph=None, custom_kb=None):
    print("="*80)
    print(" Agentic RAG 端到端评测 (End-to-End Evaluation)")
    print("="*80)
    
    graph_to_use = custom_graph if custom_graph else app_graph
    kb_to_use = custom_kb if custom_kb else kb_service
    
    print("\n[系统] 正在初始化知识库并重建索引...")
    kb_to_use.rebuild_index()
    
    # Define test cases
    test_cases = [
        {
            "id": "T1",
            "desc": "单任务检索",
            "question": "员工离职需要提前几天通知？",
            "expected_tools": ["search_knowledge_base"],
            "expected_args": {}
        },
        {
            "id": "T2",
            "desc": "多意图任务",
            "question": "这几天关于学生证补办有什么规定吗？另外今天北京天气怎么样？",
            "expected_tools": ["search_knowledge_base", "web_search_tool"],
            "expected_args": {}
        },
        {
            "id": "T3",
            "desc": "元数据过滤",
            "question": "卫津路校区的图书馆几点开门？",
            "expected_tools": ["search_knowledge_base"],
            "expected_args": {"campus": "卫津路校区"}
        },
        {
            "id": "T4",
            "desc": "日常闲聊",
            "question": "你好，你是谁？",
            "expected_tools": [],
            "expected_args": {}
        }
    ]
    
    total = len(test_cases)
    passed = 0
    
    print("\n" + "-" * 80)
    print(f"{'ID':<4} | {'Description':<12} | {'Trajectory Match':<18} | {'Param Match':<12} | {'Result':<6}")
    print("-" * 80)
    
    for i, tc in enumerate(test_cases):
        initial_state = {
            "messages": [HumanMessage(content=tc["question"])]
        }
        
        config = {"configurable": {"thread_id": f"e2e_eval_{i}"}}
        
        called_tools = set()
        captured_args_list = []
        
        try:
            for event in graph_to_use.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_state in event.items():
                    messages = node_state.get("messages", [])
                    if not messages:
                        continue
                        
                    last_msg = messages[-1]
                    
                    if node_name == "agent" and isinstance(last_msg, AIMessage):
                        if last_msg.tool_calls:
                            for tcall in last_msg.tool_calls:
                                tool_name = tcall["name"]
                                called_tools.add(tool_name)
                                captured_args_list.append(tcall["args"])
        except Exception as e:
            print(f"Error executing graph for {tc['id']}: {e}")
            
        # Verify Trajectory
        expected_tools_set = set(tc["expected_tools"])
        if not expected_tools_set:
            trajectory_match = (len(called_tools) == 0)
        else:
            trajectory_match = expected_tools_set.issubset(called_tools)
        
        # Verify Parameters
        param_match = True
        if tc["expected_args"]:
            for expected_k, expected_v in tc["expected_args"].items():
                found_match = False
                for args_dict in captured_args_list:
                    if expected_k in args_dict and args_dict[expected_k] == expected_v:
                        found_match = True
                        break
                if not found_match:
                    param_match = False
                    
        result_str = "PASS" if trajectory_match and param_match else "FAIL"
        if result_str == "PASS":
            passed += 1
            
        traj_str = "Yes" if trajectory_match else "No"
        param_str = "Yes" if param_match else "No"
        
        # Determine exact padding for Chinese characters (simple heuristic)
        desc_pad = tc['desc'].ljust(12 - len(tc['desc'].encode('gbk')) + len(tc['desc']))
        print(f"{tc['id']:<4} | {desc_pad} | {traj_str:<18} | {param_str:<12} | {result_str:<6}")
        if result_str == "FAIL":
            print(f"      [!] Expected Tools: {expected_tools_set}, Got: {called_tools}")
            if tc["expected_args"]:
                print(f"      [!] Expected Args: {tc['expected_args']}, Got: {captured_args_list}")
                
    print("-" * 80)
    pass_rate = (passed / total) * 100
    print(f"Total Cases: {total}")
    print(f"Passed: {passed}")
    print(f"Pass Rate (Task Execution & Trajectory Regression): {pass_rate:.2f}%")
    print("=" * 80)
    
if __name__ == "__main__":
    run_e2e_evaluation()
