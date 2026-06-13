import sys
import os
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config import get_settings
from scripts.evaluate_e2e import run_e2e_evaluation

def evaluate_all_models():
    settings = get_settings()
    if not settings.test_llm_models:
        print("未配置 TEST_LLM_MODELS，直接退出。请在 .env 中设置 TEST_LLM_MODELS")
        return
        
    models = [m.strip() for m in settings.test_llm_models.split(',') if m.strip()]
    print(f"准备评测以下模型: {models}")
    
    for model_name in models:
        print(f"\n\n{'='*80}")
        print(f" 开始评测模型: {model_name}")
        print(f"{'='*80}\n")
        
        # 1. 设置环境变量，覆盖 Settings 中的默认值
        os.environ["LLM_MODEL"] = model_name
        
        # 2. 清除配置缓存
        get_settings.cache_clear()
        
        # 3. 重新加载相关模块，确保新的 LLM 被初始化
        # 按依赖顺序重新加载
        import app.core.config as config_module
        importlib.reload(config_module)
        
        import app.rag.service as service_module
        importlib.reload(service_module)
        
        import app.agent.graph as graph_module
        importlib.reload(graph_module)
        
        # 4. 运行评测，传入新实例化的 graph 和 kb_service
        run_e2e_evaluation(custom_graph=graph_module.app_graph, custom_kb=graph_module.kb_service)

if __name__ == "__main__":
    evaluate_all_models()
