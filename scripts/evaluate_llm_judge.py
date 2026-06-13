import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import logging
import urllib.request
import ssl
from typing import List, Dict, Any

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage

# 引入我们在系统中已经实例化好的 LLM
from app.agent.graph import llm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class EvaluationResult(BaseModel):
    score: float = Field(description="评分，范围 0.0 到 1.0")
    reason: str = Field(description="评分的详细中文理由，指出具体缺失或命中哪些信息")

parser = PydanticOutputParser(pydantic_object=EvaluationResult)

def load_cmrc2018_sample(num_samples: int = 3) -> List[Dict[str, Any]]:
    """绕过 SSL 验证从官方仓库拉取 CMRC2018 验证集切片"""
    print(f"[数据层] 正在从公开标准库 (CMRC2018) 获取前 {num_samples} 条测试数据...")
    url = "https://raw.githubusercontent.com/ymcui/cmrc2018/master/squad-style-data/cmrc2018_dev.json"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    samples = []
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ctx) as f:
        dataset = json.loads(f.read().decode('utf-8'))
        
        # 扁平化提取 SQuAD 格式
        for article in dataset['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    ground_truth = qa['answers'][0]['text']
                    samples.append({
                        "question": question,
                        "contexts": [context], # 放入列表中以匹配 RAG 输出格式
                        "ground_truth": ground_truth
                    })
                    if len(samples) >= num_samples:
                        return samples
    return samples

def generate_answer(question: str, contexts: List[str]) -> str:
    """模拟 RAG 系统的生成阶段"""
    prompt = f"""基于以下参考上下文回答问题。如果上下文中没有答案，请说“我不知道”。
上下文：
{" ".join(contexts)}

问题：
{question}
"""
    result = llm.invoke([HumanMessage(content=prompt)])
    return result.content

def evaluate_metric(prompt_template: str, input_variables: Dict[str, Any], metric_name: str) -> EvaluationResult:
    """通用的 LLM 裁判调用函数，强制输出 JSON 结构"""
    prompt = PromptTemplate(
        template=prompt_template + "\n\n{format_instructions}",
        input_variables=list(input_variables.keys()),
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke(input_variables)
        return result
    except OutputParserException as e:
        logging.error(f"[{metric_name}] 解析 LLM 输出失败: {e}")
        return EvaluationResult(score=0.0, reason="LLM 未能按要求输出合法的 JSON 格式。")
    except Exception as e:
        logging.error(f"[{metric_name}] 评估过程发生未知错误: {e}")
        return EvaluationResult(score=0.0, reason=f"评估报错: {str(e)}")

def get_faithfulness_score(question: str, answer: str, contexts: List[str]) -> float:
    """无幻觉指数：回答是否严格基于检索到的上下文"""
    prompt = """你是专业的企业级 RAG 评测专家。
任务：评估系统生成的【回答】是否发生了“幻觉”。
规则：
1. 仔细阅读提供的【上下文】。
2. 仔细阅读系统的【回答】。
3. 凡是【回答】中出现了【上下文】中完全没有提及的实体、数字、规定或绝对性结论，一律视为幻觉。
4. 如果【回答】能被【上下文】完全支撑，分数为 1.0；如果大部分支撑只有少部分自由发挥，分数在 0.5-0.9 之间；如果完全是瞎编的，分数为 0.0。

上下文：
{contexts}

问题：{question}
回答：{answer}
"""
    result = evaluate_metric(prompt, {
        "question": question, 
        "answer": answer, 
        "contexts": "\n---\n".join(contexts)
    }, "Faithfulness")
    logging.info(f"[Faithfulness] 理由: {result.reason}")
    return result.score

def get_relevancy_score(question: str, answer: str) -> float:
    """答案相关性：回答是否一针见血解决了问题"""
    prompt = """你是专业的企业级 RAG 评测专家。
任务：评估系统的【回答】是否直接、相关且有效地回答了用户的【问题】。
规则：
1. 重点考察【回答】有没有偏题，有没有“车轱辘话”或者答非所问。
2. 如果一针见血解决了问题，分数为 1.0；如果有部分相关但不够直接，分数在 0.5-0.9 之间；如果完全不相关，分数为 0.0。

问题：{question}
回答：{answer}
"""
    result = evaluate_metric(prompt, {"question": question, "answer": answer}, "Relevancy")
    logging.info(f"[Relevancy] 理由: {result.reason}")
    return result.score

def get_context_precision_score(question: str, answer: str, contexts: List[str]) -> float:
    """上下文精度：检索出的上下文中有多少是有用的，排位越靠前越好"""
    prompt = """你是专业的企业级 RAG 评测专家。
任务：评估检索出的【上下文】列表的“精度”。
规则：
1. 给定【问题】和对应的系统【回答】。
2. 阅读提供的【上下文】，判断这些片段中，有多少比例对生成最终【回答】真正起到了支撑作用。
3. 如果大部分提供的上下文都是噪音或完全无关，分数应该偏低（0.0-0.4）。
4. 如果几乎所有提供的上下文都是高度浓缩且相关的核心信息，分数为 1.0。

问题：{question}
回答：{answer}
上下文：
{contexts}
"""
    result = evaluate_metric(prompt, {
        "question": question, 
        "answer": answer, 
        "contexts": "\n---\n".join(contexts)
    }, "ContextPrecision")
    logging.info(f"[ContextPrecision] 理由: {result.reason}")
    return result.score

def get_context_recall_score(question: str, ground_truth: str, contexts: List[str]) -> float:
    """上下文召回率：标准答案中的要点是否都在上下文中被找到了"""
    prompt = """你是专业的企业级 RAG 评测专家。
任务：评估检索出的【上下文】是否成功“召回”了人类标注的【标准答案】中的核心事实。
规则：
1. 提取【标准答案】中的核心知识点。
2. 逐一检查这些知识点是否能在【上下文】中找到对应描述。
3. 计算召回比例。如果标准答案的要点100%都在上下文中出现过，给 1.0。少一个要点就扣分，如果几乎没找到，给 0.0。

问题：{question}
标准答案：{ground_truth}
上下文：
{contexts}
"""
    result = evaluate_metric(prompt, {
        "question": question, 
        "ground_truth": ground_truth, 
        "contexts": "\n---\n".join(contexts)
    }, "ContextRecall")
    logging.info(f"[ContextRecall] 理由: {result.reason}")
    return result.score

def main():
    print("================================================================================")
    print(" Native Chinese Agentic RAG Evaluation (CMRC2018 Public Dataset) ")
    print("================================================================================")
    
    # 1. 加载公开数据集
    eval_data = load_cmrc2018_sample(num_samples=3)

    results_matrix = []

    for i, item in enumerate(eval_data):
        print(f"\n[题目 {i+1}]: {item['question']}")
        print(f" -> 标准答案: {item['ground_truth']}")
        
        # 2. 模拟 RAG 生成
        print(" -> 正在请求 LLM 生成答案...")
        generated_answer = generate_answer(item['question'], item['contexts'])
        print(f" -> 生成答案: {generated_answer}")
        
        # 3. 多维评估打分
        f_score = get_faithfulness_score(item["question"], generated_answer, item["contexts"])
        r_score = get_relevancy_score(item["question"], generated_answer)
        cp_score = get_context_precision_score(item["question"], generated_answer, item["contexts"])
        cr_score = get_context_recall_score(item["question"], item["ground_truth"], item["contexts"])
        
        results_matrix.append({
            "Faithfulness": f_score,
            "Relevancy": r_score,
            "Context_Precision": cp_score,
            "Context_Recall": cr_score
        })
        
        print(f"\n[评分报告]")
        print(f" -> Faithfulness:     {f_score:.4f}")
        print(f" -> Relevancy:        {r_score:.4f}")
        print(f" -> Context Precision:{cp_score:.4f}")
        print(f" -> Context Recall:   {cr_score:.4f}")

    # 计算平均分
    avg_scores = {
        k: sum(d[k] for d in results_matrix) / len(results_matrix)
        for k in results_matrix[0].keys()
    }

    print("\n================================================================================")
    print(" 综合平均分 (Averaged Metrics) ")
    print("================================================================================")
    for k, v in avg_scores.items():
        print(f"{k.ljust(20)}: {v:.4f}")
    print("================================================================================\n")

if __name__ == "__main__":
    main()
