import json
import os
import shutil
from pathlib import Path

DATA_DIR = Path("app/data")
UPLOAD_DIR = DATA_DIR / "uploads"

# 1. Define mock documents
MOCK_DOCUMENTS = {
    "2025最新卫津路校区图书馆管理规定.txt": """
第一条 卫津路校区图书馆作为全校教职工和学生的知识中心，2025年起实行新规。
第二条 开放时间：周一至周日 08:00 - 22:30。国家法定节假日另行通知。
第三条 借阅规则：本科生最多可借阅图书15册，借期30天。研究生可借阅20册，借期60天。
第四条 自习室预约：请通过“北洋之声”APP提前一天进行座位预约。
""",
    "2021旧版卫津路校区图书馆规章.txt": """
本规章适用于卫津路校区。
第一条 开放时间：每天 08:30 - 22:00。
第二条 本科生借阅上限为10册。
（注意：此文档已过期，仅供历史归档参考。）
""",
    "2025官方北洋园校区班车时刻表.txt": """
北洋园校区与卫津路校区通勤班车（2025版）
1. 发车频率：高峰期（07:00-09:00, 16:30-18:30）每15分钟一班。平峰期每30分钟一班。
2. 乘车点：北洋园校区东门，卫津路校区大门内广场。
3. 票价：教职工免费，学生单程3元。
4. 特殊说明：考试周期间会增开夜班车至23:00。
""",
    "北洋园校区学生日常报销指南.txt": """
学生参加学术会议、竞赛等产生的费用报销指南。
1. 所需材料：机打发票原件、支付记录截图、参会证明或获奖证书。
2. 办理地点：北洋园校区综合服务大厅12号窗口。
3. 审批流程：指导老师签字 -> 学院审核 -> 财务处入账。
4. 资金到账时间：一般为提交材料后的5个工作日。
""",
    "卫津路校区财务处报销常见问题.txt": """
卫津路校区财务处答疑：
Q1: 差旅费可以报销高铁一等座吗？
A: 原则上学生出差仅限报销高铁二等座或硬卧。教职工副高及以上职称可报销一等座。
Q2: 办理地点在哪？
A: 卫津路校区第9教学楼一层大厅。
"""
}

# 2. Define Evaluation Dataset
EVAL_DATASET = [
    {
        "question": "卫津路的图书馆晚上几点关门？",
        "expected_document_name": "2025最新卫津路校区图书馆管理规定.txt"
    },
    {
        "question": "北洋园校区的班车多久发一趟？",
        "expected_document_name": "2025官方北洋园校区班车时刻表.txt"
    },
    {
        "question": "在北洋园办理报销需要去哪个窗口？",
        "expected_document_name": "北洋园校区学生日常报销指南.txt"
    },
    {
        "question": "本科生去卫津路图书馆可以借多少本书？",
        "expected_document_name": "2025最新卫津路校区图书馆管理规定.txt"
    },
    {
        "question": "学生出差能报销高铁一等座吗，我是卫津路校区的？",
        "expected_document_name": "卫津路校区财务处报销常见问题.txt"
    }
]

def setup_mock_data():
    print("Initializing mock data for evaluation...")
    
    # Ensure directories exist
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clean previous uploads to ensure clean state
    for file in UPLOAD_DIR.glob("*.txt"):
        file.unlink()
        
    # Write mock documents
    for filename, content in MOCK_DOCUMENTS.items():
        filepath = UPLOAD_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"Created: {filename}")
        
    # Write eval dataset
    dataset_path = DATA_DIR / "eval_dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(EVAL_DATASET, f, ensure_ascii=False, indent=2)
    print(f"Created: eval_dataset.json with {len(EVAL_DATASET)} test cases.")

if __name__ == "__main__":
    setup_mock_data()
