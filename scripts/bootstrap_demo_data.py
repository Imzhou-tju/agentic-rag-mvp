from pathlib import Path

BASE = Path('app/data/uploads')
BASE.mkdir(parents=True, exist_ok=True)

files = {
    'company_intro.md': '''# 公司介绍\n\n本公司专注于企业智能助手产品，提供知识库问答、流程自动化和多模型接入能力。\n我们的产品支持 PDF、Markdown、TXT 文档导入，并强调可追溯回答与权限管理。\n''',
    'employee_handbook.txt': '''员工手册\n\n1. 考勤要求：工作日 9:30 上班，18:30 下班。\n2. 请假流程：员工需至少提前一天在系统中提交申请，由直属主管审批。\n3. 报销要求：单笔餐饮报销上限为 200 元，差旅需附发票和行程单。\n''',
    'reimbursement_policy.md': '''# 报销制度\n\n- 交通费：需提供发票。\n- 餐饮费：单笔不超过 200 元。\n- 住宿费：按城市等级执行，不同等级上限不同。\n- 提交方式：在 OA 系统中填写报销单，并上传附件。\n''',
}

for name, content in files.items():
    (BASE / name).write_text(content, encoding='utf-8')

print(f'Wrote {len(files)} demo files to {BASE}')
