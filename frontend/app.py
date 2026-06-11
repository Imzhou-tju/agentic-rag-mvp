from __future__ import annotations

import os

import requests
import streamlit as st


API_BASE = os.getenv('API_BASE', 'http://127.0.0.1:8000/api')

st.set_page_config(page_title='Enterprise Agentic RAG MVP', layout='wide')
st.title('Enterprise Agentic RAG MVP')
st.caption('上传企业文档，构建知识库，并通过 Agentic RAG 完成问答与任务执行。')

with st.sidebar:
    st.subheader('服务配置')
    api_base = st.text_input('API Base URL', value=API_BASE)
    st.markdown('任务类型说明：')
    st.markdown('- qa：知识问答')
    st.markdown('- summary：摘要')
    st.markdown('- key_points：关键要点提取')
    st.markdown('- compare：比较分析')

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader('1. 上传文档')
    files = st.file_uploader('支持 PDF / TXT / Markdown', accept_multiple_files=True)
    if st.button('上传并重建索引', type='primary'):
        if not files:
            st.warning('请先选择文件。')
        else:
            for file in files:
                resp = requests.post(
                    f'{api_base}/upload',
                    files={'file': (file.name, file.getvalue(), file.type or 'application/octet-stream')},
                    timeout=120,
                )
                if resp.ok:
                    st.success(f"{file.name} 上传成功")
                else:
                    st.error(f"{file.name} 上传失败：{resp.text}")

    if st.button('查看索引统计'):
        resp = requests.get(f'{api_base}/index/stats', timeout=30)
        if resp.ok:
            data = resp.json()
            st.json(data)
        else:
            st.error(resp.text)

with col2:
    st.subheader('2. 提问或执行任务')
    question = st.text_area('输入问题或任务', height=160, placeholder='例如：总结员工手册中的考勤要求')
    task = st.selectbox('任务类型', ['qa', 'summary', 'key_points', 'compare'])
    use_agent = st.checkbox('启用 Agent 路由', value=True)

    if st.button('运行', type='primary'):
        if not question.strip():
            st.warning('请输入问题。')
        else:
            payload = {'question': question, 'task': task, 'use_agent': use_agent}
            resp = requests.post(f'{api_base}/ask', json=payload, timeout=120)
            if resp.ok:
                data = resp.json()
                st.markdown('### 回答')
                st.write(data['answer'])

                st.markdown('### 调试信息')
                st.json(
                    {
                        'route': data['route'],
                        'rewritten_question': data.get('rewritten_question'),
                        'debug': data.get('debug', {}),
                    }
                )

                st.markdown('### 引用片段')
                if data['sources']:
                    for src in data['sources']:
                        with st.expander(f"{src['document_name']} | score={src['score']:.4f}"):
                            st.write(src['text'])
                else:
                    st.info('本次回答未使用知识库检索。')
            else:
                st.error(resp.text)
