import streamlit as st
import json

# 设置页面配置，使用宽屏模式
st.set_page_config(page_title="ArXiv Daily", layout="wide")

st.title("📚 Daily ArXiv 筛选阅读器")

# 修改了提示语，避免直接使用本地路径
uploaded_file = st.file_uploader("📁 请在此处拖入或选择你今天的 JSON 结果文件", type="json")

if uploaded_file is not None:
    # 加载 JSON 数据
    data = json.load(uploaded_file)
    meta = data.get("metadata", {})
    
    # 显示概览信息
    st.info(f"**📅 日期:** {meta.get('target_date')} | **📊 高价值论文:** {meta.get('total_hits')} 篇")
    
    # 遍历显示每一篇论文
    for i, paper in enumerate(data.get("hits_zh", [])):
        eval_info = paper.get('eval', {})
        score = eval_info.get('relevance_score', 0)
        category = eval_info.get('category', 'None')
        
        with st.container():
            # 标题部分
            st.subheader(f"[{i+1}] {paper.get('zh_title', paper.get('title'))}")
            st.caption(f"{paper.get('title')} | **得分:** {score}/5 | **类别:** {category}")
            st.write(f"**👥 作者:** {', '.join(paper.get('authors', []))}")
            
            # AI 推荐理由
            st.success(f"**💡 AI 推荐理由:** {eval_info.get('zh_reason', eval_info.get('reason', '暂无'))}")
            
            # 中文摘要（默认不展开）
            with st.expander("📝 阅读中文摘要", expanded=(score==5)):
                st.write(paper.get('zh_abstract', '暂无中文摘要'))
                
            # 英文摘要（默认不展开）
            with st.expander("📄 阅读英文摘要 (English Abstract)"):
                st.write(paper.get('abstract', 'No English abstract available.'))
                
            # 底部链接与分割线
            st.markdown(f"[🔗 前往 arXiv Abstract]({paper.get('url')})")
            st.divider()