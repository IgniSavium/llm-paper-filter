import streamlit as st
import json

# Set page configuration, use wide layout
st.set_page_config(page_title="ArXiv Daily", layout="wide")

st.title("📚 Daily ArXiv Filter")

# Modified the prompt to avoid using local paths directly
uploaded_file = st.file_uploader("📁 Please drag and drop or select your today's JSON result file here", type="json")

if uploaded_file is not None:
    # Load JSON data
    data = json.load(uploaded_file)
    meta = data.get("metadata", {})
    
    # Display overview information
    st.info(f"**📅 Date:** {meta.get('target_date')} | **📊 Related Papers:** {meta.get('total_hits')} papers")
    
    # Iterate and display each paper
    for i, paper in enumerate(data.get("hits_zh", [])):
        eval_info = paper.get('eval', {})
        score = eval_info.get('relevance_score', 0)
        category = eval_info.get('category', 'None')
        
        with st.container():
            # Title section
            st.subheader(f"[{i+1}] {paper.get('title', paper.get('zh_title'))}")
            st.caption(f"{paper.get('zh_title')} | **Score:** {score}/5 | **Category:** {category}")
            st.write(f"**👥 Authors:** {', '.join(paper.get('authors', []))}")
            
            # AI Recommendation Reason
            st.success(f"**💡 AI 推荐理由:** {eval_info.get('zh_reason', eval_info.get('reason', '暂无'))}")
            
            # 中文摘要（默认不展开）
            with st.expander("📝 中文摘要", expanded=(score==5)):
                st.write(paper.get('zh_abstract', '暂无中文摘要'))
                
            # English abstract (collapsed by default)
            with st.expander("📄 English Abstract"):
                st.write(paper.get('abstract', 'No English abstract available.'))
                
            # Bottom link and divider
            st.markdown(f"[🔗 arXiv Abstract]({paper.get('url')})")
            st.divider()