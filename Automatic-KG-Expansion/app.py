# app.py - Streamlit frontend for NER+RE with Neo4j integration

import streamlit as st
import pandas as pd
from py2neo import Graph, Node, Relationship
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="智能知识图谱扩展系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .input-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    .input-section label {
        color: white !important;
        font-weight: 600;
        font-size: 1.1rem;
    }

    .results-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    .entities-display {
        background: rgba(255,255,255,0.9);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin-bottom: 1rem;
    }

    .success-box {
        background: linear-gradient(90deg, #56ab2f, #a8e6cf);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }

    .info-box {
        background: linear-gradient(90deg, #74b9ff, #0984e3);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    div[data-testid="metric-container"] label {
        color: white !important;
        font-weight: 600;
    }

    div[data-testid="metric-container"] div {
        color: white !important;
        font-size: 1.2rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Output directory for CSV files
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Import local models
from NER.NERModel import load_ner
from RE.REModel import (
    build_cfg, load_model,
    preprocess, predict_relation
)

# Initialize session state
if "triples" not in st.session_state:
    st.session_state["triples"] = []
if "edited_df" not in st.session_state:
    st.session_state["edited_df"] = pd.DataFrame()
if "entities_str" not in st.session_state:
    st.session_state["entities_str"] = ""


# Model initialization (cached to load only once)
@st.cache_resource(show_spinner=False)
def init_models():
    ner = load_ner()
    re_cfg = build_cfg()
    re_model = load_model(re_cfg)
    return ner, re_model, re_cfg


# Page header
st.markdown('<h1 class="main-header">🧠 智能知识图谱扩展系统</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">深度学习模型驱动的知识图谱扩展系统</p>', unsafe_allow_html=True)

# Model loading
with st.spinner('🚀 正在加载模型...'):
    ner_model, re_model, re_cfg = init_models()

# Input section
with st.container():
    st.markdown('<div class="input-section">', unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        sentence = st.text_input(
            "请输入中文文本进行智能分析：",
            placeholder="例如：门诊部应设在靠近医院入口处",
            help="系统将自动识别文本中的实体并抽取它们之间的关系"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        extract_btn = st.button("🔍 开始分析", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Analysis processing
if extract_btn:
    if not sentence.strip():
        st.error("⚠️ 请先输入文本内容")
        st.stop()

    st.session_state["triples"] = []

    with st.spinner('🤖 模型正在进行分析文本...'):
        # Named Entity Recognition
        raw_output = ner_model.predict(sentence)
        try:
            if isinstance(raw_output[0][1], str) and '-' in raw_output[0][1]:
                from NER.NERModel import extract_entities

                entities = extract_entities(raw_output)
            else:
                entities = raw_output
        except Exception:
            st.error("❌ 实体识别过程中出现异常")
            st.stop()

        if len(entities) < 2:
            st.warning("⚠️ 识别到的实体不足两个，无法进行关系抽取")
            st.stop()

        st.session_state["entities_str"] = ", ".join([f"**{e}**({t})" for e, t in entities])

        # Relation Extraction
        triples = []
        progress_bar = st.progress(0)
        total_pairs = len(entities) * (len(entities) - 1) // 2

        pair_count = 0
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                h, ht = entities[i]
                t, tt = entities[j]
                sample = preprocess(sentence, h, t, ht, tt, re_cfg)
                rel, _ = predict_relation(re_model, sample)
                triples.append(dict(
                    entity=h, entitytag=ht,
                    tail=t, tailtag=tt,
                    relation=rel, 选择=True
                ))
                pair_count += 1
                progress_bar.progress(pair_count / total_pairs)

        st.session_state["triples"] = triples
        progress_bar.empty()

# Results display
if st.session_state["triples"]:
    st.markdown('<div class="results-section">', unsafe_allow_html=True)

    # Statistics
    col1, col2, col3 = st.columns(3)
    entities_count = len(st.session_state["entities_str"].split(", "))
    relations_count = len(st.session_state["triples"])

    with col1:
        st.metric("🏷️ 识别实体", f"{entities_count} 个")
    with col2:
        st.metric("🔗 抽取关系", f"{relations_count} 条")
    with col3:
        selected_count = len([t for t in st.session_state["triples"] if t.get("选择", True)])
        st.metric("✅ 待导出", f"{selected_count} 条")

    st.markdown('</div>', unsafe_allow_html=True)

    # Entity display
    st.markdown(
        f'<div class="entities-display"><strong>🏷️ 识别到的实体：</strong><br>{st.session_state["entities_str"]}</div>',
        unsafe_allow_html=True)

    # Relation table
    st.markdown("### 🔗 抽取到的关系")
    df_src = pd.DataFrame(st.session_state["triples"])
    if "选择" not in df_src.columns:
        df_src["选择"] = True

    # Reorder columns
    columns = ["选择", "entity", "entitytag", "relation", "tail", "tailtag"]
    df_src = df_src[columns]

    edited = st.data_editor(
        df_src,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "选择": st.column_config.CheckboxColumn("选择", default=True),
            "entity": st.column_config.TextColumn("头实体", width="medium"),
            "entitytag": st.column_config.TextColumn("头实体类型", width="small"),
            "relation": st.column_config.TextColumn("关系", width="medium"),
            "tail": st.column_config.TextColumn("尾实体", width="medium"),
            "tailtag": st.column_config.TextColumn("尾实体类型", width="small"),
        },
        key="editor"
    )
    st.session_state["edited_df"] = edited

    # Export button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("💾 导出到CSV并扩展知识图谱", use_container_width=True):
            sel_df = st.session_state["edited_df"]
            sel_df = sel_df[sel_df.get("选择", True)].copy()

            if sel_df.empty:
                st.error("⚠️ 请至少选择一条关系进行导出")
            else:
                data_to_save = sel_df.drop(columns=["选择"], errors='ignore')
                csv_path = OUTPUT_DIR / "output.csv"

                # CSV export
                csv_success = False
                try:
                    if csv_path.exists():
                        existing_df = pd.read_csv(csv_path, encoding="utf-8")
                        if not existing_df.empty:
                            common_columns = list(set(existing_df.columns) & set(data_to_save.columns))
                            if common_columns:
                                existing_df = existing_df[common_columns]
                                data_to_save = data_to_save[common_columns]
                        combined_df = pd.concat([existing_df, data_to_save], ignore_index=True)
                    else:
                        combined_df = data_to_save

                    combined_df.to_csv(csv_path, index=False, encoding="utf-8")
                    csv_success = True

                except Exception as e:
                    st.error(f"❌ CSV保存失败: {str(e)}")

                # Neo4j integration
                neo4j_success = False
                try:
                    graph = Graph("http://localhost:7474",
                                  auth=("neo4j", "your_password_here"))
                    graph.run("RETURN 1")  # 测试连接

                    for _, r in sel_df.iterrows():
                        h = Node(r["entitytag"], name=r["entity"])
                        t = Node(r["tailtag"], name=r["tail"])
                        graph.merge(h, r["entitytag"], "name")
                        graph.merge(t, r["tailtag"], "name")
                        graph.merge(Relationship(h, r["relation"], t))

                    neo4j_success = True

                except Exception as e:
                    st.error(f"❌ Neo4j写入失败: {str(e)}")

                # Results feedback
                if csv_success and neo4j_success:
                    st.balloons()
                    st.markdown(
                        f'<div class="success-box">🎉 成功！数据已保存到CSV文件并扩展知识图谱<br>📁 CSV文件：{csv_path}<br>📊 共导出 {len(sel_df)} 条关系</div>',
                        unsafe_allow_html=True)
                    st.markdown(
                        '<div class="info-box">💡 在Neo4j Browser中使用以下查询查看结果：<br><code>MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 50</code></div>',
                        unsafe_allow_html=True)
                elif csv_success:
                    st.warning("⚠️ CSV保存成功，但Neo4j连接失败")
                elif neo4j_success:
                    st.warning("⚠️ 知识图谱扩展成功，但CSV保存失败")
                else:
                    st.error("❌ 操作失败，请检查系统配置")

# Sidebar information
with st.sidebar:
    st.markdown("### 📋 系统信息")
    st.info("**模型状态:** ✅ 已就绪")
    st.info("**NER模型:** BiLSTM-CRF")
    st.info("**RE模型:** Transformer")
    st.info("**图数据库:** Neo4j")

    st.markdown("### 💡 使用说明")
    st.markdown("""
    1. 在输入框中输入中文文本
    2. 点击"开始分析"进行处理
    3. 查看识别的实体和关系
    4. 选择需要的关系条目
    5. 导出到CSV并扩展知识图谱
    """)

    st.markdown("### 🔧 系统要求")
    st.markdown("""
    - Neo4j 数据库服务
    - DeepKE 模型文件
    - Python 环境依赖
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "🚀 深度学习模型驱动的智能知识图谱扩展系统 | "
    "Powered by Streamlit"
    "</div>",
    unsafe_allow_html=True
)