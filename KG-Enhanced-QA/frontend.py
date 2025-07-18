import streamlit as st
import requests

# Page Configuration
st.set_page_config(
    page_title="医院建筑空间知识图谱查询系统",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 医院建筑空间知识图谱查询系统")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("系统配置")
backend_url = st.sidebar.text_input("后端服务地址", value="http://localhost:5000")


# Check service status
def check_service_status():
    try:
        response = requests.get(f"{backend_url}/health", timeout=5)
        if response.status_code == 200:
            return True, "服务正常"
        else:
            return False, f"服务异常: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"无法连接到服务: {str(e)}"


# Display service status
status, message = check_service_status()
if status:
    st.sidebar.success(f"✅ {message}")
else:
    st.sidebar.error(f"❌ {message}")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("查询界面")

    mode = st.selectbox(
        "选择查询类型",
        ["邻近关系", "冲突检测", "组成分析", "属性查询", "多跳查询","自由查询"],
        help="选择不同的查询类型来获取相应的分析结果"
    )

    examples = {
        "邻近关系": "门诊部和哪些空间相邻？",
        "冲突检测": "手术室和哪些空间存在冲突？",
        "组成分析": "ICU由哪些空间什么组成？",
        "属性查询": "CT设备有什么属性？",
        "多跳查询": "哪些空间既邻近护士站又靠近清洁通道？",
        "自由查询": "请输入您的问题..."
    }

    # User Input
    if mode == "多跳查询":
        st.markdown("#### 多跳查询配置")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**条件1**")
            space1 = st.text_input("空间1", placeholder="例如：急诊部", key="space1")
            relation1 = st.selectbox("关系1", ["邻近", "连通", "冲突"], key="rel1")

        with col_b:
            st.markdown("**条件2**")
            space2 = st.text_input("空间2", placeholder="例如：住院部", key="space2")
            relation2 = st.selectbox("关系2", ["邻近", "连通", "冲突"], key="rel2")

        if space1 and space2:
            query = f"哪些空间既{relation1}{space1}又{relation2}{space2}？"
            st.text_area("生成的查询语句", value=query, height=60, disabled=True)
        else:
            query = ""
        space = None

    elif mode != "自由查询":
        space = st.text_input(
            "请输入空间或设备名称",
            placeholder="例如：门诊部、手术室、CT设备等",
            help="输入您要查询的医疗空间、设备或其他实体名称"
        )

        query_templates = {
            "邻近关系": f"{space}和哪些空间相邻？",
            "冲突检测": f"{space}和哪些空间存在冲突？",
            "组成分析": f"{space}由什么组成？",
            "属性查询": f"{space}有什么属性？"
        }

        if space:
            query = query_templates.get(mode, f"关于{space}的信息")
            st.text_area("生成的查询语句", value=query, height=60, disabled=True)
        else:
            query = ""
    else:
        query = st.text_area(
            "请输入您的问题",
            placeholder=examples[mode],
            height=100,
            help="您可以用自然语言描述您的问题"
        )
        space = None

with col2:
    st.header("查询示例")
    st.markdown("### 🔍 示例查询")
    st.code("门诊部和哪些空间相邻？", language=None)
    st.code("手术室和哪些空间存在冲突？", language=None)
    st.code("哪些空间既邻近护士站又靠近清洁通道？", language=None)


    st.markdown("### 📋 实体类型说明")
    st.markdown("""
    - **SPC**: 建筑空间 (如门诊部、手术室)
    - **SITE**: 场地 (如园区、停车场)  
    - **EQP**: 设施设备 (如CT、MRI)
    - **MAT**: 物料 (如药品、器械、材料)
    - **PRO**: 属性 (如温度、湿度、良好通风)
    """)

# Query execution
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    query_btn = st.button("🔍 执行查询", type="primary")

with col2:
    if st.button("🔄 清空结果"):
        st.rerun()

# Query Processing
if query_btn:
    execute_query = False

    if mode == "多跳查询":
        if not query or not space1 or not space2:
            st.warning("⚠️ 请完整填写多跳查询的两个空间和关系！")
        elif not status:
            st.error("❌ 后端服务未连接，请检查服务状态！")
        else:
            execute_query = True
    elif not query or (mode != "自由查询" and not space):
        st.warning("⚠️ 请先输入查询内容！")
    elif not status:
        st.error("❌ 后端服务未连接，请检查服务状态！")
    else:
        execute_query = True

    if execute_query:
        with st.spinner("正在查询知识图谱..."):
            try:
                response = requests.post(
                    f"{backend_url}/query",
                    json={"question": query},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    structured = data.get("structured_answer", [])
                    reasoning = data.get("reasoning_answer", "")

                    st.success("✅ 查询完成！")

                    result_col1, result_col2 = st.columns([1, 1])

                    with result_col1:
                        st.subheader("📊 结构化查询结果")

                        if structured:
                            if mode == "邻近关系":
                                st.write("**邻近空间列表：**")
                                for item in structured:
                                    if isinstance(item, dict):
                                        st.write(f"• {item.get('name', item)} ({item.get('relation', '邻近')})")
                                    else:
                                        st.write(f"• {item}")

                            elif mode == "冲突检测":
                                st.write("**冲突空间列表：**")
                                for item in structured:
                                    if isinstance(item, dict):
                                        st.write(f"• {item.get('name', item)} ({item.get('relation', '冲突')})")
                                    else:
                                        st.write(f"• {item}")

                            elif mode == "组成分析":
                                st.write("**组成结构：**")
                                for item in structured:
                                    if isinstance(item, dict):
                                        st.write(f"• {item.get('name', item)} ({item.get('relation', '组成')})")
                                    else:
                                        st.write(f"• {item}")

                            elif mode == "属性查询":
                                st.write("**属性列表：**")
                                for item in structured:
                                    if isinstance(item, dict):
                                        st.write(f"• {item.get('property', item.get('name', item))}")
                                    else:
                                        st.write(f"• {item}")

                            elif mode == "多跳查询":
                                st.write("**多条件匹配结果：**")
                                for item in structured:
                                    if isinstance(item, dict):
                                        name = item.get('target', '')
                                        labels = item.get('target_labels', [])
                                        st.write(f"• {name} ({', '.join(labels)})")
                                    else:
                                        st.write(f"• {item}")
                            else:
                                st.json(structured)
                        else:
                            st.info("🔍 未找到直接的结构化结果")

                    with result_col2:
                        st.subheader("🤖 AI 分析解释")
                        if reasoning:
                            st.write(reasoning)
                        else:
                            st.info("暂无AI分析结果")

                    # visualization
                    if mode == "组成结构分析" and structured:
                        st.subheader("🎨 组成结构可视化")

                        dot_graph = "digraph G {\n"
                        dot_graph += "  rankdir=TB;\n"
                        dot_graph += "  node [shape=box, style=filled, fillcolor=lightblue];\n"
                        dot_graph += f'  "{space}" [fillcolor=lightgreen];\n'

                        for item in structured:
                            if isinstance(item, dict):
                                name = item.get('name', '')
                                relation = item.get('relation', '组成')
                                if name:
                                    dot_graph += f'  "{name}";\n'
                                    dot_graph += f'  "{space}" -> "{name}" [label="{relation}"];\n'

                        dot_graph += "}"

                        try:
                            st.graphviz_chart(dot_graph)
                        except Exception as e:
                            st.error(f"可视化生成失败: {e}")
                            st.code(dot_graph, language='dot')

                else:
                    error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
                    st.error(f"❌ 查询失败: {error_data.get('error', '未知错误')}")

            except requests.exceptions.Timeout:
                st.error("❌ 查询超时，请稍后重试")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ 网络连接错误: {str(e)}")
            except Exception as e:
                st.error(f"❌ 查询过程中发生错误: {str(e)}")

# footer
st.markdown("---")
st.markdown("### 📝 使用说明")
st.markdown("""
1. **选择查询类型**：根据您的需求选择相应的查询类型
2. **输入查询内容**：输入您要查询的空间、设备或实体名称
3. **执行查询**：点击查询按钮获取结果
4. **查看结果**：系统会显示结构化结果和AI分析解释
5. **可视化**：对于组成结构查询，系统会生成可视化图表
""")

st.markdown("---")
st.markdown("**系统状态**: 🟢 运行中 | **版本**: v1.0 | **更新时间**: 2025-07")