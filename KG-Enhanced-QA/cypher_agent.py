import re
import os
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from kg_tools import (
    get_graph,
    search_entities_by_type,
    get_entity_exact_name
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG_DIR = os.getenv("DEBUG_DIR", "./debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

try:
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )
    logger.info("LLM 初始化成功")
except Exception as e:
    logger.error(f"LLM 初始化失败: {e}")
    llm = None

def parse_direct_multi_hop_query(question: str):
    pattern = r'哪些空间既(邻近|连通|冲突)([^又]+)又(邻近|连通|冲突)([^？?]+)？?'
    match = re.search(pattern, question)

    if match:
        relation1 = match.group(1)
        entity1 = match.group(2).strip()
        relation2 = match.group(3)
        entity2 = match.group(4).strip()

        logger.info(f"直接解析多跳查询: {relation1} {entity1}, {relation2} {entity2}")

        return {
            'condition1': {'relation': relation1, 'entity': entity1},
            'condition2': {'relation': relation2, 'entity': entity2}
        }
    return None


def is_multi_hop_query(question: str):
    multi_hop_patterns = [
        r'哪些\s*\w*\s*既.*?又.*?',
        r'同时.*?和.*?',
        r'既.*?又.*?',
    ]

    for pattern in multi_hop_patterns:
        if re.search(pattern, question):
            return True
    return False


def generate_multi_hop_cypher(multi_hop_info: dict):
    """Generate Cypher for multi-hop queries"""
    if not multi_hop_info:
        return None

    cond1 = multi_hop_info['condition1']
    cond2 = multi_hop_info['condition2']

    relation_map = {
        '邻近': '邻近',
        '连通': '连通',
        '冲突': '冲突'
    }

    rel1 = relation_map.get(cond1['relation'], '邻近')
    rel2 = relation_map.get(cond2['relation'], '邻近')

    entity1 = cond1['entity']
    entity2 = cond2['entity']

    cypher = f"""
    MATCH (entity1:SPC {{name: '{entity1}'}})<-[r1:`{rel1}`]-(target)-[r2:`{rel2}`]->(entity2:SPC {{name: '{entity2}'}})
    RETURN
      '{entity1}+{entity2}' AS source,
      '{rel1},{rel2}' AS relation,
      target.name AS target,
      labels(target) AS target_labels
    """

    logger.info(f"生成的多跳查询Cypher: {cypher.strip()}")
    return cypher.strip()

def extract_entities_and_intent(question: str):
    """Identify entities, types, and intents using LLM"""
    if is_multi_hop_query(question):
        logger.info("识别为多跳查询，跳过LLM识别")
        return None, None, 'multi_hop'

    if not llm:
        return extract_entity_simple(question), None, classify_intent_simple(question)

    prompt = f"""
你是一个专业的知识图谱查询意图分析器。请从以下问题中提取核心实体、实体类型和查询意图。

**问题**:：{question}

实体类型说明：
- SPC: 建筑空间 (门诊部、手术室、ICU、急诊科、急诊部、太平间等)
- SITE: 场地 (综合医院、传染病医院、停车场等)
- EQP: 设施设备 (CT、MRI、X光机等)
- MAT: 物料 (药品、器械、材料等)
- PRO: 属性 (温度、湿度、面积、自成一区、良好通风等)

意图类型：
- adjacency: 查询邻近、相邻关系
- conflict: 查询冲突、矛盾关系
- composition: 查询组成、包含关系
- properties: 查询属性、特征
- comprehensive: 查询全面信息

重要提示：
1. 只提取具体的实体名称，不要包含动词或问句部分
2. 例如"急诊部由什么组成？"应该提取"急诊部"而不是"急诊部由什么组成"
3. 例如"门诊部和哪些空间相邻？"应该提取"门诊部"

请按以下JSON格式返回结果（只返回JSON，不要其他解释）：
{{
    "primary_entity": "主要实体名称",
    "entity_type": "实体类型(SPC/SITE/EQP/MAT/PRO)",
    "secondary_entity": "次要实体名称或null",
    "intent": "意图类型",
    "keywords": ["关键词1", "关键词2"]
}}
"""

    try:
        response = llm.invoke(prompt)  #
        # Extracting JSON String
        json_str_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if not json_str_match:
            raise ValueError("LLM返回内容中未找到有效的JSON对象。")

        result = json.loads(json_str_match.group(0))
        logger.info(f"LLM解析结果: {result}")  #

        primary_entity = result.get('primary_entity')

        # Verify that the entity is valid
        if not primary_entity or not isinstance(primary_entity, str) or len(primary_entity) > 30:
            raise ValueError(f"LLM提取的实体 '{primary_entity}' 无效或过长。")

        return primary_entity, result.get('entity_type'), result.get('intent', 'general')

    except Exception as e:
        logger.error(f"LLM实体识别失败或返回结果无效: {e}。启动回退机制...")
        return extract_entity_simple(question), None, classify_intent_simple(question)


def extract_entity_simple(question: str):
    """Entity extraction fallback method"""
    cleaned_question = question.strip('？?。！!，,')

    patterns = [
        r'^([^由？?]+?)由什么?组成',
        r'^([^由？?]+?)的组成',

        r'^([^和？?]+?)和哪些?.+?相邻',
        r'^([^和？?]+?)的邻近',

        r'^([^和？?]+?)和哪些?.+?冲突',
        r'^([^和？?]+?)的冲突',

        r'^([^有？?]+?)有什么?属性',
        r'^([^的？?]+?)的属性',

        r'^关于([^的？?，。！!]+)',

        r'^([^\s，。！？?,]{2,8})(空间|设备|设施|场地|部门|科室|房间)?[，。！？?]?',
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned_question)
        if match:
            entity = match.group(1).strip()
            if entity and len(entity) >= 2 and entity not in ['请问', '查询', '什么', '哪些', '有什么', '怎么', '如何']:
                logger.info(f"简单提取找到实体: {entity} (模式: {pattern})")
                return entity

    medical_entities = [
        '急诊部', '急诊科', '门诊部', '门诊科', '手术室', 'ICU', '重症监护室',
        '病房', '药房', '检验科', '放射科', 'CT室', 'MRI室', 'X光室',
        '护士站', '医生办公室', '候诊区', '收费处', '挂号处', '住院部'
    ]

    for entity in medical_entities:
        if entity in cleaned_question:
            logger.info(f"通过医疗实体列表找到: {entity}")
            return entity

    logger.warning(f"无法从问题中提取实体: {question}")
    return None


def classify_intent_simple(question: str):
    intent_keywords = {
        'composition': ['组成', '制成', '包含', '由什么构成', '材料', '部件', '构成', '由什么组成'],
        'adjacency': ['相邻', '邻近', '连通', '靠近', '挨着', '旁边', '和哪些空间'],
        'conflict': ['冲突', '矛盾', '不兼容', '不能同时', '互斥'],
        'properties': ['属性', '特征', '性质', '参数', '指标', '特点', '有什么属性'],
        'multi_hop': ['既...又...', '同时', '并且', '而且', '既邻近...又', '既靠近...又', '多条件'],
        'comprehensive': ['所有信息', '详细信息', '全部关系', '完整信息'],
        'general': ['是什么', '介绍', '说明', '概述']
    }

    for intent, keywords in intent_keywords.items():
        if any(kw in question for kw in keywords):
            logger.info(f"意图分类: {intent} (关键词匹配)")
            return intent

    return 'comprehensive'


def generate_comprehensive_cypher(entity_name: str, intent: str, entity_type: str = None):
    if not entity_name:
        return "MATCH (n) RETURN n.name LIMIT 10"

    entity_label = f":`{entity_type}`" if entity_type else ""

    base_queries = {
        'adjacency': [
            f"MATCH (s{entity_label} {{name: '{entity_name}'}})-[r:`邻近`]-(t) RETURN s.name as source, type(r) as relation, t.name as target, labels(t) as target_labels",
            f"MATCH (s{entity_label} {{name: '{entity_name}'}})-[r:`连通`]-(t) RETURN s.name as source, type(r) as relation, t.name as target, labels(t) as target_labels"
        ],
        'conflict': [
            f"MATCH (s{entity_label} {{name: '{entity_name}'}})-[r:`冲突`]-(t) RETURN s.name as source, type(r) as relation, t.name as target, labels(t) as target_labels"
        ],
        'composition': [
            f"MATCH (s{entity_label} {{name: '{entity_name}'}})-[r:`由…组成`]->(t) RETURN s.name as source, type(r) as relation, t.name as target, labels(t) as target_labels",
            f"MATCH (s)-[r:`由…组成`]->(t{entity_label} {{name: '{entity_name}'}}) RETURN s.name as source, type(r) as relation, t.name as target, labels(t) as target_labels"
        ],
        'properties': [
            f"MATCH (s{entity_label} {{name: '{entity_name}'}})-[r:`有…属性`]->(t) RETURN s.name as source, type(r) as relation, t.name as target, labels(t) as target_labels"
        ],
        'comprehensive': [
            f"MATCH (s{entity_label} {{name: '{entity_name}'}})-[r]-(t) RETURN s.name as source, type(r) as relation, t.name as target, labels(t) as target_labels LIMIT 20"
        ]
    }

    queries = base_queries.get(intent, base_queries['comprehensive'])

    if len(queries) > 1:
        return " UNION ".join(queries)
    else:
        return queries[0]


def analyze_kg_structure(structured_results: list, entity_name: str, intent: str):
    analysis = {
        'entity_info': {'name': entity_name, 'total_relations': len(structured_results)},
        'relation_types': {},
        'connected_entities': {},
        'domain_insights': []
    }

    for result in structured_results:
        rel_type = result.get('relation', '未知关系')
        target = result.get('target', '')
        target_labels = result.get('target_labels', [])

        if rel_type not in analysis['relation_types']:
            analysis['relation_types'][rel_type] = []
        analysis['relation_types'][rel_type].append({
            'target': target,
            'labels': target_labels
        })

        for label in target_labels:
            if label not in analysis['connected_entities']:
                analysis['connected_entities'][label] = []
            analysis['connected_entities'][label].append(target)

    if intent == 'composition':
        analysis['domain_insights'] = generate_composition_insights(analysis, entity_name)
    elif intent == 'adjacency':
        analysis['domain_insights'] = generate_adjacency_insights(analysis, entity_name)
    elif intent == 'conflict':
        analysis['domain_insights'] = generate_conflict_insights(analysis, entity_name)
    elif intent == 'properties':
        analysis['domain_insights'] = generate_properties_insights(analysis, entity_name)

    return analysis


def generate_composition_insights(analysis: dict, entity_name: str):
    insights = []

    spc_components = analysis['connected_entities'].get('SPC', [])
    eqp_components = analysis['connected_entities'].get('EQP', [])

    if spc_components:
        clinical_spaces = [s for s in spc_components if any(word in s for word in ['诊', '治疗', '检查', '手术', '科'])]
        support_spaces = [s for s in spc_components if
                          any(word in s for word in ['办公', '更衣', '贮藏', '卫生间', '污洗'])]
        service_spaces = [s for s in spc_components if
                          any(word in s for word in ['挂号', '收费', '药房', '候诊', '问讯'])]

        if clinical_spaces:
            insights.append(f"临床诊疗空间（{len(clinical_spaces)}个）：体现了{entity_name}的核心医疗功能")
        if support_spaces:
            insights.append(f"辅助支持空间（{len(support_spaces)}个）：保障日常运营和卫生要求")
        if service_spaces:
            insights.append(f"患者服务空间（{len(service_spaces)}个）：优化就医流程和体验")

    if eqp_components:
        insights.append(f"设备设施（{len(eqp_components)}个）：{entity_name}的技术支撑体系")

    total_spaces = len(spc_components)
    if total_spaces > 20:
        insights.append(f"空间配置复杂度较高（{total_spaces}个功能空间），需要精细化流线设计")

    return insights


def generate_adjacency_insights(analysis: dict, entity_name: str):
    insights = []
    adjacent_spaces = analysis['connected_entities'].get('SPC', [])

    if adjacent_spaces:
        emergency_adjacent = [s for s in adjacent_spaces if '急诊' in s]
        surgical_adjacent = [s for s in adjacent_spaces if any(word in s for word in ['手术', '麻醉', '恢复'])]

        if emergency_adjacent:
            insights.append(f"与急诊系统邻近：体现了{entity_name}在急救医疗链中的重要位置")
        if surgical_adjacent:
            insights.append(f"与手术系统邻近：支持术前术后的连贯性医疗服务")

        insights.append(f"空间邻近策略：{entity_name}的{len(adjacent_spaces)}个邻近空间形成功能聚合区")

    return insights


def generate_conflict_insights(analysis: dict, entity_name: str):
    insights = []
    conflict_spaces = analysis['connected_entities'].get('SPC', [])

    if conflict_spaces:
        noise_conflicts = [s for s in conflict_spaces if any(word in s for word in ['机房', '空调', '设备'])]
        infection_conflicts = [s for s in conflict_spaces if any(word in s for word in ['污洗', '垃圾', '太平间'])]

        if noise_conflicts:
            insights.append(f"噪声冲突：{entity_name}需要与{len(noise_conflicts)}个噪声源保持距离")
        if infection_conflicts:
            insights.append(f"感染控制冲突：院感防控要求{entity_name}与污染区域物理隔离")

        insights.append(f"规划约束：{len(conflict_spaces)}个冲突空间限制了{entity_name}的布局选择")

    return insights


def generate_properties_insights(analysis: dict, entity_name: str):
    insights = []
    properties = analysis['connected_entities'].get('PRO', [])

    if properties:
        environmental = [p for p in properties if any(word in p for word in ['温度', '湿度', '气压', '风速'])]
        spatial = [p for p in properties if any(word in p for word in ['面积', '高度', '净高', '开间'])]
        technical = [p for p in properties if any(word in p for word in ['负荷', '功率', '容量', '压力'])]

        if environmental:
            insights.append(f"环境参数控制：{len(environmental)}项指标确保{entity_name}的使用环境")
        if spatial:
            insights.append(f"空间尺度要求：{len(spatial)}项参数规范{entity_name}的建筑设计")
        if technical:
            insights.append(f"技术性能指标：{len(technical)}项参数保障{entity_name}的功能实现")

    return insights


def query_kg(question: str):
    graph = get_graph()
    if not graph:
        return [], "数据库连接失败，请检查 .env 配置或 Neo4j 服务状态。"

    # Check whether it is a multi-hop query
    if is_multi_hop_query(question):
        logger.info("处理多跳查询，跳过LLM识别")

        multi_hop_info = parse_direct_multi_hop_query(question)
        # Removed fallback to parse_multi_hop_query as it's redundant
        if multi_hop_info:
            entity1 = multi_hop_info['condition1']['entity']
            entity2 = multi_hop_info['condition2']['entity']

            multi_hop_info['condition1']['entity'] = entity1
            multi_hop_info['condition2']['entity'] = entity2

            cypher_query = generate_multi_hop_cypher(multi_hop_info)
            logger.info(f"多跳查询Cypher: {cypher_query}")

            exact_entity = f"{entity1}+{entity2}"
            intent = 'multi_hop'
            entity_type = None
        else:
            return [], "无法解析多跳查询条件，请重新描述问题。"
    else:
        primary_entity, entity_type, intent = extract_entities_and_intent(question)
        logger.info(f"识别结果 - 实体: {primary_entity}, 类型: {entity_type}, 意图: {intent}")

        if not primary_entity:
            return [], "抱歉，无法识别出具体的实体名称，请重新描述您的问题。"

        exact_entity = get_entity_exact_name(primary_entity)
        if exact_entity != primary_entity:
            logger.info(f"实体名称修正: {primary_entity} -> {exact_entity}")

        cypher_query = generate_comprehensive_cypher(exact_entity, intent, entity_type) # Pass entity_type
        logger.info(f"生成的Cypher查询: {cypher_query}")

    # Execute a query
    structured_results = []
    try:
        raw_results = graph.run(cypher_query).data()
        logger.info(f"查询返回 {len(raw_results)} 条原始结果")

        for record in raw_results:
            standardized = {
                'source': record.get('source', exact_entity),
                'relation': record.get('relation', '未知关系'),
                'target': record.get('target', ''),
                'target_labels': record.get('target_labels', [])
            }
            structured_results.append(standardized)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_q = re.sub(r'[^0-9A-Za-z]', '_', question)[:50]
        debug_info = {
            'question': question,
            'intent': intent,
            'cypher_query': cypher_query,
            'raw_results_count': len(raw_results)
        }

        with open(f"{DEBUG_DIR}/{ts}_{safe_q}_debug.json", "w", encoding="utf-8") as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)

        with open(f"{DEBUG_DIR}/{ts}_{safe_q}_query.cypher", "w", encoding="utf-8") as f:
            f.write(cypher_query)

        with open(f"{DEBUG_DIR}/{ts}_{safe_q}_result.json", "w", encoding="utf-8") as f:
            json.dump(structured_results, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"Cypher查询执行失败: {e}")
        return [], f"查询执行失败：{str(e)}"

    if not structured_results and intent != 'multi_hop':
        logger.info("尝试更宽松的查询...")
        primary_entity = exact_entity.split('+')[0] if '+' in exact_entity else exact_entity
        fallback_queries = [
            f"MATCH (s)-[r]-(t) WHERE s.name CONTAINS '{primary_entity.split()[0]}' RETURN s.name as source, type(r) as relation, t.name as target, labels(t) as target_labels LIMIT 10",
            f"MATCH (s)-[r]-(t) WHERE s.name =~ '.*{primary_entity}.*' OR t.name =~ '.*{primary_entity}.*' RETURN s.name as source, type(r) as relation, t.name as target, labels(t) as target_labels LIMIT 10"
        ]

        for fallback_query in fallback_queries:
            try:
                logger.info(f"尝试回退查询: {fallback_query}")
                raw_results = graph.run(fallback_query).data()
                if raw_results:
                    for record in raw_results:
                        standardized = {
                            'source': record.get('source', ''),
                            'relation': record.get('relation', '未知关系'),
                            'target': record.get('target', ''),
                            'target_labels': record.get('target_labels', [])
                        }
                        structured_results.append(standardized)
                    logger.info(f"回退查询找到 {len(raw_results)} 条结果")
                    break
            except Exception as e:
                logger.error(f"回退查询失败: {e}")

    kg_analysis = analyze_kg_structure(structured_results, exact_entity, intent)

    if not llm:
        return structured_results, "LLM服务不可用，无法生成详细解释。"

    enhanced_context = build_enhanced_context(question, exact_entity, intent, structured_results, kg_analysis)

    answer_prompt = f"""
你是一位资深的医疗建筑规划专家，请基于知识图谱查询结果，提供专业、深入的分析回答。

用户问题: {question}

{enhanced_context}

请按以下要求回答：
1. **直接回答核心问题**，避免"根据查询结果"等套话
2. **紧扣知识图谱结果**，清晰呈现实体间的关联与层次结构
3. **语言简洁专业**，以条理清晰的要点形式表达，突出重点、避免冗余
4. **不分小节、不加标题**

格式要求：
- 适当使用**加粗**突出关键信息
- 使用准确的专有名词
- 控制在300字以内
"""
    try:
        response = llm.invoke(answer_prompt)
        reasoning_answer = response.content.strip()
    except Exception as e:
        logger.error(f"LLM推理失败: {e}")
        reasoning_answer = "生成详细回答时出现错误，但已返回结构化查询结果。"

    return structured_results, reasoning_answer


def build_enhanced_context(question: str, entity_name: str, intent: str, structured_results: list, kg_analysis: dict):
    context_parts = []

    context_parts.append(f"查询实体: {entity_name} (共发现{len(structured_results)}个关联关系)")
    context_parts.append(f"查询类型: {intent}")

    if kg_analysis['relation_types']:
        context_parts.append("\n关系类型分布:")
        for rel_type, items in kg_analysis['relation_types'].items():
            context_parts.append(f"- {rel_type}: {len(items)}个关联实体")

    if kg_analysis['connected_entities']:
        context_parts.append("\n关联实体分类:")
        for entity_type, entities in kg_analysis['connected_entities'].items():
            type_names = {
                'SPC': '建筑空间',
                'EQP': '设施设备',
                'SITE': '场地',
                'MAT': '物料',
                'PRO': '属性参数'
            }
            type_desc = type_names.get(entity_type, entity_type)
            context_parts.append(f"- {type_desc}({entity_type}): {len(entities)}个")
            for entity in entities[:5]:
                context_parts.append(f"  • {entity}")
            if len(entities) > 5:
                context_parts.append(f"  • ...等{len(entities) - 5}个")

    if kg_analysis['domain_insights']:
        context_parts.append("\n领域知识洞察:")
        for insight in kg_analysis['domain_insights']:
            context_parts.append(f"- {insight}")

    context_parts.append("\n详细关系数据:")
    for i, result in enumerate(structured_results[:10]):
        target_info = result['target']
        if result['target_labels']:
            target_info += f" [{'/'.join(result['target_labels'])}]"
        context_parts.append(f"{i + 1}. {result['source']} --{result['relation']}--> {target_info}")

    if len(structured_results) > 10:
        context_parts.append(f"...等其他{len(structured_results) - 10}个关系")

    return "\n".join(context_parts)


def get_entity_suggestions():
    suggestions = {}
    for et in ['SPC', 'SITE', 'EQP', 'MAT', 'PRO']:
        try:
            suggestions[et] = search_entities_by_type(et, limit=20)
        except Exception as e:
            logger.error(f"获取{et}实体失败: {e}")
            suggestions[et] = []
    return suggestions