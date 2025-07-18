from py2neo import Graph
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# -------- Neo4j connection configuration -------- #
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")

_graph_instance = None

def get_graph():
    global _graph_instance
    try:
        if _graph_instance and _graph_instance.database.name:
            _graph_instance.run("RETURN 1")
            return _graph_instance
    except Exception as e:
        logger.warning(f"Neo4j 连接已失效: {e}. 尝试重新连接...")
        _graph_instance = None

    if _graph_instance is None:
        if not NEO4J_PASSWORD:
            logger.error("Neo4j 密码 (NEO4J_PASSWORD) 未在 .env 文件中设置！")
            return None
        try:
            _graph_instance = Graph(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
            _graph_instance.run("RETURN 1")
            logger.info("Neo4j 连接成功")
            return _graph_instance
        except Exception as e:
            logger.error(f"Neo4j 连接失败: {e}")
            return None

    return _graph_instance


def search_entities_by_type(entity_type: str, limit: int = 10):
    graph = get_graph()
    if not graph:
        logger.error("Neo4j 连接未建立")
        return []
    try:
        query = f"MATCH (n:{entity_type}) RETURN n.name AS name LIMIT $limit"
        result = graph.run(query, limit=limit).data()
        return [r["name"] for r in result if r.get("name")]
    except Exception as e:
        logger.error(f"搜索实体失败: {e}")
        return []


def find_entity_by_partial_name(partial_name: str, limit: int = 10):
    graph = get_graph()
    if not graph:
        logger.error("Neo4j 连接未建立")
        return []
    try:
        query = """
        MATCH (n)
        WHERE n.name CONTAINS $partial_name
        RETURN n.name AS name, labels(n) AS labels,
               CASE 
                 WHEN 'SPC' IN labels(n) THEN 'SPC'
                 WHEN 'SITE' IN labels(n) THEN 'SITE'
                 WHEN 'EQP' IN labels(n) THEN 'EQP'
                 WHEN 'MAT' IN labels(n) THEN 'MAT'
                 WHEN 'PRO' IN labels(n) THEN 'PRO'
                 ELSE 'OTHER'
               END AS entity_type
        ORDER BY length(n.name)
        LIMIT $limit
        """
        result = graph.run(query, partial_name=partial_name, limit=limit).data()
        return result
    except Exception as e:
        logger.error(f"模糊搜索失败: {e}")
        return []


def get_entity_exact_name(input_name: str):
    if not graph:
        return input_name
    try:
        exact = graph.run(
            "MATCH (n {name:$name}) RETURN n.name AS name LIMIT 1",
            name=input_name
        ).data()
        if exact:
            logger.info(f"找到精确匹配: {exact[0]['name']}")
            return exact[0]["name"]

        case_insensitive = graph.run(
            "MATCH (n) WHERE toLower(n.name) = toLower($name) RETURN n.name AS name LIMIT 1",
            name=input_name
        ).data()
        if case_insensitive:
            logger.info(f"找到大小写匹配: {case_insensitive[0]['name']}")
            return case_insensitive[0]["name"]

        partials = find_entity_by_partial_name(input_name, 5)
        if partials:
            best_match = partials[0]["name"]
            logger.info(f"使用模糊匹配: {input_name} -> {best_match}")
            return best_match

        logger.warning(f"未找到匹配实体: {input_name}")
        return input_name

    except Exception as e:
        logger.error(f"查找实体名称失败: {e}")
        return input_name