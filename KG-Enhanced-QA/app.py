from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import logging
import traceback
from cypher_agent import query_kg, get_entity_suggestions


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)
CORS(app)

@app.route('/query', methods=['POST'])
def handle_query():
    """Handle query requests with error handling"""
    try:
        if not request.is_json:
            return jsonify({'error': '请求必须是JSON格式'}), 400

        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据为空'}), 400

        question = data.get('question', '').strip()
        if not question:
            return jsonify({'error': '问题不能为空'}), 400

        if len(question) > 500:
            return jsonify({'error': '问题长度不能超过500字符'}), 400

        logger.info(f"收到查询请求: {question}")

        structured_answer, reasoning_answer = query_kg(question)

        response = {
            'structured_answer': structured_answer,
            'reasoning_answer': reasoning_answer,
            'status': 'success',
            'query': question
        }

        logger.info(f"返回结果: 结构化答案{len(structured_answer)}条")
        return jsonify(response)

    except Exception as e:
        logger.error(f"查询处理失败: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'查询处理失败: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Get entity suggestions for frontend autocomplete"""
    try:
        suggestions = get_entity_suggestions()
        return jsonify({
            'suggestions': suggestions,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"获取建议失败: {e}")
        return jsonify({
            'error': f'获取建议失败: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': '医疗空间知识图谱服务正常运行'
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '接口不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '服务器内部错误'}), 500


if __name__ == '__main__':
    logger.info("启动Flask服务...")
    app.run(host='0.0.0.0', port=5000, debug=True)