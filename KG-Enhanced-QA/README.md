# KG-Enhanced-QA: Knowledge Graph Enhanced Question Answering System

A question-answering system that combines Large Language Models (LLM) with Knowledge Graph (KG) technology for hospital architectural spatial domain queries.

## Architecture

```
├── Frontend (Streamlit)     # User interface
├── Backend API (Flask)      # RESTful service
├── Cypher Agent            # LLM-powered query processing
├── KG Tools                # Neo4j database operations
└── Knowledge Graph (Neo4j) # Graph database
```

## Features

- **Natural Language Processing**: Powered by LLM for intelligent query understanding
- **Multi-type Queries**: Adjacent relationships, conflict detection, composition analysis, attribute,Multi hop queries
- **Web Interface**: User-friendly Streamlit frontend with real-time query processing
- **Comprehensive Analysis**: Detailed reasoning and structured results

## System Requirements

- Python 3.8+
- Neo4j Database
- OpenAI-compatible API (DeepSeek, OpenAI, etc.)

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Aurora-zy00/HAS-KG.git
cd KG-Enhanced-QA
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
Create a `.env` file in the root directory:
```env
# LLM Configuration
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://api.deepseek.com
LLM_MODEL=deepseek-chat

# Neo4j Database Configuration
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Application Configuration
FLASK_PORT=5000
STREAMLIT_PORT=8501
```

4. **Start the system**
```bash
python startup.py
```

## Usage

### Web Interface
Access the Streamlit frontend at `http://localhost:8501`

**Query Types:**
- **Adjacency Relations**: "Which spaces are adjacent to the emergency department?"
- **Conflict Detection**: "Which spaces conflict with the operating room?"
- **Composition Analysis**: "What is the ICU composed of?"
- **Property Queries**: "What properties does the CT equipment have?"
- **Multi-hop Queries**: "Which spaces are both adjacent to the nursing station and close to the clean corridor?"
- - **Free query**: "Supports other natural language query"

## Entity Types

- **SPC**: Spatial entities (Emergency Department, Operating Room, ICU)
- **SITE**: Site entities (General Hospital, Parking Lot)
- **EQP**: Equipment entities (CT, MRI, X-ray)
- **MAT**: Material entities (Medications, Instruments)
- **PRO**: Property entities (Temperature, Humidity, Ventilation)

## Key Components

### Cypher Agent (`cypher_agent.py`)
- LLM-powered entity recognition and intent classification
- Dynamic Cypher query generation
- Multi-hop query processing
- Knowledge graph analysis and reasoning

### KG Tools (`kg_tools.py`)
- Neo4j database connection management
- Entity search and name matching
- Database operations and utilities

### Frontend (`frontend.py`)
- Interactive web interface
- Query type selection and configuration
- Real-time result visualization
- System status monitoring

### Backend API (`app.py`)
- RESTful API endpoints
- Error handling and validation
- CORS support for web integration

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain) for LLM integration
- Powered by [Neo4j](https://neo4j.com/) for graph database capabilities
- Frontend developed with [Streamlit](https://streamlit.io/)
- Backend API built with [Flask](https://flask.palletsprojects.com/)
