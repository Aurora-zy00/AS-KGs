# Automatic Knowledge Graph Expansion System

A deep learning-powered system for automatic knowledge graph expansion using Named Entity Recognition (NER) and Relation Extraction (RE) models. The system processes Chinese text to identify entities and extract relationships between them, then exports the results to CSV files and integrates with Neo4j graph database.

## Features

- **Named Entity Recognition**: Identifies entities in Chinese text using BiLSTM-CRF model
- **Relation Extraction**: Extracts relationships between entities using Transformer model
- **Interactive Web Interface**: User-friendly Streamlit-based interface
- **Knowledge Graph Integration**: Automatic export to Neo4j graph database
- **CSV Export**: Results can be exported to CSV format for further analysis

## System Architecture

```
Automatic KG Expansion/
├── app.py                 # Main Streamlit application
├── NER/                   # Named Entity Recognition module
│   ├── NERModel.py        # NER model implementation
│   ├── checkpoints/       # Model weights directory
│   │   └── model.pth      # Trained NER model (not included)
│   └── vocab.pkl          # Vocabulary file (not included)
├── RE/                    # Relation Extraction module
│   ├── REModel.py         # RE model implementation
│   ├── checkpoints/       # Model weights directory
│   │   └── transformer.pth # Trained RE model (not included)
│   └── data/              # Data directory
│       ├── relation.csv   # Relation types (not included)
│       └── vocab.pkl      # Vocabulary file (not included)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zhayong0208/HAS-KG.git
cd automatic-kg-expansion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Neo4j database:
   - Install Neo4j Community Edition
   - Start Neo4j service
   - Configure authentication in `app.py` (default: username=neo4j, password=password)

## Model Setup

This system requires trained models from the DeepKE project. The following files are not included in this repository and need to be obtained separately:

### NER Model Files
- `NER/checkpoints/model.pth`: BiLSTM-CRF model trained on Chinese NER dataset
- `NER/vocab.pkl`: Vocabulary file containing word2id, label2id, and id2label mappings

### RE Model Files
- `RE/checkpoints/transformer.pth`: Transformer model trained on Chinese relation extraction dataset
- `RE/data/vocab.pkl`: Vocabulary file for relation extraction
- `RE/data/relation.csv`: Relation types mapping file

### Training Your Own Models

To train these models, use the [DeepKE](https://github.com/zjunlp/DeepKE) framework:

1. **For NER Model**:
   ```bash
   # Follow DeepKE NER training instructions
   cd DeepKE/example/ner/standard
   python run.py
   ```

2. **For RE Model**:
   ```bash
   # Follow DeepKE RE training instructions
   cd DeepKE/example/re/standard
   python run.py
   ```

3. **Vocabulary Files**:
   - `vocab.pkl` files are generated during the training process
   - `relation.csv` should contain your domain-specific relation types

### File Format Examples

**relation.csv format**:
```csv
relation,index
attribute,0
connected_to,1
composed_of,2
conflict,3
made_of,4
adjacent_to,5
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Enter Chinese text in the input field

4. Click "Start Analysis" to process the text

5. Review the identified entities and extracted relations

6. Select desired relations and export to CSV/Neo4j

## Example

Input: "门诊部应设在靠近医院入口处"

Output:
- Entities: 门诊部(SPC), 医院入口处(SITE)
- Relations: 门诊部 - adjacent_to - 医院入口处


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [DeepKE](https://github.com/zjunlp/DeepKE) for the underlying NER and RE models
- [Streamlit](https://streamlit.io/) for the web interface framework
- [Neo4j](https://neo4j.com/) for graph database integration