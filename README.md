# Hospital Architectural Spatial Knowledge Graph (HAS-KG)

> ⚠️ **Notice**  
This repository is associated with an academic paper that is currently under peer review. Until the paper is officially published and this statement is removed, all contents in this repository are strictly for review and archival purposes only.  

This repository accompanies the paper “Hospital Architectural Spatial Knowledge Graph: A Framework for Incremental Construction and Scenario Applications Enabled by Generative AI and BIM.” 
It contains three loosely‑coupled sub‑projects that together cover automatic knowledge acquisition, information extraction, and knowledge‑graph‑enhanced question answering.

##  Framework Overview

Please consult the full framework description once the paper is published.  
Only the portions that the authors have chosen to release publicly are included here. Together, the three components form an intelligent design-support system for hospital architecture:

```
HAS-KG
├── 1. Knowledge Extraction & Model Training
│   └── BERT-BiLSTM-CRF/ # High-precision NER (baseline vs. DeepKE)
├── 2. Knowledge Graph Construction & Expansion  
│   └── Automatic-KG-Expansion/   # Automated KG expansion pipeline
└── 3. Knowledge Application & Querying
    └── KG-Enhanced-QA/           # Question-answering system
```

##  System Components

### 1. BERT-BiLSTM-CRF - Named Entity Recognition
- High-precision entity recognition from building design codes and regulations
- For setup and usage, see the **README** inside `BERT-BiLSTM-CRF/`

### 2. Automatic KG Expansion - Knowledge Graph Construction
- Automated pipeline for continuous knowledge graph expansion
- For setup and usage, see the **README** inside `Automatic-KG-Expansion/`

### 3. KG-Enhanced QA - Question Answering System
- Natural language interface for knowledge retrieval and reasoning
- For setup and usage, see the **README** inside `KG-Enhanced-QA/`

##  Methodology

Our approach follows a systematic pipeline:

1. **Core Corpus Construction**: Authoritative building design codes form the foundational dataset(see specific papers for details)
2. **Deep Learning Models**: Unified domain dataset with high-precision NER and RE models
3. **Automated Expansion**: Dynamic pipeline that assimilates new clauses while preserving interpretability
4. **QA System Development**: KG and LLM-based natural language retrieval and reasoning
5. **BIM Integration**: Compliance checking module connecting QA system with BIM models *(workflow described in the paper)*  
6. **Validation**: Scenario-based applications demonstrating effectiveness *(full results in the publication)* 

## Usage Scenarios

### 1. Model Training and Evaluation
- Train custom NER models on hospital architectural texts
- Evaluate model performance on domain-specific datasets
- Fine-tune for specific architectural entity types

### 2. Knowledge Graph Construction
- Parse building codes and regulations  
- Automatically extract entities and relationships  
- Incrementally expand existing knowledge graphs  

### 3. Intelligent Query Answering
- Pose natural-language queries about spatial relationships  
- Perform compliance checks for architectural designs  
- Conduct multi-hop reasoning across connected entities  

### 4. BIM Integration *(refer to the paper for in-depth examples)*
- Connect BIM  with regulatory knowledge
- Assist compliance verification
- Support design decision-making  

## Key Contributions

- **Incremental Construction**: Dynamic knowledge graph that evolves with new regulations
- **Multi-modal Integration**: Seamless connection between text processing, graph databases, and BIM
- **Intelligent Querying**: Natural language interface with advanced reasoning capabilities
- **Practical Applications**: Real-world scenario applications in hospital architectural design

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Technical Stack**: PyTorch, Transformers, Neo4j, Streamlit, Flask
- **Domain Expertise**: Hospital architectural design standards and regulations
- **Open Source**: DeepKE framework for knowledge extraction models

##  Contact

For questions about this research or collaboration opportunities, please contact:
- Email: [zhayong0208@gmail.com]
- GitHub: [@Aurora-zy00](https://github.com/Aurora-zy00)

---
