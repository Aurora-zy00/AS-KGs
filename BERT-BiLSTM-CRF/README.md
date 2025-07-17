# BERT-BiLSTM-CRF for Named Entity Recognition

This repository contains the implementation of a BERT-BiLSTM-CRF model for Named Entity Recognition (NER) tasks, particularly designed for Chinese text processing.

## Model Architecture

The model combines three key components:
- **BERT**: Pre-trained language model for contextual embeddings
- **BiLSTM**: Bidirectional LSTM for sequence modeling
- **CRF**: Conditional Random Field for label sequence optimization

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Format

The model expects data in BIO format with each line containing a token and its corresponding label separated by whitespace. Sentences are separated by empty lines.

Example format:
```
眼 B-SPC
科 I-SPC
宜 O
设 O
置 O
专 B-SPC
用 I-SPC
手 I-SPC
术 I-SPC
室 I-SPC
。 O
```

## Directory Structure

```
BERT-BiLSTM-CRF/
├── checkpoints/          # Model checkpoints (auto-created)
├── data/                 # Data directory
│   ├── train.txt        # Training data (not included)
│   ├── valid.txt        # Validation data (not included)
│   └── test.txt         # Test data (not included)
├── pretrained_models/    # Pre-trained BERT models
│   └── bert-base-chinese/
├── dataset.py           # Dataset processing
├── model.py             # Model architecture
├── train.py             # Training script
├── predict.py           # Prediction script
└── requirements.txt     # Dependencies
```

## Usage

### Training

```bash
python train.py
```

The training script will:
- Create a timestamped checkpoint directory
- Train the model and save the best performing checkpoint
- Generate training loss curves and evaluation results

### Prediction

For test set evaluation:
```bash
python predict.py test
```

For single sentence prediction:
```bash
python predict.py "Your input sentence here"
```

## Pre-trained Models

Download the `bert-base-chinese` model and place it in the `pretrained_models/` directory. You can obtain it from Hugging Face or other sources.

## Data Preparation

Since the data files are not included in this repository, you need to prepare your own training, validation, and test datasets in the BIO format described above. Place them in the `data/` directory with the following names:
- `train.txt`: Training data
- `valid.txt`: Validation data  
- `test.txt`: Test data

## Configuration

Key hyperparameters can be modified in `train.py`:
- `max_seq_length`: Maximum sequence length (default: 384)
- `batch_size`: Batch size (default: 16)
- `lr`: Learning rate (default: 2e-5)
- `epochs`: Number of training epochs (default: 15)

## Output

Training results are saved in the `checkpoints/`, including:
- `best_model.pt`: Best model checkpoint
- `config.json`: Training configuration
- `loss_curve.png`: Training loss visualization
- `results.txt`: Evaluation results