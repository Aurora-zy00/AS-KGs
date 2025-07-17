import sys
import torch
from transformers import BertTokenizerFast
from model import BertBiLstmCrf

def build_label_mapping(train_file, valid_file):
    """Build label mappings consistent with training."""
    label_set = set()
    for file in [train_file, valid_file]:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tag = line.split()[-1]
                label_set.add(tag)

    label_set.add("O")
    label_list = ["O"] + sorted([lbl for lbl in label_set if lbl != "O"])
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    num_labels = len(label_to_id)

    return label_to_id, id_to_label, num_labels


def load_test_sentences(test_file):
    """Load sentences from test file."""
    with open(test_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sentences_tokens = []
    tokens = []

    for line in lines:
        line = line.strip()
        if line == "":
            if tokens:
                sentences_tokens.append(tokens)
                tokens = []
        else:
            token = line.split()[0]
            tokens.append(token)

    if tokens:
        sentences_tokens.append(tokens)

    return sentences_tokens


def predict_sequence(model, tokenizer, tokens, id_to_label, device):
    """Predict NER labels for a sequence of tokens."""
    input_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)

    # Create CRF mask
    crf_mask = attention_mask.copy()
    crf_mask[0] = 0  # CLS
    if len(crf_mask) > 1:
        crf_mask[-1] = 0  # SEP

    # Convert to tensors
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.uint8).to(device)
    token_type_ids_tensor = torch.tensor([token_type_ids], dtype=torch.long).to(device)
    crf_mask_tensor = torch.tensor([crf_mask], dtype=torch.uint8).to(device)

    with torch.no_grad():
        emissions = model(input_ids_tensor, attention_mask=attention_mask_tensor, token_type_ids=token_type_ids_tensor)
        pred_indices_list = model.decode(emissions, crf_mask_tensor)

    pred_indices = pred_indices_list[0]
    pred_labels = [id_to_label[idx] for idx in pred_indices]

    return pred_labels


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py [\"test\" | <sentence>]")
        sys.exit(0)

    user_input = " ".join(sys.argv[1:])

    # Configuration
    train_file = "data/train.txt"
    valid_file = "data/valid.txt"
    test_file = "data/test.txt"
    model_path = "checkpoints/best_model.pt"
    pretrained_model = "pretrained_models/bert-base-chinese"

    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
    tokenizer.model_max_length = 512

    # Build label mappings
    label_to_id, id_to_label, num_labels = build_label_mapping(train_file, valid_file)

    # Load model
    model = BertBiLstmCrf(num_labels=num_labels, bert_model_name=pretrained_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    if user_input.strip().lower() == "test":
        # Predict on test set
        sentences_tokens = load_test_sentences(test_file)

        for tokens in sentences_tokens:
            pred_labels = predict_sequence(model, tokenizer, tokens, id_to_label, device)

            for token, label in zip(tokens, pred_labels):
                print(f"{token} {label}")
            print()

    else:
        # Predict on single sentence
        sentence = user_input.strip()
        if not sentence:
            print("Input sentence is empty.")
            sys.exit(0)

        # Tokenize sentence into characters
        tokens = [char for char in sentence if char.strip() != ""]

        pred_labels = predict_sequence(model, tokenizer, tokens, id_to_label, device)

        print(f"Input sentence: {sentence}")
        print("Predicted NER tags:")
        for token, label in zip(tokens, pred_labels):
            print(f"{token} {label}")


if __name__ == "__main__":
    main()