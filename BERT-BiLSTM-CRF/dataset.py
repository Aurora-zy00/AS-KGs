import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class NERDataset(Dataset):
    def __init__(
            self,
            file_path: str,
            tokenizer: BertTokenizerFast,
            label_to_id: dict,
            max_seq_length: int = 512,
    ):
        """
        Initialize NER dataset from BIO format file.

        Args:
            file_path: Path to the data file
            tokenizer: BertTokenizerFast tokenizer
            label_to_id: Label to ID mapping dictionary
            max_seq_length: Maximum sequence length including [CLS] and [SEP]
        """
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_seq_length = max_seq_length

        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id

        self.data = []
        self._load_data(file_path)

    def _load_data(self, file_path):
        """Load and process data from file."""
        tokens, labels = [], []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    if tokens:
                        self._process_sequence(tokens, labels)
                        tokens, labels = [], []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        tokens.append(parts[0])
                        labels.append(parts[-1])

            if tokens:
                self._process_sequence(tokens, labels)

    def _process_sequence(self, tokens, labels):
        """Process a single sequence and add to dataset."""
        max_tokens = self.max_seq_length - 2
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            labels = labels[:max_tokens]

        input_ids = (
                [self.cls_token_id]
                + self.tokenizer.convert_tokens_to_ids(tokens)
                + [self.sep_token_id]
        )

        cls_label_id = self.label_to_id.get("O", 0)
        sep_label_id = self.label_to_id.get("O", 0)
        label_ids = [cls_label_id] + [self.label_to_id[tag] for tag in labels] + [sep_label_id]

        assert len(input_ids) == len(label_ids)
        self.data.append((input_ids, label_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx][0], "labels": self.data[idx][1]}


def collate_fn(batch):
    """
    Custom collate function for batch processing.

    Returns:
        Tuple of (input_ids, attention_mask, token_type_ids, labels, crf_mask)
    """
    max_len = max(len(sample["input_ids"]) for sample in batch)

    batch_input_ids, batch_labels = [], []
    batch_attention_mask, batch_token_type_ids = [], []
    lengths = []

    for sample in batch:
        seq_len = len(sample["input_ids"])
        lengths.append(seq_len)
        pad_len = max_len - seq_len

        batch_input_ids.append(sample["input_ids"] + [0] * pad_len)
        batch_labels.append(sample["labels"] + [sample["labels"][0]] * pad_len)
        batch_attention_mask.append([1] * seq_len + [0] * pad_len)
        batch_token_type_ids.append([0] * max_len)

    batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)
    batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.uint8)
    batch_token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)

    # Create CRF mask: 0 for CLS/SEP/PAD positions, 1 for valid tokens
    crf_mask = batch_attention_mask.clone()
    for i, L in enumerate(lengths):
        if L > 0:
            crf_mask[i, 0] = 0  # CLS
        if L > 1:
            crf_mask[i, L - 1] = 0  # SEP

    return (
        batch_input_ids,
        batch_attention_mask,
        batch_token_type_ids,
        batch_labels,
        crf_mask,
    )