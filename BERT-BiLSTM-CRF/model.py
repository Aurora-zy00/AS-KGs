import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF


class BertBiLstmCrf(nn.Module):
    def __init__(self, num_labels, bert_model_name="bert-base-chinese", lstm_hidden=256):
        """
        BERT + BiLSTM + CRF model for Named Entity Recognition.

        Args:
            num_labels: Number of labels including "O"
            bert_model_name: Pre-trained BERT model name
            lstm_hidden: Hidden size of LSTM layer (per direction)
        """
        super(BertBiLstmCrf, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size

        self.lstm = nn.LSTM(
            input_size=self.bert_hidden_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass returning emission scores for each token.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs

        Returns:
            Emission scores of shape (batch_size, seq_len, num_labels)
        """
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        sequence_output = bert_outputs[0]
        sequence_output = self.dropout(sequence_output)

        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)

        emissions = self.classifier(lstm_output)
        return emissions

    def compute_loss(self, emissions, tags, mask):
        """
        Compute negative log-likelihood loss using CRF.

        Args:
            emissions: Emission scores including CLS and SEP positions
            tags: True labels including CLS and SEP positions
            mask: Mask tensor where False indicates CLS/SEP/PAD positions

        Returns:
            Negative log-likelihood loss
        """
        # Remove CLS position
        emissions = emissions[:, 1:, :]
        tags = tags[:, 1:]
        mask = mask[:, 1:]

        log_likelihood = self.crf(emissions, tags, mask=mask, reduction='mean')
        return -log_likelihood

    def decode(self, emissions, mask):
        """
        Decode the best label sequence using CRF.

        Args:
            emissions: Emission scores including CLS and SEP positions
            mask: Mask tensor where False indicates CLS/SEP/PAD positions

        Returns:
            List of predicted label sequences (excluding special tokens)
        """
        # Remove CLS position
        emissions = emissions[:, 1:, :]
        mask = mask[:, 1:]

        best_paths = self.crf.decode(emissions, mask=mask)
        return best_paths