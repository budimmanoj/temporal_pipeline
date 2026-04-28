# model.py  — BERT + BiLSTM + CRF  (exact match to training code)

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel

label_list = [
    "O",
    "B-DATE",     "I-DATE",
    "B-TIME",     "I-TIME",
    "B-DURATION", "I-DURATION",
    "B-SET",      "I-SET",
    "B-EVENT",    "I-EVENT",
]
label2id = {l: i for i, l in enumerate(label_list)}
id2label  = {i: l for l, i in label2id.items()}


class NERModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert    = BertModel.from_pretrained("bert-base-uncased")
        self.lstm    = nn.LSTM(768, 256, batch_first=True,
                               bidirectional=True, num_layers=2, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(512, len(label_list))
        self.crf     = CRF(len(label_list), batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        x         = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        x, _      = self.lstm(x)
        x         = self.dropout(x)
        emissions = self.fc(x)
        crf_mask  = attention_mask.bool()
        if labels is not None:
            return -self.crf(emissions, labels, mask=crf_mask, reduction="mean")
        return self.crf.decode(emissions, mask=crf_mask)
