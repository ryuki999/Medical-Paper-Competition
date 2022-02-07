import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class BaseModel(nn.Module):
    """BERTのためのBaseModel"""

    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        out = self.sigmoid(out.logits).squeeze()

        return out
