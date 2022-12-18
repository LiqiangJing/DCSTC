import torch
import torch.nn as nn
from transformers import XLNetModel


class ClassificationXlnet(nn.Module):
    def __init__(self, num_labels=2):
        super(ClassificationXlnet, self).__init__()
        # Load pre-trained bert model
        self.bert = XLNetModel.from_pretrained('xlnet-base-cased')
        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))

    def forward(self, x, length=256):
        # Encode input text
        all_hidden = self.bert(x)[0]

        # pooled_output = torch.mean(all_hidden.last_hidden_state, 1)
        pooled_output = torch.mean(all_hidden, 1)

        # Use linear layer to do the predictions
        predict = self.linear(pooled_output)

        return predict
