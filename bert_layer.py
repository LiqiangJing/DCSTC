



import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class ClassificationBert(nn.Module):
    def __init__(self, num_labels=2):
        super(ClassificationBert, self).__init__()
        # Load pre-trained bert model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))

    def forward(self, x, length=256):
        # Encode input text
        all_hidden, pooler, hidden_states = self.bert(x, output_hidden_states=True)
        # print(len(hidden_states))
        # for i in hidden_states:
        #     print(i.shape)
        pooled_output = torch.mean(all_hidden, 1)
        # Use linear layer to do the predictions
        predict = self.linear(pooled_output)
        hidden_states_layer = (hidden_states[7],  hidden_states[9],  hidden_states[12])
        return predict, hidden_states_layer


#
# model = ClassificationBert()
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(inputs['input_ids'])
