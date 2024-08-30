from positional_encoding import PositionalEncoding
import torch.nn as nn
import math

class TransformerClassificationModel(nn.Module):
    
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerClassificationModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model,vocab_size, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        self.dropout = nn.Dropout(p=0.35)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.position_encoding(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x)
        x = x.max(dim=1)[0]
        x = self.fc(x)
        return x