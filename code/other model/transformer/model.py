import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoderModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, input_dim1):
        super(TransformerEncoderModel, self).__init__()
        self.embedding1_1 = nn.Embedding(input_dim1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.d_model = d_model
        #self.resize_transform = transforms.Resize((14, 14))
        self.fusion_weight = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        
        self.fc = nn.Linear(784*d_model,10)
        self.fc1 = nn.Sequential(
                                nn.Linear(784, 64),
                                nn.ReLU(),
                                nn.Dropout(p=dropout),
                                nn.Linear(64, 10)
                                )
        #self.fc_1 = nn.Linear(64, 10)
        #self.dro = nn.Dropout(p=dropout)

    def forward(self, src, src_key_padding_mask=None):
        b,_,_ = src.shape
        #x = src.view(b,1,28,28).float()
        x1 = src.view(b,-1)
        x2 = src.view(b,-1).float()
        x = self.fc1(x2)
        
        src = self.embedding1_1(x1) * math.sqrt(self.d_model)
        
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = output.view(b,-1)
        output = self.fc(output)
        fusion_weight = torch.sigmoid(self.fusion_weight)
        output = fusion_weight * x + (1 - fusion_weight) * output
        return output
