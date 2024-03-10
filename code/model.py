import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import numpy as np

class DrugProteinInteractionModel(nn.Module):
    def __init__(self, FILTERNUM = 32, SMI_FILTER_SIZE = [4, 6, 8], PRO_FILTER_SIZE = [4, 8, 12], EMBEDDING_DIM = 128, OUTPUT_NODE = 1, FC_SIZE = [1024, 1024, 512]):
        super(DrugProteinInteractionModel, self).__init__()

        # Embedding layers
        #self.smi_embedding = nn.Embedding(SMI_DIM, EMBEDDING_DIM)
        #self.pro_embedding = nn.Embedding(PRO_DIM, EMBEDDING_DIM)

        # Drug Convolution layers
        self.drug_conv1 = nn.Conv1d(EMBEDDING_DIM, FILTERNUM, SMI_FILTER_SIZE[0])
        self.drug_conv2 = nn.Conv1d(FILTERNUM, FILTERNUM * 2, SMI_FILTER_SIZE[1])
        self.drug_conv3 = nn.Conv1d(FILTERNUM * 2, FILTERNUM * 3, SMI_FILTER_SIZE[2])

        # Protein Convolution layers
        self.protein_conv1 = nn.Conv1d(EMBEDDING_DIM, FILTERNUM, PRO_FILTER_SIZE[0])
        self.protein_conv2 = nn.Conv1d(FILTERNUM, FILTERNUM * 2, PRO_FILTER_SIZE[1])
        self.protein_conv3 = nn.Conv1d(FILTERNUM * 2, FILTERNUM * 3, PRO_FILTER_SIZE[2])

        # Attention layer
        self.attention_weight = nn.Linear(FILTERNUM * 3, FILTERNUM * 3)

        # Fully connected layers
        self.fc1 = nn.Linear(FILTERNUM * 3 * 2, FC_SIZE[0])
        self.fc2 = nn.Linear(FC_SIZE[0], FC_SIZE[1])
        self.fc3 = nn.Linear(FC_SIZE[1], FC_SIZE[2])

        # Output layer
        self.output = nn.Linear(FC_SIZE[2], OUTPUT_NODE)

    def forward(self, smi_tensor, pro_tensor):
        #smi_embed = self.smi_embedding(smi_tensor)
        #pro_embed = self.pro_embedding(pro_tensor)

        # Drug Convolution
        smi_conv1 = F.relu(self.drug_conv1(smi_tensor.permute(0, 2, 1)))
        smi_conv2 = F.relu(self.drug_conv2(smi_conv1))
        smi_conv3 = F.relu(self.drug_conv3(smi_conv2))

        # Protein Convolution
        pro_conv1 = F.relu(self.protein_conv1(pro_tensor.permute(0, 2, 1)))
        pro_conv2 = F.relu(self.protein_conv2(pro_conv1))
        pro_conv3 = F.relu(self.protein_conv3(pro_conv2))

        # Attention
        atten1 = F.relu(self.attention_weight(smi_conv3.permute(0, 2, 1)))
        atten2 = F.relu(self.attention_weight(pro_conv3.permute(0, 2, 1)))

        alph = torch.tanh(torch.bmm(atten1, atten2.permute(0, 2, 1)))

        alph_drug = torch.tanh(torch.sum(alph, dim=2))
        alph_protein = torch.tanh(torch.sum(alph, dim=1))

        drug_feature = smi_conv3 * alph_drug.unsqueeze(1)
        protein_feature = pro_conv3 * alph_protein.unsqueeze(1)

        drug_feature = F.max_pool1d(drug_feature, drug_feature.shape[2]).squeeze(2)
        protein_feature = F.max_pool1d(protein_feature, protein_feature.shape[2]).squeeze(2)

        # Concatenation
        pair_feature = torch.cat([drug_feature, protein_feature], dim=1)

        # Fully connected layers
        fc = F.leaky_relu(self.fc1(pair_feature))
        #fc = F.dropout(fc, 0.5)
        #fc = F.leaky_relu(self.fc2(fc))
        #fc = F.dropout(fc, 0.5)
        #fc = F.leaky_relu(self.fc3(fc))

        # Output layer
        #logit = self.output(fc)

        return drug_feature, protein_feature, fc#logit

def positional_encoding(sequence_length, embedding_dim):
    """
    Generates a numpy array of positional encodings.

    :param sequence_length: Length of the sequence
    :param embedding_dim: Depth of the embeddings
    :return: A numpy array of shape (sequence_length, embedding_dim)
    """
    # Initialize the position encoding matrix
    pos_enc = np.zeros((sequence_length, embedding_dim))
    
    # Get position indices and calculate the encoding
    position = np.arange(sequence_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
    
    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term)
    
    return pos_enc
    
class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_enc = positional_encoding(sequence_length, embedding_dim)
    
    def forward(self, embeddings):
        # Convert to PyTorch tensor
        pos_enc_tensor = torch.tensor(self.pos_enc, dtype=embeddings.dtype, device=embeddings.device)
        # Add positional encoding to the embeddings
        embeddings += pos_enc_tensor
        return embeddings

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        assert embedding_dim % self.num_heads == 0

        self.q_linear = nn.Linear(hidden_dim, embedding_dim)
        self.k_linear = nn.Linear(hidden_dim, embedding_dim)
        self.v_linear = nn.Linear(hidden_dim, embedding_dim)
        #self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q, k, v):
        batch_size = q.size(0)
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.embedding_dim // self.num_heads).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.embedding_dim // self.num_heads).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.embedding_dim // self.num_heads).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embedding_dim // self.num_heads)
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v).transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        #output = self.out(output)
        return output#,scores_mat

class SwinTransformer(nn.Module):
    def __init__(self, image_size, patch_size, dim, num_heads, num_layers, input_dim1, input_dim2, hidden_dim, dropout_prob, embed_dim):
        super().__init__()
        self.embedding1_1 = nn.Embedding(input_dim1, hidden_dim)
        self.embedding1_2 = nn.Embedding(input_dim2, hidden_dim)

        self.pos_encoder1 = PositionalEncoding(image_size[1], hidden_dim)
        self.pos_encoder2 = PositionalEncoding(image_size[0], hidden_dim)
        
        self.multihead_layers = nn.ModuleList([MultiHeadAttention(embed_dim, 10, hidden_dim) for _ in range(2)])
        self.relu = nn.Tanh()
        
        self.attentionDTA = DrugProteinInteractionModel()
        self.weight = nn.Parameter(torch.tensor(0.5))
        h, w = image_size
        self.embedding = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(dim * (h // patch_size) * (w // patch_size), 1024)
        self.relu1 = nn.ReLU()
        
        self.fc1 = nn.Sequential(
                                nn.Linear(1024*2, 1024),
                                nn.ReLU(),
                                nn.Dropout(p=dropout_prob),
                                nn.Linear(1024, 1)
                                )
        
        self.fc_1 = nn.Linear(1024*2, 1024)
        self.relu2 = nn.ReLU()
        self.dro = nn.Dropout(p=dropout_prob)
        self.fc_2 = nn.Linear(1024, 1)
    def forward(self, x1, x2):
        x1_embedded1 = self.embedding1_1(x1)
        x2_embedded1 = self.embedding1_2(x2)

        #x1_embedded2 = self.pos_encoder1(x1_embedded1)
        #x2_embedded2 = self.pos_encoder2(x2_embedded1)
        # x3_embedd = torch.matmul(x1_embedded1, x1_embedded1.transpose(-2, -1))

        drug, proteine, logit = self.attentionDTA(x2_embedded1,x1_embedded1)
        for layer in self.multihead_layers:
            x3_embedded1 = layer(x2_embedded1, x1_embedded1, x1_embedded1)  # Q=K=V for self-attention
            
        attention_matrix = self.relu(x3_embedded1).unsqueeze(1)
        x = self.embedding(attention_matrix)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        for block in self.blocks:
            x = block(x)
        x = x.reshape(b, -1)
        x = self.fc(x)
        x = self.relu1(x)
        xx = torch.cat([x, logit], dim=1)
        #xx_ = self.fc1(xx)
        xxx = self.fc_1(xx)
        xxx_ = self.relu2(xxx)
        xxx_ = self.dro(xxx_)
        xxxx = self.fc_2(xxx_)
        return xxxx,attention_matrix
        