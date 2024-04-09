import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, GINConv, Sequential, DenseGCNConv
import torch.nn.functional as F
import torch.nn as nn

class GCNconv(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCNconv, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.fc = torch.nn.Linear(128, num_classes)
        
        self.ret_fc = nn.Sequential(
                                nn.Linear(784, 128),
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(128, 10)
                                )
        self.fusion_weight = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x_ret = x.view(int(len(batch)/784), 784)

        x = F.tanh(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.tanh(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = global_mean_pool(x, batch)

        x = self.fc(x)
        x_ret = self.ret_fc(x_ret)
        
        fusion_weight = torch.sigmoid(self.fusion_weight)
        x_fused = fusion_weight * x + (1 - fusion_weight) * x_ret
        return F.log_softmax(x_fused, dim=1)