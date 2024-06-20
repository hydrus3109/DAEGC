import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.nn import DenseGraphConv, DMoNPooling, GCNConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from layer import GATLayer


class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout_rate =  0.2
        self.pool = DMoNPooling([hidden_size, hidden_size], embedding_size)
        # Initial GAT layer
        self.convs.append(GATConv(num_features, hidden_size, add_self_loops=True))
        self.batch_norms.append(BatchNorm1d(hidden_size))

        # Hidden GAT layers
        for _ in range(3):
            self.convs.append(GATConv(hidden_size, hidden_size, add_self_loops=True))
            self.batch_norms.append(BatchNorm1d(hidden_size))

        # Final GAT layer
        self.convs.append(GATConv(hidden_size, embedding_size, add_self_loops=True))
        self.batch_norms.append(BatchNorm1d(embedding_size))

    def forward(self, x, adj, M):
        edge_index, edge_weight = self.convert(adj)
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms[:-1]):
            x = conv(x, edge_index,edge_weight)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        z = F.normalize(x, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
    def convert(adj):
        return  from_scipy_sparse_matrix(adj)