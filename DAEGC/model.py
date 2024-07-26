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
from torch_geometric.nn import GINConv
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

    def forward(self, x, edge_index, edge_weight):
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms[:-1]):
            x = conv(x, edge_index, edge_attr = edge_weight)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        adj = to_dense_adj(edge_index)
        out, _, adj, sp1, o1, c1 = self.pool(x, adj)
        x = self.convs[-1](x, edge_index, edge_weight)
        z = F.normalize(x, p=2, dim=1)
        #A_pred = self.dot_product_decode(z)
        return out, z, sp1+o1+c1

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

"""

class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GATLayer(num_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
 
"""

"""
class GINMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GINMLP, self).__init__()

        # Initialize layers
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):

        x = self.layer1(x)
        x = F.relu(self.batch_norm(x))
        x = self.layer2(x)
        return x
class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout_rate =  0.2

        # Define a simple MLP for GINConv
        mlp = GINMLP(num_features,hidden_size,hidden_size)

        # Initial GIN layer
        self.convs.append(GINConv(mlp, train_eps=True))
        self.batch_norms.append(BatchNorm1d(hidden_size))

        # Hidden GIN layers
        for _ in range(3):
            mlp = GINMLP(hidden_size,hidden_size,hidden_size)
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(BatchNorm1d(hidden_size))

        # Final GIN layer
        mlp = GINMLP(hidden_size,hidden_size,embedding_size)
        self.convs.append(GINConv(mlp, train_eps=True))
        self.batch_norms.append(BatchNorm1d(embedding_size))

    def forward(self, x, adj, M):
        edge_index, edge_weight = self.convert(adj)
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms[:-1]):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = ReLU()(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        z = F.normalize(x, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z    
    
    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
    def convert(self, adj):
        row_indices, col_indices = torch.nonzero(adj, as_tuple=True)
        edge_index = torch.stack([row_indices, col_indices], dim=0)
        edge_weight = adj[row_indices, col_indices]
        return edge_index, edge_weight
"""
