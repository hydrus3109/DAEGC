import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


def get_dataset(dataset):
    datasets = Planetoid('./dataset', dataset)
    #datasets = create_pyg_dataset()
    return datasets


def create_pyg_dataset():
    # Load numpy arrays from files
    node_features = np.load('/content/DAEGC/DAEGC/MNIST10k.npy')
    edge_index = np.load('/content/DAEGC/DAEGC/edgeindex10k.npy')
    labels = np.load('/content/DAEGC/DAEGC/labels10k.npy')
    #min_val = np.min(node_features[:5000])
   # max_val = np.max(node_features[:5000])
    #node_features = (node_features[:5000] - min_val) / (max_val - min_val)
    # Convert numpy arrays to torch tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long) 
    # Create a PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index,y=y)
    
    return data



def data_preprocessing(dataset):
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]), torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_label = dataset.adj

    dataset.adj += torch.eye(dataset.x.shape[0])
    dataset.adj = normalize(dataset.adj, norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)

    return dataset

def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


