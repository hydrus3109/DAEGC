import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


def get_dataset(dataset):
    create_pyg_dataset()
    #datasets = Planetoid('./dataset', dataset)
    datasets = create_pyg_dataset('dataset.npy','edge_index.npy','edge_weight.npy', 'labels.npy')
    return datasets


def create_pyg_dataset(node_features_file, edge_index_file, edge_weight_file, label_file):
    # Load numpy arrays from files
    node_features = np.load(node_features_file)
    edge_index = np.load(edge_index_file)
    edge_weight = np.load(edge_weight_file)
    labels = np.load(label_file)
    # Convert numpy arrays to torch tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long) 
    # Create a PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight,y=y)
    
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


