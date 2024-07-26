import argparse
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch_geometric.loader import NeighborLoader
import utils
from model import GAT
from evaluation import eva


class lossmodel(nn.Module):
    def __init__(self, n_clusters, dropout_rate = 0):
        super(lossmodel, self).__init__()
        self.n_clusters = n_clusters
        self.dropout_rate = 0

        # Define the layers
        self.transform = nn.Sequential(
            nn.Linear(n_clusters, n_clusters),
            nn.Dropout(dropout_rate)
        )

        # Initialize weights
        nn.init.orthogonal_(self.transform[0].weight)
        nn.init.zeros_(self.transform[0].bias)

    def forward(self, x):
        return self.transform(x)


def calculate_collapse_loss(number_of_nodes, n_clusters, output, lossize):
    model = lossmodel(n_clusters).to(device)
    assignments = F.softmax(model(output), dim=1)
    cluster_sizes = torch.sum(assignments, dim=0)
    collapse_loss = torch.norm(cluster_sizes) / number_of_nodes * torch.sqrt(torch.tensor(float(n_clusters))) - 1
    return collapse_loss*lossize

def pretrain(dataset):
    model = GAT(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
    ).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = NeighborLoader(dataset, num_neighbors=[10]*2, shuffle=True, num_workers = 2, batch_size =256)
    # data process for adj
    #dataset = utils.data_preprocessing(dataset)
    #adj = dataset.adj.to(device)
    #adj_label = dataset.adj_label.to(device)
    #M = utils.get_M(adj).to(device)
    """
    edge index processing
    """
    """
    edge_index = (dataset.edge_index).to(device)
    # data and label
    x = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()
    """

    for epoch in range(args.max_epoch):
        count = 0
        for data in loader:
            #dataset = utils.data_preprocessing(data)
            #adj = dataset.adj.to(device)
            #adj_label = dataset.adj_label.to(device)
            #M = utils.get_M(adj).to(device)
            edge_index = data.edge_index.to(device)
            edge_weight = data.edge_weight.to(device)
            model.train()
            out, z, totloss = model(data.x.to(device), edge_index, edge_weight)
            #A_pred, z = model(x, edge_index)
            #loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
            out = out.squeeze()
            loss = totloss + calculate_collapse_loss(data.num_nodes,16,out,0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if count % 5 == 0:
                with torch.no_grad():
                    _, z, totloss = model(data.x.to(device), edge_index,edge_weight)
                    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                        z.data.cpu().numpy()
                    )
                    acc, nmi, ari, f1 = eva(data.y.numpy(), kmeans.labels_, epoch)
            count = count + 1
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(), f"./pretrain/predaegc_{args.name}_{epoch}.pkl"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", type=str, default="Citeseer")
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_clusters", default=6, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--embedding_size", default=16, type=int)
    parser.add_argument("--weight_decay", type=int, default=5e-3)
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
    )
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    datasets = utils.get_dataset(args.name)


    if args.name == "Citeseer":
        args.lr = 0.005
        args.k = None
        args.n_clusters = 6
        dataset = datasets[0]
    elif args.name == "Cora":
        args.lr = 0.005
        args.k = None
        args.n_clusters = 7
        dataset = datasets[0]
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 3
        dataset = datasets[0]
    elif args.name == "MNIST":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 10
        dataset = datasets
    else:
        args.k = None
        dataset = datasets[0]

    args.input_dim = dataset.num_features

    print(args)
    pretrain(dataset)
