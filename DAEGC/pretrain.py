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


def pretrain(dataset):
    model = GAT(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
    ).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = NeighborLoader(dataset, num_neighbors=[30]*2, shuffle=True, num_workers = 2, batch_size =128)
    # data process for adj
    #dataset = utils.data_preprocessing(dataset)
    #adj = dataset.adj.to(device)
    #adj_label = dataset.adj_label.to(device)
    #M = utils.get_M(adj).to(device)
    """
    edge index processing
    """
    edge_index = (dataset.edge_index).to(device)
    # data and label
    x = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()

    for epoch in range(args.max_epoch):
        count = 0
        for data in loader:
            dataset = utils.data_preprocessing(dataset)
            adj = dataset.adj.to(device)
            adj_label = dataset.adj_label.to(device)
            M = utils.get_M(adj).to(device)
            model.train()
            A_pred, z = model(x, adj, M)
            #A_pred, z = model(x, edge_index)
            loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if count % 100 == 0:
                with torch.no_grad():
                    _, z = model(x, adj, M)
                    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                        z.data.cpu().numpy()
                    )
                    acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)
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
