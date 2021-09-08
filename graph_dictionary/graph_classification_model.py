import torch
from itertools import product
import torch.nn.functional as F
import numpy as np
import math
from .get_laplacian import get_laplacian
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from graph_dictionary.utils import ASTNodeEncoder
import pandas as pd
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def check_symmetric(a):
    return check_equality(a, a.T)


def check_equality(a, b, rtol=1e-05, atol=1e-03):
    return np.allclose(a, b, rtol=rtol, atol=atol)


def get_adjacency(edge_index):
    return torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]).to(device)).to_dense()


def create_filter(laplacian, step):
    part1 = torch.diag(torch.ones(laplacian.shape[0], device=device) * 40)
    part2 = (laplacian - torch.diag(torch.ones(laplacian.shape[0], device=device)) * torch.arange(0, 2.1, step, device=device).view(
        -1, 1, 1)).matrix_power(4)
    part3 = torch.eye(laplacian.shape[0], device=device)
    return (part1.matmul(part2) + part3).matrix_power(-2)


def symmetric(X):
    return X.triu() + X.triu(1).transpose(-1, -2)


def get_class_idx(num_classes, idx, y):
    class_idx = ( [ (y[:, j].view(-1) == i).nonzero().view(-1).tolist() for i in range(num_classes) ] for j in range(y.shape[1]) )
    class_idx = [[list(set(i).intersection(set(idx.tolist()))) for i in class_i] for class_i in class_idx]
    class_idx = filter(lambda class_i: len(class_i[0]) != 0 and len(class_i[1]) != 0, class_idx)
    return list(class_idx)


def sample_negative(num_classes, idx, y):
    class_idx = get_class_idx(num_classes, idx, y)
    return [list(product(*class_i)) for class_i in class_idx]


def sample_positive(num_classes, idx, y):
    return get_class_idx(num_classes, idx, y)


class DictNet(torch.nn.Module):
    def __init__(self, dataset, step=0.1, p=2):
        super(DictNet, self).__init__()
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.step = step
        self.p = p
        # sample negative examples
        self.C = torch.nn.Parameter(torch.empty(len(torch.arange(0, 2.1, self.step)), 1))
        self.emb_dim = dataset.data.num_features
        self.node_encoder = None
        self.edge_encoder = None
        if dataset.name.lower() == 'ogbg-ppa':
            self.emb_dim = 300
            self.edge_encoder = torch.nn.Linear(7, self.emb_dim)
            self.node_encoder = torch.nn.Embedding(1, self.emb_dim) # uniform input node embedding
        elif 'mol' in dataset.name.lower():
            self.emb_dim = 300
            self.edge_encoder = BondEncoder(self.emb_dim)
            self.node_encoder = AtomEncoder(self.emb_dim)
        elif dataset.name.lower() == 'ogbg-code2':
            self.emb_dim = 300
            self.edge_encoder = torch.nn.Linear(2, self.emb_dim)
            nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
            nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))
            self.node_encoder = ASTNodeEncoder(self.emb_dim, num_nodetypes = len(nodetypes_mapping['type']), num_nodeattributes = len(nodeattributes_mapping['attr']), max_depth = 20)

    def reset_parameters(self):
        # self.W.weight = torch.nn.Parameter(get_adjacency(self.data.edge_index))
        torch.nn.init.xavier_normal_(self.C, 0.6)

    def compute_loss(self, embeddings, negative_samples, positive_samples):
        homophily_loss_1 = 0
        for class_group in negative_samples:
            homophily_loss_1_group = 0
            for group in class_group:
                homophily_loss_1_group -= torch.cdist(embeddings[[group]], embeddings[[group]]).mean()
            beta = len(class_group)/self.dataset.num_classes + 1e-13
            homophily_loss_1 += homophily_loss_1_group/beta

        homophily_loss_2 = 0
        for class_group in positive_samples:
            for group in class_group:
                homophily_loss_2 += torch.cdist(embeddings[[group]], embeddings[[group]]).mean()
                # homophily_loss_2 += torch.var(y_hat_group, dim=0).mean()

        recovery_loss = 0
        dimensions = torch.sqrt(torch.tensor(float(self.C.shape[0])))
        sparsity_loss = torch.mean((dimensions - self.C.norm(p=1, dim=0)/self.C.norm(p=2, dim=0))/(dimensions - 1))

        return recovery_loss + sparsity_loss + homophily_loss_2 + homophily_loss_1

    def forward(self, data):
        negative_samples = sample_negative(self.num_classes, torch.arange(data.num_graphs), data.y)
        positive_samples = sample_positive(self.num_classes, torch.arange(data.num_graphs), data.y)

        C = torch.nn.functional.normalize(self.C, dim=0, p=self.p)

        embeddings = torch.zeros(data.num_graphs, self.emb_dim)

        for i in range(data.num_graphs):
            g = data[i]
            x = g.x
            if self.node_encoder:
                if self.dataset.name == 'ogbg-code2':
                    node_depth = g.node_depth
                    x = self.node_encoder(x, node_depth.view(-1, ))
                else:
                    x = self.node_encoder(x)
            # if self.edge_encoder:
            #     edge_attr = self.edge_encoder(g.edge_attr)
            #     edge_attr_tensor = torch.zeros(g.num_nodes,g.num_nodes, self.emb_dim, device=device).index_put_(
            #         [g.edge_index[0], g.edge_index[1]], edge_attr)
            if g.edge_index.numel() == 0:
                y_hat = x
            else:
                L_index, L_weight = get_laplacian(g.edge_index, normalization='sym', num_nodes=g.num_nodes)
                L = torch.sparse_coo_tensor(L_index, L_weight).to_dense().to(device)
                D = create_filter(L, self.step).permute(1, 2, 0)
                L_hat = D.matmul(C).squeeze()
                A_hat = torch.eye(g.num_nodes).to(device) - L_hat

                x = A_hat.mm(x)
                # if self.edge_encoder:
                #     weighted_edge_attr_tensor = A_hat.view(g.num_nodes, g.num_nodes, 1)*edge_attr_tensor
                #     index = A_hat.nonzero().T[0]
                #     y_hat = x.index_add(0, index, weighted_edge_attr_tensor[A_hat.nonzero().T[0], A_hat.nonzero().T[1]])
                # else:
                #     y_hat = x
                y_hat = x

            embeddings[i] = y_hat.mean(dim=0)

        return self.compute_loss(embeddings, negative_samples, positive_samples)


def rewire_graph(model, dataset, keep_num_edges=False, threshold=None):
    C = model['C']
    step = 2.1/C.shape[0]

    dictionary = {}
    for i in range(len(dataset)):
        L_index, L_weight = get_laplacian(dataset[i].edge_index, normalization='sym')
        L = torch.zeros(dataset[i].num_nodes, dataset[i].num_nodes, device=L_index.device)
        L = L.index_put((L_index[0], L_index[1]), L_weight).to(device)
        D = create_filter(L, step).permute(1, 2, 0)
        dictionary[i] = D

    C = torch.nn.functional.normalize(C, dim=0, p=2)

    for i in range(len(dataset)):
        D = dictionary[i]
        L = D.matmul(C).squeeze()

        A_hat = torch.eye(L.shape[0]).to(device) - L
        A_hat = torch.nn.functional.normalize(A_hat, dim=(0, 1), p=1)
        if keep_num_edges:
            num_edges = dataset[i].num_edges
            indices = A_hat.view(-1).topk(num_edges)[1]
            rows = indices // A_hat.shape[0]
            cols = indices % A_hat.shape[0]
            edge_index = torch.stack([rows, cols], dim=0)
        else:
            A_hat = A_hat.abs() >= threshold
            edge_index = A_hat.nonzero().T
        dataset._data_list[i].edge_index = edge_index
    return dataset
