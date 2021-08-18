import torch
from itertools import product
import torch.nn.functional as F
import numpy as np
import math
from .get_laplacian import get_laplacian

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def check_symmetric(a):
    return check_equality(a, a.T)


def check_equality(a, b, rtol=1e-05, atol=1e-03):
    return np.allclose(a, b, rtol=rtol, atol=atol)


def get_adjacency(edge_index):
    return torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]).to(device)).to_dense()

def adj_to_lap(A, remove_self_loops=False):
    if remove_self_loops:
        A.fill_diagonal_(0)
    deg = A.sum(dim=0)
    deg_inv_sqrt = torch.diag(deg.pow_(-0.5))
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return torch.eye(A.shape[0], device=device) - deg_inv_sqrt.mm(A).mm(deg_inv_sqrt)

def create_filter(laplacian, step):
    part1 = torch.diag(torch.ones(laplacian.shape[0], device=device) * 40)
    part2 = (laplacian - torch.diag(torch.ones(laplacian.shape[0], device=device)) * torch.arange(0, 2.1, step, device=device).view(
        -1, 1, 1)).matrix_power(4)
    part3 = torch.eye(laplacian.shape[0], device=device)
    return (part1.matmul(part2) + part3).matrix_power(-2)

def symmetric(X):
    return X.triu() + X.triu(1).transpose(-1, -2)

def get_class_idx(num_classes, idx, y):
    class_idx = [(y == i).nonzero().view(-1).tolist() for i in range(num_classes)]
    class_idx = [list(set(i).intersection(set(idx.tolist()))) for i in class_idx]
    return class_idx

def sample_negative(num_classes, idx, y):
    class_idx = get_class_idx(num_classes, idx, y)
    return list(product(*class_idx))

def sample_positive(num_classes, idx, y):
    return get_class_idx(num_classes, idx, y)

class DictNet(torch.nn.Module):
    def __init__(self, dataset, step=0.1):
        super(DictNet, self).__init__()
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.step = step
        # sample negative examples
        self.C = torch.nn.Parameter(torch.empty(len(torch.arange(0, 2.1, self.step)), 1))

    def reset_parameters(self):
        # self.W.weight = torch.nn.Parameter(get_adjacency(self.data.edge_index))
        torch.nn.init.xavier_normal_(self.C, 0.6)

    def compute_loss(self, embeddings, negative_samples, positive_samples):
        homophily_loss_1 = 0
        for group in negative_samples:
            homophily_loss_1 -= torch.cdist(embeddings[[group]], embeddings[[group]]).mean()

        homophily_loss_2 = 0
        for group in positive_samples:
            homophily_loss_2 += torch.cdist(embeddings[[group]], embeddings[[group]]).mean()
            # homophily_loss_2 += torch.var(y_hat_group, dim=0).mean()

        beta = len(negative_samples)/self.dataset.num_classes

        recovery_loss = 0
        dimensions = torch.sqrt(torch.tensor(float(self.C.shape[0])))
        sparsity_loss = torch.mean((dimensions - self.C.norm(p=1, dim=0)/self.C.norm(p=2, dim=0))/(dimensions - 1))

        return recovery_loss + sparsity_loss + homophily_loss_2 + homophily_loss_1/beta

    def forward(self, data):
        negative_samples = sample_negative(self.num_classes, torch.arange(data.num_graphs), data.y)
        positive_samples = sample_positive(self.num_classes, torch.arange(data.num_graphs), data.y)

        C = torch.nn.functional.normalize(self.C, dim=0, p=2)

        embeddings = torch.zeros(data.num_graphs, data.x.shape[1])

        for i in range(data.num_graphs):
            g = data[i]
            L_index, L_weight = get_laplacian(g.edge_index, normalization='sym')
            L = torch.sparse_coo_tensor(L_index, L_weight).to_dense().to(device)
            D = create_filter(L, self.step).permute(1, 2, 0)
            L_hat = D.matmul(C).squeeze()
            y_hat = (torch.eye(g.num_nodes).to(device) - L_hat).mm(g.x)

            embeddings[i] = y_hat.mean(dim=0)

        return self.compute_loss(embeddings, negative_samples, positive_samples)


def rewire_graph(model, dataset, max_degree=5, threshold=0.1):
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
        A_hat = torch.nn.functional.normalize(A_hat, dim=1, p=2)

        k = min(math.floor(dataset[i].num_edges/dataset[i].num_nodes), max_degree)
        A_one_zero = torch.zeros(A_hat.shape[0], A_hat.shape[1]).to(device) \
            .index_put((torch.arange(A_hat.shape[0]).to(device).repeat_interleave(k), A_hat.abs().topk(k, dim=1, largest=True)[1].view(-1)),
                       torch.tensor(1.).to(device))
        A_one_zero = A_one_zero.masked_fill(A_hat.abs() < threshold, 0)

        edge_index = A_one_zero.nonzero().T
        dataset._data_list[i].edge_index = edge_index
    return dataset
