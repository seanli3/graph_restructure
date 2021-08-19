import torch
import torch.nn.functional as F
from numpy.random import seed as nseed
from .get_laplacian import get_laplacian
import numpy as np
import math
from torch import nn

def check_symmetric(a):
    return check_equality(a, a.T)

def check_equality(a, b, rtol=1e-05, atol=1e-03):
    return np.allclose(a, b, rtol=rtol, atol=atol)


def get_adjacency(edge_index):
    return torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1])).to_dense()


def create_filter(laplacian, b):
    return (torch.diag(torch.ones(laplacian.shape[0]) * 40).mm(
        (laplacian - torch.diag(torch.ones(laplacian.shape[0]) * b)).matrix_power(4)) + \
            torch.eye(laplacian.shape[0])).matrix_power(-2)

def create_filter(laplacian, step):
    part1 = torch.diag(torch.ones(laplacian.shape[0], device=device) * 40)
    part2 = (laplacian - torch.diag(torch.ones(laplacian.shape[0], device=device)) * torch.arange(0, 2.1, step, device=device).view(
        -1, 1, 1)).matrix_power(4)
    part3 = torch.eye(laplacian.shape[0], device=device)
    return (part1.matmul(part2) + part3).matrix_power(-2)


def symmetric(X):
    return X.triu() + X.triu(1).transpose(-1, -2)


def sample_negative(num_classes, mask, y):
    negative_samples = []
    selectedNodes = set()
    for _ in range(len(mask.nonzero(as_tuple=True)[0])):
        nodes = []
        for i in range(num_classes):
            if y[mask].bincount()[i] <= len(negative_samples):
                continue
            iter = 0
            while True:
                iter += 1
                n = np.random.choice((y.cpu() == i).logical_and(mask.cpu()).nonzero(as_tuple=True)[0])
                if n not in selectedNodes:
                    selectedNodes.add(n)
                    nodes.append(n)
                    break
                if iter > 100:
                    break
        negative_samples.append(nodes)
    return list(filter(lambda s: len(s) > 1, negative_samples))


class DictNet(torch.nn.Module):
    def __init__(self, name, split, cuda=False):
        super(DictNet, self).__init__()
        if name.lower() in ['cora', 'citeseer', 'pubmed']:
            from citation import get_dataset
        else:
            from webkb import get_dataset

        dataset = get_dataset(name, normalize_features=True, cuda=cuda)
        data = dataset[0]
        self.dataset = dataset
        self.data = data

        if name.lower() in ['cora', 'citeseer', 'pubmed']:
            self.train_mask = data.train_mask
            self.val_mask = data.val_mask
        else:
            self.train_mask = data.train_mask[split]
            self.val_mask = data.val_mask[split]

        L_index, L_weight = get_laplacian(data.edge_index, normalization='sym')
        self.L = torch.sparse_coo_tensor(L_index, L_weight).to_dense()
        # self.W = torch.nn.Embedding(data.num_nodes, data.num_nodes)
        self.D = create_filter(self.L, self.step).permute(1, 2, 0)
        self.C = torch.nn.Parameter(torch.empty(len(torch.arange(0, 2.1, self.step)), 1))

        self.A = None
        # self.norm_y = torch.nn.functional.normalize(data.x, p=2, dim=0)
        self.loss = torch.nn.MSELoss()
        self.I = torch.eye(dataset[0].num_nodes)

        # sample negative examples
        self.train_negative_samples = sample_negative(dataset.num_classes, self.train_mask, data.y)
        self.val_negative_samples = sample_negative(dataset.num_classes, self.val_mask, data.y)

    def reset_parameters(self):
        # self.W.weight = torch.nn.Parameter(get_adjacency(self.data.edge_index))
        torch.nn.init.xavier_normal_(self.C, 0.6)

    def compute_loss(self, D, x, mask):
        # C = self.C
        C = torch.nn.functional.normalize(self.C, dim=0, p=2)
        L_hat = D.matmul(C).squeeze()
        y_hat = (self.I - L_hat).mm(self.data.x)
        homophily_loss_1 = 0
        if self.training:
            negative_samples = self.train_negative_samples
        else:
            negative_samples = self.val_negative_samples

        for group in negative_samples:
            y_hat_group = y_hat[group]
            homophily_loss_1 -= torch.cdist(y_hat_group, y_hat_group).mean()

        homophily_loss_2 = 0
        for i in range(self.dataset.num_classes):
            if ((self.data.y == i).logical_and(mask)).any():
                y_hat_group = y_hat[(self.data.y == i).logical_and(mask)]
                homophily_loss_2 += torch.cdist(y_hat_group, y_hat_group).mean()
                # homophily_loss_2 += torch.var(y_hat_group, dim=0).mean()

        beta = len(negative_samples)/self.dataset.num_classes

        # return torch.frobenius_norm(y - y_hat).pow(2) + torch.nn.functional.normalize(self.C, dim=1).norm(p=1) + loss_1 + loss_2
        beta_w = 1
        beta_e = 1
        beta_h = 1
        # recovery_loss = self.loss(y, y_hat)
        recovery_loss = 0
        # sparsity_loss = beta_w * self.A.norm(p=1, dim=1).mean()
        # sparsity_loss = beta_w * self.C.weight.norm(p=1)
        dimensions =torch.sqrt(torch.tensor(float(C.shape[0])))
        sparsity_loss = torch.mean((dimensions - C.norm(p=1, dim=0)/C.norm(p=2, dim=0))/(dimensions - 1))
        # The Frobenius norm of L is added to control the distribution of the edge weights and is inspired by the approach in [27].
        # edge_weight_loss = beta_e * torch.frobenius_norm(self.L, dim=1).pow(2).mean()
        edge_weight_loss =0
        laplacian_loss = 0

        return recovery_loss + sparsity_loss + edge_weight_loss + laplacian_loss + homophily_loss_2 + homophily_loss_1/beta

    def forward(self, mask):
        x = self.data.x

        return self.compute_loss(self.D, x, mask)
