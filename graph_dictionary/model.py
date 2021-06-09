import torch
import torch.nn.functional as F
from numpy.random import seed as nseed
from webkb import get_dataset
from citation import get_dataset as get_citation_data, run as run_citation
import numpy as np
import math
from sklearn.linear_model import OrthogonalMatchingPursuit


def check_symmetric(a):
    return check_equality(a, a.T)


def check_equality(a, b, rtol=1e-05, atol=1e-03):
    return np.allclose(a, b, rtol=rtol, atol=atol)


def get_adjacency(edge_index):
    return torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1])).to_dense()


def adj_to_lap(A, remove_self_loops=False):
    if remove_self_loops:
        A.fill_diagonal_(0)
    deg = A.sum(dim=0)
    deg_inv_sqrt = torch.diag(deg.pow_(-0.5))
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return torch.eye(A.shape[0]) - deg_inv_sqrt.mm(A).mm(deg_inv_sqrt)


def create_filter(laplacian, b):
    return (torch.diag(torch.ones(laplacian.shape[0]) * 40).mm(
        (laplacian - torch.diag(torch.ones(laplacian.shape[0]) * b)).matrix_power(4)) + \
            torch.eye(laplacian.shape[0])).pinverse().matrix_power(2)


# # Text for the above functions
# dataset = get_dataset(DATASET, normalize_features=True)
# data= dataset[0]
# edge_index = data.edge_index
# L_index, L_weight = get_laplacian(edge_index, normalization='sym')
# L = torch.sparse_coo_tensor(L_index, L_weight).to_dense()
# assert check_equality(adj_to_lap(get_adjacency(edge_index)), L)


class DictNet(torch.nn.Module):
    def __init__(self, name, n_nonzero_coefs):
        super(DictNet, self).__init__()
        dataset = get_dataset(name, normalize_features=True, self_loop=True)
        data = dataset[0]
        self.dataset = dataset
        self.data = data
        self.n_nonzero_coefs = n_nonzero_coefs

        self.train_mask = data.train_mask[0]
        self.val_mask = data.val_mask[0]
        self.num_filters = math.ceil(2.1 / 0.25)

        self.W = torch.nn.Parameter(torch.Tensor(data.num_nodes, data.num_nodes))
        self.C = None
        self.norm_y = torch.nn.functional.normalize(data.x, p=2, dim=0)

        # sample negative examples
        selectedNodes = set()
        self.negative_samples = []
        for _ in range(math.ceil(len(self.train_mask.nonzero(as_tuple=True)[0]) / dataset.num_classes)):
            nodes = []
            for i in range(dataset.num_classes):
                if data.y[self.train_mask].bincount()[i] <= len(self.negative_samples):
                    continue
                iter = 0
                while True:
                    iter += 1
                    n = np.random.choice((data.y == i).logical_and(self.train_mask).nonzero(as_tuple=True)[0])
                    if n not in selectedNodes:
                        selectedNodes.add(n)
                        nodes.append(n)
                        break
                    if iter > 100:
                        break
            self.negative_samples.append(nodes)

    def reset_parameters(self):
        self.W = torch.nn.Parameter(get_adjacency(self.data.edge_index))
        torch.nn.init.sparse(self.L)

    def omp_step(self, D, y):
        # This will normalize atoms D
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)
        D = torch.nn.functional.normalize(D.detach(), p=2, dim=0)
        omp.fit(D, self.norm_y)
        # TODO: renormalize atoms D
        self.C = torch.FloatTensor(omp.coef_).T

    def compute_loss(self, D, y):
        y_hat = D.mm(self.C)
        loss_1 = 0
        # for group in self.negative_samples:
        #     if len(group) > 0:
        #         loss_1 -= y[group].var()
        loss_2 = 0
        #
        # for i in range(self.dataset.num_classes):
        #     if ((self.data.y == 1).logical_and(mask)!= 0).any():
        #         loss_2 += y[(self.data.y == i).logical_and(mask).nonzero(as_tuple=True)[0]].var()

        # return torch.frobenius_norm(y - y_hat).pow(2) + torch.nn.functional.normalize(self.C, dim=1).norm(p=1) + loss_1 + loss_2
        beta_w = 1
        recovery_loss = torch.frobenius_norm(y - y_hat).pow(2)
        sparsity_loss = beta_w * self.W.norm(p=1)
        # The Frobenius norm of L is added to control the distribution of the edge weights and is inspired by the approach in [27].
        # edge_weight_loss = beta * torch.frobenius_norm(self.L).pow(2)
        edge_weight_loss = 0
        laplacian_loss = 0

        return recovery_loss + sparsity_loss + edge_weight_loss + laplacian_loss

    def forward(self, mask):
        L = adj_to_lap(self.W)
        filters = [create_filter(L, b) for b in torch.arange(0, 2.1, 0.25)]
        D = torch.cat(filters, dim=1)
        y = self.data.x
        self.omp_step(D, y)

        return self.compute_loss(D, y)
