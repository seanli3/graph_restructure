import math
from torch_geometric.utils import get_laplacian
import torch
from torch import nn
import torch.nn.functional as F
from .utils import create_filter, dot_product, create_filter_sparse
from config import DEVICE
from models.utils import get_random_signals

device = DEVICE


class SpectralSimilarityEncoder(torch.nn.Module):
    def __init__(self, data, step, exact, with_node_feature=True, with_rand_signal=True, sparse=False):
        super(SpectralSimilarityEncoder, self).__init__()
        self.step = step
        L_index, L_weight = get_laplacian(data.edge_index, normalization='sym')
        L = torch.sparse_coo_tensor(L_index, L_weight, device=device, size=(data.num_nodes, data.num_nodes))
        if not sparse:
            L = L.to_dense()
        self.exact = exact
        self.sparse = sparse

        if exact:
            e, D = torch.linalg.eigh(L)
        else:
            if sparse:
                D = create_filter_sparse(L, self.step)
            else:
                D = create_filter(L, self.step).permute(1, 0, 2)
        del L

        if with_rand_signal:
            random_signals = get_random_signals(data.num_nodes, data.num_features)
            if with_node_feature:
                x = torch.cat((data.x, random_signals), dim=1)
            else:
                x = random_signals
        else:
            x = data.x

        self.windows = math.ceil(2/self.step)

        if sparse:
            self.D = []
            for Di in D:
                D_x = Di.mm(x)
                D_x = D_x.where(D_x.abs() > 1e-8, torch.tensor(0., device=device))
                self.D.append(D_x.to_sparse_coo())
            self.D = torch.stack(self.D, 2)
            self.D = self.D.to_dense()
        else:
            self.D = D.matmul(x).permute(0, 2, 1)
        del D

        self.w1 = nn.Parameter(torch.empty(self.windows, 64, device=device))
        self.w2 = nn.Parameter(torch.empty(64, 32, device=device))
        self.w3 = nn.Parameter(torch.empty(32, 1, device=device))

    def sim(self, batch):
        return dot_product(self(batch))

    def dist(self, D_batch=None):
        emb = self(D_batch)
        d = torch.nn.functional.pdist(emb, p=2)
        return d

    def forward(self, D_batch):
        if D_batch is not None:
            x = self.D[D_batch].matmul(self.w1)
        else:
            x = self.D.matmul(self.w1)
        x = nn.functional.relu(x)
        x = x.matmul(self.w2)
        x = nn.functional.relu(x)
        x = x.matmul(self.w3)
        return x.squeeze()

    def reset_parameters(self):
        nn.init.normal_(self.w1)
        nn.init.normal_(self.w2)
        nn.init.normal_(self.w3)

    def dist_triplet_loss(self, d, dist_diff_indices, batch_size, eps=0.05):
        triu_indices = torch.triu_indices(row=batch_size, col=batch_size, offset=1, device=device)
        dist = torch.zeros(batch_size, batch_size, device=device)
        dist[triu_indices[0], triu_indices[1]] = d.pow(2)
        dist = dist + dist.T
        intra_class_edge_indices = dist_diff_indices[:, 0].T
        inter_class_edge_indices = dist_diff_indices[:, 1].T
        intra_class_dist = dist[intra_class_edge_indices[0], intra_class_edge_indices[1]]
        inter_class_dist = dist[inter_class_edge_indices[0], inter_class_edge_indices[1]]
        l = (intra_class_dist-inter_class_dist + eps).clamp(min=0).sum()
        return l
