import math
from collections import OrderedDict
from torch_geometric.utils import get_laplacian
import torch
from torch import nn
import torch.nn.functional as F
from .utils import create_filter


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class CosineSimilarity(nn.Module):
    def __init__(self, eps=1e-8):
        super(CosineSimilarity, self).__init__()
        self.eps = eps

    def forward(self, a):
        a_n, b_n = a.norm(dim=1)[:, None], a.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=self.eps)
        b_norm = a / torch.clamp(b_n, min=self.eps)
        sim_mt = torch.mm(a_norm, b_norm.T)
        return sim_mt


class NodeFeatureSimilarityEncoder(torch.nn.Module):
    def __init__(self, data, layers, name):
        super(NodeFeatureSimilarityEncoder, self).__init__()

        self.name = name
        self.layers = nn.Sequential(
            OrderedDict(
                {
                    'lin1': nn.Linear(data.num_features, layers[0]), 'tanh1': nn.Tanh(),
                    'lin2': nn.Linear(layers[0], layers[1]), 'tanh2': nn.Tanh(),
                    'lin3': nn.Linear(layers[1], layers[2]), 'tanh3': nn.Tanh(),
                    'sim': CosineSimilarity()
                }
            )
        )

        self.outputs = {}

        self.layers[0].register_forward_hook(self.get_activation('lin1'))
        self.layers[2].register_forward_hook(self.get_activation('lin2'))
        self.layers[4].register_forward_hook(self.get_activation('lin3'))
        self.layers[6].register_forward_hook(self.get_activation('sim'))

        self.x = data.x

    def get_activation(self, name):
        def hook(module, input, output):
            self.outputs[name] = output
        return hook

    def forward(self):
        output = self.layers(self.x)
        return output

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def loss(self, a_hat, a):
        mask = self.sample_negative_edge_mask(a)
        masked_a_hat = torch.zeros_like(a).where(mask, torch.relu(a_hat)).triu(diagonal=1)
        return F.mse_loss(masked_a_hat, a.triu(diagonal=1))

    def sample_negative_edge_mask(self, a):
        positive_edges= a.triu(diagonal=1).nonzero()
        negative_edges = (1 - a).triu(diagonal=1).nonzero()
        sampled_indices = torch.randperm(negative_edges.shape[0])[:positive_edges.shape[0]]
        negative_edges = negative_edges[sampled_indices].T
        mask = torch.zeros_like(a).bool().index_put((negative_edges[0], negative_edges[1]), torch.tensor(True))
        return mask


class SpectralSimilarityEncoder(torch.nn.Module):
    def __init__(self, data, x, step, name):
        super(SpectralSimilarityEncoder, self).__init__()
        self.name = name
        self.step = step
        L_index, L_weight = get_laplacian(data.edge_index, normalization='sym')
        self.L = torch.sparse_coo_tensor(L_index, L_weight, device=device).to_dense()
        self.D = create_filter(self.L, self.step).permute(1, 2, 0)

        self.A = None
        self.windows = math.ceil(2.1/self.step)

        self.x = x

        self.layers = nn.Sequential(
            OrderedDict(
                {
                    # 'lin1': torch.nn.Linear(self.windows, self.windows * 2),
                    'lin3': torch.nn.Linear(self.windows, self.windows*2),
                    'sig2': nn.Tanh(),
                    'lin2': nn.Linear(self.windows*2, 1),
                    'tanh': nn.Tanh()
                }
            )
        )

        self.sim = CosineSimilarity()

        self.outputs = {}

        # self.layers[0].register_forward_hook(self.get_activation('lin1'))
        # self.layers[2].register_forward_hook(self.get_activation('lin2'))
        # self.sim.register_forward_hook(self.get_activation('sim'))

    def get_activation(self, name):
        def hook(module, input, output):
            self.outputs[name] = output
        return hook

    def forward(self):
        L_hat = self.layers(self.D).squeeze()
        x_hat = L_hat.mm(self.x)
        return self.sim(x_hat)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def loss(self, a_hat, a):
        mask = self.sample_negative_edge_mask(a)
        masked_a_hat = torch.zeros_like(a).where(mask, torch.relu(a_hat)).triu(diagonal=1)
        return F.mse_loss(masked_a_hat, a.triu(diagonal=1))

    def sample_negative_edge_mask(self, a):
        positive_edges= a.triu(diagonal=1).nonzero()
        negative_edges = (1 - a).triu(diagonal=1).nonzero()
        sampled_indices = torch.randperm(negative_edges.shape[0])[:positive_edges.shape[0]]
        negative_edges = negative_edges[sampled_indices].T
        mask = torch.zeros_like(a).bool().index_put((negative_edges[0], negative_edges[1]), torch.tensor(True))
        return mask
