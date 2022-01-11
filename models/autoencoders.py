import math
from collections import OrderedDict
from torch_geometric.utils import get_laplacian, negative_sampling
import torch
from torch import nn
import torch.nn.functional as F
import shap
from .utils import create_filter
from config import USE_CUDA
import itertools

device = torch.device('cuda') if torch.cuda.is_available() and USE_CUDA else torch.device('cpu')


def cosine_sim(a):
    eps = 1e-8
    if len(a.shape) == 2:
        a = a.view(1, a.shape[0], a.shape[1])
    a_n, b_n = a.norm(dim=2)[:, None], a.norm(dim=2)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps).view(a.shape[0], a.shape[1], 1)
    b_norm = a / torch.clamp(b_n, min=eps).view(a.shape[0], a.shape[1], 1)
    sim_mt = a_norm.matmul(b_norm.permute(0, 2, 1))
    return sim_mt



class NodeFeatureSimilarityEncoder(torch.nn.Module):
    def __init__(self, data, layers, name):
        super(NodeFeatureSimilarityEncoder, self).__init__()

        self.name = name
        self.dist = torch.nn.functional.pdist
        layer_size = [data.num_features] + layers
        self.layers = nn.Sequential(
            *list(itertools.chain(*[
                [nn.Linear(layer_size[l], layer_size[l + 1], bias=False)] for l in range(len(layers))
            ]))
        )

        self.outputs = {}

        # self.layers[0].register_forward_hook(self.get_activation('lin1'))
        # self.layers[2].register_forward_hook(self.get_activation('lin2'))
        # self.layers[4].register_forward_hook(self.get_activation('lin3'))
        # self.layers[6].register_forward_hook(self.get_activation('sim'))

        self.x = data.x
        self.data = data

    def get_activation(self, name):
        def hook(module, input, output):
            self.outputs[name] = output
        return hook

    def sim(self):
        return cosine_sim(self())

    def forward(self):
        output = self.layers(self.x)
        output = torch.nn.functional.normalize(output, p=2, dim=1)
        return output

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                torch.nn.init.normal_(layer.weight)
        self.layers.to(device)

    def loss(self, a_hat, a, mask=None):
        a_hat = a_hat.squeeze()
        # negative_mask = self.sample_negative_edge_mask(self.data.edge_index)
        # positive_mask = a > 0
        # masked_a_hat = torch.relu(a_hat).where(negative_mask.logical_or(positive_mask), torch.zeros(1, device=device)).triu(diagonal=1)
        # masked_a = torch.zeros_like(a, device=device).where(positive_mask.logical_and(negative_mask.logical_not()), torch.relu(a).triu(diagonal=1))
        # return F.mse_loss(masked_a_hat, masked_a)
        if mask is not None:
            masked_a = a[mask][:,mask]
            masked_a_hat = a_hat[mask][:,mask]
            return F.mse_loss(masked_a_hat, masked_a)
        else:
            return F.mse_loss(a_hat, a)

    def sample_negative_edge_mask(self, edge_index):
        positive_edges = edge_index.shape[1]
        negative_edges = negative_sampling(edge_index, num_nodes=self.data.num_nodes, method="sparse",
                                           num_neg_samples=positive_edges, force_undirected=True)
        mask = torch.zeros(self.data.num_nodes, self.data.num_nodes, device=device).bool().index_put((negative_edges[0], negative_edges[1]),
                                                                   torch.tensor(True, device=device))
        return mask


class SpectralSimilarityEncoder(torch.nn.Module):
    def __init__(self, data, x, step, name):
        super(SpectralSimilarityEncoder, self).__init__()
        self.name = name
        self.step = step
        L_index, L_weight = get_laplacian(data.edge_index, normalization='sym')
        L = torch.sparse_coo_tensor(L_index, L_weight, device=device, size=(data.num_nodes, data.num_nodes)).to_dense()
        self.D = create_filter(L, self.step).permute(1, 2, 0)
        self.data = data

        self.A = None
        self.windows = math.ceil(2.1/self.step)

        self.x = x

        self.layers = nn.Sequential(
            OrderedDict(
                {
                    'lin1': torch.nn.Linear(self.windows, self.windows),
                    'sig1': nn.Tanh(),
                    # 'lin2': torch.nn.Linear(self.windows*2, self.windows),
                    # 'sig2': nn.Tanh(),
                    'lin3': nn.Linear(self.windows, 1, bias=False),
                    'tanh': nn.Tanh()
                }
            )
        )

        self.linear = nn.Linear(x.shape[1], 16, bias=False)

        self.dist = torch.nn.functional.pdist

        self.outputs = {}

        # self.layers[0].register_forward_hook(self.get_activation('lin1'))
        # self.layers[2].register_forward_hook(self.get_activation('lin2'))
        # self.sim.register_forward_hook(self.get_activation('sim'))

    def get_activation(self, name):
        def hook(module, input, output):
            self.outputs[name] = output
        return hook

    def sim(self):
        return cosine_sim(self())

    def forward(self, input_weights=None):
        if input_weights is not None:
            x_D = self.D.mul(input_weights)
        else:
            x_D = self.D

        L_hat = self.layers(x_D).squeeze()
        # L_hat = self.layers(x_D.permute(0, 1, 2).view(self.windows, 1, -1)).squeeze()
        x_hat = L_hat.matmul(self.x)
        # return self.dist(x_hat, p=1)
        x_hat = torch.nn.functional.normalize(x_hat, p=2, dim=1)
        return x_hat

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                torch.nn.init.normal_(layer.weight)
        self.linear.reset_parameters()
        torch.nn.init.normal_(self.linear.weight)
        self.layers.to(device)

    def loss(self, a_hat, a, mask=None):
        a_hat = a_hat.squeeze()
        # negative_mask = self.sample_negative_edge_mask(self.data.edge_index)
        # positive_mask = a > 0
        # masked_a_hat = torch.relu(a_hat).where(negative_mask.logical_or(positive_mask), torch.zeros(1, device=device)).triu(diagonal=1)
        # masked_a = torch.zeros_like(a, device=device).where(positive_mask.logical_and(negative_mask.logical_not()), torch.relu(a).triu(diagonal=1))
        # return F.mse_loss(masked_a_hat, masked_a)
        if mask is not None:
            masked_a = a[mask][:,mask]
            masked_a_hat = a_hat[mask][:,mask]
            return F.mse_loss(masked_a_hat, masked_a)
        else:
            return F.mse_loss(a_hat, a)

    def sample_negative_edge_mask(self, edge_index):
        positive_edges = edge_index.shape[1]
        negative_edges = negative_sampling(edge_index, num_nodes=self.data.num_nodes, method="sparse",
                                           num_neg_samples=positive_edges, force_undirected=True)
        mask = torch.zeros(self.data.num_nodes, self.data.num_nodes, device=device).bool().index_put((negative_edges[0], negative_edges[1]),
                                                                   torch.tensor(True, device=device))
        return mask

    def explain(self, target):
        explainer = shap.explainers.Permutation(self(), target)
        shap_values = explainer(target)
        shap.plots.bar(shap_values)


class LowPassSimilarityEncoder(torch.nn.Module):
    def __init__(self, data, x, name):
        super(LowPassSimilarityEncoder, self).__init__()
        self.name = name
        L_index, L_weight = get_laplacian(data.edge_index, normalization='sym')
        L = torch.sparse_coo_tensor(L_index, L_weight, device=device).to_dense()
        self.D = torch.eye(data.num_nodes, data.num_nodes, device=device) - L
        self.data = data
        self.x = x

        self.dist = torch.nn.functional.pdist

        self.outputs = {}

        # self.sim.register_forward_hook(self.get_activation('sim'))

    def get_activation(self, name):
        def hook(module, input, output):
            self.outputs[name] = output
        return hook

    def sim(self):
        return cosine_sim(self())

    def forward(self):
        x_hat = self.D.mm(self.x)
        x_hat = torch.nn.functional.normalize(x_hat, p=2, dim=1)
        return x_hat

    def reset_parameters(self):
        pass

    def loss(self, a_hat, a, mask=None):
        # negative_mask = self.sample_negative_edge_mask(self.data.edge_index)
        # positive_mask = a > 0
        # masked_a_hat = torch.relu(a_hat).where(negative_mask.logical_or(positive_mask), torch.zeros(1, device=device)).triu(diagonal=1)
        # masked_a = torch.zeros_like(a, device=device).where(positive_mask.logical_and(negative_mask.logical_not()), torch.relu(a).triu(diagonal=1))
        # return F.mse_loss(masked_a_hat, masked_a)
        if mask is not None:
            masked_a = a[mask][:,mask]
            masked_a_hat = a_hat[mask][:,mask]
            return F.mse_loss(masked_a_hat, masked_a)
        else:
            return F.mse_loss(a_hat, a)

    def sample_negative_edge_mask(self, edge_index):
        positive_edges = edge_index.shape[1]
        negative_edges = negative_sampling(edge_index, num_nodes=self.data.num_nodes, method="sparse",
                                           num_neg_samples=positive_edges, force_undirected=True)
        mask = torch.zeros(self.data.num_nodes, self.data.num_nodes, device=device).bool().index_put((negative_edges[0], negative_edges[1]),
                                                                   torch.tensor(True, device=device))
        return mask