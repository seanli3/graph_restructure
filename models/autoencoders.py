import math
from collections import OrderedDict
from torch_geometric.utils import get_laplacian, negative_sampling
import torch
from torch import nn
import torch.nn.functional as F
import shap
from .utils import create_filter
from config import USE_CUDA, DEVICE
import itertools

device = DEVICE


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
        self.layers.to(device)
        self.outputs = {}
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
        if mask is not None:
            masked_a = a[mask][:,mask]
            masked_a_hat = a_hat[mask][:,mask]
            return F.mse_loss(masked_a_hat, masked_a)
        else:
            return F.mse_loss(a_hat, a)


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
        self.windows = math.ceil(2./self.step)

        self.layers = nn.Sequential(
            nn.Linear(self.windows, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 64, bias=False),
            nn.Tanh(),
            nn.Linear(64, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 1, bias=False),
            nn.ReLU(),
        )
        self.layers.to(device)
        self.x = x


        self.linear = nn.Linear(x.shape[1], 16, bias=False)

        self.dist = torch.nn.functional.pdist

        self.outputs = {}

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
        if mask is not None:
            masked_a = a[mask][:,mask]
            masked_a_hat = a_hat[mask][:,mask]
            return F.mse_loss(masked_a_hat, masked_a)
        else:
            return F.mse_loss(a_hat, a)

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
        if mask is not None:
            masked_a = a[mask][:,mask]
            masked_a_hat = a_hat[mask][:,mask]
            return F.mse_loss(masked_a_hat, masked_a)
        else:
            return F.mse_loss(a_hat, a)

