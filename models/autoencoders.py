import math
from torch_geometric.utils import get_laplacian
import torch
from torch import nn
import torch.nn.functional as F
from .utils import create_filter, cosine_sim
from config import DEVICE
import itertools

device = DEVICE


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
        a_hat = a_hat.squeeze().clip(min=0)
        a = a.squeeze().clip(min=0)
        if mask is not None:
            masked_a = a.triu()[mask][:,mask]
            masked_a_hat = a_hat.triu()[mask][:,mask]
            return F.mse_loss(masked_a_hat, masked_a)
        else:
            return F.mse_loss(a_hat.triu(), a.triu())


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
            nn.Linear(self.windows, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            nn.Tanh(),
            nn.Linear(16, 1, bias=False),
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
        a_hat = a_hat.squeeze().clip(min=0)
        a = a.squeeze().clip(min=0)
        if mask is not None:
            masked_a = a.triu()[mask][:,mask]
            masked_a_hat = a_hat.triu()[mask][:,mask]
            return F.mse_loss(masked_a_hat, masked_a)
        else:
            return F.mse_loss(a_hat.triu(), a.triu())

    def explain(self, target):
        import shap
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
        a_hat = a_hat.squeeze().clip(min=0)
        a = a.squeeze().clip(min=0)
        if mask is not None:
            masked_a = a.triu()[mask][:,mask]
            masked_a_hat = a_hat.triu()[mask][:,mask]
            return F.mse_loss(masked_a_hat, masked_a)
        else:
            return F.mse_loss(a_hat.triu(), a.triu())

