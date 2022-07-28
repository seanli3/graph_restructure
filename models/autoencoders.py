import math
from torch_geometric.utils import get_laplacian
import torch
from torch import nn
import torch.nn.functional as F
from .utils import create_filter, dot_product
from config import DEVICE
import itertools
from models.utils import get_random_signals

device = DEVICE


class NodeFeatureSimilarityEncoder(torch.nn.Module):
    def __init__(self, data, layers, name):
        super(NodeFeatureSimilarityEncoder, self).__init__()

        self.name = name
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
        return dot_product(self())

    def forward(self):
        output = self.layers(self.x)
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
    def __init__(self, data, step, name, exact, with_node_feature=True, with_rand_signal=True):
        super(SpectralSimilarityEncoder, self).__init__()
        self.name = name
        self.step = step
        L_index, L_weight = get_laplacian(data.edge_index, normalization='sym')
        L = torch.sparse_coo_tensor(L_index, L_weight, device=device, size=(data.num_nodes, data.num_nodes)).to_dense()
        self.exact = exact

        if exact:
            e, self.D = torch.linalg.eigh(L)
        else:
            self.D = create_filter(L, self.step).permute(1, 2, 0)
        self.data = data

        if with_rand_signal:
            random_signals = get_random_signals(data.num_nodes, data.num_features)
            if with_node_feature:
                self.x = torch.cat((data.x, random_signals), dim=1)
            else:
                self.x = random_signals
        else:
            self.x = data.x

        self.A = None
        self.windows = math.ceil(2/self.step)

        if exact:
            self.layers = nn.Sequential(
                nn.Linear(data.num_nodes, 32, bias=False)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(self.windows, 64, bias=False),
                nn.ReLU(),
                nn.Linear(64, 32, bias=False),
                nn.ReLU(),
                nn.Linear(32, 1, bias=False),
            )
        self.layers.to(device)

        self.outputs = {}

    def get_activation(self, name):
        def hook(module, input, output):
            self.outputs[name] = output
        return hook

    def sim(self, batch):
        return dot_product(self(batch))

    def dist(self,batch=None):
        return torch.nn.functional.pdist(self(batch), p=2)

    def forward(self, batch, input_weights=None):
        if input_weights is not None:
            if batch is not None:
                x_D = self.D.mul(input_weights)[batch, :, :][:, batch, :]
                x = self.x[batch]
            else:
                x_D = self.D.mul(input_weights)
                x = self.x
        else:
            if batch is not None:
                x_D = self.D[batch, :, :][:, batch, :]
                x = self.x[batch]
            else:
                x_D = self.D
                x = self.x

        L_hat = self.layers(x_D).squeeze()
        if self.exact:
            x_hat = L_hat
        else:
            x_hat = L_hat.matmul(x)
        return x_hat

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                torch.nn.init.normal_(layer.weight)
        self.layers.to(device)

    def sim_mse_loss(self, s, a, mask=None):
        s = s.squeeze().clip(min=0)
        a = a.squeeze().clip(min=0)
        if mask is not None:
            masked_a = a.triu()[mask][:,mask]
            masked_a_hat = s.triu()[mask][:,mask]
            return F.mse_loss(masked_a_hat, masked_a)
        else:
            return F.mse_loss(s.triu(), a.triu())

    def sim_npair_loss(self, x, positive_nodes, negative_nodes, mask=None):
        x = x.squeeze().clip(min=0)
        # https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#contrastive-loss
        node_indices = mask.nonzero().view(-1)
        s_masked = x[node_indices]
        positive_loss = torch.exp(s_masked.view(s_masked.shape[0], 1, s_masked.shape[1]).matmul(
                        s_masked[positive_nodes].permute(0, 2, 1)
                       )).clamp(max=9e23).squeeze()
        negative_loss_sum = s_masked.view(s_masked.shape[0], 1, s_masked.shape[1]).matmul(
                                s_masked[negative_nodes].permute(0, 2, 1)
                            ).sum(dim=2).clamp(max=9e23).squeeze()
        positive_loss = positive_loss.where(torch.isnan(positive_loss).logical_not(), torch.tensor(9e23, device=device))
        negative_loss_sum= negative_loss_sum.where(torch.isnan(negative_loss_sum).logical_not(), torch.tensor(9e23, device=device))
        loss = -torch.log(positive_loss/(positive_loss + negative_loss_sum))
        return loss.sum()

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

    def dist_contrastive_loss(self, d, dist_diff_indices, eps=0.05):
        triu_indices = torch.triu_indices(row=self.data.num_nodes, col=self.data.num_nodes, offset=1, device=device)
        dist = torch.zeros(self.data.num_nodes, self.data.num_nodes, device=device)
        dist[triu_indices[0], triu_indices[1]] = d.pow(2)
        dist = dist + dist.T
        intra_class_edge_indices = dist_diff_indices[:, 0].T
        inter_class_edge_indices = dist_diff_indices[:, 1].T
        intra_class_dist = dist[intra_class_edge_indices[0], intra_class_edge_indices[1]]
        inter_class_dist = (eps - dist[inter_class_edge_indices[0], inter_class_edge_indices[1]]).clamp(min=0)
        l = (intra_class_dist + inter_class_dist).sum()
        return l

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
        self.outputs = {}

    def get_activation(self, name):
        def hook(module, input, output):
            self.outputs[name] = output
        return hook

    def sim(self):
        return dot_product(self())

    def forward(self):
        x_hat = self.D.mm(self.x)
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

