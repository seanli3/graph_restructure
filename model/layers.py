import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_filter import Filter
from torch_sparse import spmm
import pygsp
from math import sqrt


class GaussFilter(nn.Module):
    def __init__(self, k: int):
        super(GaussFilter, self).__init__()
        self.k = k
        self.reset_parameters()
        self.centers = torch.linspace(0, 2, k)

    def reset_parameters(self):
        self.bandwidths = nn.Parameter(torch.full([1, self.k], sqrt(2/self.k)))
        self.gains = nn.Parameter(torch.ones(1, self.k))
        self.amps = nn.Parameter(torch.full([1, self.k], 8.0))

    def forward(self, x):
        return self.gains.mul(torch.exp(-self.amps*(x.view(-1, 1) - self.centers).pow(2)/self.bandwidths.pow(2)))


class HeatFilter(nn.Module):
    def __init__(self, out_channels, lmax, device, tau=0.2):
        super(HeatFilter, self).__init__()
        self.device = device
        self.lmax = lmax
        # self.t = nn.Parameter(torch.empty(1, out_channels, device=self.device))
        self.t = tau
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.uniform_(self.t)
        pass

    def forward(self, x):
        return torch.exp(-self.t * x.view(-1,1) / self.lmax)


class AnalysisFilter(nn.Module):
    def __init__(self, out_channel, device):
        super(AnalysisFilter, self).__init__()
        self.device=device
        self.out_channel = out_channel
        self.layers = nn.Sequential(nn.Linear(1, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, 64),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(64, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, out_channel),
                                    nn.ReLU(inplace=True))
        self.to(device)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.to(self.device)

    def forward(self, x):
        return self.layers(x.view(-1,1))


class GraphSpectralFilterLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, G, in_features, out_features, dropout, out_channels, device="cpu", concat=True,
                 order=16, pre_training=False, filter="analysis", method="chebyshev", k=5, threshold=None,
                 Kb=18, Ka=2, Tmax=200, tau=0.2):
        super(GraphSpectralFilterLayer, self).__init__()
        self.G = G
        self.k = k
        self.tau = tau
        self.threshold = threshold
        self.Kb = Kb
        self.Ka = Ka
        self.Tmax = Tmax
        self.device=device
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.out_channels = out_channels
        self.method = method
        self.pre_training = pre_training
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # self.mlp = nn.Sequential(
        #     nn.Linear(in_features, 512, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(512, out_features, bias=False),
        # )
        self.order = order
        self.N = G.n_vertices
        self.filter_type = filter
        if self.filter_type == 'analysis':
            self.filter_kernel = AnalysisFilter(out_channel=self.out_channels, device=self.device)
        else:
            self.filter_kernel = HeatFilter(out_channels=self.out_channels, device=self.device, lmax=self.G.lmax,
                                            tau=self.tau)
        self.filter = Filter(self.G, self.filter_kernel, nf=self.out_channels, device=self.device, order=self.order,
                             method=self.method, Kb=self.Kb, Ka=self.Ka, Tmax=self.Tmax)
        self.concat = concat

        self.to(self.device)
        self.linear.to(self.device)
        self.filter_kernel.to(self.device)

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.linear.reset_parameters()
        # for layer in self.mlp:
        #     if hasattr(layer, 'reset_parameters'):
        #         layer.reset_parameters()
        if self.filter_type == 'analysis':
            self.filter_kernel = AnalysisFilter(out_channel=self.out_channels, device=self.device)
        else:
            self.filter_kernel = HeatFilter(out_channels=self.out_channels, device=self.device, lmax=self.G.lmax, tau=self.tau)
        self.filter = Filter(self.G, self.filter_kernel, nf=self.out_channels, device=self.device, order=self.order,
                             method=self.method, Kb=self.Kb, Ka=self.Ka, Tmax=self.Tmax)

        if self.pre_training:
            if self.out_channels > 1:
                itersine = pygsp.filters.Itersine(self.G, self.out_channels)
            else:
                itersine = pygsp.filters.Heat(self.G, self.out_channels)
            k_optimizer = torch.optim.Adam(self.filter_kernel.parameters(), lr=5e-4, weight_decay=1e-5)
            x = torch.rand(1000, device=self.device).view(-1, 1) * 2
            y = torch.FloatTensor(itersine.evaluate(x.cpu().view(-1))).to(self.device).T
            val_x = torch.rand(1000, device=self.device).view(-1, 1) * 2
            val_y = torch.FloatTensor(itersine.evaluate(val_x.cpu().view(-1))).to(self.device).T
            for _ in range(2000):
                self.filter_kernel.train()
                k_optimizer.zero_grad()
                predictions = self.filter_kernel(x)
                loss = F.mse_loss(input=predictions, target=y, reduction="mean")
                # if _ % 1000 == 0:
                #     self.filter_kernel.eval()
                #     val_predictions = self.filter_kernel(val_x)
                #     val_loss = F.mse_loss(input=val_predictions, target=val_y, reduction="mean")
                #     print(
                #         'kernel training epoch {} loss {} validation loss {}'.format(_, str(loss.item()),
                #                                                                      str(val_loss.item())))
                #     self.filter_kernel.train()
                loss.backward()
                k_optimizer.step()
        self.to(self.device)
        self.linear.to(self.device)
        self.filter_kernel.to(self.device)

    def forward(self, input):
        h = self.linear(input)
        assert not torch.isnan(h).any()

        attention = self.filter()

        if self.threshold is not None:
            attention = torch.where(attention > self.threshold, attention, torch.tensor([-9e15], device=self.device))
        else:
            ret = torch.topk(attention, k=self.k, dim=1)
            attention.fill_(-9e15)
            attention.scatter_(1, ret.indices, ret.values)

        attention = attention.softmax(1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        nzidx = attention.nonzero(as_tuple=True)
        h_prime = spmm(nzidx, attention[nzidx], self.N*self.out_channels, self.N, h)

        if self.concat:
            return h_prime.view(self.out_channels, self.N, self.out_features).permute(1, 0, 2).reshape(self.N, -1), attention
        else:
            return h_prime.view(self.out_channels, self.N, self.out_features).mean(dim=0), attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


