import torch.nn as nn
import torch.nn.functional as F
from benchmark.model.layers import GraphSpectralFilterLayer, AnalysisFilter
import copy


class GraphDecimation(nn.Module):
    def __init__(self, G, nfeat, nhid, nclass, feature_dropout, spectrum_dropout, alpha, nheads,
                 order, attention_dropout, kernel=None):
        super(GraphDecimation, self).__init__()
        self.G = G
        self.G.estimate_lmax()
        self.feature_dropout = nn.Dropout(p=feature_dropout)

        self.analysis = GraphSpectralFilterLayer(self.G, nfeat, nhid, dropout=attention_dropout,
                                                 out_channels=nheads,
                                                 spectrum_dropout=spectrum_dropout, alpha=alpha,
                                                 kernel=kernel, order=order)

        self.synthesis = GraphSpectralFilterLayer(self.G, nhid * nheads, nclass, dropout=attention_dropout,
                                                  out_channels=1,
                                                  spectrum_dropout=spectrum_dropout, alpha=alpha,
                                                  kernel=None, order=order)

    def forward(self, x):
        x = self.feature_dropout(x)
        x = self.analysis(x)
        x = self.feature_dropout(x)
        x = F.elu(self.synthesis(x))
        return F.log_softmax(x, dim=1)


class DeepGraphDecimation(nn.Module):
    def __init__(self, G, nfeat, nhid, nclass, feature_dropout, spectrum_dropout, alpha, nheads,
                 order, attention_dropout, kernel=None, layers=0):
        super(DeepGraphDecimation, self).__init__()
        self.G = G
        self.G.estimate_lmax()
        self.feature_dropout = nn.Dropout(p=feature_dropout)

        self.analysis = GraphSpectralFilterLayer(self.G, nfeat, nhid,
                                                 dropout=attention_dropout,
                                                 out_channels=nheads,
                                                 spectrum_dropout=spectrum_dropout, alpha=alpha,
                                                 kernel=kernel, order=order)

        self.layers = []
        for _ in range(layers):
            k = AnalysisFilter(nheads, spectrum_dropout)
            if kernel:
                k.load_state_dict(copy.deepcopy(kernel.state_dict()))
            self.layers.append(GraphSpectralFilterLayer(self.G, nhid * nheads, nhid, dropout=0,
                                 out_channels=nheads,
                                 spectrum_dropout=spectrum_dropout, alpha=alpha,
                                 kernel=k, order=order))


        self.synthesis = GraphSpectralFilterLayer(self.G, nhid * nheads, nclass, dropout=0,
                                                  out_channels=1,
                                                  spectrum_dropout=spectrum_dropout, alpha=alpha,
                                                  kernel=None, order=order)

    def forward(self, x):
        x = self.feature_dropout(x)
        x = self.analysis(x)
        for layer in self.layers:
            x = self.feature_dropout(x)
            x = layer(x)
        x = self.feature_dropout(x)
        x = F.elu(self.synthesis(x))
        return F.log_softmax(x, dim=1)


class GraphDecimationMlp(nn.Module):
    def __init__(self, G, nfeat, nhid, nclass, feature_dropout, spectrum_dropout, mlp_hidden, alpha, nheads,
                 order, attention_dropout, mlp_dropout, kernel=None):
        super(GraphDecimationMlp, self).__init__()
        self.G = G
        self.G.estimate_lmax()
        self.feature_dropout = nn.Dropout(p=feature_dropout)

        self.analysis = GraphSpectralFilterLayer(self.G, nfeat, nhid, dropout=attention_dropout,
                                                 out_channels=nheads,
                                                 spectrum_dropout=spectrum_dropout, alpha=alpha,
                                                 kernel=kernel, order=order)

        self.out_att = nn.Sequential(
            nn.Linear(nhid*nheads, mlp_hidden),
            nn.Dropout(p=mlp_dropout),
            nn.LeakyReLU(),
            nn.Linear(mlp_hidden, nclass),
            nn.Dropout(p=mlp_dropout),
            nn.LeakyReLU(),
            # nn.Tanh()
        )

    def forward(self, x):
        x = self.feature_dropout(x)
        x = self.analysis(x)
        x = self.feature_dropout(x)
        x = F.elu(self.out_att(x))
        return F.log_softmax(x, dim=1)
