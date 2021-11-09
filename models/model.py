import math
import os

import pandas as pd
import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder
from torch_geometric.utils import get_laplacian

from config import EDGE_LOGIT_THRESHOLD
from .utils import (
    create_filter, sample_negative_graphs, sample_positive_graphs, sample_positive_nodes_nce, ASTNodeEncoder,
    sample_negative_nodes_nce, sample_negative_nodes_cont, sample_positive_nodes_cont, sample_positive_nodes_dict,
    sample_negative_nodes_dict, sample_positive_nodes_naive, sample_negative_nodes_naive, sample_positive_nodes_naive_2, sample_negative_nodes_naive_2
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class RewireNetNodeClassification(torch.nn.Module):
    def __init__(self, dataset, split, step, objective="naive_loss"):
        super(RewireNetNodeClassification, self).__init__()
        self.step = step
        data = dataset[0]
        self.dataset = dataset
        self.data = data

        if dataset.name.lower() in ['cora', 'citeseer', 'pubmed']:
            self.train_mask = data.train_mask
            self.val_mask = data.val_mask
        else:
            self.train_mask = data.train_mask[split]
            self.val_mask = data.val_mask[split]

        L_index, L_weight = get_laplacian(data.edge_index, normalization='sym')
        self.L = torch.sparse_coo_tensor(L_index, L_weight, device=device).to_dense()
        self.D = create_filter(self.L, self.step).permute(1, 2, 0)

        self.A = None
        self.I = torch.eye(dataset[0].num_nodes, device=device)
        self.K = 10

        self.objective = objective.lower()
        [self.train_positive_nodes, self.train_negative_nodes] = self.sample_nodes(self.train_mask)
        [self.val_positive_nodes, self.val_negative_nodes] = self.sample_nodes(self.val_mask)

        # random Guassian signals for simulating spectral clustering
        # epsilon - error bound
        epsilon = 0.25
        # random_signal_size = math.floor(
        #     6 / (math.pow(epsilon, 2) / 2 - math.pow(epsilon, 3) / 3) * math.log(self.data.num_nodes)
        # )
        random_signal_size = self.data.num_features
        random_signal = torch.normal(
            0, math.sqrt(1 / random_signal_size), size=(self.data.num_nodes, random_signal_size), device=device
        )

        self.x = self.data.x
        self.random_signal = random_signal

        # self.C = torch.nn.Parameter(torch.empty(len(torch.arange(0, 2.1, self.step)), 1, device=device))
        self.windows = math.ceil(2.1/self.step)
        self.spectral_mlp = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(self.windows, self.windows*2), torch.nn.ReLU(),
            torch.nn.Linear(self.windows*2, 1),
        ) for _ in range(2)])
        self.feature_mlp = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.data.num_features, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, 64), ) for _ in range(3)
        ])

        self.final_mlp = torch.nn.Linear(64*3, self.dataset.num_classes, bias=False)

    def reset_parameters(self):
        # torch.nn.init.xavier_normal_(self.C, 0.6)  # torch.nn.init.xavier_normal_(self.W)
        # torch.nn.init.xavier_normal_(self.W, 0.5)  # torch.nn.init.xavier_normal_(self.W)
        for mlp in self.feature_mlp:
            for layer in mlp:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        for mlp in self.spectral_mlp:
            for layer in mlp:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def sample_nodes(self, mask):
        if self.objective == 'nce_loss':
            positive_nodes = sample_positive_nodes_nce(mask, self.data.y)
            negative_nodes = sample_negative_nodes_nce(mask, self.data.y, self.K)
        elif self.objective in ['contrastive_loss', 'triplet_loss']:
            positive_nodes = sample_positive_nodes_cont(mask, self.data.y, self.K)
            negative_nodes = sample_negative_nodes_cont(mask, self.data.y, self.K)
        elif self.objective == 'lifted_struct_loss':
            positive_nodes = sample_positive_nodes_dict(mask, self.data.y, 5)
            negative_nodes = sample_negative_nodes_dict(mask, self.data.y, 30)
        elif self.objective == 'npair_loss':
            positive_nodes = sample_positive_nodes_dict(mask, self.data.y, 5)
            negative_nodes = sample_negative_nodes_dict(mask, self.data.y, 100)
        elif self.objective == 'naive_loss':
            positive_nodes = sample_positive_nodes_naive_2(mask, self.data.y)
            negative_nodes = sample_negative_nodes_naive_2(mask, self.data.y)
        else:
            positive_nodes = []
            negative_nodes = []
        return [positive_nodes, negative_nodes]

    def nce_loss(self, x, mask):
        x_masked = x[mask]
        if self.training:
            positive_nodes = self.train_positive_nodes
            negative_nodes = self.train_negative_nodes
        else:
            positive_nodes = self.val_positive_nodes
            negative_nodes = self.val_negative_nodes

        x_positive = x[positive_nodes].view(-1, x.shape[1], 1)
        x_negative = x[negative_nodes].permute(0, 2, 1)

        nce_loss = -torch.mean(
            torch.log(torch.sigmoid(x_masked.matmul(x_positive))).view(-1) + torch.log(
                torch.sigmoid(-x_masked.matmul(x_negative))
            ).sum(dim=2).view(-1)
        )

        return nce_loss

    def contrastive_loss(self, x, mask):
        x_masked = x[mask].view(-1, 1, x.shape[1])

        if self.training:
            positive_nodes = self.train_positive_nodes
            negative_nodes = self.train_negative_nodes
        else:
            positive_nodes = self.val_positive_nodes
            negative_nodes = self.val_negative_nodes

        x_positive = x[positive_nodes]
        x_negative = x[negative_nodes]
        e = 0.8
        loss = torch.frobenius_norm(x_masked - x_positive, dim=2).pow(2).sum(dim=1) + torch.max(
            torch.zeros(1), e - torch.frobenius_norm(
                x_masked - x_negative, dim=2
            )
        ).pow(2).sum(dim=1)
        return loss.mean()

    def triplet_loss(self, x, mask):
        x_masked = x[mask].view(-1, 1, x.shape[1])
        if self.training:
            positive_nodes = self.train_positive_nodes
            negative_nodes = self.train_negative_nodes
        else:
            positive_nodes = self.val_positive_nodes
            negative_nodes = self.val_negative_nodes

        x_positive = x[positive_nodes]
        x_negative = x[negative_nodes]
        e = 0.2
        loss = torch.relu(
            torch.frobenius_norm(x_masked - x_positive, dim=2).pow(2) - torch.frobenius_norm(
                x_masked - x_negative, dim=2
            ).pow(2) + e
        )
        return loss.mean()

    def lifted_struct_loss(self, x, mask):
        if self.training:
            positive_nodes = self.train_positive_nodes
            negative_nodes = self.train_negative_nodes
        else:
            positive_nodes = self.val_positive_nodes
            negative_nodes = self.val_negative_nodes

        e = 0.02

        # https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#contrastive-loss
        node_indices = mask.nonzero().view(-1)
        x_masked = x[node_indices]
        D_ij = torch.frobenius_norm(x_masked.view(-1, 1, x.shape[1]) - x_masked[positive_nodes], dim=2)
        D_ik = torch.frobenius_norm(x_masked.view(-1, 1, x.shape[1]) - x_masked[negative_nodes], dim=2)
        D_jl = torch.frobenius_norm(
            x_masked[positive_nodes].view(positive_nodes.shape[0], positive_nodes.shape[1], 1, x.shape[1]) - x_masked[
                negative_nodes[positive_nodes]], dim=3
        )
        loss = torch.mean(
            torch.sum(
                D_ij + torch.log(
                    torch.exp(e - D_ik).sum(dim=1).view(-1, 1) + torch.exp(e - D_jl).sum(dim=2)
                ), dim=1
            )
        )

        return loss

    def npair_loss(self, x, mask):
        if self.training:
            positive_nodes = self.train_positive_nodes
            negative_nodes = self.train_negative_nodes
        else:
            positive_nodes = self.val_positive_nodes
            negative_nodes = self.val_negative_nodes

        # https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#contrastive-loss
        node_indices = mask.nonzero().view(-1)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        x_masked = x[node_indices]

        loss = torch.log(
            1 + torch.exp(
                x_masked.view(x_masked.shape[0], 1, x_masked.shape[1]).matmul(x_masked[negative_nodes].permute(0, 2, 1)) -
                x_masked.view(x_masked.shape[0], 1, x_masked.shape[1]).matmul(x_masked[positive_nodes].permute(0, 2, 1)).mean(dim=2, keepdim=True)
            ).squeeze().sum(dim=1)
        ).mean()

        l2_regularizer = x.norm(p=2, dim=1).mean()

        return loss + 0.001*l2_regularizer
        # return loss

    def sparsity_loss(self):
        # Normalize learnable coefficient C, is it necessary?
        C = torch.nn.functional.normalize(self.C, dim=0, p=1)
        # sparsity loss?
        dimensions = torch.sqrt(torch.tensor(float(C.shape[0])))
        sparsity_loss = torch.mean((dimensions - C.norm(p=1, dim=0) / C.norm(p=2, dim=0)) / (dimensions - 1))
        return sparsity_loss

    def edge_weight_loss(self, x):
        # The Frobenius norm of L is added to control the distribution of the edge weights and is inspired by the approach in [27].
        edge_weight_loss = torch.frobenius_norm(self.decode_all(x), dim=1).pow(2).mean()
        return edge_weight_loss

    def naive_loss(self, x, mask):
        # masked_x = x[mask]
        masked_x = x
        e = 0.5
        if self.training:
            positive_nodes = self.train_positive_nodes
            negative_nodes = self.train_negative_nodes
        else:
            positive_nodes = self.val_positive_nodes
            negative_nodes = self.val_negative_nodes

        # homophily loss 1: distance between pairs in negative samples
        homophily_loss_1 = 0
        for group in negative_nodes:
            # group embeddings by class
            x_group = masked_x[group]
            # add mean of distance to loss
            # homophily_loss_1 += torch.relu(e - torch.pdist(x_group)).mean()
            homophily_loss_1 += e - torch.pdist(x_group).clamp(max=20).mean()

        # homophily loss 2: distance between pairs between pairs of the same class
        homophily_loss_2 = 0
        # iterate over classes
        for group in positive_nodes:
            x_group = masked_x[group]
            homophily_loss_2 += torch.pdist(x_group).clamp(max=20).mean()  # homophily_loss_2 += torch.var(y_hat_group, dim=0).mean()

        '''
            we have a lot more negative samples than positive ones,
            divide homophily_loss_1 by beta (the average negative samples per class)
        '''
        # beta = len(negative_nodes) / self.dataset.num_classes
        l2_loss = 0.01*sum([s.norm(p=2) for s in self.parameters()])
        return homophily_loss_2/self.dataset.num_classes + homophily_loss_1/len(negative_nodes) + l2_loss

    def e2e_loss(self, x, mask):
        logits = F.log_softmax(self.final_mlp(x), dim=1)
        return torch.nn.functional.nll_loss(logits[mask], self.data.y[mask])

    def loss(self, x, mask):
        if self.objective == 'nce_loss':
            homo_loss = self.nce_loss(x, mask)
        elif self.objective == 'contrastive_loss':
            homo_loss = self.contrastive_loss(x, mask)
        elif self.objective == 'triplet_loss':
            homo_loss = self.triplet_loss(x, mask)
        elif self.objective == 'lifted_struct_loss':
            homo_loss = self.lifted_struct_loss(x, mask)
        elif self.objective == 'npair_loss':
            homo_loss = self.npair_loss(x, mask)
        elif self.objective == 'naive_loss':
            homo_loss = self.naive_loss(x, mask)
        elif self.objective == 'e2e_loss':
            homo_loss = self.e2e_loss(x, mask)
        else:
            homo_loss = 0
        # return homo_loss + self.sparsity_loss() + self.edge_weight_loss(x)
        return homo_loss

    '''
    I - L is a low-pass filter used by GCN, the loss function 
    encourages node embeddings of the same label to be close under
    this low-pass filter
    '''

    def forward(self):
        # Normalize learnable coefficient C, is it necessary?
        # C = torch.nn.functional.normalize(self.C, dim=0, p=2)
        # C = torch.nn.functional.normalize(self.C, dim=0, p=1)
        # C = torch.relu(self.C)
        # C = self.C
        # Construct a filtered laplacian by sum over all filter banks with C
        # L_hat = self.D.matmul(C).squeeze()
        L_hat = self.spectral_mlp[0](self.D).squeeze()
        # I - L low-pass filter
        y_hat_1 = (self.I - L_hat).mm(self.feature_mlp[0](self.x))

        y_hat_2 = self.spectral_mlp[1](self.D).squeeze().mm(self.feature_mlp[1](self.random_signal))
        y_hat_3 = self.feature_mlp[2](self.x)

        # y_hat = torch.cat((y_hat.view(self.data.num_nodes, self.data.num_features, 2), self.data.x.view(x.shape[0], -1, 1)), dim=2)
        # y_hat = y_hat.matmul(torch.nn.functional.normalize(self.W, p=1, dim=0)).squeeze()

        # return torch.nn.functional.normalize(y_hat, p=2, dim=1)
        # return y_hat.mm(self.W)
        return torch.cat([y_hat_1, y_hat_2, y_hat_3], dim=1)
        # return torch.cat([y_hat_1, y_hat_2, y_hat_3], dim=1)

    def decode_all(self, z):
        return z @ z.t()

    def transform_edges(self, dataset):
        y_hat = self()
        with torch.no_grad():
            if self.objective == 'nce_loss':
                dataset.data.edge_index = self.get_edges_cosin(y_hat)
            elif self.objective in ['contrastive_loss', 'lifted_struct_loss', 'npair_loss', 'naive_loss']:
                dataset.data.edge_index = self.get_edges_pdist(y_hat)
        return dataset

    def get_edges_cosin(self, x):
        edge_logits = self.decode_all(x)
        A = edge_logits.where(edge_logits > EDGE_LOGIT_THRESHOLD, torch.tensor(0., device=device))
        index = A.nonzero().T
        return index.detach()

    def get_edges_pdist(self, x):
        dist = torch.nn.functional.pdist(x, p=2)
        e = 1
        indices = (dist < e).nonzero().view(-1)
        num_nodes = x.shape[0]
        rows = indices // num_nodes
        cols = indices % num_nodes
        return torch.stack((torch.cat((rows, cols)), torch.cat((cols, rows))), dim=0).detach()


class RewireNetGraphClassification(torch.nn.Module):
    def __init__(self, dataset, step):
        super(RewireNetGraphClassification, self).__init__()
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.step = step
        # self.windows = math.ceil(2.1/self.step)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(11, 44), torch.nn.ReLU(), torch.nn.Linear(44, 1), torch.nn.ReLU(), torch.nn.Linear(11, 1)
            )
        self.emb_dim = dataset[0].num_features
        self.node_encoder = None
        self.edge_encoder = None
        if dataset.name.lower() == 'ogbg-ppa':
            self.emb_dim = 300
            self.edge_encoder = torch.nn.Linear(7, self.emb_dim)
            self.node_encoder = torch.nn.Embedding(1, self.emb_dim)  # uniform input node embedding
        elif 'mol' in dataset.name.lower():
            self.emb_dim = 300
            self.edge_encoder = BondEncoder(self.emb_dim)
            self.node_encoder = AtomEncoder(self.emb_dim)
        elif dataset.name.lower() == 'ogbg-code2':
            self.emb_dim = 300
            self.edge_encoder = torch.nn.Linear(2, self.emb_dim)
            nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
            nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))
            self.node_encoder = ASTNodeEncoder(
                self.emb_dim, num_nodetypes=len(nodetypes_mapping['type']),
                num_nodeattributes=len(nodeattributes_mapping['attr']), max_depth=20
            )

        self.C = torch.nn.Parameter(torch.empty(len(torch.arange(0, 2.1, self.step)), 1))

        # epsilon - error bound
        self.epsilon = 0.25
        avg_num_nodes = max(dataset.data.num_nodes)
        self.random_signal_size = math.floor(
            6 / (math.pow(self.epsilon, 2) / 2 - math.pow(self.epsilon, 3) / 3) * math.log(avg_num_nodes)
        )

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(
            self.C, 0.6
        )
        # for layer in self.mlp:
        #     if hasattr(layer, 'reset_parameters'):
        #         layer.reset_parameters()

    def compute_loss(self, embeddings, negative_samples, positive_samples):
        # first part of homophily loss: graph embeddings from different groups should be far-apart
        homophily_loss_1 = 0
        for class_group in negative_samples:
            homophily_loss_1_group = 0
            for group in class_group:
                homophily_loss_1_group -= torch.cdist(embeddings[[group]], embeddings[[group]]).mean()
            beta = len(class_group) / self.dataset.num_classes + 1e-13
            homophily_loss_1 += homophily_loss_1_group / beta

        # second part of homophily loss: graph embeddings from same groups should be close
        homophily_loss_2 = 0
        for class_group in positive_samples:
            for group in class_group:
                homophily_loss_2 += torch.cdist(
                    embeddings[[group]], embeddings[[group]]
                ).mean()  # homophily_loss_2 += torch.var(y_hat_group, dim=0).mean()

        # sparsity loss? what's this?
        dimensions = torch.sqrt(torch.tensor(float(self.C.shape[0])))
        sparsity_loss = torch.mean((dimensions - self.C.norm(p=1, dim=0) / self.C.norm(p=2, dim=0)) / (dimensions - 1))

        return sparsity_loss + homophily_loss_2 + homophily_loss_1

    def decode_all(self, z):
        return z @ z.t()

    def transform_edges(self, dataset):
        for i, data in enumerate(dataset):
            g = data
            g.to(device)
            x = g.x
            if self.node_encoder:
                if self.dataset.name == 'ogbg-code2':
                    node_depth = g.node_depth
                    x = self.node_encoder(x, node_depth.view(-1, ))
                else:
                    x = self.node_encoder(x)
            # if self.edge_encoder:
            #     edge_attr = self.edge_encoder(g.edge_attr)
            #     edge_attr_tensor = torch.zeros(g.num_nodes,g.num_nodes, self.emb_dim, device=device).index_put_(
            #         [g.edge_index[0], g.edge_index[1]], edge_attr)
            L_index, L_weight = get_laplacian(g.edge_index, normalization='sym', num_nodes=g.num_nodes)
            L = torch.sparse_coo_tensor(L_index, L_weight).to_dense().to(device)
            D = create_filter(L, self.step).permute(1, 2, 0)
            # L_hat = self.mlp(D).squeeze()
            C = torch.nn.functional.normalize(self.C, dim=0, p=2)
            L_hat = D.matmul(C).squeeze()
            A_hat = torch.eye(g.num_nodes).to(device) - L_hat
            y_hat = A_hat.mm(x)
            y_hat = torch.nn.functional.normalize(y_hat, p=2, dim=1)
            edge_logits = self.decode_all(y_hat)
            A = edge_logits.where(edge_logits > EDGE_LOGIT_THRESHOLD, torch.tensor(0., device=device))
            index = A.nonzero().T
            edge_index = index.detach()
            # edge_weight = A[index[0], index[1]].detach()
            data.edge_index = edge_index
            # data.edge_weight = edge_weight
            dataset._data_list[i] = data
        return dataset

    def forward(self, data):
        negative_samples = sample_negative_graphs(self.num_classes, torch.arange(data.num_graphs), data.y)
        positive_samples = sample_positive_graphs(self.num_classes, torch.arange(data.num_graphs), data.y)

        embeddings = torch.zeros(data.num_graphs, self.emb_dim + self.random_signal_size)
        sparsity_loss = 0

        for i in range(data.num_graphs):
            g = data[i]
            x = g.x
            if self.node_encoder:
                if self.dataset.name == 'ogbg-code2':
                    node_depth = g.node_depth
                    x = self.node_encoder(x, node_depth.view(-1, ))
                else:
                    x = self.node_encoder(x)

            # random Guassian signals for simulating spectral clustering
            random_signal = torch.normal(
                0, math.sqrt(1 / self.random_signal_size), size=(g.num_nodes, self.random_signal_size), device=device
            )

            x = torch.cat([x, random_signal], dim=1)
            # if self.edge_encoder:
            #     edge_attr = self.edge_encoder(g.edge_attr)
            #     edge_attr_tensor = torch.zeros(g.num_nodes,g.num_nodes, self.emb_dim, device=device).index_put_(
            #         [g.edge_index[0], g.edge_index[1]], edge_attr)
            if g.edge_index.numel() == 0:
                y_hat = x
            else:
                L_index, L_weight = get_laplacian(g.edge_index, normalization='sym', num_nodes=g.num_nodes)
                L = torch.sparse_coo_tensor(L_index, L_weight).to_dense().to(device)
                D = create_filter(L, self.step).permute(1, 2, 0)

                # L_hat = self.mlp(D).squeeze()
                C = torch.nn.functional.normalize(self.C, dim=0, p=2)
                L_hat = D.matmul(C).squeeze()
                A_hat = torch.eye(g.num_nodes).to(device) - L_hat

                y_hat = A_hat.mm(x)
                y_hat = torch.nn.functional.normalize(y_hat, p=2, dim=1)
                edge_logits = self.decode_all(y_hat)

                sparsity_loss += torch.relu(edge_logits - EDGE_LOGIT_THRESHOLD).norm(
                    p=1
                ) / edge_logits.numel()  # if self.edge_encoder:  #     weighted_edge_attr_tensor = A_hat.view(g.num_nodes, g.num_nodes, 1)*edge_attr_tensor  #     index = A_hat.nonzero().T[0]  #     y_hat = x.index_add(0, index, weighted_edge_attr_tensor[A_hat.nonzero().T[0], A_hat.nonzero().T[1]])  # else:  #     y_hat = x

            embeddings[i] = y_hat.mean(dim=0)
            sparsity_loss /= data.num_graphs

        return self.compute_loss(embeddings, negative_samples, positive_samples) + sparsity_loss
