import math
import os
import pandas as pd
import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder
from torch_geometric.utils import get_laplacian
from .utils import (
    create_filter, check_symmetric, get_adjacency, get_class_idx, sample_negative_graphs, sample_positive_graphs,
    sample_negative, ASTNodeEncoder,
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class RewireNetNodeClassification(torch.nn.Module):
    def __init__(self, dataset, split, step):
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
        self.C = torch.nn.Parameter(torch.empty(len(torch.arange(0, 2.1, self.step)), 1, device=device))

        self.A = None
        self.loss = torch.nn.MSELoss()
        self.I = torch.eye(dataset[0].num_nodes, device=device)

        # sample negative examples
        self.train_negative_samples = sample_negative(dataset.num_classes, self.train_mask, data.y)
        self.val_negative_samples = sample_negative(dataset.num_classes, self.val_mask, data.y)

        # random Guassian signals for simulating spectral clustering
        # epsilon - error bound
        epsilon = 0.25
        random_signal_size = math.floor(
            6 / (math.pow(epsilon, 2) / 2 - math.pow(epsilon, 3) / 3) * math.log(self.data.num_nodes)
        )
        random_signal = torch.normal(
            0, math.sqrt(1 / random_signal_size), size=(self.data.num_nodes, random_signal_size), device=device
        )

        self.x = torch.cat([self.data.x, random_signal], dim=1)

    def reset_parameters(self):
        # self.W.weight = torch.nn.Parameter(get_adjacency(self.data.edge_index))
        torch.nn.init.xavier_normal_(self.C, 0.6)

    def compute_loss(self, y_hat, mask, C):
        homophily_loss_1 = 0
        if self.training:
            negative_samples = self.train_negative_samples
        else:
            negative_samples = self.val_negative_samples

        # homophily loss 1: distance between pairs in negative samples
        for group in negative_samples:
            # group embeddings by class
            y_hat_group = y_hat[group]
            # add mean of distance to loss
            homophily_loss_1 -= torch.cdist(y_hat_group, y_hat_group).mean()

        # homophily loss 2: distance between pairs between pairs of the same class
        homophily_loss_2 = 0
        # iterate over classes
        for i in range(self.dataset.num_classes):
            # only calculate loss when we have samples for ths class
            if ((self.data.y == i).logical_and(mask)).any():
                # group embeddings by class
                y_hat_group = y_hat[(self.data.y == i).logical_and(mask)]
                # add mean of distance to loss
                homophily_loss_2 += torch.cdist(
                    y_hat_group, y_hat_group
                ).mean()  # homophily_loss_2 += torch.var(y_hat_group, dim=0).mean()

        # sparsity loss?
        dimensions = torch.sqrt(torch.tensor(float(C.shape[0])))
        sparsity_loss = torch.mean((dimensions - C.norm(p=1, dim=0) / C.norm(p=2, dim=0)) / (dimensions - 1))

        # The Frobenius norm of L is added to control the distribution of the edge weights and is inspired by the approach in [27].
        # edge_weight_loss = beta_e * torch.frobenius_norm(self.L, dim=1).pow(2).mean()
        edge_weight_loss = 0

        '''
            we have a lot more negative samples than positive ones,
            divide homophily_loss_1 by beta (the average negative samples per class)
        '''
        beta = len(negative_samples) / self.dataset.num_classes

        return sparsity_loss + edge_weight_loss + homophily_loss_2 + homophily_loss_1 / beta

    '''
    I - L is a low-pass filter used by GCN, the loss function 
    encourages node embeddings of the same label to be close under
    this low-pass filter
    '''

    def forward(self, mask):
        x = self.x
        # Normalize learnable coefficient C, is it necessary?
        C = torch.nn.functional.normalize(self.C, dim=0, p=2)
        # Construct a filtered laplacian by sum over all filter banks with C
        L_hat = self.D.matmul(C).squeeze()

        # I - L low-pass filter
        y_hat = (self.I - L_hat).mm(x)

        # concatenate filtered feature with original node feature?
        # y_hat = torch.cat(y_hat, self.data.x)
        return self.compute_loss(y_hat, mask, C)

    def decode_all(self, z):
        return z @ z.t()

    def transform_edges(self, dataset):
        x = self.x
        D = self.D
        C = torch.nn.functional.normalize(self.C, dim=0, p=2)
        L_hat = D.matmul(C).squeeze()
        y_hat = (self.I - L_hat).mm(x)
        A_prob = torch.sigmoid(self.decode_all(y_hat))
        A = A_prob.where(A_prob > 0.5, torch.tensor(0., device=device))
        index = A.nonzero().T
        edge_index = index.detach()
        # edge_weight = A[index[0], index[1]].detach()
        dataset.data.edge_index = edge_index
        # data.edge_weight = edge_weight
        return dataset


class RewireNetGraphClassification(torch.nn.Module):
    def __init__(self, dataset, step):
        super(RewireNetGraphClassification, self).__init__()
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.step = step
        # self.windows = math.ceil(2.1/self.step)
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(self.windows, 2*self.windows), torch.nn.ReLU(), torch.nn.Linear(2*self.windows, 1))
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
        # for layer in self.mlp:  #     if hasattr(layer, 'reset_parameters'):  #         layer.reset_parameters()

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
        sparsity_loss = torch.mean((dimensions - self.C.norm(p=1, dim=0)/self.C.norm(p=2, dim=0))/(dimensions - 1))

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

            x = A_hat.mm(x)

            A_prob = torch.sigmoid(self.decode_all(x))
            A = A_prob.where(A_prob > 0.5, torch.tensor(0., device=device))
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

                x = A_hat.mm(x)

                # edge_logits = self.decode_all(x)
                edge_logits = self.decode_all(torch.nn.functional.normalize(x, p=2, dim=1))

                sparsity_loss += -edge_logits.norm(p=1) / (g.num_nodes * g.num_nodes)
                # sparsity_loss += edge_logits.var()
                # deg = torch.diag(A.sum(dim=1).pow(-0.5))
                # A_tilde = deg.mm(A).mm(deg)
                # sparsity_loss += A.norm(p=1)/(g.num_nodes*g.num_nodes)

                # if self.edge_encoder:
                #     weighted_edge_attr_tensor = A_hat.view(g.num_nodes, g.num_nodes, 1)*edge_attr_tensor
                #     index = A_hat.nonzero().T[0]
                #     y_hat = x.index_add(0, index, weighted_edge_attr_tensor[A_hat.nonzero().T[0], A_hat.nonzero().T[1]])
                # else:
                #     y_hat = x
                y_hat = x

            embeddings[i] = y_hat.mean(dim=0)
            sparsity_loss /= data.num_graphs

        return self.compute_loss(embeddings, negative_samples, positive_samples) + sparsity_loss
