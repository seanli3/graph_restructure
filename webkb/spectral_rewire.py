import torch
from graph_dictionary.model import create_filter
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import get_laplacian as gl
from random import seed as rseed
from numpy.random import seed as nseed
from citation import get_dataset, run
import numpy as np
import networkx as nx
import math

#%%

# dataset = get_dataset('Cornell', self_loop=True)
# data = dataset[0]

DATASET='Cora'

dataset = get_dataset(DATASET, True)
data = dataset[0]
random = True
remove_ratio = 0.9
add_ratio = 0.7


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

#%%

def get_random_edgeList(adj, numSamples):
    edges = adj.nonzero(as_tuple=False)
    indices = torch.tensor(np.random.choice(range(len(edges)), numSamples, replace=False))
    return edges[indices]

L_index, L_weight = gl(data.edge_index, normalization='sym')
L = torch.sparse_coo_tensor(L_index, L_weight).to_dense()


comple_A = 1 - torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.shape[1])).to_dense()
comple_L_index, comple_L_weight = gl(comple_A.nonzero(as_tuple=False).T, normalization='sym')
comple_L = torch.sparse_coo_tensor(comple_L_index, comple_L_weight).to_dense()
train_mask = data.train_mask[0]


if not random:
    filters = [create_filter(L, b) for b in torch.arange(0, 2.1, 0.25)]
    com_filters = [create_filter(comple_L, b) for b in torch.arange(0, 2.1, 0.25)]

#%%
def get_edgeList_proposal(num_samples, complement=False):
    losses = []
    fs = com_filters if complement else filters
    adj = comple_A if complement else A
    for f in fs:
        x_hat = f.mm(data.x)
        loss = 0
        for i in range(dataset.num_classes):
            loss += x_hat[(data.y == i).logical_and(train_mask).nonzero(as_tuple=True)[0]].var()

        for group in negative_samples:
            if len(group) > 0:
                loss -= x_hat[group].var()
        losses.append(loss)
    losses = torch.tensor(losses)

    max_filter_index = losses.argsort(descending=True)[:3]
    min_filter_index = losses.argsort(descending=False)[:3]
    coef1 = torch.stack([filters[fi].masked_fill(adj == 0, 0).abs() for fi in max_filter_index], dim=0).sum(dim=0)
    coef2 = torch.stack([filters[fi].masked_fill(adj == 0, 0).abs() for fi in min_filter_index], dim=0).sum(dim=0)
    coef = (coef2 - coef1).abs() if complement else (coef1 - coef2).abs()

    m = coef.view(-1).topk(num_samples, largest=not complement)[1].tolist()
    m = torch.tensor(m) % L.numel()
    indices = torch.stack((m // L.shape[0], m % L.shape[0])).T

    return indices

#%%

selectedNodes = set()
negative_samples = []
for _ in range(math.ceil(len(train_mask.nonzero(as_tuple=True)[0]) / dataset.num_classes)):
    group = []
    for i in range(dataset.num_classes):
        candidates = set((data.y[train_mask] == i).nonzero(as_tuple=True)[0].tolist()) - selectedNodes
        if len(candidates) == 0:
            continue
        n = np.random.choice(list(candidates))
        selectedNodes.add(n)
        group.append(n)
    negative_samples.append(group)



connected = True
edges_to_remove = []
edges_to_add = []

if not connected:
    A = torch.sparse_coo_tensor(data.edge_index, torch.ones(dataset[0].num_edges)).to_dense()


    for _ in range(1):
        if random:
            edges_to_remove = get_random_edgeList(A, math.floor(remove_ratio*data.num_edges))
            edges_to_add = get_random_edgeList(comple_A, math.floor(add_ratio*data.num_edges))
            for edge in edges_to_remove:
                A[edge[0], edge[1]] = 0
                A[edge[1], edge[0]] = 0
            for index in edges_to_add:
                A[index[0], index[1]] = 1
                A[index[1], index[0]] = 1
        else:
            edges_to_remove = get_edgeList_proposal(math.floor(remove_ratio*data.num_edges))
            edges_to_add = get_edgeList_proposal(math.floor(add_ratio*data.num_edges), complement=True)
            # add_edge_candidates = get_edgeList_proposal(1 - A)
            for index in edges_to_remove:
                A[index[0], index[1]] = 0
                A[index[1], index[0]] = 0
            for index in edges_to_add:
                A[index[0], index[1]] = 1
                A[index[1], index[0]] = 1
    connected = nx.is_connected(nx.Graph(A.numpy()).to_undirected())
    if not connected:
        print('Disconnected! try again')


def get_laplacian(edge_index):
    L_index, L_weight = gl(edge_index, normalization=None)
    L = torch.sparse_coo_tensor(L_index, L_weight).to_dense()
    return L

class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 256)
        self.conv2 = GCNConv(256, 256)
        self.conv3 = GCNConv(256, 256)
        self.conv4 = GCNConv(256, 256)
        self.conv5 = GCNConv(256, dataset.num_classes)
        self.L = get_laplacian(dataset[0].edge_index)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.conv5.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        print('0', x.T.mm(self.L).mm(x).diag().mean())
        x = F.relu(self.conv1(x, edge_index))
        print('1', x.T.mm(self.L).mm(x).diag().mean())
        x = F.relu(self.conv2(x, edge_index))
        print('2', x.T.mm(self.L).mm(x).diag().mean())
        x = F.relu(self.conv3(x, edge_index))
        print('3', x.T.mm(self.L).mm(x).diag().mean())
        x = F.relu(self.conv4(x, edge_index))
        print('4', x.T.mm(self.L).mm(x).diag().mean())
        x = self.conv5(x, edge_index)

        return F.log_softmax(x, dim=1), x


use_dataset = lambda : get_dataset(DATASET, True, edges_to_remove=edges_to_remove, edges_to_add=edges_to_add)

run(use_dataset, Net, 1, 2000, 0.01, 0.0005, 100)





#%%
