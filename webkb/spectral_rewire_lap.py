import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import get_laplacian, add_self_loops
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
random = False
remove_ratio = 0.1
add_ratio = 0.4


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

#%%

edge_index, _ = add_self_loops(data.edge_index)
L_index, L_weight = get_laplacian(edge_index, normalization='sym')
L = torch.sparse_coo_tensor(L_index, L_weight).to_dense()
print(data.x.T.mm(L).mm(data.x))


train_mask = data.train_mask[0]

comple_A = 1 - torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1])).to_dense()
comple_L_index, comple_L_weight = get_laplacian(comple_A.nonzero(as_tuple=False).T, normalization='sym')
comple_L = torch.sparse_coo_tensor(comple_L_index, comple_L_weight).to_dense()
com_lam, com_vec = comple_L.symeig(True)
com_filters = [com_vec.mm(torch.diag((40*((com_lam - b).pow(4)) + 1).pow(-2))).mm(com_vec.T) for b in torch.arange(0, 2.1, 0.25)]

#%%
def rewire_graph(L, complement=False):
    if not complement:
        lam, vec = L.symeig(True)
        filters = [vec.mm(torch.diag((40 * ((lam - b).pow(4)) + 1).pow(-2))).mm(vec.T) for b in
                   torch.arange(0, 2.1, 0.25)]

    losses = []
    fs = com_filters if complement else filters
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

    min_filter_index = losses.argsort(descending=False)[:2]

    new_L = torch.sum(torch.stack([fs[i] for i in min_filter_index]), dim=0)
    return new_L

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



A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1])).to_dense()

edges_to_remove = []
edges_to_add = []


for _ in range(5):
    L = rewire_graph(L)
    L += rewire_graph(L, True)
A_hat = torch.eye(data.num_nodes, data.num_nodes) - L


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.w_1 = torch.nn.Parameter(torch.empty(dataset.num_features, 256))
        self.w_2 = torch.nn.Parameter(torch.empty(256, dataset.num_classes))

    def reset_parameters(self):
        glorot(self.w_1)
        glorot(self.w_2)

    def forward(self, data):
        x = F.relu(A_hat.mm(data.x).mm(self.w_1))
        x = F.dropout(x, p=0.8, training=self.training)
        x = A_hat.mm(x).mm(self.w_2)

        return F.log_softmax(x, dim=1), x


use_dataset = lambda : get_dataset(DATASET, True)

run(use_dataset, Net, 1, 2000, 0.01, 0.0005, 10)





#%%
