import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import get_laplacian
from random import seed as rseed
from numpy.random import seed as nseed
from webkb import get_dataset, run
from citation import get_dataset as get_citation_data, run as run_citation
import numpy as np
import networkx as nx
import math

#%%

# dataset = get_dataset('Cornell', self_loop=True)
# data = dataset[0]

dataset = get_dataset('Cornell')
data = dataset[0]

#%%

L_index, L_weight = get_laplacian(data.edge_index)
L = torch.sparse_coo_tensor(L_index, L_weight)
A = torch.sparse_coo_tensor(data.edge_index, torch.ones(dataset[0].num_edges)).to_dense()
D = torch.diag(A.sum(dim=0))
beta_1 = 1e2
beta_2 = 1e2
# train_mask = data.train_mask[0]
train_mask = data.train_mask
random = False


def get_random_edgeList(adj, numSamples):
    edges = adj.nonzero(as_tuple=False)
    indices = torch.tensor(np.random.choice(range(len(edges)), numSamples, replace=False))
    return edges[indices]

#%%
def get_edgeList_proposal(adj):
    randomNodeOrderTemp = np.random.permutation(data.num_nodes)
    nodePairsOut = []
    matchedNodesTemp = set()

    numSamples = math.floor(0.3*data.num_nodes)

    for firstNode in randomNodeOrderTemp:
        if firstNode not in matchedNodesTemp:
            unmatchedNeighborsTemp = [index for index, item in enumerate(adj[firstNode]) if
                                      item > 0 and index not in matchedNodesTemp]
            if len(unmatchedNeighborsTemp) > 0:
                secondNode = np.random.choice(unmatchedNeighborsTemp)
                nodePairsOut.append(sorted([firstNode, secondNode]))
                matchedNodesTemp.add(firstNode)
                matchedNodesTemp.add(secondNode)
        if len(nodePairsOut) >= numSamples:
            break

    proposedEdgeListOut = nodePairsOut
    return torch.tensor(proposedEdgeListOut)

#%%

def compute_loss(weight, e):
    b_e = torch.zeros(data.num_nodes).view(-1,1)
    b_e[e[0]] = 1
    b_e[e[1]] = -1
    delta_L = weight*b_e.mm(b_e.T)
    loss_1 = delta_L.norm('fro').pow(2)
    loss_2 = 0
    loss_3 = 0
    selectedNodes= set()
    negative_samples = []
    D_hat = torch.clone(D)
    D_hat[e[0], e[0]] += weight
    D_hat[e[1], e[1]] += weight
    if (torch.diag(D_hat) ==0).any():
        return np.inf
    D_hat_inv = D_hat.inverse()
    x_hat = D_hat_inv.mm(delta_L).mm(data.x)
    for _ in range(math.ceil(len(train_mask.nonzero(as_tuple=True)[0])/dataset.num_classes)):
        nodes = []
        for i in range(dataset.num_classes):
            if data.y[train_mask[0]].bincount()[i] <= len(negative_samples):
                continue
            while True:
                n = np.random.choice((data.y == i).logical_and(train_mask).nonzero(as_tuple=True)[0])
                if n not in selectedNodes:
                    selectedNodes.add(n)
                    nodes.append(n)
                    break
        negative_samples.append(nodes)
    for group in negative_samples:
            loss_2 += x_hat[group].var()
    for i in range(dataset.num_classes):
            loss_3 -= x_hat[(data.y == i).logical_and(train_mask).nonzero(as_tuple=True)[0]].var()
    return loss_1 + beta_1 * loss_2 + beta_2 * loss_3
#%%
edges_to_remove = []
edges_to_add = []
connected = False
ratio = 0.1
if not connected:
    for _ in range(10):
        if random:
            del_edge_candidates = get_random_edgeList(A, math.floor(ratio*0.3*data.num_nodes))
            add_edge_candidates = get_random_edgeList(1-A, math.floor(ratio*0.3*data.num_nodes))
            for edge in del_edge_candidates:
                A[edge[0], edge[1]] = 0
                A[edge[1], edge[0]] = 0
                edges_to_remove.append(edge)
            for edge in add_edge_candidates:
                A[edge[0], edge[1]] = 1
                A[edge[1], edge[0]] = 1
                edges_to_add.append(edge)
        else:
            del_edge_candidates = get_edgeList_proposal(A)
            add_edge_candidates = get_edgeList_proposal(1 - A)
            del_loss = []
            add_loss = []
            for e in del_edge_candidates:
                del_loss.append(compute_loss(-1, e))

            for e in add_edge_candidates:
                add_loss.append(compute_loss(1, e))

            index_offset = len(del_loss)
            losses = np.array(del_loss + add_loss)

            budget = math.ceil(ratio * (len(del_edge_candidates) + len(add_edge_candidates)))
            # budget = math.ceil(ratio * (len(del_edge_candidates)))
            edge_index = losses.argsort()[:budget]

            for index in edge_index:
                if index < index_offset:
                    edges_to_remove.append([del_edge_candidates[index][0], del_edge_candidates[index][1]])
                    A[del_edge_candidates[index][0], del_edge_candidates[index][1]] = 0
                    A[del_edge_candidates[index][1], del_edge_candidates[index][0]] = 0
                else:
                    edges_to_add.append([
                        add_edge_candidates[index - index_offset][0],
                        add_edge_candidates[index - index_offset][1]
                    ])
                    A[add_edge_candidates[index - index_offset][0], add_edge_candidates[index - index_offset][1]] = 1
                    A[add_edge_candidates[index - index_offset][1], add_edge_candidates[index - index_offset][0]] = 1
    connected = nx.is_connected(nx.Graph(A.numpy()).to_undirected())
    if not connected:
        print('Disconnected! try again')


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 256)
        self.conv2 = GCNConv(256, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1), x


use_dataset = lambda : get_dataset('Cornell', True, edges_to_remove=edges_to_remove, edges_to_add=edges_to_add, self_loop=True)

run(use_dataset, Net, 1, 2000, 0.01, 0.0005, 100)





#%%
