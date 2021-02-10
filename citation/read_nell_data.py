import sys
import os.path as osp
from itertools import repeat
import numpy as np
import scipy.sparse as sp
import os
import networkx as nx

import torch
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_nell_data(folder, prefix):
    processed_data_path = "{}/{}.data.npz".format(folder, prefix)
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    items = [read_file(folder, prefix, name) for name in names]
    x, tx, allx, y, ty, ally, graph, test_index = items
    train_index = torch.arange(y.size(0), dtype=torch.long)
    val_index = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)

    # Find relation nodes, add them as zero-vecs into the right position
    test_idx_reorder = test_index
    test_idx_range = np.sort(test_idx_reorder)
    test_idx_range_full = range(allx.shape[0], len(graph))
    isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range - allx.shape[0], :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    ty_extended[test_idx_range - allx.shape[0], :] = ty
    ty = ty_extended

    x = sp.vstack((allx, tx)).tolil()
    x[test_idx_reorder, :] = x[test_idx_range, :]
    y = sp.vstack((ally, ty)).tolil()
    y[test_idx_reorder, :] = y[test_idx_range, :]

    # print("Creating feature vectors for relations - this might take a while...")
    # x_extended = sp.hstack((x, sp.lil_matrix((x.shape[0], len(isolated_node_idx)))),
    #                               dtype=np.int32).todense()
    # x_extended[isolated_node_idx, x.shape[1]:] = np.eye(len(isolated_node_idx))
    # x = torch.tensor(x_extended).float()

    x = torch.tensor(x.todense()).float()
    y = torch.tensor(y.todense()).max(dim=1)[1]

    train_mask = index_to_mask(train_index, size=y.size(0))
    val_mask = index_to_mask(val_index, size=y.size(0))
    test_mask = index_to_mask(test_index, size=y.size(0))

    edge_index = edge_index_from_dict(graph, num_nodes=y.size(0))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def read_file(folder, prefix, name):
    path = osp.join(folder, 'nell_data', 'ind.{}.{}'.format(prefix.lower(), name))

    if name == 'test.index':
        return read_txt_array(path, dtype=torch.long)

    with open(path, 'rb') as f:
        if sys.version_info > (3, 0):
            out = pickle.load(f, encoding='latin1')
        else:
            out = pickle.load(f)

    if name == 'graph':
        return out

    out = out.todense() if hasattr(out, 'todense') else out
    out = torch.Tensor(out)
    return out


def edge_index_from_dict(graph_dict, num_nodes=None):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    # NOTE: There are duplicated edges and self loops in the datasets. Other
    # implementations do not remove them!
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask
