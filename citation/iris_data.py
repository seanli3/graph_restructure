import os.path as osp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
import numpy as np
import networkx as nx



def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

class Iris(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None):
        super(Iris, self).__init__(None, transform, pre_transform)
        iris = load_iris()
        # Store features matrix in X
        x = torch.tensor(iris.data).float()
        # Store target vector in
        y = torch.tensor(iris.target)

        train_idx, test_idx, _, _ = train_test_split(list(range(x.shape[0])), list(range(x.shape[0])), test_size=0.6,
                                                     random_state=42)
        val_idx, test_idx, _, _ = train_test_split(test_idx, test_idx, test_size=0.5)

        train_mask = index_to_mask(torch.tensor(train_idx), y.shape[0])
        val_mask = index_to_mask(torch.tensor(val_idx), y.shape[0])
        test_mask = index_to_mask(torch.tensor(test_idx), y.shape[0])

        G = nx.wheel_graph(x.shape[0])

        data = Data(x=x, edge_index=torch.tensor(list(G.edges)).T, y=y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        self.data = data

        self.slices = {
            'x': torch.tensor((0, x.shape[0])),
            'y': torch.tensor((0, y.shape[0])),
            'edge_index': torch.tensor((0, data.edge_index.shape[1])),
            'train_mask': torch.tensor((0, data.train_mask.shape[0])),
            'val_mask': torch.tensor((0, data.val_mask.shape[0])),
            'test_mask': torch.tensor((0, data.test_mask.shape[0])),
        }


