import os.path as osp
from .webkb_data import WebKB
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, dropout_adj
from random import sample
from torch.nn import functional as F
from torch_geometric.utils import to_networkx
import networkx as nx
import torch
import networkx as nx
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.utils import is_undirected, to_undirected, get_laplacian


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def matching_labels_distribution(dataset):
    # Build graph
    adj = coo_matrix(
        (np.ones(dataset[0].num_edges),
        (dataset[0].edge_index[0].cpu().numpy(), dataset[0].edge_index[1].cpu().numpy())),
        shape=(dataset[0].num_nodes, dataset[0].num_nodes))
    G = nx.Graph(adj)

    hop_1_matching_percent = []
    hop_2_matching_percent = []
    hop_3_matching_percent = []
    for n in range(dataset.data.num_nodes):
        hop_1_neighbours = list(nx.ego_graph(G, n, 1).nodes())
        hop_2_neighbours = list(nx.ego_graph(G, n, 2).nodes())
        hop_3_neighbours = list(nx.ego_graph(G, n, 3).nodes())
        node_label = dataset[0].y[n]
        hop_1_labels = dataset[0].y[hop_1_neighbours]
        hop_2_labels = dataset[0].y[hop_2_neighbours]
        hop_3_labels = dataset[0].y[hop_3_neighbours]
        matching_1_labels = node_label == hop_1_labels
        matching_2_labels = node_label == hop_2_labels
        matching_3_labels = node_label == hop_3_labels
        hop_1_matching_percent.append(matching_1_labels.float().sum()/matching_1_labels.shape[0])
        hop_2_matching_percent.append(matching_2_labels.float().sum()/matching_2_labels.shape[0])
        hop_3_matching_percent.append(matching_3_labels.float().sum()/matching_3_labels.shape[0])

    return hop_1_matching_percent, hop_2_matching_percent, hop_3_matching_percent


def get_dataset(name, normalize_features=False, transform=None, edge_dropout=None, node_feature_dropout=None,
                dissimilar_t = 1, cuda=False, permute_masks=None, lcc=False, self_loop=False,
                edges_to_remove=None, edges_to_add=None, dummy_nodes = 0, removal_nodes = 0, features=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = WebKB(path, name)
    dataset.data.y = dataset.data.y.long()
    if features is not None:
        dataset.data.x = features

    if self_loop:
        dataset.data.edge_index = add_self_loops(dataset.data.edge_index)[0]

    if not is_undirected(dataset.data.edge_index):
        dataset.data.edge_index = to_undirected(dataset.data.edge_index)

    dataset.data, dataset.slices = dataset.collate([dataset.data])

    if normalize_features:
        dataset.transform = T.NormalizeFeatures()

    lcc_mask = None
    if lcc:  # select largest connected component
        data_ori = dataset[0]
        data_nx = to_networkx(data_ori)
        data_nx = data_nx.to_undirected()
        print("Original #nodes:", data_nx.number_of_nodes())
        data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
        print("#Nodes after lcc:", data_nx.number_of_nodes())
        lcc_mask = list(data_nx.nodes)

    if permute_masks is not None:
        dataset.data = permute_masks(dataset.data, dataset.num_classes, lcc_mask=lcc_mask)
        for key in dataset.data.keys:
            if key not in dataset.slices:
                dataset.slices[key] = torch.tensor([0, dataset.data[key].shape[0]])

    # dataset.data.node_map = node_map
    dataset.data, dataset.slices = dataset.collate([dataset.data])
    del dataset._data_list
    # assert (nx.is_connected(to_networkx(dataset[0]).to_undirected()))

    if transform:
        dataset = transform(dataset)

    if cuda:
        dataset.data.to('cuda')
        if hasattr(dataset, '_data_list') and dataset._data_list:
            for d in dataset._data_list:
                d.to('cuda')

    return dataset
