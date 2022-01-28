import os.path as osp
import torch.cuda
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import Amazon, Planetoid, Coauthor, Reddit, Flickr, Yelp, WebKB, WikipediaNetwork, Actor
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils import is_undirected, to_undirected, to_networkx
from config import USE_CUDA, DEVICE
from dataset.dataset import PygNcDataset
import networkx as nx
import numpy as np
from dataset.seeds import development_seed, test_seeds


device = DEVICE


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def set_train_val_test_split(
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
    rnd_state = np.random.RandomState(development_seed)
    num_nodes = data.y.shape[0]
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)

    train_masks = []
    val_masks = []
    test_masks = []

    seeds = test_seeds
    for seed in seeds:
        test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

        train_idx = []
        rnd_state = np.random.RandomState(seed)
        for c in range(data.y.max() + 1):
            class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
            train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

        val_idx = [i for i in development_idx if i not in train_idx]

        def get_mask(idx):
            mask = torch.zeros(num_nodes, dtype=torch.bool)
            mask[idx] = 1
            return mask
        train_masks.append(get_mask(train_idx))
        val_masks.append(get_mask(val_idx))
        test_masks.append(get_mask(test_idx))

    data.train_mask = torch.stack(train_masks, dim=1)
    data.val_mask = torch.stack(val_masks, dim=1)
    data.test_mask = torch.stack(test_masks, dim=1)

    return data


def get_dataset(name, normalize_features=False, transform=None,
                self_loop=False, features=None, split='public', lcc=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    if name.lower() in ['computers', 'photo']:
        dataset = Amazon(path, name)
    elif name.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name, split=split)
    elif name.lower() in ['cs', 'physics']:
        dataset = Coauthor(path, name)
    elif name.lower() in ['reddit']:
        dataset = Reddit(path)
    elif name.lower() in ['flickr']:
        dataset = Flickr(path)
    elif name.lower() in ['yelp']:
        dataset = Yelp(path)
    elif name.lower() in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(path, name)
    elif name.lower() in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(path, name, True)
    elif name.lower() in ['actor']:
        dataset = Actor(path)
    else:
        dataset = PygNcDataset(name=name, root=path, transform=T.ToSparseTensor())

    dataset.data.y = dataset.data.y.long()
    if features is not None:
        dataset.data.x = features

    if self_loop:
        dataset.data.edge_index = add_self_loops(dataset.data.edge_index)[0]

    if not is_undirected(dataset.data.edge_index):
        dataset.data.edge_index = to_undirected(dataset.data.edge_index)
        if hasattr(dataset[0], 'adj_t'):
            dataset.data.adj_t = dataset[0].adj_t.to_symmetric()

    if not hasattr(dataset[0], 'train_mask'):
        split_idx = dataset.get_idx_split()
        dataset.data.train_mask = split_idx['train'].to(device)
        dataset.data.val_mask = split_idx['valid'].to(device)
        dataset.data.test_mask = split_idx['test'].to(device)

    if normalize_features:
        dataset.transform = T.NormalizeFeatures()

    dataset.data, dataset.slices = dataset.collate([dataset.data])
    if hasattr(dataset, '_data_list'):
        del dataset._data_list

    dataset.data.to(device)
    if hasattr(dataset, '_data_list') and dataset._data_list:
        for d in dataset._data_list:
            d.to(device)

    if lcc:
        print("Original #nodes:", dataset[0].num_nodes)
        lcc = get_largest_connected_component(dataset)

        x_new = dataset.data.x[lcc]
        y_new = dataset.data.y[lcc]

        row, col = dataset.data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))

        data = Data(
            x=x_new, edge_index=torch.LongTensor(edges), y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
        set_train_val_test_split(data)
        dataset.data = data
        print("#Nodes after lcc:", dataset[0].num_nodes)

    return dataset

    if transform:
        dataset = transform(dataset)

    return dataset
