import os.path as osp
from torch_geometric.datasets import Amazon, Planetoid, Coauthor, Reddit, Flickr, Yelp
from .webkb_data import WebKB
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import is_undirected, to_undirected


def get_dataset(name, normalize_features=False, transform=None, cuda=False,
                self_loop=False, features=None, split='full'):
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
    else:
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

    dataset.data, dataset.slices = dataset.collate([dataset.data])
    if hasattr(dataset, '_data_list'):
        del dataset._data_list

    if transform:
        dataset = transform(dataset)

    if cuda:
        dataset.data.to('cuda')
        if hasattr(dataset, '_data_list') and dataset._data_list:
            for d in dataset._data_list:
                d.to('cuda')

    return dataset
