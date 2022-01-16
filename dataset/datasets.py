import os.path as osp
import torch.cuda
from torch_geometric.datasets import Amazon, Planetoid, Coauthor, Reddit, Flickr, Yelp, WebKB, WikipediaNetwork, Actor
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils import is_undirected, to_undirected
from config import USE_CUDA, DEVICE

device = DEVICE

def get_dataset(name, normalize_features=False, transform=None,
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
    elif name.lower() in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(path, name)
    elif name.lower() in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(path, name, True)
    elif name.lower() in ['actor']:
        dataset = Actor(path)
    else:
        dataset = PygNodePropPredDataset(name=name, root=path, transform=T.ToSparseTensor())

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

    if transform:
        dataset = transform(dataset)

    return dataset
