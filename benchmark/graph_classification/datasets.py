import os.path as osp
import torch
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from ogb.graphproppred import PygGraphPropPredDataset
from graph_dictionary.utils import get_vocab_mapping, augment_edge, encode_y_to_arr, decode_arr_to_seq
from torchvision import transforms


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def get_dataset(name, sparse=True, cleaned=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    if name.upper() in ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COLLAB']:
        dataset = TUDataset(path, name, cleaned=cleaned)
        dataset.data.edge_attr = None
        if dataset.data.x is None:
            max_degree = 0
            degs = []
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            if max_degree < 1000:
                dataset.transform = T.OneHotDegree(max_degree)
            else:
                deg = torch.cat(degs, dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                dataset.transform = NormalizedDegree(mean, std)

    elif 'ogbg' in name.lower():
        if 'mol' in name.lower():
            dataset = PygGraphPropPredDataset(root=path, name=name)
        elif 'code2' in name.lower():
            dataset = PygGraphPropPredDataset(root=path, name=name)
            num_vocab = 5000
            max_seq_len = 5
            split_idx = dataset.get_idx_split()
            vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], num_vocab)
            dataset.transform = transforms.Compose([augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len)])
        else:
            dataset = PygGraphPropPredDataset(root=path, name=name)
            dataset.transform = add_zeros

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset[torch.tensor(indices)]

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    # Using dataset.transform will perform transformation everytime data is accessed, so disable it for perfomrnace
    # if transform is not None and dataset.transform is not None:
    #     dataset.transform = T.Compose([dataset.transform, transform])
    # elif transform is not None:
    #     dataset.transform = transform

    # transforming whole graph at once is more performant
    if transform:
        dataset = transform(dataset)
    return dataset
