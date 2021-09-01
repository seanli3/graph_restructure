import os.path as osp
import torch
from torch_geometric.datasets import TUDataset
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Optional
from kernel.utils import scatter_add_deterministic
from ogb.graphproppred import PygGraphPropPredDataset
from graph_dictionary.utils import get_vocab_mapping, augment_edge, encode_y_to_arr, decode_arr_to_seq
from torchvision import transforms



def degree(index, num_nodes: Optional[int] = None,
           dtype: Optional[int] = None):
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`
    """
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N, ), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0), ), dtype=out.dtype, device=out.device)
    return scatter_add_deterministic(one, index)


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


class OneHotDegree(object):
    r"""Adds the node degree as one hot encodings to the node features.

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """

    def __init__(self, max_degree, in_degree=False, cat=True):
        self.max_degree = max_degree
        self.in_degree = in_degree
        self.cat = cat

    def __call__(self, data):
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = degree(idx, data.num_nodes, dtype=torch.long)
        deg = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.max_degree)


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def get_dataset(name, sparse=True, cleaned=False):
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
                dataset.transform = OneHotDegree(max_degree)
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
            dataset = PygGraphPropPredDataset(root=path, name=name, transform=add_zeros)


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

    return dataset
