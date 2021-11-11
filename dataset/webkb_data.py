import os.path as osp

import torch
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
import numpy as np


class WebKB(InMemoryDataset):
    r"""The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/new_data')

    split_file_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/splits')

    datasets = ['cornell', 'texas', 'wisconsin', 'chameleon', 'film', 'squirrel']

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in self.datasets

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def split_file_names(self):
        return [
            '_split_0.6_0.2_0.npz',
            '_split_0.6_0.2_1.npz',
            '_split_0.6_0.2_2.npz',
            '_split_0.6_0.2_3.npz',
            '_split_0.6_0.2_4.npz',
            '_split_0.6_0.2_5.npz',
            '_split_0.6_0.2_6.npz',
            '_split_0.6_0.2_7.npz',
            '_split_0.6_0.2_8.npz',
            '_split_0.6_0.2_9.npz'
        ]

    @property
    def processed_file_names(self):
        return 'data.pt'

    @property
    def raw_split_file_paths(self):
        r"""The filepaths to find in order to skip the download."""
        files = list(map(lambda name: self.name + name, self.split_file_names))
        return [osp.join(self.raw_dir, f) for f in files]


    def _download(self):
        from torch_geometric.data.dataset import files_exist, makedirs
        if files_exist(self.raw_paths) and files_exist(self.raw_split_file_paths):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)
        for name in self.split_file_names:
            download_url(f'{self.split_file_url}/{self.name}{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            if self.name == 'film':
                features = torch.zeros(len(x), 932)
                for i, idx in enumerate(x):
                    features[i, torch.LongTensor(idx)] = 1
                x = features
            else:
                x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.float)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_mask = []
        val_mask = []
        test_mask = []
        for file_path in self.raw_split_file_paths:
            with np.load(file_path) as splits_file:
                train_mask.append(splits_file['train_mask'])
                val_mask.append(splits_file['val_mask'])
                test_mask.append(splits_file['test_mask'])
        train_mask = torch.BoolTensor(train_mask)
        val_mask = torch.BoolTensor(val_mask)
        test_mask = torch.BoolTensor(test_mask)

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)
