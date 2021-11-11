import os.path as osp
import networkx as nx
from .webkb_data import WebKB

def print_dataset(dataset):
    data = dataset[0]
    print('Name', dataset.name)
    print('Nodes', data.num_nodes)
    print('Edges', data.num_edges)
    print('Features', data.num_features)
    print('Classes', dataset.num_classes)
    print('Label rate', len(data.train_mask[0].nonzero().tolist()) / data.num_nodes)
    G = nx.Graph(data.edge_index.T.tolist())
    print('Density', nx.density(G))

for name in ['Cornell', 'Texas', 'Wisconsin']:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = WebKB(path, name)
    print_dataset(dataset)


for name in ['Cora', 'CiteSeer', 'PubMed']:
    print_dataset(get_planetoid_dataset(name))
