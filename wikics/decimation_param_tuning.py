import argparse
from model.layers import GraphSpectralFilterLayer
from model.spectral_filter import Graph
import torch
import torch.nn.functional as F
from random import seed as rseed
from numpy.random import seed as nseed
from wikics import get_dataset, run
from ax import optimize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=20)
arg = parser.parse_args()

args = {
    'dataset': 'WikiCS',
    'runs': 1,
    'epochs': 1000,
    'alpha': 0.2,
    'seed': 729,
    'lr': 0.005,
    'weight_decay': 0.0005,
    'patience': 50,
    'hidden': 32,
    'heads': 16,
    'dropout': 0.8,
    'normalize_features': True,
    'pre_training': True,
    'cuda': True,
    'order': 13,
    'edge_dropout': 0,
    'node_feature_dropout': 0,
    'filter': 'analysis'
}


def decimation(args):
    print(args)
    rseed(args['seed'])
    nseed(args['seed'])
    torch.manual_seed(args['seed'])

    args['cuda'] = args['cuda'] and torch.cuda.is_available()


    # if args['cuda']:
    #     print("-----------------------Training on CUDA-------------------------")

    if args['cuda']:
        torch.cuda.manual_seed(args['seed'])
        torch.set_default_tensor_type('torch.cuda.FloatTensor')


    class Net(torch.nn.Module):
        def __init__(self, dataset):
            super(Net, self).__init__()
            data = dataset.data
            adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.num_edges))
            self.G = Graph(adj)
            self.G.estimate_lmax()

            self.analysis = GraphSpectralFilterLayer(self.G, dataset.num_node_features, args['hidden'],
                                                     dropout=args['dropout'], out_channels=args['heads'], filter=args['filter'],
                                                     pre_training=args['pre_training'], device='cuda' if args['cuda'] else 'cpu',
                                                     alpha=args['alpha'], order=args['order'])
            # self.mlp = nn.Sequential(nn.Linear(args['hidden * args['heads, 128),
            #                             nn.ReLU(inplace=True),
            #                             nn.Linear(128, 64),
            #                             nn.ReLU(inplace=True),
            #                             nn.Linear(64, 32),
            #                             nn.ReLU(inplace=True),
            #                             nn.Linear(32, dataset.num_classes),
            #                             nn.ReLU(inplace=True))

            # self.W = torch.zeros(args['hidden * args['heads, dataset.num_classes)

            self.synthesis = GraphSpectralFilterLayer(self.G, args['hidden'] * args['heads'], dataset.num_classes, filter=args['filter'],
                                                      device='cuda' if args['cuda'] else 'cpu', dropout=args['dropout'],
                                                      out_channels=1, alpha=args['alpha'], pre_training=False,
                                                      order=args['order'])

        def reset_parameters(self):
            self.analysis.reset_parameters()
            # torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
            # for layer in self.mlp:
            #     if hasattr(layer, 'reset_parameters'):
            #         layer.reset_parameters()
            self.synthesis.reset_parameters()

        def forward(self, data):
            x = data.x
            x = F.dropout(x, p=args['dropout'], training=self.training)
            x, att1 = self.analysis(x)
            x = F.dropout(x, p=args['dropout'], training=self.training)
            x, att2 = self.synthesis(x)
            x = F.elu(x)
            # x = F.elu(x.mm(self.W))
            # x = self.mlp(x)
            return F.log_softmax(x, dim=1), att1, att2

    use_dataset = lambda: get_dataset(args['dataset'], args['normalize_features'], edge_dropout=args['edge_dropout'],
                                      permute_masks=None, cuda=args['cuda'],
                                      node_feature_dropout=args['node_feature_dropout'], self_loop=args['self_loop'])

    return run(use_dataset, Net, args['runs'], args['epochs'], args['lr'], args['weight_decay'],
        args['patience'], None)


best_parameters, best_values, _, _ = optimize(
 parameters=[{'name': 'dataset', 'type': 'fixed', 'value': args['dataset']},
    {'name': 'runs', 'type': 'fixed', 'value': 1},
    {'name': 'epochs', 'type': 'fixed', 'value': 2000},
    {'name': 'alpha', "type": "fixed", "value": 0.2},
    {'name': 'seed', 'type': 'fixed', 'value': 729},
    {'name': 'lr', "type": "range", "bounds": [0.0005, 0.01], "log_scale": True},
    {'name': 'weight_decay', 'type': 'range', "bounds": [0.000001, 0.005], "log_scale": True},
    {'name': 'patience', 'type': 'fixed', 'value': 100},
    {'name': 'hidden', 'type': 'range', "bounds": [8, 88], "log_scale": True},
    {'name': 'heads', 'type': 'range', "bounds": [1, 12]},
    {'name': 'dropout', "type": "range", "bounds": [0.1, 0.9]},
    {'name': 'self_loop', "type": "fixed", "value": False},
    {'name': 'normalize_features', 'type': 'fixed', 'value': True},
    {'name': 'pre_training', 'type': 'fixed', 'value': False},
    {'name': 'cuda', 'type': 'fixed', 'value': args['cuda']},
    {'name': 'order', 'type': 'fixed', 'value': 16},
    {'name': 'edge_dropout', 'type': 'fixed', 'value': 0},
    {'name': 'node_feature_dropout', 'type': 'fixed', 'value': 0},
    {'name': 'filter', 'type': 'fixed', 'value': 'analysis'}],
    evaluation_function=decimation,
    total_trials=arg.trials,
    minimize=False)

print(best_parameters, best_values)
