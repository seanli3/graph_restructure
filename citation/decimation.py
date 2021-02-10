import argparse
from model.layers import GraphSpectralFilterLayer
from model.spectral_filter import Graph
import torch
import torch.nn.functional as F
from torch import nn
from random import seed as rseed
from numpy.random import seed as nseed
from citation import get_dataset, random_planetoid_splits, run, random_coauthor_amazon_splits
from citation.train_eval import evaluate
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--seed', type=int, default=729, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=7.530100210192558e-05)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--hidden', type=int, default=63)
parser.add_argument('--heads', type=int, default=14)
parser.add_argument('--output_heads', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.6174883141474811)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--self_loop', action='store_true')
parser.add_argument('--lcc', type=bool, default=False)
parser.add_argument('--pre_training', action='store_true')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--order', type=int, default=15, help='Chebyshev polynomial order')
parser.add_argument('--edge_dropout', type=float, default=0)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--node_feature_dropout', type=float, default=0)
parser.add_argument('--filter', type=str, default='analysis')
parser.add_argument('--split', type=str, default='full')
parser.add_argument('--method', type=str, default='chebyshev')
parser.add_argument('--dissimilar_t', type=float, default=1)
parser.add_argument('--tau', type=float, default=0.2)
parser.add_argument('--threshold', type=float, default=None)
parser.add_argument('--Kb', type=int, default=18)
parser.add_argument('--Ka', type=int, default=2)
parser.add_argument('--Tmax', type=int, default=200)
args = parser.parse_args()
print(args)


rseed(args.seed)
nseed(args.seed)
torch.manual_seed(args.seed)

args.cuda = args.cuda and torch.cuda.is_available()

if args.cuda:
    print("-----------------------Training on CUDA-------------------------")
    torch.cuda.manual_seed(args.seed)

class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        data = dataset[0]
        if args.cuda:
            data.to('cuda')

        self.G = Graph(data)

        self.analysis = GraphSpectralFilterLayer(self.G, dataset.num_node_features, args.hidden, method=args.method,
                                                 dropout=args.dropout, out_channels=args.heads, filter=args.filter,
                                                 pre_training=args.pre_training, device='cuda' if args.cuda else 'cpu',
                                                 order=args.order, concat=True, k=args.k, tau=args.tau,
                                                 threshold=args.threshold, Kb=args.Kb, Ka=args.Ka, Tmax=args.Tmax)
        # self.mlp = nn.Sequential(nn.Linear(args.hidden * args.heads, 128),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(128, 64),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(64, 32),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(32, dataset.num_classes),
        #                             nn.ReLU(inplace=True))

        # self.W = torch.nn.Parameter(torch.zeros(args.hidden * args.heads, dataset.num_classes))

        self.synthesis = GraphSpectralFilterLayer(self.G, args.hidden * args.heads, dataset.num_classes,
                                                  method=args.method, dropout=args.dropout,
                                                  out_channels=args.output_heads, filter=args.filter,
                                                  pre_training=False, device='cuda' if args.cuda else 'cpu',
                                                  order=args.order, concat=False, k=args.k, tau=args.tau,
                                                  threshold=args.threshold, Kb=args.Kb, Ka=args.Ka, Tmax=args.Tmax)
        if args.cuda:
            self.to('cuda')

    def reset_parameters(self):
        self.analysis.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # for layer in self.mlp:
        #     if hasattr(layer, 'reset_parameters'):
        #         layer.reset_parameters()
        self.synthesis.reset_parameters()
        if args.cuda:
            self.to('cuda')

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.analysis(x)[0]
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.synthesis(x)[0]
        x = F.elu_(x)
        # x = F.elu(x.mm(self.W))
        # x = F.elu(self.mlp(x))
        return F.log_softmax(x, dim=1), None, None


permute_masks = None
if args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
    permute_masks = random_planetoid_splits if args.random_splits else None
elif args.dataset == "CS" or args.dataset == "Physics":
    permute_masks = random_coauthor_amazon_splits
elif args.dataset == "Computers" or args.dataset == "Photo":
    permute_masks = random_coauthor_amazon_splits


use_dataset = lambda : get_dataset(args.dataset, args.normalize_features, edge_dropout=args.edge_dropout, self_loop=args.self_loop,
                                    permute_masks=permute_masks, cuda=args.cuda, lcc=args.lcc, split=args.split,
                                    node_feature_dropout=args.node_feature_dropout, dissimilar_t=args.dissimilar_t)

run(use_dataset, Net, args.runs, args.epochs, args.lr, args.weight_decay, args.patience, cuda=args.cuda)
