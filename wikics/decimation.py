import argparse
from model.layers import GraphSpectralFilterLayer
from model.spectral_filter import Graph
import torch
import torch.nn.functional as F
from random import seed as rseed
from numpy.random import seed as nseed
from wikics import get_dataset, run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--alpha', type=float, default=0.7709619178612326)
parser.add_argument('--seed', type=int, default=729, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=7.530100210192558e-05)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--hidden', type=int, default=63)
parser.add_argument('--heads', type=int, default=14)
parser.add_argument('--output_heads', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.6174883141474811)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--lcc', type=bool, default=False)
parser.add_argument('--pre_training', action='store_true')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--order', type=int, default=15, help='Chebyshev polynomial order')
parser.add_argument('--edge_dropout', type=float, default=0)
parser.add_argument('--node_feature_dropout', type=float, default=0)
parser.add_argument('--filter', type=str, default='analysis')
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--dissimilar_t', type=float, default=1)
args = parser.parse_args()
print(args)


rseed(args.seed)
nseed(args.seed)
torch.manual_seed(args.seed)

args.cuda = args.cuda and torch.cuda.is_available()


if args.cuda:
    print("-----------------------Training on CUDA-------------------------")
    torch.cuda.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        data = dataset.data
        adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.num_edges))
        self.G = Graph(adj)
        self.G.estimate_lmax()

        self.analysis = GraphSpectralFilterLayer(self.G, dataset.num_node_features, args.hidden,
                                                 dropout=args.dropout, out_channels=args.heads, filter=args.filter,
                                                 pre_training=args.pre_training, device='cuda' if args.cuda else 'cpu',
                                                 alpha=args.alpha, order=args.order, concat=True)
        # self.mlp = nn.Sequential(nn.Linear(args.hidden * args.heads, 128),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(128, 64),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(64, 32),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(32, dataset.num_classes),
        #                             nn.ReLU(inplace=True))

        # self.W = torch.nn.Parameter(torch.zeros(args.hidden * args.heads, dataset.num_classes))

        self.synthesis = GraphSpectralFilterLayer(self.G, args.hidden * args.heads, dataset.num_classes, filter=args.filter,
                                                  device='cuda' if args.cuda else 'cpu', dropout=args.dropout,
                                                  out_channels=args.output_heads, alpha=args.alpha, pre_training=False,
                                                  order=args.order, concat=False)

    def reset_parameters(self):
        self.analysis.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # for layer in self.mlp:
        #     if hasattr(layer, 'reset_parameters'):
        #         layer.reset_parameters()
        self.synthesis.reset_parameters()

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.analysis(x)[0]
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.synthesis(x)[0]
        x = F.elu(x)
        # x = F.elu(x.mm(self.W))
        # x = F.elu(self.mlp(x))
        return F.log_softmax(x, dim=1), None, None


permute_masks = None


use_dataset = lambda : get_dataset(args.dataset, args.normalize_features, edge_dropout=args.edge_dropout,
                                    permute_masks=permute_masks, cuda=args.cuda, lcc=args.lcc,
                                    node_feature_dropout=args.node_feature_dropout, dissimilar_t=args.dissimilar_t)

run(use_dataset, Net, args.runs, args.epochs, args.lr, args.weight_decay, args.patience, split=args.split)
