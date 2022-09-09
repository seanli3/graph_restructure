import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import SGConv
from random import seed as rseed
from numpy.random import seed as nseed
from benchmark.node_classification.train_eval import run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--seed', type=int, default=729, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--normalize_features', type=bool, default=False)
parser.add_argument('--K', type=int, default=2)
parser.add_argument('--rewired', action='store_true')
parser.add_argument('--num_edges', type=float, default=3000)
parser.add_argument('--rewirer_step', type=float, default=0.1)
parser.add_argument('--max_node_degree', type=int, default=1000)
parser.add_argument('--run_split', type=int, default=None)
parser.add_argument('--with_node_feature', action='store_true')
parser.add_argument('--with_rand_signal', action='store_true')
parser.add_argument('--edge_step', type=int, default=None)
parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--h_den', type=float, default=None)

args = parser.parse_args()


rseed(args.seed)
nseed(args.seed)
torch.manual_seed(args.seed)


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = SGConv(
            dataset.num_features, dataset.num_classes, K=args.K, cached=True)

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1), None


run(args.dataset, Net, args.rewired, args.runs, args.epochs, args.lr, args.weight_decay, args.patience,
    run_split=args.run_split, num_edges=args.num_edges, eps=args.eps, rewirer_step=args.rewirer_step,
    max_node_degree=args.max_node_degree, with_node_feature=args.with_node_feature,
    with_rand_signal=args.with_rand_signal, edge_step=args.edge_step, h_den=args.h_den)
