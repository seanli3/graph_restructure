import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from random import seed as rseed
from numpy.random import seed as nseed
from benchmark.node_classification.train_eval import run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=729, help='Random seed.')
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lcc', type=bool, default=False)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--num_hops', type=int, default=3)
parser.add_argument('--edge_dropout', type=float, default=0)
parser.add_argument('--node_feature_dropout', type=float, default=0)
parser.add_argument('--dissimilar_t', type=float, default=1)
parser.add_argument('--rewired', action='store_true')
parser.add_argument('--num_edges', type=float, default=3000)
parser.add_argument('--rewirer_mode', type=str, default='supervised')
parser.add_argument('--rewirer_step', type=float, default=0.2)
parser.add_argument('--run_split', type=int, default=None)
parser.add_argument('--model_indices', nargs="+", type=int, default=[0,1])
parser.add_argument('--max_node_degree', type=int, default=10)
parser.add_argument('--with_node_feature', action='store_true')
parser.add_argument('--with_rand_signal', action='store_true')

args = parser.parse_args()


rseed(args.seed)
nseed(args.seed)
torch.manual_seed(args.seed)

class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = ChebConv(dataset.num_features, args.hidden, args.num_hops)
        self.conv2 = ChebConv(args.hidden, dataset.num_classes, args.num_hops)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), None


run(args.dataset, Net, args.rewired, args.runs, args.epochs, args.lr, args.weight_decay, args.patience,
    run_split=args.run_split, num_edges=args.num_edges, model_indices=args.model_indices,
    rewirer_mode=args.rewirer_mode, rewirer_step=args.rewirer_step, max_node_degree=args.max_node_degree,
    with_node_feature = args.with_node_feature, with_rand_signal = args.with_rand_signal)
