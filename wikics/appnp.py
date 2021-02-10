import argparse
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from random import seed as rseed
from numpy.random import seed as nseed

from citation import get_dataset, random_planetoid_splits, run, random_coauthor_amazon_splits

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--seed', type=int, default=729, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--lcc', type=bool, default=False)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--edge_dropout', type=float, default=0)
parser.add_argument('--node_feature_dropout', type=float, default=0)
parser.add_argument('--dissimilar_t', type=float, default=1)
args = parser.parse_args()


rseed(args.seed)
nseed(args.seed)
torch.manual_seed(args.seed)


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1), None


if args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
    permute_masks = random_planetoid_splits if args.random_splits else None
elif args.dataset == "CS" or args.dataset == "Physics":
    permute_masks = random_coauthor_amazon_splits
elif args.dataset == "Computers" or args.dataset == "Photo":
    permute_masks = random_coauthor_amazon_splits

use_dataset = lambda : get_dataset(args.dataset, args.normalize_features, edge_dropout=args.edge_dropout,
                                    permute_masks=permute_masks, lcc=args.lcc,
                                    node_feature_dropout=args.node_feature_dropout, dissimilar_t=args.dissimilar_t)

run(use_dataset, Net, args.runs, args.epochs, args.lr, args.weight_decay, args.patience)


