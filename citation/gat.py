import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from random import seed as rseed
from numpy.random import seed as nseed

from citation import get_dataset, random_planetoid_splits, run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--dummy_nodes', type=int, default=0)
parser.add_argument('--removal_nodes', type=int, default=0)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--hidden', type=int, default=8)
parser.add_argument('--lcc', type=bool, default=False)
parser.add_argument('--seed', type=int, default=729, help='Random seed.')
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--output_heads', type=int, default=1)
parser.add_argument('--edge_dropout', type=float, default=0)
parser.add_argument('--node_feature_dropout', type=float, default=0)
parser.add_argument('--dissimilar_t', type=float, default=1)
parser.add_argument('--split', type=str, default='full')
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

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
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=args.dropout, training=self.training)
        x, _ = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=args.dropout, training=self.training)
        x, _ = self.conv2(x, edge_index, return_attention_weights=True)
        # att1 = torch.zeros(data.num_nodes, data.num_nodes, args.heads)
        # att2 = torch.zeros(data.num_nodes, data.num_nodes)
        # att1[list(map(lambda x: torch.tensor(x), edge_index_1.tolist()))] = att_val_1
        # att2[list(map(lambda x: torch.tensor(x), edge_index_2.tolist()))] = att_val_2.view(-1)
        return F.log_softmax(x, dim=1), x


permute_masks = random_planetoid_splits if args.random_splits else None

use_dataset = lambda : get_dataset(args.dataset, args.normalize_features, edge_dropout=args.edge_dropout,
                                    permute_masks=permute_masks, split=args.split, lcc=args.lcc, cuda=args.cuda,
                                    node_feature_dropout=args.node_feature_dropout, dissimilar_t=args.dissimilar_t,
                                    dummy_nodes = args.dummy_nodes, removal_nodes = args.removal_nodes)

run(use_dataset, Net, args.runs, args.epochs, args.lr, args.weight_decay, args.patience)
