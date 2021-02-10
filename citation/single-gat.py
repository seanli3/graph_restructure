import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from random import seed as rseed
from numpy.random import seed as nseed

from citation import get_planetoid_dataset, random_planetoid_splits, run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=8)
parser.add_argument('--seed', type=int, default=729, help='Random seed.')
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--lcc', type=bool, default=False)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--output_heads', type=int, default=1)
parser.add_argument('--edge_dropout', type=float, default=0)
parser.add_argument('--node_feature_dropout', type=float, default=0)
args = parser.parse_args()

rseed(args.seed)
nseed(args.seed)
torch.manual_seed(args.seed)

class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)

        self.W = torch.nn.Parameter(torch.zeros(args.hidden * args.heads, dataset.num_classes))

    def reset_parameters(self):
        self.conv1.reset_parameters()
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=args.dropout, training=self.training)
        x, (edge_index_1, att_val_1) = self.conv1(x, edge_index, return_attention_weights=True)
        j = F.dropout(x, p=args.dropout, training=self.training)
        x = F.elu(x.mm(self.W))
        att1 = torch.zeros(data.num_nodes, data.num_nodes, args.heads)
        att1[list(map(lambda x: torch.tensor(x), edge_index_1.tolist()))] = att_val_1
        return F.log_softmax(x, dim=1), att1, None


dataset = get_planetoid_dataset(args.dataset, args.normalize_features, edge_dropout=args.edge_dropout,
                                node_feature_dropout=args.node_feature_dropout)

permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, permute_masks)
