import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from random import seed as rseed
from torch_geometric.utils import get_laplacian
from numpy.random import seed as nseed

from webkb import get_dataset, run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--dummy_nodes', type=int, default=0)
parser.add_argument('--removal_nodes', type=int, default=0)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--seed', type=int, default=729, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--lcc', type=bool, default=False)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--edge_dropout', type=float, default=0)
parser.add_argument('--node_feature_dropout', type=float, default=0)
parser.add_argument('--dissimilar_t', type=float, default=1)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()


rseed(args.seed)
nseed(args.seed)
torch.manual_seed(args.seed)

args.cuda = args.cuda and torch.cuda.is_available()

def create_filter(laplacian, b):
    return (torch.diag(torch.ones(laplacian.shape[0]) * 40).mm(
        (laplacian - torch.diag(torch.ones(laplacian.shape[0]) * b)).matrix_power(4)) + \
            torch.eye(laplacian.shape[0])).matrix_power(-2)

for split in range(10):
    model = torch.load('./{}_best_model_split_{}.pt'.format(args.dataset, split))
    C = model['C']
    # C = C.clip_(min=0)
    C = torch.nn.functional.normalize(C, dim=0, p=2)
    # C = C / C.sum(0)

    dataset = get_dataset(args.dataset, args.normalize_features, edge_dropout=args.edge_dropout,
               permute_masks=None, cuda=args.cuda, lcc=args.lcc,
               node_feature_dropout=args.node_feature_dropout, dissimilar_t=args.dissimilar_t)

    data = dataset[0]
    L_index, L_weight = get_laplacian(data.edge_index, normalization='sym')
    laplacian = torch.sparse_coo_tensor(L_index, L_weight).to_dense()
    filters = [create_filter(laplacian, b) for b in torch.arange(0, 2.1, 0.1)]
    # D = torch.stack([f.mm(data.x) for f in filters], dim=2)
    L = torch.stack(filters, dim=2).matmul(C).squeeze().abs() > 0.01
    # L = laplacian.abs() > 0.05
    A = torch.sparse_coo_tensor(data.edge_index, torch.ones(dataset[0].num_edges)).to_dense()

    edges_to_add = (L != 0).logical_and((A == 0)).nonzero()
    edges_to_remove = (A != 0).logical_and((L == 0)).nonzero()

    class Net(torch.nn.Module):
        def __init__(self, dataset):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, args.hidden)
            self.conv2 = GCNConv(args.hidden, dataset.num_classes)

        def reset_parameters(self):
            self.conv1.reset_parameters()
            self.conv2.reset_parameters()

        def forward(self, data):
            x = data.x

            edge_index = data.edge_index
            # edge_index = torch.eye(x.shape[0], x.shape[0]).nonzero().T
            edge_weight = None
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, p=args.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return F.log_softmax(x, dim=1), x


    use_dataset = lambda : get_dataset(args.dataset, args.normalize_features, edge_dropout=args.edge_dropout,
                          permute_masks=None, lcc=args.lcc, edges_to_remove=edges_to_remove, edges_to_add=edges_to_add,
                          node_feature_dropout=args.node_feature_dropout, dissimilar_t=args.dissimilar_t,
                          dummy_nodes = args.dummy_nodes, removal_nodes = args.removal_nodes)

    run(use_dataset, Net, args.runs, args.epochs, args.lr, args.weight_decay, args.patience, split=split)