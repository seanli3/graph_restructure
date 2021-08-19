import argparse
import torch
from random import seed as rseed
from numpy.random import seed as nseed
from webkb import get_dataset, run
from torch.nn import functional as F
from .get_laplacian import get_laplacian
from torch_geometric.nn.inits import glorot

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--seed', type=int, default=729, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--lcc', type=bool, default=False)
parser.add_argument('--pre_training', action='store_true')
parser.add_argument('--cuda', action='store_true')
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

def create_filter(laplacian, b):
    return (torch.diag(torch.ones(laplacian.shape[0]) * 40).mm(
        (laplacian - torch.diag(torch.ones(laplacian.shape[0]) * b)).matrix_power(4)) + \
            torch.eye(laplacian.shape[0])).matrix_power(-2)

for split in range(10):
    if args.cuda:
        print("-----------------------Training on CUDA-------------------------")
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    class Net(torch.nn.Module):
        def __init__(self, dataset):
            super(Net, self).__init__()
            self.W1 = torch.nn.Parameter(torch.Tensor(dataset.num_features, args.hidden))
            self.W2 = torch.nn.Parameter(torch.Tensor(args.hidden, dataset.num_classes))
            self.I = torch.eye(dataset[0].num_nodes)
            self.dropout = args.dropout

            data = dataset[0]
            L_index, L_weight = get_laplacian(data.edge_index, normalization='sym')
            laplacian = torch.sparse_coo_tensor(L_index, L_weight).to_dense()
            self.filters = [create_filter(laplacian, b) for b in torch.arange(0, 2.1, 0.1)]
            D = torch.stack(self.filters, dim=2)

            model = torch.load('../webkb/{}_best_model_split_{}.pt'.format(args.dataset, split))
            C = model['C']
            self.L = D.matmul(C).squeeze()

        def reset_parameters(self):
            glorot(self.W1)
            glorot(self.W2)

        def forward(self, data):
            L_hat = self.I - self.L
            x = F.relu(L_hat.mm(data.x).mm(self.W1))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = L_hat.mm(x).mm(self.W2)

            return F.log_softmax(x, dim=1), x

    permute_masks = None

    use_dataset = lambda : get_dataset(args.dataset, args.normalize_features, edge_dropout=args.edge_dropout,
                                       permute_masks=permute_masks, cuda=args.cuda, lcc=args.lcc,
                                       node_feature_dropout=args.node_feature_dropout, dissimilar_t=args.dissimilar_t)

    run(use_dataset, Net, args.runs, args.epochs, args.lr, args.weight_decay, args.patience, split=split)
