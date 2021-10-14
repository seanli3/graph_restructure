import argparse
import torch
from random import seed as rseed
from numpy.random import seed as nseed
from webkb import get_dataset, run
from torch import nn
from torch_geometric.utils import get_laplacian
from graph_dictionary.utils import create_filter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--seed', type=int, default=729, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=7.530100210192558e-05)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--hidden1', type=int, default=256)
parser.add_argument('--hidden2', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--lcc', type=bool, default=False)
parser.add_argument('--pre_training', type=bool, default=False)
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


if args.cuda:
    print("-----------------------Training on CUDA-------------------------")
    torch.cuda.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


for split in range(10):
    model = torch.load('./{}_best_model_split_{}.pt'.format(args.dataset, split))
    C = model['C']

    class Net(torch.nn.Module):
        def __init__(self, dataset):
            super(Net, self).__init__()
            self.layers = nn.Sequential(nn.Linear(dataset[0].num_node_features, args.hidden1),
                                        nn.Dropout(args.dropout),
                                        nn.Linear(args.hidden1, args.hidden2),
                                        nn.Dropout(args.dropout),
                                        nn.Linear(args.hidden2, dataset.num_classes),
                                        nn.ELU(inplace=True),
                                        # nn.ReLU(inplace=True),
                                        nn.LogSoftmax(dim=1))

            data = dataset[0]
            L_index, L_weight = get_laplacian(data.edge_index, normalization='sym')
            laplacian = torch.sparse_coo_tensor(L_index, L_weight).to_dense()
            self.filters = [create_filter(laplacian, b).mm(data.x) for b in torch.arange(0, 2.1, 0.2)]
            self.D = torch.cat(self.filters, dim=1)
            self.x = self.D.mm(C)

        def reset_parameters(self):
            for layer in self.layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        def forward(self, data):
            return self.layers(self.x), None, None

    permute_masks = None

    use_dataset = lambda : get_dataset(args.dataset, args.normalize_features, edge_dropout=args.edge_dropout,
                                        permute_masks=permute_masks, cuda=args.cuda, lcc=args.lcc,
                                        node_feature_dropout=args.node_feature_dropout, dissimilar_t=args.dissimilar_t)

    run(use_dataset, Net, args.runs, args.epochs, args.lr, args.weight_decay, args.patience, split=split)
