import argparse
import torch
from random import seed as rseed
from numpy.random import seed as nseed
from webkb import get_dataset, run
from config import SAVED_MODEL_PATH_NODE_CLASSIFICATION
from torch import nn
from torch_geometric.utils import get_laplacian
from models.model import RewireNetNodeClassification

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--seed', type=int, default=729, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--rewired', action='store_true')
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


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.layers = nn.Sequential(nn.Linear(192, 128),
                                    nn.Dropout(0.2),
                                    nn.Linear(128, 64),
                                    nn.Linear(64, dataset.num_classes),
                                    nn.ELU(inplace=True),
                                    # nn.ReLU(inplace=True),
                                    nn.LogSoftmax(dim=1))

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, data):
        return self.layers(data.x), None, None


def use_dataset(split):
    dataset = get_dataset(args.dataset, args.normalize_features, cuda=args.cuda, lcc=args.lcc)
    if args.rewired:
        rewirer_state = torch.load(SAVED_MODEL_PATH_NODE_CLASSIFICATION.format(args.dataset, split))
        step = rewirer_state['step']
        rewirer = RewireNetNodeClassification(dataset, split, step)
        rewirer.load_state_dict(rewirer_state['model'])
        def transform_x(d):
            d.data.x = rewirer().detach()
            return d
        new_dataset = get_dataset(
            args.dataset, args.normalize_features, cuda=args.cuda, lcc=args.lcc, transform=transform_x
        )
    else:
        new_dataset = dataset
    return new_dataset

run(use_dataset, Net, args.runs, args.epochs, args.lr, args.weight_decay, args.patience, cuda=args.cuda)
