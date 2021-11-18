import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from config import USE_CUDA
from random import seed as rseed
from numpy.random import seed as nseed
from pathlib import Path
from benchmark.node_classification.train_eval import run

path = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--seed', type=int, default=729, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--rewired', action='store_true')
args = parser.parse_args()


rseed(args.seed)
nseed(args.seed)
torch.manual_seed(args.seed)

device = torch.device('cuda') if torch.cuda.is_available() and USE_CUDA else torch.device('cpu')

if device == torch.device('cuda'):
    print("-----------------------Training on CUDA-------------------------")
    torch.cuda.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden )
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.adj_t if hasattr(data, 'adj_t') else data.edge_index
        edge_weight = None

        # edge_index = torch.eye(x.shape[0], x.shape[0]).nonzero().T
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1), x


run(args.dataset, Net, args.rewired, args.runs, args.epochs, args.lr, args.weight_decay, args.patience)
