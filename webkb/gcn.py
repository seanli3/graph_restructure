import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from random import seed as rseed
from torch_geometric.utils import get_laplacian
from numpy.random import seed as nseed
from graph_dictionary.utils import create_filter
from pathlib import Path
from webkb import get_dataset, run

path = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--dummy_nodes', type=int, default=0)
parser.add_argument('--removal_nodes', type=int, default=0)
parser.add_argument('--epochs', type=int, default=2000)
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
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if args.cuda:
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
        x, edge_index = data.x, data.edge_index
        edge_weight = None

        # edge_index = torch.eye(x.shape[0], x.shape[0]).nonzero().T
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1), x


def use_dataset(split):
    saved = torch.load(path / '../graph_dictionary/{}_best_model_split_{}.pt'.format(args.dataset, split))
    C = saved['C']
    # C = torch.nn.functional.normalize(C, dim=0, p=1)
    C = torch.nn.functional.normalize(C, dim=0, p=2)
    step = saved['step'] if 'step' in saved else 0.2

    dataset = get_dataset(args.dataset, args.normalize_features, edge_dropout=args.edge_dropout,
               permute_masks=None, cuda=args.cuda, lcc=args.lcc,
               node_feature_dropout=args.node_feature_dropout, dissimilar_t=args.dissimilar_t)
    data = dataset[0]
    L_index, L_weight = get_laplacian(data.edge_index, normalization='sym')
    laplacian = torch.sparse_coo_tensor(L_index, L_weight, device=device).to_dense()
    D = create_filter(laplacian, step).permute(1,2,0)
    # D = torch.stack([f.mm(data.x) for f in filters], dim=2)
    L = D.matmul(C).squeeze()

    A_hat = torch.eye(L.shape[0], device=device) - L
    A_hat = torch.nn.functional.normalize(A_hat, dim=1, p=2)

    # Sparsify edges
    k = 5
    A_one_zero = torch.zeros(A_hat.shape[0], A_hat.shape[1], device=device)\
        .index_put((torch.arange(A_hat.shape[0], device=device).repeat_interleave(k), A_hat.topk(k, dim=1, largest=True)[1].view(-1)), torch.tensor(1., device=device))
    A_one_zero.masked_fill_(A_hat.abs() < 0.3, 0)

    # A_hat.masked_fill_(A_hat.abs() < 0.01, 0)
    # edge_index = A_hat.nonzero(as_tuple=False).T
    # edge_weight = A_hat[A_hat > 0]

    # L = laplacian.abs() > 0.05
    A = torch.sparse_coo_tensor(data.edge_index, torch.ones(dataset[0].num_edges), device=device).to_dense()

    edges_to_add = (A_one_zero != 0).logical_and((A == 0)).nonzero()
    edges_to_remove = (A != 0).logical_and((A_one_zero == 0)).nonzero()

    return get_dataset(args.dataset, args.normalize_features, edge_dropout=args.edge_dropout,
            permute_masks=None, lcc=args.lcc, edges_to_remove=edges_to_remove, edges_to_add=edges_to_add,
            node_feature_dropout=args.node_feature_dropout, dissimilar_t=args.dissimilar_t,
            dummy_nodes = args.dummy_nodes, removal_nodes = args.removal_nodes, cuda=args.cuda)

run(use_dataset, Net, args.runs, args.epochs, args.lr, args.weight_decay, args.patience, cuda=args.cuda)