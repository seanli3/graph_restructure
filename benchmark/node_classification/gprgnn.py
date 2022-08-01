import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import argparse
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from random import seed as rseed
from config import USE_CUDA, DEVICE
from numpy.random import seed as nseed
from benchmark.node_classification.train_eval import run


parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--dprate', type=float, default=0.5)
parser.add_argument('--C', type=int)
parser.add_argument('--Init', type=str,
                    choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                    default='PPR')
parser.add_argument('--Gamma', default=None)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--seed', type=int, default=729, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--run_split', type=int, default=None)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--rewired', action='store_true')
parser.add_argument('--rewirer_step', type=float, default=0.2)
parser.add_argument('--model_indices', nargs="+", type=int, default=[0,1])
parser.add_argument('--num_edges', type=float, default=3000)
parser.add_argument('--rewirer_mode', type=str, default='supervised')
parser.add_argument('--lcc', action='store_true')
parser.add_argument('--loss', type=str, default='triplet')
parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--max_node_degree', type=int, default=10)
parser.add_argument('--with_node_feature', action='store_true')
parser.add_argument('--with_rand_signal', action='store_true')
args = parser.parse_args()


rseed(args.seed)
nseed(args.seed)
torch.manual_seed(args.seed)

device = DEVICE

if USE_CUDA:
    torch.cuda.manual_seed(args.seed)


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(torch.nn.Module):
    def __init__(self, dataset):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1), None


run(args.dataset, GPRGNN, args.rewired, args.runs, args.epochs, args.lr, args.weight_decay, args.patience,
    run_split=args.run_split, num_edges=args.num_edges, model_indices=args.model_indices,
    rewirer_mode=args.rewirer_mode, rewirer_step=args.rewirer_step, lcc=args.lcc, loss=args.loss, eps=args.eps,
    max_node_degree=args.max_node_degree, with_node_feature=args.with_node_feature,
    with_rand_signal=args.with_rand_signal, edge_step=args.edge_step)
