import argparse
import math
import torch
import torch.nn.functional as F
from config import USE_CUDA, DEVICE
from random import seed as rseed
from numpy.random import seed as nseed
from pathlib import Path
from benchmark.node_classification.train_eval import run
from models.utils import get_normalized_adj, cosine_sim

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
parser.add_argument('--dropout', type=float, default=0.9)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--rewired', action='store_true')
parser.add_argument('--rewirer_step', type=float, default=0.2)
parser.add_argument('--model_indices', nargs="+", type=int, default=[0,1])
parser.add_argument('--num_edges', type=float, default=3000)
parser.add_argument('--rewirer_mode', type=str, default='supervised')
parser.add_argument('--run_split', type=int, default=None)
args = parser.parse_args()


rseed(args.seed)
nseed(args.seed)
torch.manual_seed(args.seed)

device = DEVICE

if USE_CUDA:
    torch.cuda.manual_seed(args.seed)


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def reset_parameters(self):
        pass

    def forward(self, data):
        x, edge_index, train_mask, y = data.x, data.adj_t if hasattr(data, 'adj_t') else data.edge_index, data.train_mask[:, args.run_split], data.y
        edge_weight = None
        label = torch.nn.functional.one_hot(y)
        train_label = label.float()
        train_label[train_mask.logical_not()] = torch.zeros(y.max() + 1, device=device)
        norm_adj = get_normalized_adj(edge_index, edge_weight, x.shape[0])

        bad_counter = 0
        pre_nodes_without_labels_fea = torch.tensor(math.nan, device=device)
        fea_bad_counter = 0
        threshold = 0.5
        nodes_without_labels = x.shape[0] - train_mask.count_nonzero()
        for i in range(1000):
            pre_nodes_without_labels = nodes_without_labels
            known_mask = train_label.max(1)[0] != 0
            nodes_without_labels = x.shape[0] - known_mask.count_nonzero()
            if  nodes_without_labels == 0:
                break
            if pre_nodes_without_labels == nodes_without_labels:
                bad_counter += 1
            else:
                threshold = 0.5
                bad_counter = 0
            # print( 'nodes without labels:,', str(nodes_without_labels))
            z = train_label
            z = norm_adj.matrix_power(1+bad_counter).mm(z)
            mask = (z.max(1)[0] >= 0.5).logical_and(known_mask.logical_not())
            train_label[mask] = (z[mask] >= 0.5).float()
            if bad_counter > 1:
                unknown_mask = known_mask.logical_not()
                x_unknown = x[unknown_mask]
                # sim = x_unknown.mm(x.T)
                sim = cosine_sim(x).squeeze().fill_diagonal_(0)
                unknown_labels = train_label[sim.argmax(1)]
                if pre_nodes_without_labels_fea == nodes_without_labels.float():
                    sim = sim.where(sim < threshold, torch.tensor(threshold-0.1, device=device))
                    threshold -= 0.1
                else:
                    threshold = 0.5
                fea_simimlar_mask = sim.max(1)[0] >= threshold
                fea_simimlar_idx = sim.argmax(1)
                fea_similar_labels = train_label[fea_simimlar_idx]
                mask_to_set = unknown_mask.logical_and(fea_simimlar_mask)
                train_label[mask_to_set] = fea_similar_labels[mask_to_set]
                bad_counter = 0
                pre_nodes_without_labels_fea = nodes_without_labels

        return train_label, x


run(args.dataset, Net, args.rewired, args.runs, args.epochs, args.lr, args.weight_decay, args.patience, run_split=args.run_split,
    num_edges=args.num_edges, model_indices=args.model_indices, rewirer_mode=args.rewirer_mode, rewirer_step=args.rewirer_step)
