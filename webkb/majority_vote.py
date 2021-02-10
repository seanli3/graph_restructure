import argparse
import networkx as nx
from scipy.sparse import coo_matrix
import numpy as np
import time
import torch
from torch import tensor
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from random import choice, seed as rseed
from numpy.random import seed as nseed
from collections import Counter
from torch.nn import functional as F
from citation.train_eval import evaluate

from citation import get_dataset, random_planetoid_splits, run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--hidden', type=int, default=8)
parser.add_argument('--seed', type=int, default=729, help='Random seed.')
parser.add_argument('--lcc', type=bool, default=False)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--output_heads', type=int, default=1)
parser.add_argument('--edge_dropout', type=float, default=0)
parser.add_argument('--node_feature_dropout', type=float, default=0)
parser.add_argument('--dissimilar_t', type=float, default=1)
args = parser.parse_args()

rseed(args.seed)
nseed(args.seed)
torch.manual_seed(args.seed)

permute_masks = random_planetoid_splits if args.random_splits else None

def multimode(eles, num_classes):
    res = []
    if(len(eles) == 0):
        eles = list(range(num_classes))
    counter = Counter(eles)
    temp = counter.most_common(1)[0][1]
    for ele in eles:
      if eles.count(ele) == temp:
        res.append(ele)
    return list(set(res))

use_dataset = lambda : get_dataset(args.dataset, args.normalize_features, edge_dropout=args.edge_dropout,
                                    permute_masks=permute_masks,
                                    node_feature_dropout=args.node_feature_dropout, dissimilar_t=args.dissimilar_t, lcc=args.lcc)

val_losses, train_accs, val_accs, test_accs, f1, durations = [], [], [], [], [], []

class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.num_classes = dataset.num_classes
        adj = coo_matrix(
            (np.ones(dataset[0].num_edges),
             (dataset[0].edge_index[0].numpy(), dataset[0].edge_index[1].numpy())),
            shape=(dataset[0].num_nodes, dataset[0].num_nodes))
        self.G = nx.Graph(adj)

    def forward(self, data):
        train_nodes = set(data.train_idx.nonzero().view(-1).tolist())

        logits = []
        for n in range(data.num_nodes):
            hop_1_neighbours = set(nx.ego_graph(self.G, n, 1).nodes())
            hop_1_labels = data.y[list(hop_1_neighbours)]
            label = choice(multimode(hop_1_labels.tolist(), self.num_classes))
            logits.append(F.one_hot(torch.tensor(label), self.num_classes).float())
        return torch.stack(logits, 0), None

if torch.cuda.is_available():
    torch.cuda.synchronize()

if torch.cuda.is_available():
    torch.cuda.synchronize()

for _ in range(args.runs):
    dataset = use_dataset()
    model = Net(dataset)

    t_start = time.perf_counter()

    best_val_loss = float('inf')
    best_val_acc = float(0)

    eval_info = evaluate(model, dataset[0])

    t_end = time.perf_counter()
    durations.append(t_end - t_start)

    val_losses.append(eval_info['val_loss'])
    train_accs.append(eval_info['train_acc'])
    val_accs.append(eval_info['val_acc'])
    test_accs.append(eval_info['test_acc'])
    f1.append(eval_info['f1_score'])
    durations.append(t_end - t_start)

val_losses, train_accs, val_accs, test_accs, f1, duration = tensor(val_losses), tensor(train_accs), tensor(val_accs), \
                                                            tensor(test_accs), tensor(f1), tensor(durations)

print(
    'Val Loss: {:.4f}, Train Accuracy: {:.3f} ± {:.3f}, Val Accuracy: {:.3f} ± {:.3f}, Test Accuracy: {:.3f} ± {:.3f}, F1: {:.3f} ± {:.3f}, Duration: {:.3f}'.
    format(val_losses.mean().item(),
           train_accs.mean().item(),
           train_accs.std().item(),
           val_accs.mean().item(),
           val_accs.std().item(),
           test_accs.mean().item(),
           test_accs.std().item(),
           f1.mean().item(),
           f1.std().item(),
           duration.mean().item()))
