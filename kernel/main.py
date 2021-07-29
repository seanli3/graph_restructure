from itertools import product

import argparse
from kernel.datasets import get_dataset
from kernel.train_eval import cross_validation_with_val_set

from kernel.gcn import GCN, GCNWithJK
from kernel.graph_sage import GraphSAGE, GraphSAGEWithJK
from kernel.gin import GIN0, GIN0WithJK, GIN, GINWithJK
from kernel.graclus import Graclus
from kernel.top_k import TopK
from kernel.sag_pool import SAGPool
from kernel.diff_pool import DiffPool
from kernel.edge_pool import EdgePool
from kernel.global_attention import GlobalAttentionNet
from kernel.set2set import Set2SetNet
from kernel.sort_pool import SortPool
from kernel.asap import ASAP
from random import seed as rseed
from numpy.random import seed as nseed
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--seed', type=int, default=49)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

rseed(args.seed)
nseed(args.seed)
torch.manual_seed(args.seed)
# torch.use_deterministic_algorithms(True)

args.cuda = args.cuda and torch.cuda.is_available()

if args.cuda:
    print("-----------------------Training on CUDA-------------------------")
    torch.cuda.manual_seed(args.seed)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

# layers = [1, 2, 3, 4, 5]
# hiddens = [16, 32, 64, 128]
layers = [1, 2]
hiddens = [32]
# datasets = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']# , 'COLLAB']
datasets = ['PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']# , 'COLLAB']
nets = [
    # GCNWithJK,
    # GraphSAGEWithJK,
    # GIN0WithJK,
    # GINWithJK,
    # Graclus,
    # TopK,
    # SAGPool,
    # DiffPool,
    # EdgePool,
    GCN,
    # GraphSAGE,
    # GIN0,
    # GIN,
    # GlobalAttentionNet,
    # Set2SetNet,
    # SortPool,
    # ASAP,
]


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, val_loss, test_acc))


results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print('-----\n{} - {}'.format(dataset_name, Net.__name__))
    for num_layers, hidden in product(layers, hiddens):
        dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
        model = Net(dataset, num_layers, hidden)
        loss, acc, std = cross_validation_with_val_set(
            dataset,
            model,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            logger=None,
        )
        if loss < best_result[0]:
            best_result = (loss, acc, std)

    desc = '{:.3f} ± {:.3f}'.format(best_result[1], best_result[2])
    print('Best result - {}'.format(desc))
    results += ['{} - {}: {}'.format(dataset_name, model, desc)]
print('-----\n{}'.format('\n'.join(results)))
