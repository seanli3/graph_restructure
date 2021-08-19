from itertools import product

import argparse
from kernel.datasets import get_dataset
from kernel.train_eval import cross_validation_with_val_set

from kernel.gcn import GCN, GCNWithJK
from kernel.graph_sage import GraphSAGE, GraphSAGEWithJK
from kernel.gin import GIN0, GIN0WithJK, GIN, GINWithJK
from kernel.top_k import TopK
from kernel.diff_pool import DiffPool
from kernel.global_attention import GlobalAttentionNet
from kernel.set2set import Set2SetNet
from kernel.sort_pool import SortPool
from random import seed as rseed
from numpy.random import seed as nseed
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--seed', type=int, default=172)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--rewired', action='store_true')
parser.add_argument('--dataset', type=str)
parser.add_argument('--net', type=str)
parser.add_argument('--layers', type=int)
parser.add_argument('--hiddens', type=int)
parser.add_argument('--max_degree', type=int, default=5)
parser.add_argument('--threshold', type=float, default=0.01)

args = parser.parse_args()

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

args.cuda = args.cuda and torch.cuda.is_available()

if args.layers:
    layers = [args.layers]
else:
    layers = [1, 2, 3, 4, 5]

if args.hiddens:
    hiddens = [args.hiddens]
else:
    hiddens = [16, 32, 64, 128]

if args.dataset:
    datasets = [args.dataset]
else:
    datasets = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']# , 'COLLAB']

if args.net:
    nets = [eval(args.net)]
else:
    nets = [
        GCNWithJK,
        GraphSAGEWithJK,
        GIN0WithJK,
        GINWithJK,
        TopK,
        GCN,
        GraphSAGE,
        GIN0,
        GIN,
        GlobalAttentionNet,
        Set2SetNet,
        SortPool,
    ]


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, val_loss, test_acc))

results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0, 0, 0)  # (loss, acc, std)
    for num_layers, hidden in product(layers, hiddens):
        print('-----\n{} - {} - hidden {} - layers - {} - max_degree {} - threshold {}'\
              .format(dataset_name, Net.__name__, str(hidden), str(num_layers),\
                      str(args.max_degree), str(args.threshold)))
        rseed(args.seed)
        nseed(args.seed)
        torch.manual_seed(args.seed)

        args.cuda = args.cuda and torch.cuda.is_available()

        if args.cuda:
            print("-----------------------Training on CUDA-------------------------")
            torch.cuda.manual_seed(args.seed)
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_num_threads(8)

        dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
        model = Net(dataset, num_layers, hidden)
        loss, acc, std = cross_validation_with_val_set(
            dataset,
            model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            logger=None,
            rewired=args.rewired,
            max_degree=args.max_degree,
            threshold=args.threshold
        )
        if loss < best_result[0]:
            best_result = (loss, acc, std, num_layers, hidden)

    desc = '{:.3f} ± {:.3f}, hidden: {}, layer: {}'.format(best_result[1], best_result[2], best_result[4], best_result[3])
    print('Best result - {}'.format(desc))
    results += ['{} - {}: {}'.format(dataset_name, model, desc)]
print('-----\n{}'.format('\n'.join(results)))
