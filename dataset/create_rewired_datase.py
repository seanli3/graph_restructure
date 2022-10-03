from __future__ import division
import argparse
import torch
from dataset.datasets import get_dataset
from models.encoder_node_classification_fast import Rewirer
from config import SEED, DEVICE

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--step', type=float, default=0.1)
parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--max_node_degree', type=int, default=5)
parser.add_argument('--edge_step', type=int, default=None)
args = parser.parse_args()
print(args)

device = DEVICE

def run(dataset_name, normalize_features=True, run_split=None, rewirer_layers=[256, 128, 64], rewirer_step=0.2,
        lcc=False, eps=0.1, max_node_degree=5, with_node_feature=True, with_rand_signal=True, edge_step=None,
        h_den=None):
    dataset = get_dataset(dataset_name, normalize_features, lcc=lcc, h_den=h_den)
    if len(dataset.data.train_mask.shape) > 1:
        has_splits = True
    else:
        has_splits = False

    from random import seed as rseed
    from numpy.random import seed as nseed
    rseed(SEED)
    nseed(SEED)
    torch.manual_seed(SEED)

    if device == torch.device('cuda'):
        # print("-----------------------Training on CUDA-------------------------")
        torch.cuda.manual_seed(SEED)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset = Rewirer.rewire(dataset, run_split if has_splits else None, eps=eps, max_node_degree=max_node_degree,
                             step=rewirer_step, layers=rewirer_layers, with_node_feature=with_node_feature,
                             with_rand_signal=with_rand_signal, edge_step=edge_step, h_den=h_den)
    torch.save(dataset, '{}_split_{}_eps_{}_max_node_degree_{}_step_{}_layers_{}_edges_{}.pt'.format(
        dataset_name, run_split, eps, max_node_degree, rewirer_step, rewirer_layers, dataset.data.num_edges
    ))


run(args.dataset, run_split=args.split, rewirer_step=args.step, eps=args.eps, max_node_degree=args.max_node_degree,
    edge_step=args.edge_step)