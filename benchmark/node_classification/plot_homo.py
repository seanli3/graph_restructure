from dataset.datasets import get_dataset
from models.encoder_node_classification import Rewirer
from models.utils import device
import torch
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--step', type=float, default=0.1)
    parser.add_argument('--split', type=int, default=None)
    parser.add_argument('--mode', type=str, default='supervised')
    parser.add_argument('--lcc', action='store_true')
    parser.add_argument('--loss', type=str, default='triplet')
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--early_stop', action='store_true')
    args = parser.parse_args()

    print('Running plot on {}, with step={} split={} mode={} lcc={} loss={} eps={}'
          .format(args.dataset, args.step, args.split, args.mode, args.lcc, args.loss, args.eps))

    if args.loss not in ['triplet', 'contrastive', 'npair', 'mse']:
        raise RuntimeError('loss has to be one of triplet, contrastive, npair, mse')
    DATASET = args.dataset
    dataset = get_dataset(DATASET, normalize_features=True, lcc=args.lcc)
    data = dataset[0]

    rewirer = Rewirer(data, step=args.step, layers=[256, 128, 64], DATASET=DATASET, mode=args.mode, split=args.split,
                     loss=args.loss, eps=args.eps)
    rewirer.load()
    rewirer.plot_homophily(
        dataset, [0],
        dataset[0].val_mask[:, args.split] if args.split is not None else dataset[0].val_mask,
        early_stop=args.early_stop
    )
