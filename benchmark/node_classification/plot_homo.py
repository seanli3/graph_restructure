from dataset.datasets import get_dataset
from models.encoder_node_classification import Rewirer
from models.utils import device
import torch

dataset = get_dataset('wisconsin', True)

if len(dataset.data.train_mask.shape) > 1:
    splits = dataset.data.train_mask.shape[1]
    has_splits = True
else:
    splits = 1
    has_splits = False

for split in range(1, 2):
    if has_splits:
        print('Split:', split)
    rewirer = Rewirer(
        dataset[0], DATASET=dataset.name, step=0.05, layers=[256, 128, 64], mode='supervised',
        split=split if has_splits else None
    )
    rewirer.load()
    rewirer.plot_homophily(dataset, [0], dataset[0].train_mask[:,split].logical_or(dataset[0].val_mask[:, split]))
    # rewirer.plot_homophily(dataset, [0], dataset[0].val_mask[:, split])
