from dataset.datasets import get_dataset
from models.encoder_node_classification import Rewirer

dataset = get_dataset('Cora', True)

if len(dataset.data.train_mask.shape) > 1:
    splits = dataset.data.train_mask.shape[1]
    has_splits = True
else:
    splits = 1
    has_splits = False

for split in range(splits):
    if has_splits:
        print('Split:', split)
    rewirer = Rewirer(
        dataset[0], DATASET=dataset.name, step=0.1, layers=[256, 128, 64], mode='supervised',
        split=split if has_splits else None
    )
    rewirer.load()
    rewirer.plot_homophily(dataset, [2])