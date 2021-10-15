import torch
from tqdm import tqdm
import math
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from config import OGB_SAVED_MODEL_PATH_GRAPH_CLASSIFICATION, TU_SAVED_MODEL_PATH_GRAPH_CLASSIFICATION
from kernel.train_eval import k_fold
from models.model import RewireNetGraphClassification
from kernel.datasets import get_dataset
from pathlib import Path
import argparse
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--step', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

path = Path(__file__).parent

def train_rewirer(dataset, Model, train_idx, val_idx, test_idx, batch_size,
                  epochs, lr, weight_decay, patience, step):
        model = Model(dataset, step)
        model.to(device).reset_parameters()

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        best_val_loss = float('inf')
        eval_info_early_model = {}
        bad_counter = 0

        pbar = tqdm(range(0, epochs))
        for epoch in pbar:
            model.train()
            # Predict and calculate loss for user factor and bias
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                        weight_decay=weight_decay)  # learning rate

            train_loss = 0
            for data in train_loader:
                optimizer.zero_grad()
                data = data.to(device)
                loss = model(data)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            train_loss = train_loss / math.ceil(len(train_loader.dataset)/batch_size)

            val_loss = 0
            model.eval()
            for data in val_loader:
                data = data.to(device)
                with torch.no_grad():
                    val_loss = model(data)
            val_loss = val_loss / math.ceil(len(val_loader.dataset)/batch_size)

            pbar.set_description('Epoch: {}, training loss: {:.6f}, validation loss: {:.6f}'.format(
                epoch, train_loss, val_loss))

            if val_loss < best_val_loss:
                eval_info_early_model['train_loss'] = train_loss
                eval_info_early_model['val_loss'] = val_loss
                eval_info_early_model['epoch'] = epoch
                eval_info_early_model['step'] = step
                eval_info_early_model['model'] = deepcopy(model.state_dict())
                best_val_loss = val_loss
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter == patience:
                    break
        return eval_info_early_model


def train_tu(dataset, batch_size=128, epochs=200, lr=0.01, weight_decay=0.0005, patience=10, step=0.1):
    train_indices, test_indices, val_indices = k_fold(dataset, splits_dir="../kernel/splits")
    for i in range(len(train_indices)):
        dataset._data_list = None
        rewirer_model = train_rewirer(dataset, RewireNetGraphClassification, train_indices[i], val_indices[i],
                                      test_indices[i], batch_size, epochs, lr, weight_decay, patience, step)
        torch.save(rewirer_model, TU_SAVED_MODEL_PATH_GRAPH_CLASSIFICATION.format(dataset.name, i))


def train_ogb(dataset, batch_size=128, epochs=2000, lr=0.01, weight_decay=0.0005, patience=10, step=0.1):
    split_idx = dataset.get_idx_split()
    train_indices, test_indices, val_indices = split_idx["train"], split_idx['test'], split_idx['valid']
    dataset._data_list = None
    rewirer_model = train_rewirer(dataset, RewireNetGraphClassification, train_indices, val_indices,
                                  test_indices, batch_size, epochs, lr, weight_decay, patience, step)
    torch.save(rewirer_model, OGB_SAVED_MODEL_PATH_GRAPH_CLASSIFICATION.format(dataset.name))


# datasets = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']# , 'COLLAB']
# datasets = ['MUTAG', 'PROTEINS']# , 'COLLAB']
datasets = [args.dataset]# , 'COLLAB']

for dataset_name in datasets:
    dataset = get_dataset(dataset_name)
    print(args)
    if dataset_name.upper() in ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COLLAB']:
        train_tu(dataset, step=args.step, lr=args.lr, batch_size=args.batch_size)
    elif 'ogbg' in dataset_name.lower():
        train_ogb(dataset, step=args.step, lr=args.lr, batch_size=args.batch_size)
