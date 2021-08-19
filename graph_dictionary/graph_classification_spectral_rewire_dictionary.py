import torch
from tqdm import tqdm
import math
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from kernel.train_eval import k_fold
from graph_dictionary.graph_classification_model import DictNet
from kernel.datasets import get_dataset
from pathlib import Path

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
            optimizer = torch.optim.Adam([model.C], lr=lr,
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

            pbar.set_description('Epoch: {}, training loss: {:.4f}, validation loss: {:.4f}'.format(
                epoch, train_loss, val_loss))

            if val_loss < best_val_loss:
                eval_info_early_model['train_loss'] = train_loss
                eval_info_early_model['val_loss'] = val_loss
                eval_info_early_model['epoch'] = epoch
                eval_info_early_model['C'] = torch.clone(model.C.detach())
                best_val_loss = val_loss
                bad_counter = 0
                # torch.save(eval_info_early_model, './{}_best_model_split_{}.pt'.format(DATASET, split))
            else:
                bad_counter += 1
                if bad_counter == patience:
                    break
        return eval_info_early_model


def train(dataset, batch_size=128, epochs=2000, lr=0.01, weight_decay=0.0005, patience=10, step=0.1):
    train_indices, test_indices, val_indices = k_fold(dataset, splits_dir="../kernel/splits")
    for i in range(len(train_indices)):
        dataset._data_list = None
        rewirer_model = train_rewirer(dataset, DictNet, train_indices[i], val_indices[i],
                                              test_indices[i], batch_size, epochs, lr, weight_decay, patience, step)
        torch.save(rewirer_model, path / '../kernel/saved_models/{}_dataset_split_{}.pt'.format(dataset.name, i))


datasets = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']# , 'COLLAB']

for dataset_name in datasets:
    dataset = get_dataset(dataset_name)
    train(dataset, step=0.1, lr=0.002, batch_size=256)