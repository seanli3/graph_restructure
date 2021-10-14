import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from kernel.diff_pool import DiffPool
from graph_dictionary.model import RewireNetGraphClassification
from graph_dictionary.utils import rewire_graph
from pathlib import Path
from kernel.datasets import get_dataset


path = Path(__file__).parent

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def cross_validation_with_val_set(dataset_name, Net, epochs, batch_size, lr,
                                  lr_decay_factor,lr_decay_step_size, weight_decay,
                                  num_layers, hidden, logger=None, rewired=False):

    val_losses, accs, durations = [], [], []
    folds = 10
    dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds=folds))):

        if rewired:
            rewirer_state = torch.load(path / '../kernel/saved_models/{}_dataset_split_{}.pt'.format(dataset.name, fold))
            step = rewirer_state['model']['step']
            rewirer = RewireNetGraphClassification(dataset, step)
            rewirer.load_state_dict(rewirer_state['model'])
            rewirer.eval()
            rewirer.to(device)
            new_dataset = get_dataset(dataset_name, sparse=Net != DiffPool, transform=rewirer.transform_edges)
        else:
            new_dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
        model = Net(new_dataset, num_layers, hidden)

        train_dataset = new_dataset[train_idx]
        test_dataset = new_dataset[test_idx]
        val_dataset = new_dataset[val_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.reset_parameters()
        model.to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss = train(model, optimizer, train_loader)
            val_losses.append(eval_loss(model, val_loader))
            accs.append(eval_acc(model, test_loader))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_losses[-1],
                'test_acc': accs[-1],
            }

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(loss_mean, acc_mean, acc_std, duration_mean))

    return loss_mean, acc_mean, acc_std


def k_fold(dataset, splits_dir="./splits", folds=10):
    train_file_name = path / './splits/{}_train.pt'.format(dataset.name)
    val_file_name = path / './splits/{}_val.pt'.format(dataset.name)
    test_file_name = path / './splits/{}_test.pt'.format(dataset.name)

    try:
        print('Loading split files...')
        train_indices = torch.load(train_file_name)
        val_indices = torch.load(val_file_name)
        test_indices = torch.load(test_file_name)
    except FileNotFoundError:
        print('Split files not found, creating them...')
        skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

        test_indices, train_indices = [], []
        for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
            test_indices.append(torch.from_numpy(idx).to(torch.long))

        val_indices = [test_indices[i - 1] for i in range(folds)]

        for i in range(folds):
            train_mask = torch.ones(len(dataset), dtype=torch.bool)
            train_mask[test_indices[i]] = 0
            train_mask[val_indices[i]] = 0
            train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

        torch.save(train_indices, path / '{}/{}_train.pt'.format(splits_dir, dataset.name))
        torch.save(val_indices, path / '{}/{}_val.pt'.format(splits_dir, dataset.name))
        torch.save(test_indices, path / '{}/{}_test.pt'.format(splits_dir, dataset.name))
    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)
