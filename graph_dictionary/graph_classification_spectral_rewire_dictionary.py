import torch
from tqdm import tqdm
from torch_geometric.utils import get_laplacian
import math
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from graph_dictionary.graph_classification_model import create_filter
from kernel.train_eval import k_fold
from graph_dictionary.graph_classification_model import DictNet
from kernel.datasets import get_dataset

device = torch.device('cpu')

def train_rewirer(dataset, Model, train_idx, val_idx, test_idx, batch_size,
                  epochs, lr, weight_decay, patience):
        model = Model(dataset)
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


def rewire_graphs(dataset):
    train_indices, test_indices, val_indices = k_fold(dataset, splits_dir="../kernel/splits")
    for i in range(len(train_indices)):
        dataset._data_list = None
        best_model = train_rewirer(dataset, DictNet, train_indices[i], val_indices[i],
                                              test_indices[i], 128, 2000, 0.005, 0.0005, 10)
        rewired_dataset = rewire_graph(best_model, dataset)
        torch.save(rewired_dataset, '../kernel/splits/{}_dataset_split_{}.pt'.format(dataset.name, i))

def rewire_graph(model, dataset):
    dictionary = {}
    step = 0.2
    for i in range(len(dataset)):
        L_index, L_weight = get_laplacian(dataset[i].edge_index, normalization='sym')
        L = torch.sparse_coo_tensor(L_index, L_weight).to_dense().to(device)
        filters = [create_filter(L, b) for b in torch.arange(0, 2.1, step).to(device)]
        D = torch.stack(filters, dim=2)
        dictionary[i] = D

    C = model['C']
    C = torch.nn.functional.normalize(C, dim=0, p=2)

    for i in range(len(dataset)):
        D = dictionary[i]
        L = D.matmul(C).squeeze()

        A_hat = torch.eye(L.shape[0]).to(device) - L
        A_hat = torch.nn.functional.normalize(A_hat, dim=1, p=2)

        # k = math.floor(dataset[i].num_edges/dataset[i].num_nodes)
        # A_one_zero = torch.zeros(A_hat.shape[0], A_hat.shape[1]).to(device) \
        #     .index_put((torch.arange(A_hat.shape[0]).to(device).repeat_interleave(k), A_hat.abs().topk(k, dim=1, largest=True)[1].view(-1)),
        #                torch.tensor(1.).to(device))
        A_one_zero = A_hat.masked_fill(A_hat.abs() < 0.1, 0)

        edge_index = A_one_zero.nonzero().T
        dataset._data_list[i].edge_index = edge_index
    return dataset


# datasets = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']# , 'COLLAB']
datasets = ['REDDIT-BINARY']# , 'COLLAB']

for dataset_name in datasets:
    dataset = get_dataset(dataset_name)
    rewire_graphs(dataset)