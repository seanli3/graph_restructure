import torch
from tqdm import tqdm
from torch_geometric.utils import get_laplacian
import math


def train_rewirer(dataset, Model, train_idx, val_idx, epochs, lr, weight_decay, patience):
        model = Model(dataset, train_idx, val_idx)
        model.reset_parameters()

        best_val_loss = float('inf')
        eval_info_early_model = {}
        bad_counter = 0

        pbar = tqdm(range(0, epochs))
        for epoch in pbar:
            model.train()
            # Predict and calculate loss for user factor and bias
            optimizer = torch.optim.Adam([model.C], lr=lr,
                                        weight_decay=weight_decay)  # learning rate
            loss = model()
            # Backpropagate
            loss.backward()
            # Update the parameters
            optimizer.step()
            optimizer.zero_grad()

            model.eval()
            with torch.no_grad():
                val_loss = model()
            pbar.set_description('Epoch: {}, training loss: {:.2f}, validation loss: {:.2f}'.format(epoch, loss.item(), val_loss.item()))


            if val_loss < best_val_loss:
                eval_info_early_model['train_loss'] = loss
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


def create_filter(laplacian, b):
    return (torch.diag(torch.ones(laplacian.shape[0]) * 40).mm(
        (laplacian - torch.diag(torch.ones(laplacian.shape[0]) * b)).matrix_power(4)) + \
            torch.eye(laplacian.shape[0])).matrix_power(-2)


def rewire_graph(model, dataset):
    dictionary = {}
    step = 0.2
    for i in range(len(dataset)):
        L_index, L_weight = get_laplacian(dataset[i].edge_index, normalization='sym')
        L = torch.sparse_coo_tensor(L_index, L_weight).to_dense()
        filters = [create_filter(L, b) for b in torch.arange(0, 2.1, step)]
        D = torch.stack(filters, dim=2)
        dictionary[i] = D

    C = model['C']
    C = torch.nn.functional.normalize(C, dim=0, p=2)

    for i in range(len(dataset)):
        D = dictionary[i]
        L = D.matmul(C).squeeze()

        A_hat = torch.eye(L.shape[0]) - L
        A_hat = torch.nn.functional.normalize(A_hat, dim=1, p=2)

        k = math.ceil(dataset[i].num_edges/dataset[i].num_nodes/2)
        # k = 3
        A_one_zero = torch.zeros(A_hat.shape[0], A_hat.shape[1]) \
            .index_put((torch.arange(A_hat.shape[0]).repeat_interleave(k), A_hat.abs().topk(k, dim=1, largest=True)[1].view(-1)),
                       torch.tensor(1.))
        A_one_zero.masked_fill_(A_hat.abs() < 0.01, 0)

        edge_index = A_one_zero.nonzero().T
        dataset._data_list[i].edge_index = edge_index
