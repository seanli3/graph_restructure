import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, remove_self_loops
from random import seed as rseed
from torch_geometric.nn.inits import glorot
from torch.optim import Adam
import time
from numpy.random import seed as nseed
from webkb import get_dataset, run as run_webkb
from citation import get_dataset as get_citation_data, run as run_citation
import numpy as np
import networkx as nx
import math
from sklearn.linear_model import OrthogonalMatchingPursuit
from graph_dictionary.model import DictNet

# %%

DATASET = 'Chameleon'

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


def run(name, Model, runs, epochs, lr, weight_decay, patience):
    val_losses, durations = [], []
    model = Model(name, 5)
    dataset = model.dataset
    data = model.data

    for _ in range(runs):
        print('Runs:', _)
        model.reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        t_start = time.perf_counter()

        best_val_loss = float('inf')
        eval_info_early_model = {}
        bad_counter = 0

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            loss = model(model.train_mask)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = model(model.val_mask)

            outs = {}
            outs['train_loss'] = loss
            outs['val_loss'] = val_loss
            outs['epoch'] = epoch

            if epoch % 10 == 0:
                print(outs)

            if outs['val_loss'] < best_val_loss:
                eval_info_early_model = outs
                eval_info_early_model['C'] = torch.clone(model.C.detach())
                eval_info_early_model['W'] = torch.clone(model.W.detach())
                eval_info_early_model['A'] = torch.clone(model.A.detach())
                best_val_loss = np.min((best_val_loss, outs['val_loss']))
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter == patience:
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

        val_losses.append(eval_info_early_model.get('val_loss'))
        durations.append(t_end - t_start)

    val_losses, duration = torch.tensor(val_losses), torch.tensor(durations)

    print('Val Loss: {:.4f} ± {:.3f}, Duration: {:.3f}, Epoch: {}'.
          format(val_losses.mean().item(),
                 val_losses.std().item(),
                 duration.mean().item(),
                 eval_info_early_model['epoch']))
    torch.save(eval_info_early_model, './best_model.pt')
    return eval_info_early_model


eval_info_early_model = run(DATASET, DictNet, 1, 2000, 0.01, 0.005, 100)


# features_hat = eval_info_early_model['D'].mm(eval_info_early_model['C']).detach()
# L = eval_info_early_model['L'].detach()

# use_dataset = lambda: get_dataset(DATASET, True, self_loop=True, features=features_hat)
#
# run_webkb(use_dataset, Net, 1, 2000, 0.01, 0.0005, 100)

#
#
#
#
# #%%
