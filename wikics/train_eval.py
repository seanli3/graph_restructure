
from __future__ import division

import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.metrics import f1_score
import numpy as np
# from torch_sparse import spmm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run(use_dataset, Model, runs, epochs, lr, weight_decay, patience, logger=None, split=0):
    val_losses, train_accs, val_accs, test_accs, test_macro_f1s, durations = [], [], [], [], [], []

    dataset = use_dataset()
    data = dataset[0]

    for _ in range(1):
        train_mask = data.train_mask[:, split].view(-1)
        val_mask = data.val_mask[:, split].view(-1)
        test_mask = data.test_mask

        for _ in range(runs):
            model = Model(dataset)
            model.to(device).reset_parameters()
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_start = time.perf_counter()

            best_val_loss = float('inf')
            best_val_acc = float(0)
            eval_info_early_model = None
            bad_counter = 0

            for epoch in range(1, epochs + 1):
                train(model, optimizer, data, train_mask)
                eval_info = evaluate(model, data, train_mask, val_mask, test_mask)
                eval_info['epoch'] = epoch
                if epoch % 100 == 0:
                    print(eval_info)

                if logger is not None:
                    logger(eval_info)

                if eval_info['val_acc'] > best_val_acc or eval_info['val_loss'] < best_val_loss:
                    if eval_info['val_acc'] >= best_val_acc and eval_info['val_loss'] <= best_val_loss:
                        eval_info_early_model = eval_info
                        # torch.save(model.state_dict(), './best_{}_appnp.pkl'.format(dataset.name))
                    best_val_acc = np.max((best_val_acc, eval_info['val_acc']))
                    best_val_loss = np.min((best_val_loss, eval_info['val_loss']))
                    bad_counter = 0
                else:
                    bad_counter += 1
                    if bad_counter == patience:
                        break

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_end = time.perf_counter()
            durations.append(t_end - t_start)

            val_losses.append(eval_info_early_model['val_loss'])
            train_accs.append(eval_info_early_model['train_acc'])
            val_accs.append(eval_info_early_model['val_acc'])
            test_accs.append(eval_info_early_model['test_acc'])
            test_macro_f1s.append(eval_info_early_model['test_macro_f1'])
            durations.append(t_end - t_start)

    val_losses, train_accs, val_accs, test_accs, test_macro_f1s, duration = tensor(val_losses), tensor(train_accs), tensor(val_accs), \
                                                                            tensor(test_accs), tensor(test_macro_f1s), tensor(durations)

    print('Val Loss: {:.4f} ± {:.3f}, Train Accuracy: {:.3f} ± {:.3f}, Val Accuracy: {:.3f} ± {:.3f}, Test Accuracy: {:.3f} ± {:.3f}, Macro-F1: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(val_losses.mean().item(),
                 val_losses.std().item(),
                 train_accs.mean().item(),
                 train_accs.std().item(),
                 val_accs.mean().item(),
                 val_accs.std().item(),
                 test_accs.mean().item(),
                 test_accs.std().item(),
                 test_macro_f1s.mean().item(),
                 test_macro_f1s.std().item(),
                 duration.mean().item()))
    return test_accs.mean().item()


def train(model, optimizer, data, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data)[0]
    # coefficients = torch.eye(filterbanks[0].shape[0], filterbanks[0].shape[1])
    # for c in filterbanks:
    #     coefficients = spmm(c.indices(), c.values(), c.shape[0], c.shape[1], coefficients)
    # discrimative_loss = coefficients.mean()
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()


def evaluate(model, data, train_mask, val_mask, test_mask):
    model.eval()

    with torch.no_grad():
        logits = model(data)[0]

    outs = {}
    for key, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = f1_score(data.y[mask].cpu(), logits[mask].max(1)[1].cpu(), average='micro')
        outs['{}_macro_f1'.format(key)] = f1_score(data.y[mask].cpu(), logits[mask].max(1)[1].cpu(), average='macro')

    return outs
