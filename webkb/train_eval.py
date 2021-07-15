from __future__ import division

import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
# from torch_sparse import spmm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run(use_dataset, Model, runs, epochs, lr, weight_decay, patience, logger=None):
    val_losses, train_accs, val_accs, test_accs, test_macro_f1s, durations = [], [], [], [], [], []
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for _ in range(runs):
        print('Runs:', _)
        for split in range(10):
            print('Split:', split)
        
            dataset = use_dataset(split)
            data = dataset[0]

            model = Model(dataset)
            model.to(device).reset_parameters()
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            t_start = time.perf_counter()

            best_val_loss = float('inf')
            best_val_acc = float(0)
            eval_info_early_model = None
            bad_counter = 0


            pbar = tqdm(range(0, epochs))
            for epoch in pbar:
                train(model, optimizer, data, split)
                eval_info = evaluate(model, data, split)
                eval_info['epoch'] = epoch
                if epoch % 10 == 0:
                    pbar.set_description(
                        'Epoch: {}, train loss: {:.2f}, val loss: {:.2f}, train acc: {:.4f}, val acc: {:.4f},'
                        'test loss: {:.2f}, test acc: {:.4f}'
                            .format(
                                epoch, eval_info['train_loss'], eval_info['val_loss'], eval_info['train_acc'],
                                eval_info['val_acc'], eval_info['test_loss'], eval_info['test_acc']
                            )
                    )

                if logger is not None:
                    logger(eval_info)

                if eval_info['val_acc'] > best_val_acc or eval_info['val_loss'] < best_val_loss:
                    if eval_info['val_acc'] >= best_val_acc and eval_info['val_loss'] <= best_val_loss:
                        eval_info_early_model = eval_info
                        # torch.save(model.state_dict(), './best_{}_single_dec_split_{}.pkl'.format(dataset.name, split))
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

    print('Val Loss: {:.4f} ± {:.3f}, Train Accuracy: {:.3f} ± {:.3f}, Val Accuracy: {:.3f} ± {:.3f}, Test Accuracy: {:.3f} ± {:.3f}, Macro-F1: {:.3f} ± {:.3f}, Duration: {:.3f}, Epoch: {}'.
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
                 duration.mean().item(),
                 eval_info_early_model['epoch']))

    # print('row_diff:', cal_row_diff(model, data, split), 'col_diff:', cal_col_diff(model, data, split))
    return test_accs.mean().item()



def cal_col_diff(model, data, split):
    with torch.no_grad():
        model.eval()
        _, embeddings = model(data)
        test_embeddings = embeddings[data.test_mask[split]]
        normalized_test_embeddings = test_embeddings/torch.linalg.norm(test_embeddings, 1, dim=0)
        index = list(range(test_embeddings.shape[1]))
        sum = 0
        for _ in range(test_embeddings.shape[1]):
            index.insert(0, index.pop())
            sum += torch.linalg.norm(normalized_test_embeddings - normalized_test_embeddings[:, index], 2, dim=0).sum()
        row_diff = sum / pow(test_embeddings.shape[1], 2)
    return row_diff



def cal_row_diff(model, data, split):
    with torch.no_grad():
        model.eval()
        _, embeddings = model(data)
        test_embeddings = embeddings[data.test_mask[split]]
        normalized_test_embeddings = test_embeddings/torch.linalg.norm(test_embeddings, 1, dim=1).view(-1, 1)
        # calculate row-diff
        index = list(range(test_embeddings.shape[0]))
        sum = 0
        for _ in range(test_embeddings.shape[0]):
            index.insert(0, index.pop())
            sum += torch.linalg.norm(normalized_test_embeddings - normalized_test_embeddings[index], 2, dim=1).sum()
        row_diff = sum / pow(test_embeddings.shape[0], 2)
    return row_diff

def train(model, optimizer, data, split):
    model.train()
    optimizer.zero_grad()
    out = model(data)[0]
    # coefficients = torch.eye(filterbanks[0].shape[0], filterbanks[0].shape[1])
    # for c in filterbanks:
    #     coefficients = spmm(c.indices(), c.values(), c.shape[0], c.shape[1], coefficients)
    # discrimative_loss = coefficients.mean()
    loss = F.nll_loss(out[data.train_mask[split]], data.y[data.train_mask[split]])
    loss.backward()
    optimizer.step()


def evaluate(model, data, split):
    model.eval()

    with torch.no_grad():
        logits = model(data)[0]

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)][split]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = f1_score(data.y[mask].cpu(), logits[mask].max(1)[1].cpu(), average='micro')
        outs['{}_macro_f1'.format(key)] = f1_score(data.y[mask].cpu(), logits[mask].max(1)[1].cpu(), average='macro')

    return outs
