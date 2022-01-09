from __future__ import division
import time
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import AdamW
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
from dataset.datasets import get_dataset
from models.encoder_node_classification import Rewirer
from config import USE_CUDA, SEED

device = torch.device('cuda') if torch.cuda.is_available() and USE_CUDA else torch.device('cpu')


def run(dataset_name, Model, rewired, runs, epochs, lr, weight_decay, patience, normalize_features=True,
        rewirer_mode='supervised', rewirer_layers=[256, 128, 64], rewirer_step=0.2, edge_per=0.8, model_indices=[0,1]):

    dataset = get_dataset(dataset_name, normalize_features)
    if len(dataset.data.train_mask.shape) > 1:
        splits = dataset.data.train_mask.shape[1]
        has_splits = True
    else:
        splits = 1
        has_splits = False

    val_losses, train_accs, val_accs, test_accs, test_macro_f1s, durations = [], [], [], [], [], []
    for split in range(splits):
        from random import seed as rseed
        from numpy.random import seed as nseed
        rseed(SEED)
        nseed(SEED)
        torch.manual_seed(SEED)

        if device == torch.device('cuda'):
            print("-----------------------Training on CUDA-------------------------")
            torch.cuda.manual_seed(SEED)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        if has_splits:
            print('Split:', split)
        if rewired:
            dataset = get_dataset(dataset_name, normalize_features)
            rewirer = Rewirer(
                dataset[0], DATASET=dataset.name, step=rewirer_step, layers=rewirer_layers,
                mode=rewirer_mode, split=split if has_splits else None)
            rewirer.load()
            dataset = get_dataset(dataset_name, normalize_features,
                                  transform=lambda d:rewirer.rewire(d, model_indices,edge_per)
                                  )

        if device == torch.device('cuda'):
            torch.cuda.synchronize()

        data = dataset[0]

        for _ in range(runs):
            print('Runs:', _)

            model = Model(dataset)
            model.to(device).reset_parameters()
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            t_start = time.perf_counter()

            best_val_loss = float('inf')
            best_val_acc = float(0)
            eval_info_early_model = None
            bad_counter = 0


            pbar = tqdm(range(0, epochs))
            for epoch in pbar:
                train(model, optimizer, data, split=split if has_splits else None)
                eval_info = evaluate(model, data, split=split if has_splits else None)
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

                if eval_info['val_acc'] > best_val_acc:
                    eval_info_early_model = eval_info
                    # torch.save(model.state_dict(), './best_{}_single_dec_split_{}.pkl'.format(dataset.name, split))
                    best_val_acc = np.max((best_val_acc, eval_info['val_acc']))
                    best_val_loss = np.min((best_val_loss, eval_info['val_loss']))
                    bad_counter = 0
                else:
                    bad_counter += 1
                    if bad_counter == patience:
                        break

            if device == torch.device('cuda'):
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
        mask = data.test_mask[split] if split is not None else data.test_mask
        test_embeddings = embeddings[mask]
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
        mask = data.test_mask[split] if split is not None else data.test_mask
        test_embeddings = embeddings[mask]
        normalized_test_embeddings = test_embeddings/torch.linalg.norm(test_embeddings, 1, dim=1).view(-1, 1)
        # calculate row-diff
        index = list(range(test_embeddings.shape[0]))
        sum = 0
        for _ in range(test_embeddings.shape[0]):
            index.insert(0, index.pop())
            sum += torch.linalg.norm(normalized_test_embeddings - normalized_test_embeddings[index], 2, dim=1).sum()
        row_diff = sum / pow(test_embeddings.shape[0], 2)
    return row_diff

def train(model, optimizer, data, split=None):
    model.train()
    optimizer.zero_grad()
    out = model(data)[0]
    mask = data.train_mask[:, split] if split is not None else data.train_mask
    loss = F.nll_loss(out[mask], data.y[mask].view(-1))
    loss.backward()
    optimizer.step()


def evaluate(model, data, split=None):
    model.eval()

    with torch.no_grad():
        logits = model(data)[0]

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)][:, split] if split is not None else data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask].view(-1)).item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = f1_score(data.y[mask].cpu(), logits[mask].max(1)[1].cpu(), average='micro')
        outs['{}_macro_f1'.format(key)] = f1_score(data.y[mask].cpu(), logits[mask].max(1)[1].cpu(), average='macro')

    return outs
