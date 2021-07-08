import torch
from numpy.random import seed as nseed
import numpy as np
from graph_dictionary.model import DictNet
from tqdm import tqdm

# %%

DATASET = 'Squirrel'

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


def run(name, Model, runs, epochs, lr, weight_decay, patience):
    for split in range(10):
        print('Split:', split)
        model = Model(name, split)
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
            loss = model(model.train_mask)
            # Backpropagate
            loss.backward()
            # Update the parameters
            optimizer.step()
            optimizer.zero_grad()

            # # predict and calculate loss for item factor and bias
            # optimizer = torch.optim.SGD([model.W], lr=1,
            #                             weight_decay=1e-5)  # learning rate
            # loss = model(model.train_mask)
            # # Backpropagate
            # loss.backward()
            #
            # # Update the parameters
            # optimizer.step()
            # optimizer.zero_grad()

            model.eval()
            with torch.no_grad():
                val_loss = model(model.val_mask)
            pbar.set_description('Epoch: {}, training loss: {:.2f}, validation loss: {:.2f}'.format(epoch, loss.item(), val_loss.item()))


            if val_loss < best_val_loss:
                eval_info_early_model['train_loss'] = loss
                eval_info_early_model['val_loss'] = val_loss
                eval_info_early_model['epoch'] = epoch
                eval_info_early_model['C'] = torch.clone(model.C.detach())
                # eval_info_early_model['W'] = torch.clone(model.W.detach())
                # eval_info_early_model['A'] = torch.clone(model.A.detach())
                best_val_loss = val_loss
                bad_counter = 0
                torch.save(eval_info_early_model, './{}_best_model_split_{}.pt'.format(DATASET, split))
            else:
                bad_counter += 1
                if bad_counter == patience:
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    return eval_info_early_model


eval_info_early_model = run(DATASET, DictNet, 1, 1000, 0.005, 0.0005, 10)
print()


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
