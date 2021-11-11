import torch
from numpy.random import seed as nseed
import numpy as np

from dataset.datasets import get_dataset
from models.autoencoders import NodeFeatureSimilarityEncoder, SpectralSimilarityEncoder
import math
from tqdm import tqdm
import argparse
from config import SAVED_MODEL_PATH_NODE_CLASSIFICATION_NO_SPLIT
from copy import deepcopy
from torch_geometric.utils import remove_self_loops, to_dense_adj
from torch_scatter import scatter_add


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_normalized_adj(edge_index, edge_weight, num_nodes):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    row, col = edge_index[0], edge_index[1]
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    # Compute A_norm = D^{-1} A.
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()


def get_adj(edge_index, edge_weight, num_nodes):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    return to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()


# random Guassian signals for simulating spectral clustering
# epsilon - error bound
def get_random_signals(num_nodes, epsilon=0.25):
    random_signal_size = math.floor(
        6 / (math.pow(epsilon, 2) / 2 - math.pow(epsilon, 3) / 3) * math.log(num_nodes)
    )
    random_signal = torch.normal(
        0, math.sqrt(1 / random_signal_size), size=(num_nodes, random_signal_size), device=device
    )
    return random_signal


class Rewirer(torch.nn.Module):
    def __init__(self, data, DATASET, step=0.2, layers=[256, 128, 64]):
        super(Rewirer, self).__init__()
        self.data = data
        self.DATASET = DATASET
        random_signals = get_random_signals(data.num_nodes)

        self.fea_sim_model = NodeFeatureSimilarityEncoder(data, layers=layers, name='fea')
        self.struct_sim_model = SpectralSimilarityEncoder(data, random_signals, step=step, name='struct')
        self.conv_sim_model = SpectralSimilarityEncoder(data, data.x, step=step, name="conv")
        self.models = [self.fea_sim_model, self.struct_sim_model, self.conv_sim_model]
        # self.models = [self.fea_sim_model, self.struct_sim_model]

    def train(self, epochs, lr, weight_decay, patience, step):
        data = self.data
        for model in self.models:
            model.reset_parameters()
            best_loss = float('inf')
            best_model = {}
            bad_counter = 0

            adj = get_adj(data.edge_index, data.edge_attr, data.num_nodes)

            pbar = tqdm(range(0, epochs))

            for epoch in pbar:
                model.train()
                # Predict and calculate loss for user factor and bias
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, weight_decay=weight_decay
                )  # learning rate
                x_hat = model()
                loss = model.loss(x_hat, adj)
                # Backpropagate
                loss.backward()
                # Update the parameters
                optimizer.step()
                optimizer.zero_grad()

                pbar.set_description('Epoch: {}, loss: {:.2f}'.format(epoch, loss.item()))

                if loss < best_loss:
                    best_model['train_loss'] = loss
                    best_model['loss'] = loss
                    best_model['epoch'] = epoch
                    best_model['step'] = step
                    best_model['model'] = deepcopy(model.state_dict())
                    best_loss = loss
                    bad_counter = 0
                else:
                    bad_counter += 1
                    if bad_counter == patience:
                        break
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            torch.save(best_model, SAVED_MODEL_PATH_NODE_CLASSIFICATION_NO_SPLIT.format(model.name + '_' + self.DATASET.lower()))

    def load(self):
        for model in self.models:
            saved_model = torch.load(SAVED_MODEL_PATH_NODE_CLASSIFICATION_NO_SPLIT.format(model.name + '_' + self.DATASET.lower()))
            model.load_state_dict(saved_model['model'])

    def rewire(self, dataset):
        a = torch.sparse_coo_tensor(dataset[0].edge_index, torch.ones(dataset[0].num_edges),
                                    (dataset[0].num_nodes, dataset[0].num_nodes), device=device)\
            .to_dense()
        eps_1 = 0.9
        eps_2 = 0.
        for model in self.models:
            a_hat = model()
            # a += torch.zeros_like(a_hat).masked_fill_(a_hat > eps_1, 1)
            # a += torch.zeros_like(a_hat).masked_fill_(a_hat < eps_2, -1)
            a += a_hat
        a = a.triu(diagonal=1)
        a = a + a.T
        dataset.data.edge_index = (a >= 2).nonzero().T
        # v, i = torch.topk(a.flatten(), int(dataset[0].num_edges/5))
        # edges = torch.tensor(np.array(np.unravel_index(i.cpu().numpy(), a.shape)), device=device).detach()
        # dataset.data.edge_index = edges
        return dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--step', type=float, default=0.1)
    args = parser.parse_args()

    DATASET = args.dataset

    print(DATASET)

    cuda = torch.cuda.is_available()
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cuda:
        print("-----------------------Training on CUDA-------------------------")
        torch.cuda.manual_seed(seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset = get_dataset(DATASET, normalize_features=True, cuda=cuda)
    data = dataset[0]

    module = Rewirer(data, step=0.2, layers=[256, 128, 64], DATASET=DATASET)
    module.train(epochs=5000, lr=args.lr, weight_decay=0.0005, patience=100, step=args.step)