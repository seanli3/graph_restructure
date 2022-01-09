import torch
from numpy.random import seed as nseed
from torch_geometric.utils import get_laplacian, negative_sampling
import numpy as np
from itertools import product, combinations
from models.utils import homophily, our_homophily_measure
from dataset.datasets import get_dataset
from models.autoencoders import NodeFeatureSimilarityEncoder, SpectralSimilarityEncoder
import math
from tqdm import tqdm
import argparse
from config import SAVED_MODEL_DIR_NODE_CLASSIFICATION, USE_CUDA, SEED
from copy import deepcopy
from torch_geometric.utils import remove_self_loops, to_dense_adj
from torch_scatter import scatter_add


device = torch.device('cuda') if torch.cuda.is_available() and USE_CUDA else torch.device('cpu')

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
    return to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes).squeeze()


# random Guassian signals for simulating spectral clustering
# epsilon - error bound
def get_random_signals(num_nodes, size=None, epsilon=0.25):
    if size is None:
        random_signal_size = math.floor(
            6 / (math.pow(epsilon, 2) / 2 - math.pow(epsilon, 3) / 3) * math.log(num_nodes)
        )
    else:
        random_signal_size = size
    random_signal = torch.normal(
        0, math.sqrt(1 / random_signal_size), size=(num_nodes, random_signal_size),
        device=device, generator=torch.Generator(device).manual_seed(SEED)
    )
    return random_signal


def create_label_sim_matrix(data, mask):
    community = torch.zeros(data.num_nodes, data.num_nodes, device=device)
    for c in range(dataset.num_classes):
        index = (data.y == c).logical_and(mask).nonzero().view(-1).tolist()
        indices = torch.tensor(list(combinations(index, 2))).T
        if len(indices) > 0:
            community.index_put_((indices[0], indices[1]), torch.tensor(1.0, device=device))
    return community


def get_masked_edges(adj, mask):
    subgraph_adj = adj[mask][:,mask]
    return subgraph_adj.nonzero().T


class Rewirer(torch.nn.Module):
    def __init__(self, data, DATASET, step=0.2, layers=[128, 64], mode="supervised", split=None):
        super(Rewirer, self).__init__()
        self.data = data
        self.DATASET = DATASET
        random_signals = get_random_signals(data.x.shape[0])

        self.fea_sim_model = NodeFeatureSimilarityEncoder(data, layers=layers, name='fea')
        self.struct_sim_model = SpectralSimilarityEncoder(data, random_signals, step=step, name='struct')
        self.conv_sim_model = SpectralSimilarityEncoder(data, data.x, step=step, name="conv")
        if mode == 'unsupervised':
            self.models = [self.fea_sim_model, self.struct_sim_model]
        else:
            self.models = [self.fea_sim_model, self.struct_sim_model, self.conv_sim_model]
        self.mode = mode
        self.split = split

    def train(self, epochs, lr, weight_decay, patience, step):
        if self.mode == 'supervised':
            train_mask = self.data.train_mask[:, self.split] if self.split is not None else self.data.train_mask
            val_mask = self.data.train_mask[:, self.split] if self.split is not None else self.data.train_mask
            self.train_supervised(epochs, lr, weight_decay, patience, step, train_mask, val_mask)
        elif self.mode == 'unsupervised':
            self.train_unsupervised(epochs, lr, weight_decay, patience, step)
        elif self.mode == 'vae':
            self.train_vae(epochs, lr, weight_decay, patience, step)

    def train_supervised(self, epochs, lr, weight_decay, patience, step, train_mask, val_mask):
        data = self.data
        community = create_label_sim_matrix(data, train_mask)
        val_community = create_label_sim_matrix(data, val_mask)

        for model in self.models:
            model.reset_parameters()
            best_loss = float('inf')
            best_model = {}
            bad_counter = 0

            pbar = tqdm(range(0, epochs))

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            ) if len(list(model.parameters())) > 0 else None

            for epoch in pbar:
                model.train()
                x_hat = model()
                loss = model.loss(x_hat, community, train_mask)
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_loss = model.loss(x_hat, val_community, val_mask)
                pbar.set_description('Epoch: {}, loss: {:.2f} val loss: {:.2f}'.format(epoch, loss.item(), val_loss.item()))

                if val_loss < best_loss:
                    best_model['train_loss'] = loss
                    best_model['val_loss'] = val_loss
                    best_model['epoch'] = epoch
                    best_model['step'] = step
                    best_model['model'] = deepcopy(model.state_dict())
                    best_loss = val_loss
                    bad_counter = 0
                else:
                    bad_counter += 1
                    if bad_counter == patience:
                        break

            file_name = SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/{}_{}_{}'.format(
                    self.DATASET.lower(), self.mode, model.name
                ) + '.pt' if self.split is None else SAVED_MODEL_DIR_NODE_CLASSIFICATION + \
                '/{}_{}_{}_split_{}'.format(
                    self.DATASET.lower(), self.mode, model.name,  self.split
                ) + '.pt'
            torch.save(
                best_model, file_name
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def train_vae(self, epochs, lr, weight_decay, patience, step):
        data = self.data
        adj = get_adj(data.edge_index, data.edge_attr, data.num_nodes)

        for model in self.models:
            model.reset_parameters()
            best_loss = float('inf')
            best_model = {}
            bad_counter = 0

            pbar = tqdm(range(0, epochs))

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

            for epoch in pbar:
                model.train()
                x_hat = model()
                num_samples = data.edge_index.shape[1]//3 * 2
                negative_sample_mask = self.sample_negative_edge_mask(data.edge_index, num_samples)
                positive_sample_mask = self.sample_edge_mask(data.edge_index, num_samples)
                mask = negative_sample_mask.logical_or(positive_sample_mask)
                loss = model.loss(x_hat.masked_select(mask), adj.masked_select(mask))
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
            torch.save(
                best_model, SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/{}_{}_{}'.format(
                    self.DATASET.lower(), self.mode, model.name
                ) + '.pt'
            )

    def train_unsupervised(self, epochs, lr, weight_decay, patience, step):
        for model in self.models:
            model.reset_parameters()

        best_loss_struct = float('inf')
        best_loss_fea = float('inf')
        best_model_fea = {}
        best_model_struct = {}
        bad_counter = 0

        pbar = tqdm(range(0, epochs))

        optimizers = [
            torch.optim.AdamW(
                self.fea_sim_model.parameters(), lr=lr, weight_decay=weight_decay
            ),
            torch.optim.AdamW(
                self.struct_sim_model.parameters(), lr=lr, weight_decay=weight_decay
            ), ]
        loss_struct = torch.tensor(float('inf'))
        loss_fea = torch.tensor(float('inf'))
        for epoch in pbar:
            if epoch // 10 % 2 == 0:
                self.struct_sim_model.eval()
                self.fea_sim_model.train()

                x_hat_fea = self.fea_sim_model()
                x_hat_struct = self.struct_sim_model()

                loss_fea = self.fea_sim_model.loss(x_hat_fea, x_hat_struct)
                loss_fea.backward()
                optimizers[0].step()
                optimizers[0].zero_grad()
            else:
                self.struct_sim_model.train()
                self.fea_sim_model.eval()

                x_hat_fea = self.fea_sim_model()
                x_hat_struct = self.struct_sim_model()

                loss_struct = self.struct_sim_model.loss(x_hat_fea, x_hat_struct)
                loss_struct.backward()
                optimizers[1].step()
                optimizers[1].zero_grad()

            pbar.set_description('Epoch: {}, fea loss: {:.2f}, struct loss: {:.2f}'\
                                 .format(epoch, loss_fea.item(), loss_struct.item()))

            if loss_fea < best_loss_fea:
                best_model_fea['train_loss'] = best_loss_fea
                best_model_fea['loss'] = best_loss_fea
                best_model_fea['epoch'] = epoch
                best_model_fea['step'] = step
                best_model_fea['model'] = deepcopy(self.fea_sim_model.state_dict())
                best_loss_fea = loss_fea
                bad_counter = 0
                torch.save(
                    best_model_fea, SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/{}_{}_{}'.format(
                        self.DATASET.lower(), self.mode, self.fea_sim_model.name
                    ) + '.pt'
                )

            elif loss_struct < best_loss_struct:
                best_model_struct['train_loss'] = best_loss_struct
                best_model_struct['loss'] = best_loss_struct
                best_model_struct['epoch'] = epoch
                best_model_struct['step'] = step
                best_model_struct['model'] = deepcopy(self.struct_sim_model.state_dict())
                best_loss_struct = loss_struct
                bad_counter = 0
                torch.save(
                    best_model_struct, SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/{}_{}_{}'.format(
                        self.DATASET.lower(), self.mode, self.struct_sim_model.name
                    ) + '.pt'
                )
            else:
                bad_counter += 1
                if bad_counter == patience:
                    break
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def sample_negative_edge_mask(self, edge_index, num_samples):
        negative_edges = negative_sampling(
            edge_index, num_nodes=self.data.num_nodes, method="sparse", num_neg_samples=num_samples,
            force_undirected=True
            )
        mask = torch.zeros(self.data.num_nodes, self.data.num_nodes, device=device).bool().index_put(
            (negative_edges[0], negative_edges[1]), torch.tensor(True, device=device)
            )
        return mask

    def sample_edge_mask(self, edge_index, num_samples):
        sampled_edges = edge_index[:, torch.randperm(edge_index.shape[1])][:, :num_samples]
        mask = torch.zeros(self.data.num_nodes, self.data.num_nodes, device=device).bool().index_put(
            (sampled_edges[0], sampled_edges[1]), torch.tensor(True, device=device)
        )
        return mask

    def load(self):
        if self.mode in ['vae', 'unsupervised']:
            for model in self.models:
                saved_model = torch.load(
                    SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/{}_{}_{}'.format(
                        self.DATASET.lower(), self.mode, model.name
                    ) + '.pt'
                    )
                model.load_state_dict(saved_model['model'])
        elif self.mode == 'supervised':
            for model in self.models:
                file_name = SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/{}_{}_{}'.format(
                    self.DATASET.lower(), self.mode, model.name
                ) + '.pt' if self.split is None else SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/{}_{}_{}_split_{}'.format(
                    self.DATASET.lower(), self.mode, model.name, self.split
                ) + '.pt'
                saved_model = torch.load(file_name)
                model.load_state_dict(saved_model['model'])

    def rewire(self, dataset, model_indices, edge_per):
        # a = torch.sparse_coo_tensor(dataset[0].edge_index, torch.ones(dataset[0].num_edges),
        #                             (dataset[0].num_nodes, dataset[0].num_nodes), device=device)\
        #     .to_dense()
        with torch.no_grad():
            a = torch.zeros(dataset[0].num_nodes, dataset[0].num_nodes)
            # a = get_normalized_adj(dataset[0].edge_index, None, dataset[0].num_nodes)
            # triu_indices = torch.triu_indices(self.data.num_nodes, self.data.num_nodes, offset=1)
            for model in map(self.models.__getitem__, model_indices):
                a_hat = model().squeeze()
                # a_hat = torch.zeros(data.num_nodes, data.num_nodes).index_put((triu_indices[0],triu_indices[1]), a_hat)
                # a += torch.zeros_like(a_hat).masked_fill_(a_hat > eps_1, 1)
                # a += torch.zeros_like(a_hat).masked_fill_(a_hat < eps_2, -1)
                a += a_hat
            a += torch.eye(a.shape[1], a.shape[1])*-9e9
            # print(a.min(), a.max())
            # print("edges before: ", dataset.data.num_edges, "edges after: ", (a > threshold).count_nonzero().item())
            new_dataset = deepcopy(dataset)
            # new_dataset.data.edge_index = (a > threshold).nonzero().T
            v, i = torch.topk(a.flatten(), int(dataset[0].num_edges*edge_per))
            edges = torch.tensor(np.array(np.unravel_index(i.cpu().numpy(), a.shape)), device=device).detach()
            print("edges before: ", dataset.data.num_edges, " edges after: ", edges.shape[1])
            print('homophily before:', our_homophily_measure(dataset[0].edge_index, dataset[0].y),
                  'homophily after:', our_homophily_measure(edges, dataset[0].y))
            new_dataset.data.edge_index = edges
            return new_dataset

    def plot_homophily_regular(self, dataset, model_indices):
        # a = torch.sparse_coo_tensor(dataset[0].edge_index, torch.ones(dataset[0].num_edges),
        #                             (dataset[0].num_nodes, dataset[0].num_nodes), device=device)\
        #     .to_dense()
        with torch.no_grad():
            xi = []
            yi = []
            # ori_homo = our_homophily_measure(
            #     get_masked_edges(get_adj(dataset[0].edge_index, None, dataset[0].num_nodes),
            #                      dataset[0].val_mask[:, self.split]), dataset[0].y[dataset[0].val_mask[:, self.split]]
            #     )
            ori_homo = our_homophily_measure( dataset[0].edge_index, dataset[0].y )
            for n_edges in range(1, 30, 1):
                if n_edges == 0:
                    edges = torch.tensor([[]])
                else:
                    a = torch.zeros(dataset[0].num_nodes, dataset[0].num_nodes)
                    # a = get_normalized_adj(dataset[0].edge_index, None, dataset[0].num_nodes)

                    # triu_indices = torch.triu_indices(self.data.num_nodes, self.data.num_nodes, offset=1)
                    for model in map(self.models.__getitem__, model_indices):
                        a_hat = model().squeeze()
                        # a_hat = torch.zeros(data.num_nodes, data.num_nodes).index_put((triu_indices[0],triu_indices[1]), a_hat)
                        # a += torch.zeros_like(a_hat).masked_fill_(a_hat > eps_1, 1)
                        # a += torch.zeros_like(a_hat).masked_fill_(a_hat < eps_2, -1)
                        a += a_hat
                    a += torch.eye(a.shape[1], a.shape[1])*-9e9
                    v, idx = torch.topk(a, n_edges, dim=1)
                    edges = []
                    for i in range(idx.shape[1]):
                        edges += list(zip(range(dataset[0].num_nodes), idx[:,i].cpu().tolist()))
                    edges = torch.tensor(edges).T
                    dir_g = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), (dataset[0].num_nodes, dataset[0].num_nodes)).to_dense()
                    undir_g = dir_g + dir_g.T

                    edges = undir_g.nonzero().T

                # new_homophily = our_homophily_measure(get_masked_edges(adj, dataset[0].val_mask[:, self.split]),
                #                           dataset[0].y[dataset[0].val_mask[:, self.split]])
                new_homophily = our_homophily_measure(edges, dataset[0].y)
                print("edge percent: ", n_edges, "edges before: ", dataset.data.num_edges,
                      " edges after: ", edges.shape[1], 'homophily before:', ori_homo,
                      'homophily after:', new_homophily)
                xi.append(n_edges)
                yi.append(new_homophily)
            from matplotlib import pyplot as plt
            plt.plot(xi, yi)
            plt.axhline(y=ori_homo, color='r', linestyle='-')
            plt.title(dataset.name + ", model indices:" + str(model_indices))
            plt.show()

    def plot_homophily(self, dataset, model_indices):
        # a = torch.sparse_coo_tensor(dataset[0].edge_index, torch.ones(dataset[0].num_edges),
        #                             (dataset[0].num_nodes, dataset[0].num_nodes), device=device)\
        #     .to_dense()
        with torch.no_grad():
            xi = []
            yi = []
            # ori_homo = our_homophily_measure(
            #     get_masked_edges(get_adj(dataset[0].edge_index, None, dataset[0].num_nodes),
            #                      dataset[0].val_mask[:, self.split]), dataset[0].y[dataset[0].val_mask[:, self.split]]
            #     )
            ori_homo = our_homophily_measure(dataset[0].edge_index, dataset[0].y).item()

            for edge_per in torch.arange(0.05, 700, 10):
                # a = torch.zeros(dataset[0].num_nodes, dataset[0].num_nodes)
                a = get_normalized_adj(dataset[0].edge_index, None, dataset[0].num_nodes)
                # triu_indices = torch.triu_indices(self.data.num_nodes, self.data.num_nodes, offset=1)
                for model in map(self.models.__getitem__, model_indices):
                    model.to(device)
                    a_hat = model().squeeze()
                    # a_hat = torch.zeros(data.num_nodes, data.num_nodes).index_put((triu_indices[0],triu_indices[1]), a_hat)
                    # a += torch.zeros_like(a_hat).masked_fill_(a_hat > eps_1, 1)
                    # a += torch.zeros_like(a_hat).masked_fill_(a_hat < eps_2, -1)
                    a += a_hat
                a += torch.eye(a.shape[1], a.shape[1], device=device) * -9e9
                # print(a.min(), a.max())
                # print("edges before: ", dataset.data.num_edges, "edges after: ", (a > threshold).count_nonzero().item())
                # new_dataset.data.edge_index = (a > threshold).nonzero().T
                v, i = torch.topk(a.flatten(), int(dataset[0].num_edges * edge_per))
                edges = torch.tensor(np.array(np.unravel_index(i.cpu().numpy(), a.shape)), device=device).detach()

                # new_homophily = our_homophily_measure(get_masked_edges(adj, dataset[0].val_mask[:, self.split]),
                #                           dataset[0].y[dataset[0].val_mask[:, self.split]])
                new_homophily = our_homophily_measure(edges, dataset[0].y)
                print(
                    "edge percent: ", edge_per, "edges before: ", dataset.data.num_edges, " edges after: ",
                    edges.shape[1], 'homophily before:', ori_homo, 'homophily after:', new_homophily
                    )
                xi.append(edge_per.item())
                yi.append(new_homophily.item())
            from matplotlib import pyplot as plt
            plt.plot(xi, yi)
            plt.axhline(y=ori_homo, color='r', linestyle='-')
            plt.title(dataset.name + ", model indices:" + str(model_indices) + ", split:" + str(self.split))
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--step', type=float, default=0.1)
    parser.add_argument('--split', type=int, default=None)
    parser.add_argument('--mode', type=str, default='supervised')
    args = parser.parse_args()

    DATASET = args.dataset

    print(DATASET)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available() and USE_CUDA:
        print("-----------------------Training on CUDA-------------------------")
        torch.cuda.manual_seed(seed)
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset = get_dataset(DATASET, normalize_features=True)
    data = dataset[0]

    module = Rewirer(data, step=args.step, layers=[256, 128, 64], DATASET=DATASET, mode=args.mode, split=args.split)
    module.train(epochs=10000, lr=args.lr, weight_decay=0.0005, patience=100, step=args.step)