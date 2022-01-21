import torch
from numpy.random import seed as nseed
from torch_geometric.utils import negative_sampling
import numpy as np
from models.utils import our_homophily_measure, get_adj, get_random_signals, create_label_sim_matrix, \
    get_normalized_adj, cosine_sim
from dataset.datasets import get_dataset
from models.autoencoders import NodeFeatureSimilarityEncoder, SpectralSimilarityEncoder
from tqdm import tqdm
import argparse
from config import SAVED_MODEL_DIR_NODE_CLASSIFICATION, USE_CUDA, DEVICE
from copy import deepcopy

device = DEVICE


# random Guassian signals for simulating spectral clustering
# epsilon - error bound


class Rewirer(torch.nn.Module):
    def __init__(self, data, DATASET, step=0.2, layers=[128, 64], mode="supervised", split=None, exact=False):
        super(Rewirer, self).__init__()
        self.data = data
        self.DATASET = DATASET
        random_signals = get_random_signals(data.num_nodes)

        if mode == 'unsupervised':
            self.fea_sim_model = NodeFeatureSimilarityEncoder(data, layers=layers, name='fea')
            self.struct_sim_model = SpectralSimilarityEncoder(data, random_signals, step=step, name='struct', exact=exact)
            self.conv_sim_model = SpectralSimilarityEncoder(data, data.x, step=step, name="conv", exact=exact)
            self.models = [self.fea_sim_model, self.struct_sim_model]
        else:
            # self.fea_sim_model = NodeFeatureSimilarityEncoder(data, layers=layers, name='fea')
            self.struct_sim_model = SpectralSimilarityEncoder(data, random_signals, step=step, name='struct', exact=exact)
            # self.conv_sim_model = SpectralSimilarityEncoder(data, data.x, step=step, name="conv")
            self.models = [self.struct_sim_model]

        self.mode = mode
        self.split = split

    def train(self, epochs, lr, weight_decay, patience, step):
        if self.mode == 'supervised':
            train_mask = self.data.train_mask[:, self.split] if self.split is not None else self.data.train_mask
            val_mask = self.data.train_mask[:, self.split] if self.split is not None else self.data.train_mask
            self.train_supervised(epochs, lr, weight_decay, patience, step, train_mask, val_mask)
            # self.train_supervised_alt(epochs, lr, weight_decay, patience, step, train_mask, val_mask)
        elif self.mode == 'kmeans':
            train_mask = self.data.train_mask[:, self.split] if self.split is not None else self.data.train_mask
            val_mask = self.data.train_mask[:, self.split] if self.split is not None else self.data.train_mask
            self.train_supervised_kmeans(epochs, lr, weight_decay, patience, step, train_mask, val_mask)
        elif self.mode == 'unsupervised':
            self.train_unsupervised(epochs, lr, weight_decay, patience, step)
        elif self.mode == 'vae':
            self.train_vae(epochs, lr, weight_decay, patience, step)

    def train_supervised_alt(self, epochs, lr, weight_decay, patience, step, train_mask, val_mask):
        data = self.data
        community = create_label_sim_matrix(data)
        train_community_mask = train_mask.float().view(-1, 1).mm(train_mask.float().view(1, -1)).bool()
        best_loss_struct = float('inf')
        best_loss_fea = float('inf')

        optimizers = [
            torch.optim.AdamW(
                self.fea_sim_model.parameters(), lr=lr, weight_decay=weight_decay
            ),
            torch.optim.AdamW(
                self.struct_sim_model.parameters(), lr=lr, weight_decay=weight_decay
            ), ]

        loss_struct = torch.tensor(float('inf'))
        loss_fea = torch.tensor(float('inf'))
        self.fea_sim_model.reset_parameters()
        self.struct_sim_model.reset_parameters()

        best_loss = float('inf')
        bad_counter = 0
        best_model_fea = {}
        best_model_struct = {}

        pbar = tqdm(range(0, epochs))

        for epoch in pbar:
            if epoch // 100 % 2 == 0:
                self.struct_sim_model.eval()
                self.fea_sim_model.train()
                x_hat_fea = self.fea_sim_model.sim()

                if epoch < 200:
                    loss_fea = self.fea_sim_model.loss(x_hat_fea, community, train_mask)
                else:
                    with torch.no_grad():
                        x_hat_struct = self.struct_sim_model.sim()
                    self_sup_mask = (x_hat_struct > 0.9).logical_or(x_hat_struct < -0.9).logical_and(train_community_mask.logical_not())
                    target = community.where(train_community_mask, torch.tensor(0.).to(device)) + x_hat_struct.where(self_sup_mask, torch.tensor(0.).to(device))
                    loss_fea = self.fea_sim_model.loss(x_hat_fea, target)
                loss_fea.backward()
                optimizers[0].step()
                optimizers[0].zero_grad()
            else:
                self.struct_sim_model.train()
                self.fea_sim_model.eval()
                x_hat_struct = self.struct_sim_model.sim()
                if epoch < 200:
                    loss_struct = self.struct_sim_model.loss(x_hat_struct, community, train_mask)
                else:
                    with torch.no_grad():
                        x_hat_fea = self.fea_sim_model.sim()
                    self_sup_mask = (x_hat_fea > 0.9).logical_or(x_hat_fea < -0.9).logical_and(self_sup_mask.logical_not())
                    target = community.where(train_community_mask, torch.tensor(0.).to(device)) + x_hat_fea.where(self_sup_mask, torch.tensor( 0.).to( device))
                    loss_struct = self.struct_sim_model.loss(x_hat_struct, target)
                loss_struct.backward()
                optimizers[1].step()
                optimizers[1].zero_grad()

            pbar.set_description('Epoch: {}, fea loss: {:.3f}, struct loss: {:.3f}' \
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

    def train_supervised(self, epochs, lr, weight_decay, patience, step, train_mask, val_mask):
        data = self.data
        community = create_label_sim_matrix(data)

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
                x_hat = model.sim()
                loss = model.loss(x_hat, community, train_mask)
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_loss = model.loss(x_hat, community, val_mask)
                pbar.set_description('Epoch: {}, loss: {:.3f} val loss: {:.3f}'.format(epoch, loss.item(), val_loss.item()))

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
            print(
                'Model saved for Epoch: {}, loss: {:.3f} val loss: {:.3f}'.format(
                    best_model['epoch'], best_model['train_loss'].item(), best_model['val_loss'].item()
                )
            )

    def train_supervised_kmeans(self, epochs, lr, weight_decay, patience, step, train_mask, val_mask):
        data = self.data
        community = create_label_sim_matrix(data)

        for model in self.models:
            model.reset_parameters()
            best_acc = float(0)
            best_model = {}
            bad_counter = 0

            pbar = tqdm(range(0, epochs))

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            ) if len(list(model.parameters())) > 0 else None

            for epoch in pbar:
                model.train()
                x_hat = model.sim()
                loss = model.loss(x_hat, community, train_mask)
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_loss = model.loss(x_hat, community, val_mask)
                train_acc, val_acc, test_acc = self.kmeans([self.models.index(model)], self.split)
                pbar.set_description(
                    'Epoch: {}, loss: {:.3f} val loss: {:.3f} train acc {:.3f}, val acc {:.3f}, test acc {:.3f}'.format(
                        epoch, loss.item(), val_loss.item(), train_acc, val_acc, test_acc
                    )
                )

                if val_acc > best_acc:
                    best_model['train_loss'] = loss
                    best_model['val_loss'] = val_loss
                    best_model['epoch'] = epoch
                    best_model['step'] = step
                    best_model['model'] = deepcopy(model.state_dict())
                    best_acc = val_acc
                    bad_counter = 0
                else:
                    bad_counter += 1
                    if bad_counter == patience:
                        break

            file_name = SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/{}_{}_{}'.format(
                self.DATASET.lower(), self.mode, model.name
            ) + '.pt' if self.split is None else SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/{}_{}_{}_split_{}'.format(
                self.DATASET.lower(), self.mode, model.name, self.split
            ) + '.pt'
            torch.save(
                best_model, file_name
            )
            print(
                'Model saved for Epoch: {}, loss: {:.3f} val loss: {:.3f}'.format(
                    best_model['epoch'], best_model['train_loss'].item(), best_model['val_loss'].item()
                )
            )

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
                x_hat = model.sim()
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

                pbar.set_description('Epoch: {}, loss: {:.3f}'.format(epoch, loss.item()))

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
            torch.save(
                best_model, SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/{}_{}_{}'.format(
                    self.DATASET.lower(), self.mode, model.name
                ) + '.pt'
            )
            print(
                'Model saved for Epoch: {}, loss: {:.3f} val loss: {:.3f}'.format(
                    best_model['epoch'], best_model['train_loss'].item(), best_model['loss'].item()
                )
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

                x_hat_fea = self.fea_sim_model.sim()
                x_hat_struct = self.struct_sim_model.sim()

                loss_fea = self.fea_sim_model.loss(x_hat_fea, x_hat_struct)
                loss_fea.backward()
                optimizers[0].step()
                optimizers[0].zero_grad()
            else:
                self.struct_sim_model.train()
                self.fea_sim_model.eval()

                x_hat_fea = self.fea_sim_model.sim()
                x_hat_struct = self.struct_sim_model.sim()

                loss_struct = self.struct_sim_model.loss(x_hat_fea, x_hat_struct)
                loss_struct.backward()
                optimizers[1].step()
                optimizers[1].zero_grad()

            pbar.set_description('Epoch: {}, fea loss: {:.3f}, struct loss: {:.3f}'\
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
                print(
                    'Model saved for Epoch: {}, loss: {:.3f} val loss: {:.3f}'.format(
                        best_model_fea['epoch'], best_model_fea['train_loss'].item(), best_model_fea['loss'].item()
                    )
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
                print(
                    'Model saved for Epoch: {}, loss: {:.3f} val loss: {:.3f}'.format(
                        best_model_struct['epoch'], best_model_struct['train_loss'].item(), best_model_struct['loss'].item()
                    )
                )
            else:
                bad_counter += 1
                if bad_counter == patience:
                    break

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
                saved_model = torch.load(file_name, map_location=device)
                model.load_state_dict(saved_model['model'])

    def rewire(self, dataset, model_indices, num_edges):
        # a = torch.sparse_coo_tensor(dataset[0].edge_index, torch.ones(dataset[0].num_edges),
        #                             (dataset[0].num_nodes, dataset[0].num_nodes), device=device)\
        #     .to_dense()
        with torch.no_grad():
            # a = get_normalized_adj(dataset[0].edge_index, None, dataset[0].num_nodes)
            # triu_indices = torch.triu_indices(self.data.num_nodes, self.data.num_nodes, offset=1)
            fea_sim = cosine_sim(dataset[0].x).squeeze().fill_diagonal_(0)
            a = get_normalized_adj(dataset[0].edge_index, None, dataset[0].num_nodes)
            a += fea_sim
            for model in map(self.models.__getitem__, model_indices):
                a_hat = model.sim().squeeze_().fill_diagonal_(0)
                # a_hat = torch.zeros(data.num_nodes, data.num_nodes).index_put((triu_indices[0],triu_indices[1]), a_hat)
                # a += torch.zeros_like(a_hat).masked_fill_(a_hat > eps_1, 1)
                # a += torch.zeros_like(a_hat).masked_fill_(a_hat < eps_2, -1)
                a = a + a_hat
            a += torch.eye(a.shape[1], a.shape[1], device=device)*-9e9
            # print(a.min(), a.max())
            # print("edges before: ", dataset.data.num_edges, "edges after: ", (a > threshold).count_nonzero().item())
            new_dataset = deepcopy(dataset)
            # new_dataset.data.edge_index = (a > threshold).nonzero().T
            v, i = torch.topk(a.flatten(), int(num_edges))
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
                    a = torch.zeros(dataset[0].num_nodes, dataset[0].num_nodes, device=device)
                    # a = get_normalized_adj(dataset[0].edge_index, None, dataset[0].num_nodes)

                    # triu_indices = torch.triu_indices(self.data.num_nodes, self.data.num_nodes, offset=1)
                    for model in map(self.models.__getitem__, model_indices):
                        a_hat = model.sim().squeeze()
                        # a_hat = torch.zeros(data.num_nodes, data.num_nodes).index_put((triu_indices[0],triu_indices[1]), a_hat)
                        # a += torch.zeros_like(a_hat).masked_fill_(a_hat > eps_1, 1)
                        # a += torch.zeros_like(a_hat).masked_fill_(a_hat < eps_2, -1)
                        a += a_hat
                    a += torch.eye(a.shape[1], a.shape[1], device=device)*-9e9
                    v, idx = torch.topk(a, n_edges, dim=1)
                    edges = []
                    for i in range(idx.shape[1]):
                        edges += list(zip(range(dataset[0].num_nodes), idx[:,i].cpu().tolist()))
                    edges = torch.tensor(edges, device=device).T
                    dir_g = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1], device=device), (dataset[0].num_nodes, dataset[0].num_nodes)).to_dense()
                    undir_g = dir_g + dir_g.T

                    edges = undir_g.nonzero().T

                # new_homophily = our_homophily_measure(get_masked_edges(adj, dataset[0].val_mask[:, self.split]),
                #                           dataset[0].y[dataset[0].val_mask[:, self.split]])
                new_homophily = our_homophily_measure(edges, dataset[0].y)
                print("edge percent: ", n_edges, "edges before: ", dataset.data.num_edges,
                      " edges after: ", edges.shape[1], 'homophily before:', ori_homo,
                      'homophily after:', new_homophily)
                xi.append(n_edges)
                yi.append(new_homophily.cpu())
            from matplotlib import pyplot as plt
            yi = torch.tensor(yi)
            gradient = yi[1:] - yi[:-1]
            plt.plot(xi, yi)
            plt.plot(xi[:-1], gradient)
            plt.axhline(y=ori_homo.cpu(), color='r', linestyle='-')
            plt.title(dataset.name + ", model indices:" + str(model_indices))
            plt.show()

    def plot_homophily(self, dataset, model_indices, mask):
        with torch.no_grad():
            xi = []
            yi = []
            ori_homo = our_homophily_measure(dataset[0].edge_index, dataset[0].y.where(mask, torch.tensor(-1, device=device))).item()
            fea_sim = cosine_sim(dataset[0].x).squeeze().fill_diagonal_(0)

            for num_edges in torch.arange(dataset[0].num_nodes//3, dataset[0].num_nodes*100, dataset[0].num_nodes//2):
                a = get_normalized_adj(dataset[0].edge_index, None, dataset[0].num_nodes)
                a += fea_sim
                for model in map(self.models.__getitem__, model_indices):
                    model.to(device)
                    a_hat = model.sim().squeeze_().fill_diagonal_(0)
                    # a_hat = torch.zeros(data.num_nodes, data.num_nodes).index_put((triu_indices[0],triu_indices[1]), a_hat)
                    # a += torch.zeros_like(a_hat).masked_fill_(a_hat > eps_1, 1)
                    # a += torch.zeros_like(a_hat).masked_fill_(a_hat < eps_2, -1)
                    a = a + a_hat
                a += torch.eye(a.shape[1], a.shape[1], device=device) * -9e9
                # print(a.min(), a.max())
                # print("edges before: ", dataset.data.num_edges, "edges after: ", (a > threshold).count_nonzero().item())
                # new_dataset.data.edge_index = (a > threshold).nonzero().T
                v, i = torch.topk(a.flatten(), int(num_edges))
                edges = torch.tensor(np.array(np.unravel_index(i.cpu().numpy(), a.shape)), device=device).detach()

                # new_homophily = our_homophily_measure(get_masked_edges(adj, dataset[0].val_mask[:, self.split]),
                #                           dataset[0].y[dataset[0].val_mask[:, self.split]])
                new_homophily = our_homophily_measure(edges, dataset[0].y.where(mask, torch.tensor(-1, device=device)))
                print(
                    "edges: ", num_edges, "edges before: ", dataset.data.num_edges, " edges after: ",
                    edges.shape[1], 'homophily before:', ori_homo, 'homophily after:', new_homophily
                    )
                xi.append(num_edges.item())
                yi.append(new_homophily.item())
            from matplotlib import pyplot as plt
            yi = torch.tensor(yi)
            gradient = yi[1:] - yi[:-1]
            plt.plot(xi, yi)
            # plt.plot(xi[:-1], gradient)
            plt.axhline(y=ori_homo, color='r', linestyle='-')
            plt.title(dataset.name + ", model indices:" + str(model_indices) + ", split:" + str(self.split))
            plt.show()

    def kmeans(self, model_indices, split):
        from kmeans_pytorch import kmeans
        from sklearn.metrics import f1_score
        with torch.no_grad():
            for model in self.models:
                model.to(device)
            x = torch.cat([self.models[i]() for i in model_indices], dim=1)
            num_classes = self.data.y.max().item() + 1
            cluster_ids_x, cluster_centers = kmeans(
                X=x, num_clusters=num_classes, device=device
            )
            predicted = torch.zeros_like(cluster_ids_x, device=device)
            y = self.data.y
            train_mask = self.data.train_mask[:, split] if split is not None else data.train_mask
            val_mask = self.data.val_mask[:, split] if split is not None else data.val_mask
            test_mask = self.data.test_mask[:, split] if split is not None else data.test_mask
            for i in range(num_classes):
                predicted[cluster_ids_x == i] = y[cluster_ids_x == i].bincount().argmax()

            train_acc = f1_score(
                self.data.y.cpu()[train_mask],
                predicted[train_mask].cpu(),
                average='micro'
            )
            val_acc = f1_score(
                self.data.y.cpu()[val_mask], predicted[val_mask].cpu(), average='micro'
            )
            test_acc = f1_score(
                self.data.y.cpu()[test_mask], predicted[test_mask].cpu(), average='micro'
            )
            # print('train acc: {:.3f}, val acc: ${:.3f}, test acc: ${:.3f}'.format(train_acc, val_acc, test_acc))
            return train_acc, val_acc, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--step', type=float, default=0.1)
    parser.add_argument('--split', type=int, default=None)
    parser.add_argument('--mode', type=str, default='supervised')
    parser.add_argument('--exact', action='store_true')
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

    module = Rewirer(data, step=args.step, layers=[256, 128, 64], DATASET=DATASET, mode=args.mode, split=args.split, exact=args.exact)
    module.train(epochs=10000, lr=args.lr, weight_decay=0.0005, patience=100, step=args.step)
