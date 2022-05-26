import torch
from numpy.random import seed as nseed
from torch_geometric.utils import negative_sampling, to_undirected, add_self_loops, homophily
import numpy as np
from models.utils import our_homophily_measure, get_adj, get_random_signals, create_label_sim_matrix, \
    get_normalized_adj, dot_product, get_distance_diff_indices, sample_positive_nodes_dict, sample_negative_nodes_dict
from dataset.datasets import get_dataset
from models.autoencoders import NodeFeatureSimilarityEncoder, SpectralSimilarityEncoder
from tqdm import tqdm
import argparse
from config import SAVED_MODEL_DIR_NODE_CLASSIFICATION, USE_CUDA, DEVICE
from copy import deepcopy
from sklearn.utils import gen_batches
import networkx as nx

device = DEVICE


# random Guassian signals for simulating spectral clustering
# epsilon - error bound


class Rewirer(torch.nn.Module):
    def __init__(self, data, DATASET, step=0.2, layers=[128, 64], mode="supervised", split=None, exact=False,
                 loss="triplet", eps=0.1, dry_run=False):
        super(Rewirer, self).__init__()
        self.data = data
        self.DATASET = DATASET
        self.loss = loss
        self.eps = eps
        self.dry_run = dry_run

        if mode == 'unsupervised':
            self.fea_sim_model = NodeFeatureSimilarityEncoder(data, layers=layers, name='fea')
            self.struct_sim_model = SpectralSimilarityEncoder(data, step=step, name='struct', exact=exact)
            self.conv_sim_model = SpectralSimilarityEncoder(data, data.x, step=step, name="conv", exact=exact)
            self.models = [self.fea_sim_model, self.struct_sim_model]
        else:
            # self.fea_sim_model = NodeFeatureSimilarityEncoder(data, layers=layers, name='fea')
            self.struct_sim_model = SpectralSimilarityEncoder(data, step=step, name='struct', exact=exact)
            # self.conv_sim_model = SpectralSimilarityEncoder(data, data.x, step=step, name="conv")
            self.models = [self.struct_sim_model]

        self.mode = mode
        self.split = split

    def train(self, epochs, lr, weight_decay, patience, step, sample_size):
        if self.mode == 'supervised':
            train_mask = self.data.train_mask[:, self.split] if self.split is not None else self.data.train_mask
            val_mask = self.data.val_mask[:, self.split] if self.split is not None else self.data.val_mask
            self.train_supervised(epochs, lr, weight_decay, patience, step, train_mask, val_mask, sample_size)
        elif self.mode == 'kmeans':
            train_mask = self.data.train_mask[:, self.split] if self.split is not None else self.data.train_mask
            val_mask = self.data.val_mask[:, self.split] if self.split is not None else self.data.val_mask
            self.train_supervised_kmeans(epochs, lr, weight_decay, patience, step, train_mask, val_mask)
        elif self.mode == 'unsupervised':
            self.train_unsupervised(epochs, lr, weight_decay, patience, step)
        elif self.mode == 'vae':
            self.train_vae(epochs, lr, weight_decay, patience, step)

    def train_supervised(self, epochs, lr, weight_decay, patience, step, train_mask, val_mask, sample_size):
        data = self.data
        # community_mask = create_label_sim_matrix(data).bool()
        community = create_label_sim_matrix(data)
        train_idx = train_mask.nonzero().view(-1)
        val_idx = val_mask.nonzero().view(-1)

        if self.loss == 'npair':
            train_positive_nodes = sample_positive_nodes_dict(train_mask, self.data.y, 1)
            train_negative_nodes = sample_negative_nodes_dict(train_mask, self.data.y, (self.data.y.max().item() + 1) * 2)
            val_positive_nodes = sample_positive_nodes_dict(val_mask, self.data.y, 1)
            val_negative_nodes = sample_negative_nodes_dict(val_mask, self.data.y, (self.data.y.max().item() + 1) * 2)
        else:
            train_negative_nodes = None
            train_positive_nodes = None
            val_positive_nodes = None
            val_negative_nodes = None

        for model in self.models:
            model.reset_parameters()
            best_loss_mean = float('inf')
            best_model = {}
            bad_counter = 0

            pbar = tqdm(range(0, epochs))

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            ) if len(list(model.parameters())) > 0 else None

            file_name = SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/{}_{}_{}_{}_{}'.format(
                self.DATASET.lower(), self.mode, model.name, self.loss, self.eps
            ) + '.pt' if self.split is None else SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/{}_{}_{}_{}_{}_split_{}'.format(
                self.DATASET.lower(), self.mode, model.name, self.loss, self.eps, self.split
            ) + '.pt'

            # batch_size = data.num_nodes
            batch_size = 256

            for epoch in pbar:
                train_losses = []
                val_losses = []

                # Training
                batches = gen_batches(train_idx.shape[0], batch_size, min_batch_size=1)
                for batch_num, batch in enumerate(batches):
                    train_loss = self.train_batch(community, model, optimizer, sample_size, train_idx[batch],
                                                  train_mask[batch], train_negative_nodes, train_positive_nodes)
                    train_losses.append(train_loss.item())

                # Validation
                batches = gen_batches(val_idx.shape[0], batch_size, min_batch_size=1)
                for batch in batches:
                    val_loss = self.val_batch(community, model, sample_size, val_idx[batch],
                                              val_mask[batch], val_negative_nodes, val_positive_nodes)
                    val_losses.append(val_loss.item())

                # Compute loss
                train_loss_mean = torch.tensor(train_losses).mean().item()
                val_loss_mean = torch.tensor(val_losses).mean().item()
                pbar.set_description('Epoch: {}, loss: {:.5f} val loss: {:.5f}'.format(
                    epoch, train_loss_mean, val_loss_mean)
                )

                # Epoch finish
                if val_loss_mean < best_loss_mean:
                    best_model['train_loss'] = train_loss_mean
                    best_model['val_loss'] = val_loss_mean
                    best_model['epoch'] = epoch
                    best_model['step'] = step
                    best_model['model'] = deepcopy(model.state_dict())
                    best_loss_mean = val_loss_mean
                    bad_counter = 0
                    if not self.dry_run:
                        torch.save(
                            best_model, file_name
                        )
                else:
                    bad_counter += 1
                    if bad_counter == patience:
                        break
            print(
                'Model saved for Epoch: {}, loss: {:.5f} val loss: {:.5f}'.format(
                    best_model['epoch'], best_model['train_loss'], best_model['val_loss']
                )
            )

    def val_batch(self, community, model, sample_size, val_idx, val_mask, val_negative_nodes,
                  val_positive_nodes):
        val_dist_diff_indices = get_distance_diff_indices(
            community[val_idx, :][:, val_idx], num_samples=sample_size
        )
        model.eval()
        if self.loss in ['contrastive', 'triplet']:
            x_hat = model.dist(val_idx)
        elif self.loss in ['mse']:
            x_hat = model.sim(val_idx)
        elif self.loss in ['npair']:
            x_hat = model()
        if self.loss == 'contrastive':
            val_loss = model.dist_contrastive_loss(x_hat, val_dist_diff_indices)
        elif self.loss == 'triplet':
            val_loss = model.dist_triplet_loss(x_hat, val_dist_diff_indices, val_idx.shape[0], self.eps)
        elif self.loss == 'mse':
            val_loss = model.sim_mse_loss(x_hat, community, val_mask)
        elif self.loss == 'npair':
            val_loss = model.sim_npair_loss(x_hat, val_positive_nodes, val_negative_nodes, val_mask)
        return val_loss

    def train_batch(self, community, model, optimizer, sample_size, train_idx, train_mask,
                    train_negative_nodes, train_positive_nodes):
        train_dist_diff_indices = get_distance_diff_indices(
            community[train_idx, :][:, train_idx], num_samples=sample_size
        )
        model.train()
        optimizer.zero_grad()
        if self.loss in ['contrastive', 'triplet']:
            x_hat = model.dist(train_idx)
        elif self.loss in ['mse']:
            x_hat = model.sim(train_idx)
        elif self.loss in ['npair']:
            x_hat = model()
        if self.loss == 'contrastive':
            train_loss = model.dist_contrastive_loss(x_hat, train_dist_diff_indices, self.eps)
        elif self.loss == 'triplet':
            train_loss = model.dist_triplet_loss(x_hat, train_dist_diff_indices, train_idx.shape[0], self.eps)
        elif self.loss == 'mse':
            train_loss = model.sim_mse_loss(x_hat, community, train_mask)
        elif self.loss == 'npair':
            train_loss = model.sim_npair_loss(x_hat, train_positive_nodes, train_negative_nodes, train_mask)
        train_loss.backward()
        optimizer.step()
        return train_loss

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
            if not self.dry_run:
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
            if not self.dry_run:
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
                if not self.dry_run:
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
                if not self.dry_run:
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
                file_name = SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/{}_{}_{}_{}_{}'.format(
                    self.DATASET.lower(), self.mode, model.name, self.loss, self.eps
                ) + '.pt' if self.split is None else SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/{}_{}_{}_{}_{}_split_{}'\
                    .format(self.DATASET.lower(), self.mode, model.name, self.loss, self.eps, self.split) + '.pt'
                saved_model = torch.load(file_name, map_location=device)
                model.load_state_dict(saved_model['model'])

    def get_rewired_edges_regular(self, model_indices, num_edges):
        with torch.no_grad():
            if self.loss in ['triplet', 'contrastive']:
                triu_indices = torch.triu_indices(self.data.num_nodes, self.data.num_nodes, 1)
                dist = torch.zeros(triu_indices.shape[1], device=device)
                for model in map(self.models.__getitem__, model_indices):
                    model.to(device)
                    dist += model.dist()
                D = torch.zeros(self.data.num_nodes, self.data.num_nodes, device=device)
                D[triu_indices[0], triu_indices[1]] = dist
                D = D+D.T
                D.fill_diagonal_(9e15)
                _, idx = torch.topk(D, int(num_edges), dim=0, largest=False)
                edges = torch.stack((
                    idx.view(-1),
                    torch.arange(self.data.num_nodes, device=device).repeat(num_edges)
                ), dim=0)
            else:
                sim = torch.zeros(self.data.num_nodes, self.data.num_nodes, device=device)
                for model in map(self.models.__getitem__, model_indices):
                    model.to(device)
                    sim += model.sim().squeeze_().fill_diagonal_(0)
                _, idx = torch.topk(sim.flatten(), int(num_edges), dim=0)
                edges = torch.stack((
                    idx.view(-1),
                    torch.arange(self.data.num_nodes, device=device).repeat(num_edges)
                ), dim=0)
            return edges

    @classmethod
    def rewire(cls, dataset, model_indices, num_edges, split, loss='triplet', eps='0.1', max_node_degree=10,
               step=0.1, layers=[256, 128, 64]):
        SAVED_DISTANCE_MATRIX = SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/dist_mat_{}_{}_{}_{}_{}_{}'.format(
            dataset.name, split, loss, eps, step, layers
        ) + '.pt'
        from os.path import exists
        file_exists = exists(SAVED_DISTANCE_MATRIX)
        print(SAVED_DISTANCE_MATRIX)
        print(file_exists)

        if not file_exists:
            rewirer = Rewirer(
                dataset[0], DATASET=dataset.name, step=step, layers=layers,
                mode='supervised', split=split, loss=loss, eps=eps)
            rewirer.load()
            if loss in ['triplet', 'contrastive']:
                dist, D = rewirer.get_dist_matrix(model_indices, max_node_degree)
            else:
                dist, D = rewirer.get_sim_matrix(model_indices, max_node_degree)
            torch.save([dist, D], SAVED_DISTANCE_MATRIX)

        dist, D = torch.load(SAVED_DISTANCE_MATRIX)
        if loss in ['triplet', 'contrastive']:
            _, idx = torch.topk(D, int(max_node_degree), dim=0, largest=False)
            edges = torch.stack(
                (idx.view(-1), torch.arange(dataset.data.num_nodes, device=device).repeat(max_node_degree)),
                dim=0
            )
            A_2 = torch.zeros(dataset.data.num_nodes, dataset.data.num_nodes, device=device).bool()
            A_2[edges[0], edges[1]] = True

            triu_indices = torch.triu_indices(dataset.data.num_nodes, dataset.data.num_nodes, 1)
            _, idx = torch.topk(dist, int(num_edges), dim=0, largest=False)
            edges = to_undirected(triu_indices[:, idx])
            A_1 = torch.zeros(dataset.data.num_nodes, dataset.data.num_nodes, device=device).bool()
            A_1[edges[0], edges[1]] = True
        else:
            _, idx = torch.topk(D, int(max_node_degree), dim=0, largest=True)
            edges = torch.stack(
                (idx.view(-1), torch.arange(dataset.data.num_nodes, device=device).repeat(max_node_degree)), dim=0
            )
            A_2 = torch.zeros(dataset.data.num_nodes, dataset.data.num_nodes, device=device).bool()
            A_2[edges[0], edges[1]] = True

            _, idx = torch.topk(D.flatten(), int(num_edges))
            edges = torch.tensor(np.array(np.unravel_index(idx.cpu().numpy(), sim.shape)), device=device).detach()
            edges = to_undirected(edges)
            A_1 = torch.zeros(dataset.data.num_nodes, dataset.data.num_nodes, device=device).bool()
            A_1[edges[0], edges[1]] = True

        A = A_1.logical_and_(A_2)
        new_edge_index = A.nonzero().T
        new_dataset = deepcopy(dataset)

        G = nx.Graph(new_edge_index.T.tolist())
        G = G.to_undirected()
        print(new_edge_index.shape[1],
              our_homophily_measure(new_edge_index, dataset[0].y).item(),
              our_homophily_measure(
                  new_edge_index, dataset[0].y.where(dataset[0].train_mask[:, split], torch.tensor(-1, device=device))
              ).item(),
              our_homophily_measure(
                  new_edge_index, dataset[0].y.where(dataset[0].val_mask[:, split], torch.tensor(-1, device=device))
              ).item(),
              our_homophily_measure(
                  new_edge_index, dataset[0].y.where(dataset[0].test_mask[:, split], torch.tensor(-1, device=device))
              ).item(),
              homophily(new_edge_index, dataset[0].y, method='edge'),
              homophily(new_edge_index, dataset[0].y, method='node'),
              homophily(new_edge_index, dataset[0].y, method='edge_insensitive'),
              new_edge_index.shape[1]/dataset[0].num_nodes,
              nx.density(G),
              end=' ')
        # print("edges before: ", dataset.data.num_edges, " edges after: ", new_edge_index.shape[1])
        # print('homophily before:', our_homophily_measure(dataset[0].edge_index, dataset[0].y),
        #       'homophily after:', our_homophily_measure(new_edge_index, dataset[0].y))
        new_edges_set = set([tuple(i) for i in new_edge_index.T.tolist()])
        old_edges_set = set([tuple(i) for i in dataset[0].edge_index.T.tolist()])
        # edges_added = len(new_edges_set.difference(old_edges_set))
        # edges_removed = len(old_edges_set.difference(new_edges_set))
        # edges_preserved = len(old_edges_set.intersection(new_edges_set))
        # print('Edges added: {}, edges removed: {}, edges preserved: {}'.format(edges_added, edges_removed, edges_preserved))

        new_dataset.data.edge_index = new_edge_index
        return new_dataset

    def get_sim_matrix(self, model_indices, max_node_degree):
        sim = torch.zeros(self.data.num_nodes, self.data.num_nodes, device=device)
        for model in map(self.models.__getitem__, model_indices):
            model.to(device)
            sim += model.sim().squeeze_()
        sim.fill_diagonal_(-9e15)
        return sim, None

    def get_dist_matrix(self, model_indices, max_node_degree):
        triu_indices = torch.triu_indices(self.data.num_nodes, self.data.num_nodes, 1)
        dist = torch.zeros(triu_indices.shape[1], device=device)
        for model in map(self.models.__getitem__, model_indices):
            model.to(device)
            dist += model.dist()
        D = torch.zeros(self.data.num_nodes, self.data.num_nodes, device=device)
        D[triu_indices[0], triu_indices[1]] = dist
        D = D + D.T
        D.fill_diagonal_(9e15)
        return dist, D

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
    parser.add_argument('--lcc', action='store_true')
    parser.add_argument('--loss', type=str, default='triplet')
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--sample_size', type=int, default=5)
    args = parser.parse_args()

    if args.loss not in ['triplet', 'contrastive', 'npair', 'mse']:
        raise RuntimeError('loss has to be one of triplet, contrastive, npair, mse')

    print('Running rewire on {}, with lr={} step={} split={} mode={} lcc={} loss={} eps={} dry_run={} sample_size={}'
          .format(args.dataset, args.lr, args.step, args.split, args.mode, args.lcc, args.loss, args.eps, args.dry_run,
                  args.sample_size))

    DATASET = args.dataset

    print(DATASET)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available() and USE_CUDA:
        print("-----------------------Training on CUDA-------------------------")
        torch.cuda.manual_seed(seed)
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset = get_dataset(DATASET, normalize_features=True, lcc=args.lcc, split="public")
    data = dataset[0]

    module = Rewirer(data, step=args.step, layers=[256, 128, 64], DATASET=DATASET, mode=args.mode, split=args.split,
                     exact=args.exact, loss=args.loss, eps=args.eps, dry_run=args.dry_run)
    module.train(epochs=10000, lr=args.lr, weight_decay=0.0005, patience=100, step=args.step, sample_size=args.sample_size)
