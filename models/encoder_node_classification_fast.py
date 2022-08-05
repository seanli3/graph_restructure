import torch
from numpy.random import seed as nseed
import numpy as np
from torch_geometric.utils import homophily
from models.utils import create_label_sim_matrix, get_distance_diff_indices, find_optimal_edges, our_homophily_measure, get_distance_diff_indices_sparse
from dataset.datasets import get_dataset
from models.autoencoders import SpectralSimilarityEncoder
from tqdm import tqdm
import argparse
from config import SAVED_MODEL_DIR_NODE_CLASSIFICATION, USE_CUDA, DEVICE
from copy import deepcopy
from sklearn.utils import gen_batches
import networkx as nx
import time

device = DEVICE


class Rewirer(torch.nn.Module):
    def __init__(self, data, DATASET, step=0.2, layers=[128, 64], mode="supervised", split=None, exact=False,
                 loss="triplet", eps=0.1, dry_run=False, with_rand_signal=True, with_node_feature=True):
        super(Rewirer, self).__init__()
        self.data = data
        self.DATASET = DATASET
        self.loss = loss
        self.eps = eps
        self.dry_run = dry_run
        self.with_node_feature = with_node_feature
        self.with_rand_signal = with_rand_signal

        self.struct_sim_model = SpectralSimilarityEncoder(data, step=step, name='struct', exact=exact,
                                                          with_node_feature=with_node_feature,
                                                          with_rand_signal=with_rand_signal, sparse=True)
        self.models = [self.struct_sim_model]

        self.mode = mode
        self.split = split

    def train(self, epochs, lr, weight_decay, patience, step, sample_size):
        train_mask = self.data.train_mask[:, self.split] if self.split is not None else self.data.train_mask
        val_mask = self.data.val_mask[:, self.split] if self.split is not None else self.data.val_mask
        self.train_supervised(epochs, lr, weight_decay, patience, step, train_mask, val_mask, sample_size)

    def train_supervised(self, epochs, lr, weight_decay, patience, step, train_mask, val_mask, sample_size):
        data = self.data
        train_idx = train_mask.nonzero().view(-1)
        val_idx = val_mask.nonzero().view(-1)

        for model in self.models:
            model.reset_parameters()
            best_loss_mean = float('inf')
            best_model = {}
            bad_counter = 0

            pbar = tqdm(range(0, epochs))

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            ) if len(list(model.parameters())) > 0 else None

            file_name = SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/fast_{}_{}_{}_{}_{}_{}_{}'.format(
                self.DATASET.lower(), self.mode, model.name, self.loss,
                self.eps, self.with_node_feature, self.with_rand_signal
            ) + '.pt' if self.split is None else SAVED_MODEL_DIR_NODE_CLASSIFICATION \
            + '/fast_{}_{}_{}_{}_{}_split_{}_{}_{}'.format(
                self.DATASET.lower(), self.mode, model.name, self.loss,
                self.eps, self.split, self.with_node_feature, self.with_rand_signal
            ) + '.pt'

            # batch_size = data.num_nodes
            batch_size =256
            train_batches = list(gen_batches(train_idx.shape[0], batch_size, min_batch_size=1))
            val_batches = list(gen_batches(val_idx.shape[0], batch_size, min_batch_size=1))

            train_dist_diff_indices = []
            val_dist_diff_indices = []
            train_D_batch = []
            val_D_batch = []

            y = data.y.view(-1)
            for batch in train_batches:
                train_dist_diff_indices.append(
                    get_distance_diff_indices_sparse(
                        train_idx[batch], y, num_samples=sample_size
                    )
                )
                train_D_batch.append(
                    model.D.index_select(0, train_idx[batch]).index_select(1, train_idx[batch])
                )
            for batch in val_batches:
                val_dist_diff_indices.append(
                    get_distance_diff_indices_sparse(
                        val_idx[batch], y, num_samples=sample_size
                    )
                )
                val_D_batch.append(
                    model.D.index_select(0, val_idx[batch]).index_select(1, val_idx[batch])
                )

            for epoch in pbar:
                train_losses = []
                val_losses = []

                # Training
                for batch_num, batch in enumerate(train_batches):
                    train_loss = self.train_batch(train_dist_diff_indices[batch_num], model, optimizer,
                                                  train_idx[batch], train_D_batch[batch_num])
                    train_losses.append(train_loss.item())
                # # Validation
                for batch_num, batch in enumerate(val_batches):
                    if len(val_dist_diff_indices[batch_num]) > 0:
                        val_loss = self.val_batch(val_dist_diff_indices[batch_num], model, val_idx[batch],
                                                  val_D_batch[batch_num])
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

    def val_batch(self, val_dist_diff_indices, model, val_idx, D_batch):
        model.eval()
        x_hat = model.dist(val_idx, D_batch)
        val_loss = model.dist_triplet_loss(x_hat, val_dist_diff_indices, val_idx.shape[0], self.eps)
        return val_loss

    def train_batch(self, train_dist_diff_indices, model, optimizer, train_idx, D_batch):
        model.train()
        optimizer.zero_grad()
        cur = time.time()
        x_hat = model.dist(train_idx, D_batch)
        print('forward time:', time.time() - cur)
        cur = time.time()
        train_loss = model.dist_triplet_loss(x_hat, train_dist_diff_indices, train_idx.shape[0], self.eps)
        print('loss time:', time.time() - cur)
        cur = time.time()
        train_loss.backward()
        print('backward time:', time.time() - cur)
        optimizer.step()
        return train_loss

    def load(self):
        for model in self.models:
            file_name = SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/fast_{}_{}_{}_{}_{}_{}_{}'.format(
                self.DATASET.lower(), self.mode, model.name, self.loss, self.eps, self.with_node_feature,
                self.with_rand_signal
            ) + '.pt' if self.split is None else SAVED_MODEL_DIR_NODE_CLASSIFICATION + \
                                                 '/fast_{}_{}_{}_{}_{}_split_{}_{}_{}'.format(
                                                     self.DATASET.lower(), self.mode, model.name, self.loss, self.eps,
                                                     self.split,
                                                     self.with_node_feature, self.with_rand_signal) + '.pt'
            saved_model = torch.load(file_name, map_location=device)
            model.load_state_dict(saved_model['model'])


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

    @classmethod
    def rewire(cls, dataset, model_indices, num_edges, split, loss='triplet', eps='0.1', max_node_degree=10,
               step=0.1, layers=[256, 128, 64], with_node_feature=True, with_rand_signal=True, edge_step=None):
        SAVED_DISTANCE_MATRIX = SAVED_MODEL_DIR_NODE_CLASSIFICATION + '/fast_dist_mat_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            dataset.name, split, loss, eps, step, layers, with_node_feature, with_rand_signal
        ) + '.pt'
        from os.path import exists
        file_exists = exists(SAVED_DISTANCE_MATRIX)

        if not file_exists:
            rewirer = Rewirer(
                dataset[0], DATASET=dataset.name, step=step, layers=layers, mode='supervised', split=split, loss=loss,
                eps=eps, with_node_feature=with_node_feature, with_rand_signal=with_rand_signal)
            rewirer.load()
            dist, D = rewirer.get_dist_matrix(model_indices, max_node_degree)
            torch.save([dist, D], SAVED_DISTANCE_MATRIX)

        dist, D = torch.load(SAVED_DISTANCE_MATRIX)

        _, idx = torch.topk(D, int(max_node_degree), dim=0, largest=False)
        val_mask = dataset[0].y.where(dataset[0].val_mask[:, split], torch.tensor(-1, device=device))
        edges = find_optimal_edges(dataset.data.num_nodes, dist, val_mask, step=edge_step)

        new_edge_index = edges
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
              new_edge_index.shape[1] / dataset[0].num_nodes,
              nx.density(G),
              sep=',',
              end=',')

        new_dataset.data.edge_index = new_edge_index
        return new_dataset


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
    parser.add_argument('--with_node_feature', action='store_true')
    parser.add_argument('--with_rand_signal', action='store_true')
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
        torch.cuda.manual_seed(seed)
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset = get_dataset(DATASET, normalize_features=True, lcc=args.lcc, split="public")
    data = dataset[0]

    module = Rewirer(data, step=args.step, layers=[256, 128, 64], DATASET=DATASET, mode=args.mode, split=args.split,
                     exact=args.exact, loss=args.loss, eps=args.eps, dry_run=args.dry_run,
                     with_node_feature=args.with_node_feature, with_rand_signal=args.with_rand_signal)
    module.train(epochs=1000, lr=args.lr, weight_decay=0.0005, patience=100, step=args.step,
                 sample_size=args.sample_size)
