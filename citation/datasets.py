import os.path as osp
from torch_geometric.datasets import Planetoid, PPI, Amazon, Reddit, Coauthor, Flickr, Yelp
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, dropout_adj, get_laplacian
from random import sample
from torch.nn import functional as F
from torch_geometric.utils import to_networkx
import networkx as nx
import torch
import networkx as nx
from scipy.sparse import coo_matrix
import numpy as np
from .nell import Nell
from .iris_data import Iris


def matching_labels_distribution(dataset):
    # Build graph
    adj = coo_matrix(
        (np.ones(dataset[0].num_edges),
        (dataset[0].edge_index[0].numpy(), dataset[0].edge_index[1].numpy())),
        shape=(dataset[0].num_nodes, dataset[0].num_nodes))
    G = nx.Graph(adj)

    hop_1_matching_percent = []
    hop_2_matching_percent = []
    hop_3_matching_percent = []
    for n in range(dataset.data.num_nodes):
        hop_1_neighbours = list(nx.ego_graph(G, n, 1).nodes())
        hop_2_neighbours = list(nx.ego_graph(G, n, 2).nodes())
        hop_3_neighbours = list(nx.ego_graph(G, n, 3).nodes())
        node_label = dataset[0].y[n]
        hop_1_labels = dataset[0].y[hop_1_neighbours]
        hop_2_labels = dataset[0].y[hop_2_neighbours]
        hop_3_labels = dataset[0].y[hop_3_neighbours]
        matching_1_labels = node_label == hop_1_labels
        matching_2_labels = node_label == hop_2_labels
        matching_3_labels = node_label == hop_3_labels
        hop_1_matching_percent.append(matching_1_labels.float().sum()/matching_1_labels.shape[0])
        hop_2_matching_percent.append(matching_2_labels.float().sum()/matching_2_labels.shape[0])
        hop_3_matching_percent.append(matching_3_labels.float().sum()/matching_3_labels.shape[0])

    return hop_1_matching_percent, hop_2_matching_percent, hop_3_matching_percent

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, dissimilar_mask=None, lcc_mask=None):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        if dissimilar_mask is None and lcc_mask is None:
            index = (data.y == i).nonzero().view(-1)
        elif dissimilar_mask is not None and lcc_mask is None:
            index = (data.y == i).logical_and(dissimilar_mask).nonzero().view(-1)
        elif dissimilar_mask is None and lcc_mask is not None:
            index = (data.y == i).logical_and(lcc_mask).nonzero().view(-1)
        else:
            index = (data.y == i).logical_and(dissimilar_mask).logical_and(lcc_mask).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    if len(rest_index) < 60:
        raise Exception('Not enough val and test data')

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:min(len(rest_index)//3, 500)], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[min(len(rest_index)//3, 500):max(len(rest_index)*2//3, 1500)], size=data.num_nodes)

    return data


def random_coauthor_amazon_splits(data, num_classes, lcc_mask):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing

    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data


def get_dataset(name, normalize_features=False, transform=None, edge_dropout=None, node_feature_dropout=None,
                dissimilar_t = 1, cuda=False, permute_masks=None, lcc=False, split="public", self_loop=True,
                dummy_nodes=0, removal_nodes=0, substitute=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    if name in ['Computers', 'Photo']:
        dataset = Amazon(path, name)
    elif name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(path, name, split=split)
    elif name in ['CS', 'Physics']:
        dataset = Coauthor(path, name)
    elif name in ['Reddit']:
        dataset = Reddit(path)
    elif name in ['Flickr']:
        dataset = Flickr(path)
    elif name in ['Yelp']:
        dataset = Yelp(path)
    elif name in ['nell.0.1', 'nell.0.01', 'nell.0.001']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Nell')
        dataset = Nell(path, 'Nell', name)
    elif name.lower() == 'iris':
        dataset = Iris()

    # # Removing high-degree nodes
    # print('remove nodes:', removal_nodes)
    # for _ in range(removal_nodes):
    #     adj = torch.sparse_coo_tensor(dataset[0].edge_index, torch.ones(dataset[0].edge_index.shape[1])).to_dense()
    #     max_degree_node = adj.sum(dim=0).argmax().item()
    #     new_indices = list(range(adj.shape[0]))
    #     new_indices.remove(max_degree_node)
    #     adj = adj[new_indices][:, new_indices]
    #     dataset.data.edge_index = adj.nonzero(as_tuple=False).T
    #     dataset.data.x = torch.cat((dataset.data.x[:max_degree_node, :], dataset.data.x[max_degree_node+1:, :]))
    #     dataset.data.y = torch.cat((dataset.data.y[:max_degree_node], dataset.data.y[max_degree_node+1:]))
    #     dataset.data.train_mask = torch.cat((dataset.data.train_mask[:max_degree_node], dataset.data.train_mask[max_degree_node+1:]))
    #     dataset.data.val_mask = torch.cat((dataset.data.val_mask[:max_degree_node], dataset.data.val_mask[max_degree_node+1:]))
    #     dataset.data.test_mask = torch.cat((dataset.data.test_mask[:max_degree_node], dataset.data.test_mask[max_degree_node+1:]))
    #     dataset.data.num_nodes = dataset.data.x.shape[0]
    #     dataset.__data_list__ = None
    #     dataset.data, dataset.slices = dataset.collate([dataset.data])
    #
    #     # edge_index, edge_weight = get_laplacian(dataset.data.edge_index, normalization="sym")
    #     # L = torch.sparse_coo_tensor(edge_index, edge_weight).to_dense()
    #     # l2 = torch.symeig(L)[0]
    #     # from matplotlib import pyplot as plt
    #     # plt.plot(torch.arange(L.shape[0]), l2)
    #     # plt.title('remove {} nodes'.format(_+1))
    #     # plt.savefig('./remove_{}_nodes.png'.format(_ + 1))
    #     # plt.clf()
    #
    # # Adding high-degree dummy nodes
    # print('dummy nodes:', dummy_nodes)
    # for _ in range(dummy_nodes):
    #     adj = torch.sparse_coo_tensor(dataset[0].edge_index, torch.ones(dataset[0].edge_index.shape[1])).to_dense()
    #     # edge_index, edge_weight = get_laplacian(dataset.data.edge_index, normalization="sym")
    #     # L = torch.sparse_coo_tensor(edge_index, edge_weight).to_dense()
    #     # l1 = torch.symeig(L)[0]
    #
    #     new_col = torch.dropout(torch.ones(adj.shape[0], 1).float(), 0.8, train=True).bool().float()
    #     adj = torch.cat((adj, new_col), dim=1)
    #     new_row=torch.cat((new_col.T, torch.ones(1,1).float()), dim=1)
    #     adj = torch.cat((adj, new_row), dim=0)
    #     dataset.data.edge_index = adj.nonzero(as_tuple=False).T
    #     dataset.data.x = torch.cat((dataset.data.x, dataset.data.x[0].view(1, -1)))
    #     dataset.data.y = torch.cat((dataset.data.y, dataset.data.y[0].view(1)))
    #     dataset.data.train_mask = torch.cat((dataset.data.train_mask, torch.ones(1).bool()))
    #     dataset.data.val_mask = torch.cat((dataset.data.val_mask, torch.zeros(1).bool()))
    #     dataset.data.test_mask = torch.cat((dataset.data.test_mask, torch.zeros(1).bool()))
    #     dataset.data.num_nodes = dataset.data.x.shape[0]
    #     dataset.__data_list__ = None
    #     dataset.data, dataset.slices = dataset.collate([dataset.data])
    #
    #     # edge_index, edge_weight = get_laplacian(dataset.data.edge_index, normalization="sym")
    #     # L = torch.sparse_coo_tensor(edge_index, edge_weight).to_dense()
    #     # l2 = torch.symeig(L)[0]
    #     # from matplotlib import pyplot as plt
    #     # plt.plot(torch.arange(L.shape[0]), l2)
    #     # plt.title('{} dummy nodes'.format(_+1))
    #     # plt.savefig('./{}_dummy_100.png'.format(_ + 1))
    #     # plt.clf()


    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    if self_loop:
        dataset.data.edge_index = add_self_loops(dataset.data.edge_index)[0]

    if edge_dropout:
        edge_list = dataset.data.edge_index
        num_edges = edge_list.shape[1]
        edge_list, _ = dropout_adj(edge_list, p=edge_dropout, force_undirected=True)
        print('Edge dropout rate: {:.4f}'.format(1 - edge_list.shape[1] / num_edges))
        dataset.data.edge_index = edge_list
    if node_feature_dropout:
        num_nodes = dataset.data.num_nodes
        drop_indices = sample(list(range(num_nodes)), int(node_feature_dropout * num_nodes))
        dataset.data.x.index_fill_(0, torch.tensor(drop_indices).cpu(), 0)
        print('Node feature dropout rate: {:.4f}' .format(len(drop_indices)/num_nodes))

    if dissimilar_t < 1 and not permute_masks:
        label_distributions = torch.tensor(matching_labels_distribution(dataset)).cpu()
        dissimilar_neighbhour_train_mask = dataset[0]['train_mask']\
            .logical_and(label_distributions[0] <= dissimilar_t)
        dissimilar_neighbhour_val_mask = dataset[0]['val_mask']\
            .logical_and(label_distributions[0] <= dissimilar_t)
        dissimilar_neighbhour_test_mask = dataset[0]['test_mask']\
            .logical_and(label_distributions[0] <= dissimilar_t)
        dataset.data.train_mask = dissimilar_neighbhour_train_mask
        dataset.data.val_mask = dissimilar_neighbhour_val_mask
        dataset.data.test_mask = dissimilar_neighbhour_test_mask

    lcc_mask = None
    if lcc:  # select largest connected component
        data_ori = dataset[0]
        data_nx = to_networkx(data_ori)
        data_nx = data_nx.to_undirected()
        print("Original #nodes:", data_nx.number_of_nodes())
        data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
        print("#Nodes after lcc:", data_nx.number_of_nodes())
        lcc_mask = index_to_mask(torch.tensor(list(data_nx.nodes)), size=dataset[0].num_nodes)
        adj = torch.sparse_coo_tensor(dataset[0].edge_index, torch.ones(dataset[0].edge_index.shape[1])).to_dense()
        adj = adj[lcc_mask][:, lcc_mask]
        dataset.data.x = dataset.data.x[lcc_mask]
        dataset.data.y = dataset.data.y[lcc_mask]
        dataset.data.train_mask = dataset.data.train_mask[lcc_mask]
        dataset.data.val_mask = dataset.data.val_mask[lcc_mask]
        dataset.data.test_mask = dataset.data.test_mask[lcc_mask]
        dataset.data.edge_index = adj.nonzero(as_tuple=False).T
        dataset.data, dataset.slices = dataset.collate([dataset.data])
        del dataset.__data_list__
        assert(nx.is_connected(to_networkx(dataset[0]).to_undirected()))


    if permute_masks is not None:
        # label_distributions = torch.tensor(matching_labels_distribution(dataset)).cpu()
        dataset.data = permute_masks(dataset.data, dataset.num_classes, lcc_mask=lcc_mask)
        for key in dataset.data.keys:
            if key not in dataset.slices:
                dataset.slices[key] = torch.tensor([0, dataset.data[key].shape[0]])

    if substitute:
        adj = torch.tensor(coo_matrix(
            (np.ones(dataset[0].num_edges),
             (dataset[0].edge_index[0].numpy(), dataset[0].edge_index[1].numpy())),
            shape=(dataset[0].num_nodes, dataset[0].num_nodes)).todense())

        node_map = {}
        adj2 = torch.zeros(adj.shape)
        dataset.data.old_edge_index = dataset.data.edge_index
        dataset.data.old_x = dataset.data.x
        dataset.data.old_y = dataset.data.y
        dataset.data.old_train_mask = dataset.data.train_mask
        dataset.data.old_val_mask = dataset.data.val_mask
        dataset.data.old_test_mask = dataset.data.test_mask

        while (adj.sum(0) > 3).any():
            asum = adj.sum(0).view(-1)
            idx = (asum > 3).nonzero(as_tuple=True)[0][0]
            num = asum[idx].int().item()
            padding = torch.zeros(num, adj.shape[0])

            loop_base = torch.zeros(num)
            loop_base[1] = loop_base[-1] = 1
            loop = []
            for i in range(num):
                loop.append(loop_base.roll(i, 0))
            loop = torch.stack(loop, dim=0)

            adj2 = torch.cat((adj2, padding), dim=0)
            adj2 = torch.cat((adj2, torch.cat((padding.T, loop), dim=0)), dim=1)
            adj2 = adj2[1:, 1:]

            padding[torch.arange(num), adj[idx].nonzero(as_tuple=True)[0]] = 1
            adj = torch.cat((adj, padding), dim=0)
            zeros = torch.zeros(num, num)
            # adj = torch.cat((adj, torch.cat((padding.T, zeros), dim=0)), dim=1)
            adj = torch.cat((adj, torch.cat((padding.T, loop), dim=0)), dim=1)
            adj = adj[torch.arange(adj.shape[0]) != idx]
            adj = adj[:, torch.arange(adj.shape[0] + 1) != idx]

            dataset.data.x = torch.cat((dataset.data.x[torch.arange(dataset.data.x.shape[0]) != idx],
                                       dataset.data.old_x[idx].repeat(num,1)), dim=0)
            dataset.data.y = torch.cat((dataset.data.y[torch.arange(dataset.data.y.shape[0]) != idx],
                                        dataset.data.old_y[idx].repeat(num)), dim=0)

            dataset.data.train_mask = torch.cat((dataset.data.train_mask[torch.arange(dataset.data.train_mask.shape[0]) != idx],
                                        dataset.data.old_train_mask[idx].repeat(num)), dim=0)
            dataset.data.val_mask = torch.cat((dataset.data.val_mask[torch.arange(dataset.data.val_mask.shape[0]) != idx],
                                                 dataset.data.old_val_mask[idx].repeat(num)), dim=0)
            dataset.data.test_mask = torch.cat((dataset.data.test_mask[torch.arange(dataset.data.test_mask.shape[0]) != idx],
                                               dataset.data.old_test_mask[idx].repeat(num)), dim=0)


            for nid in node_map:
                node_map[nid] = node_map[nid]-1

            node_map[idx.item()] = (torch.arange(num) + adj.shape[0] - num)


        dataset.data.edge_index = adj.nonzero(as_tuple=False).T
        dataset.data.edge_index2 = adj2.nonzero(as_tuple=False).T
        dataset.data.node_map = node_map
        dataset.data, dataset.slices = dataset.collate([dataset.data])
        del dataset.__data_list__

    if cuda:
        dataset.data.to('cuda')
        if hasattr(dataset, '__data_list__') and dataset.__data_list__:
            for d in dataset.__data_list__:
                d.to('cuda')

    torch.save(dataset, './sparsified_graph.pt')

    return dataset
    # return torch.load('./sparsified_graph.pt')
