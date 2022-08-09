import math
from itertools import product

import numpy as np
import torch
from numpy.random import seed as nseed
from torch_geometric.utils import get_laplacian, to_dense_adj, to_undirected, add_self_loops
from config import SEED
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops
from config import DEVICE
from torch_sparse import spspmm


device = DEVICE


def compat_matrix_edge_idx(edge_idx, labels):
    """
     c x c compatibility matrix, where c is number of classes
     H[i,j] is proportion of endpoints that are class j
     of edges incident to class i nodes
     "Generalizing GNNs Beyond Homophily"
     treats negative labels as unlabeled
     """
    edge_index = edge_idx
    src_node, targ_node = edge_index[0,:], edge_index[1,:]
    labeled_nodes = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    label = labels.squeeze().to(device)
    c = label.max()+1
    H = torch.zeros((c,c)).to(device)
    src_label = label[src_node[labeled_nodes]]
    targ_label = label[targ_node[labeled_nodes]]
    label_idx = torch.cat((src_label.unsqueeze(0), targ_label.unsqueeze(0)), axis=0)
    for k in range(c):
        sum_idx = torch.where(src_label == k)[0]
        add_idx = targ_label[sum_idx]
        scatter_add(torch.ones_like(add_idx, device=device).to(H.dtype), add_idx, out=H[k,:], dim=-1)
    return H


def normalized_compat_matrix_edge_idx(edge_idx, label):
    label = label.squeeze()
    c = label.max() + 1
    H = compat_matrix_edge_idx(edge_idx, label)
    nonzero_label = label[label >= 0]
    counts = nonzero_label.unique(return_counts=True)[1]
    complete_graph_edges = counts.view(-1,1).mm(counts.view(1, -1))
    complete_graph_edges = complete_graph_edges - torch.diag(counts)
    # H2 = H / complete_graph_edges
    num_nodes = label.shape[0]
    total = num_nodes * (num_nodes - 1)
    H2 = H / complete_graph_edges
    H2[H2.isnan()] = 0
    H1 = H / torch.sum(H, axis=1, keepdims=True)
    H1[H1.isnan()] = 0
    # val = H1 * (1 - H2)
    val = H2
    return val


def homophily(edge_index, label):
    """
    our measure \hat{h}
    treats negative labels as unlabeled
    """
    label = label.squeeze()
    nonzero_label = label[label >= 0]
    c = label.max()+1
    counts = nonzero_label.unique(return_counts=True)[1]
    H = compat_matrix_edge_idx(edge_index, label)
    H = H / torch.sum(H, axis=1, keepdims=True)
    nonzero_label = label[label >= 0]
    proportions = counts.float() / nonzero_label.shape[0]
    val = 0
    for k in range(c):
        class_add = torch.clamp(H[k,k] - proportions[k], min=0)
        if not torch.isnan(class_add):
            # only add if not nan
            val += class_add
    val /= c-1
    return val


def our_homophily_measure(edge_index, label):
    """
    our measure \hat{h}
    treats negative labels as unlabeled
    """
    label = label.squeeze()
    H = compat_matrix_edge_idx(edge_index, label)
    nonzero_label = label[label >= 0]
    counts = nonzero_label.unique(return_counts=True)[1].float()
    complete_graph_edges = counts.view(-1,1).mm(counts.view(1, -1))
    try:
        h = H/complete_graph_edges
    except RuntimeError as e:
        # print('Missing labels')
        return torch.tensor(0)
    h_homo = h.diag()
    h_hete = (h.triu(1) + h.tril(-1)).max(0).values
    # ret = max(h_hete, h_homo) * (h_homo - h_hete) / density
    # return 1 / (1 + torch.exp(- ret))
    return (h_homo - h_hete).min()/2+0.5


    # complete_graph_edges = counts.view(-1,1).mm(counts.view(1, -1))
    # # complete_graph_edges = complete_graph_edges - torch.diag(counts)
    # complete_graph_edges = complete_graph_edges + torch.diag(counts)
    # complete_graph_edges[torch.eye(c,c).bool()] = (complete_graph_edges.diag().float()/2).long()
    #
    # h_homo = H.diag() / complete_graph_edges.diag()
    # h_hete = (H / complete_graph_edges).fill_diagonal_(0).sum(dim=1) / (c-1)
    #
    # h_homo[h_homo.isnan()] = 0
    # h_hete[h_hete.isnan()] = 0
    # # h_homo = h_homo.where(h_homo > 0.1, 0.1)
    # # h_hete = h_hete.where(h_hete > 0.1, 0.1)
    # density_cap = 0.1
    # cap_deg = density_cap*(counts - 1)/2
    # sparsity_factor = 10
    # # scale_factor = sparsity_factor*(counts - 1)/(2*cap_deg) # 100
    # scale_factor = sparsity_factor * 20
    # h_homo = torch.tanh(scale_factor * h_homo)
    # h_hete = torch.tanh(scale_factor * h_hete)
    # # h_homo = h_homo / ((num_nodes - counts) / counts)
    #
    # val = (h_homo - h_hete).mean()
    # return val

class ASTNodeEncoder(torch.nn.Module):
    '''
        Input:
            x: default node feature. the first and second column represents node type and node attributes.
            depth: The depth of the node in the AST.

        Output:
            emb_dim-dimensional vector

    '''
    def __init__(self, emb_dim, num_nodetypes, num_nodeattributes, max_depth):
        super(ASTNodeEncoder, self).__init__()

        self.max_depth = max_depth

        self.type_encoder = torch.nn.Embedding(num_nodetypes, emb_dim)
        self.attribute_encoder = torch.nn.Embedding(num_nodeattributes, emb_dim)
        self.depth_encoder = torch.nn.Embedding(self.max_depth + 1, emb_dim)

    def forward(self, x, depth):
        depth[depth > self.max_depth] = self.max_depth
        return self.type_encoder(x[:, 0]) + self.attribute_encoder(x[:, 1]) + self.depth_encoder(depth)


def get_vocab_mapping(seq_list, num_vocab):
    '''
        Input:
            seq_list: a list of sequences
            num_vocab: vocabulary size
        Output:
            vocab2idx:
                A dictionary that maps vocabulary into integer index.
                Additioanlly, we also index '__UNK__' and '__EOS__'
                '__UNK__' : out-of-vocabulary term
                '__EOS__' : end-of-sentence

            idx2vocab:
                A list that maps idx to actual vocabulary.

    '''

    vocab_cnt = {}
    vocab_list = []
    for seq in seq_list:
        for w in seq:
            if w in vocab_cnt:
                vocab_cnt[w] += 1
            else:
                vocab_cnt[w] = 1
                vocab_list.append(w)

    cnt_list = np.array([vocab_cnt[w] for w in vocab_list])
    topvocab = np.argsort(-cnt_list, kind='stable')[:num_vocab]

    print('Coverage of top {} vocabulary:'.format(num_vocab))
    print(float(np.sum(cnt_list[topvocab])) / np.sum(cnt_list))

    vocab2idx = {vocab_list[vocab_idx]: idx for idx, vocab_idx in enumerate(topvocab)}
    idx2vocab = [vocab_list[vocab_idx] for vocab_idx in topvocab]

    # print(topvocab)
    # print([vocab_list[v] for v in topvocab[:10]])
    # print([vocab_list[v] for v in topvocab[-10:]])

    vocab2idx['__UNK__'] = num_vocab
    idx2vocab.append('__UNK__')

    vocab2idx['__EOS__'] = num_vocab + 1
    idx2vocab.append('__EOS__')

    # test the correspondence between vocab2idx and idx2vocab
    for idx, vocab in enumerate(idx2vocab):
        assert (idx == vocab2idx[vocab])

    # test that the idx of '__EOS__' is len(idx2vocab) - 1.
    # This fact will be used in decode_arr_to_seq, when finding __EOS__
    assert (vocab2idx['__EOS__'] == len(idx2vocab) - 1)

    return vocab2idx, idx2vocab


def augment_edge(data):
    '''
        Input:
            data: PyG data object
        Output:
            data (edges are augmented in the following ways):
                data.edge_index: Added next-token edge. The inverse edges were also added.
                data.edge_attr (torch.Long):
                    data.edge_attr[:,0]: whether it is AST edge (0) for next-token edge (1)
                    data.edge_attr[:,1]: whether it is original direction (0) or inverse direction (1)
    '''

    ##### AST edge
    edge_index_ast = data.edge_index
    edge_attr_ast = torch.zeros((edge_index_ast.size(1), 2))

    ##### Inverse AST edge
    edge_index_ast_inverse = torch.stack([edge_index_ast[1], edge_index_ast[0]], dim=0)
    edge_attr_ast_inverse = torch.cat(
        [torch.zeros(edge_index_ast_inverse.size(1), 1), torch.ones(edge_index_ast_inverse.size(1), 1)], dim=1
    )

    ##### Next-token edge

    ## Obtain attributed nodes and get their indices in dfs order
    # attributed_node_idx = torch.where(data.node_is_attributed.view(-1,) == 1)[0]
    # attributed_node_idx_in_dfs_order = attributed_node_idx[torch.argsort(data.node_dfs_order[attributed_node_idx].view(-1,))]

    ## Since the nodes are already sorted in dfs ordering in our case, we can just do the following.
    attributed_node_idx_in_dfs_order = torch.where(data.node_is_attributed.view(-1, ) == 1)[0]

    ## build next token edge
    # Given: attributed_node_idx_in_dfs_order
    #        [1, 3, 4, 5, 8, 9, 12]
    # Output:
    #    [[1, 3, 4, 5, 8, 9]
    #     [3, 4, 5, 8, 9, 12]
    edge_index_nextoken = torch.stack(
        [attributed_node_idx_in_dfs_order[:-1], attributed_node_idx_in_dfs_order[1:]], dim=0
    )
    edge_attr_nextoken = torch.cat(
        [torch.ones(edge_index_nextoken.size(1), 1), torch.zeros(edge_index_nextoken.size(1), 1)], dim=1
    )

    ##### Inverse next-token edge
    edge_index_nextoken_inverse = torch.stack([edge_index_nextoken[1], edge_index_nextoken[0]], dim=0)
    edge_attr_nextoken_inverse = torch.ones((edge_index_nextoken.size(1), 2))

    data.edge_index = torch.cat(
        [edge_index_ast, edge_index_ast_inverse, edge_index_nextoken, edge_index_nextoken_inverse], dim=1
    )
    data.edge_attr = torch.cat(
        [edge_attr_ast, edge_attr_ast_inverse, edge_attr_nextoken, edge_attr_nextoken_inverse], dim=0
    )

    return data


def encode_y_to_arr(data, vocab2idx, max_seq_len):
    '''
    Input:
        data: PyG graph object
        output: add y_arr to data
    '''

    # PyG >= 1.5.0
    seq = data.y

    # PyG = 1.4.3
    # seq = data.y[0]

    data.y_arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len)

    return data


def encode_seq_to_arr(seq, vocab2idx, max_seq_len):
    '''
    Input:
        seq: A list of words
        output: add y_arr (torch.Tensor)
    '''

    augmented_seq = seq[:max_seq_len] + ['__EOS__'] * max(0, max_seq_len - len(seq))
    return torch.tensor(
        [[vocab2idx[w] if w in vocab2idx else vocab2idx['__UNK__'] for w in augmented_seq]], dtype=torch.long
    )


def decode_arr_to_seq(arr, idx2vocab):
    '''
        Input: torch 1d array: y_arr
        Output: a sequence of words.
    '''

    eos_idx_list = torch.nonzero(
        arr == len(idx2vocab) - 1, as_tuple=False
    )  # find the position of __EOS__ (the last vocab in idx2vocab)
    if len(eos_idx_list) > 0:
        clippted_arr = arr[: torch.min(eos_idx_list)]  # find the smallest __EOS__
    else:
        clippted_arr = arr

    return list(map(lambda x: idx2vocab[x], clippted_arr.cpu()))


def test():
    seq_list = [['a', 'b'], ['a', 'b', 'c', 'df', 'f', '2edea', 'a'], ['eraea', 'a', 'c'], ['d'],
                ['4rq4f', 'f', 'a', 'a', 'g']]
    vocab2idx, idx2vocab = get_vocab_mapping(seq_list, 4)
    print(vocab2idx)
    print(idx2vocab)
    print()
    assert (len(vocab2idx) == len(idx2vocab))

    for vocab, idx in vocab2idx.items():
        assert (idx2vocab[idx] == vocab)

    for seq in seq_list:
        print(seq)
        arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len=4)[0]
        # Test the effect of predicting __EOS__
        # arr[2] = vocab2idx['__EOS__']
        print(arr)
        seq_dec = decode_arr_to_seq(arr, idx2vocab)

        print(arr)
        print(seq_dec)
        print('')


# Don't use, doesn't converge
def jacobi_inv(m, k):
    X = torch.zeros_like(m, device=device)
    D_inv = torch.diag_embed(1/m.diagonal(dim1=1, dim2=2))
    LU = m.tril(-1) + m.triu(1)
    for _ in range(k):
        X = D_inv - D_inv.matmul(LU).matmul(X)
    return X


def neumann_inv(m, k):
    I = torch.diag_embed(torch.ones(m.shape[0], m.shape[1], device=device))
    ret = I
    for l in range(k):
        ret = ret.matmul(I + (I-m).matrix_power(int(math.pow(2,l))))
    if ret.isnan().any() or ret.isinf().any():
        raise RuntimeError('NaN or Inf in Neumann approximation, try reduce order')
    return ret


def neumann_inv_sparse(m, I, k):
    ret = I
    size = I.shape[0]
    ret = ret.coalesce()
    ret_index = ret.indices()
    ret_values = ret.values()
    I_m = (I - m).coalesce()
    for l in range(k):
        I_m_pow_index = I_m.indices()
        I_m_pow_values = I_m.values()
        for _ in range(1, int(math.pow(2,l))):
            I_m_pow_index, I_m_pow_values = spspmm(I_m_pow_index, I_m_pow_values, I_m_pow_index, I_m_pow_values, size, size, size)
            # I_m_pow_values = I_m_pow_values.clamp(max=1, min=-1)
            # I_m_pow_index = I_m_pow_index[:, I_m_pow_values.abs() > 0.001]
            # I_m_pow_values = I_m_pow_values[I_m_pow_values.abs() > 0.001]
        I_m_pow = torch.sparse_coo_tensor(I_m_pow_index, I_m_pow_values, device=device, size=(size, size))
        IIm = (I + I_m_pow).coalesce()
        IIm_index = IIm.indices()
        IIm_values = IIm.values()
        IIm_index = IIm_index[:, IIm_values.abs() > 1e-6]
        IIm_values = IIm_values[IIm_values.abs() > 1e-6]
        ret_index, ret_values = spspmm(ret_index, ret_values, IIm_index, IIm_values, size, size, size)
        # ret_index = ret_index[:, ret_values.abs() > 0.001]
        # ret_values = ret_values[ret_values.abs() > 0.001]
        if ret_values.isnan().any() or ret_values.isinf().any():
            raise RuntimeError('NaN or Inf in Neumann approximation, try reduce order')
    return ret_index, ret_values


def create_filter(laplacian, step, order=3, neumann_order=10):
    num_nodes = laplacian.shape[0]
    a = torch.arange(-step/2, 2-step/2, step, device=device).view(-1, 1, 1)
    s = 4/step
    m = order*2
    e = 2*math.pow(s, 2*m)/(math.pow(s, 2*m)-1)-2 + 1e-6

    if len(laplacian.shape) > 1:
        I = torch.eye(num_nodes, device=device)
        B = ((laplacian - a * I) / (2 + e)).matrix_power(m) + I / math.pow(s, m)
        ret = neumann_inv(B, neumann_order)
        # ret = B.matrix_power(-1)
        ret = (I / math.pow(s, m)).matmul(ret)
    else:
        ret = 1/s.pow(m)*(((laplacian - a)/(2+e)).pow(m) + 1/s.pow(m)).pow(-1)
    return ret.float()

def create_filter_sparse(laplacian, step, order=3, neumann_order=4):
    num_nodes = laplacian.shape[0]
    a = torch.arange(0, 2, step).tolist()
    s = 4/step
    m = order*2
    e = 2*math.pow(s, 2*m)/(math.pow(s, 2*m)-1)-2 + 1e-6

    I_index = torch.tensor([range(num_nodes), range(num_nodes)])
    I_values = torch.ones(num_nodes)
    I = torch.sparse_coo_tensor(I_index, I_values, device=device, size=(num_nodes, num_nodes))
    ret = []
    for ai in a:
        B = ((laplacian - ai * I) / (2 + e))
        B = B.coalesce()
        # B_index = B.indices()
        # B_values = B.values()
        B_index = B.indices()[:, B.values().abs() > 1e-3]
        B_values = B.values()[B.values().abs() > 1e-3]
        for _ in range(1, m):
            B_index, B_values = spspmm(B_index, B_values, B_index, B_values, num_nodes, num_nodes, num_nodes, coalesced=True)
            B_index = B_index[:, B_values.abs() > 1e-3]
            B_values = B_values[B_values.abs() > 1e-3]
        B = torch.sparse_coo_tensor(B_index, B_values, device=device, size=(num_nodes, num_nodes))
        B = B + I / math.pow(s, m)
        ret_index, ret_values = neumann_inv_sparse(B, I, neumann_order)
        Ism = (I/math.pow(s, m)).coalesce()
        Ism_index = Ism.indices()
        Ism_values = Ism.values()
        ret_index, ret_values = spspmm(Ism_index, Ism_values, ret_index, ret_values, num_nodes, num_nodes, num_nodes)
        # ret_index = ret_index[:, ret_values.abs() > 0.001]
        # ret_values = ret_values[ret_values.abs() > 0.0001]
        ret.append(torch.sparse_coo_tensor(ret_index, ret_values, device=device, size=(num_nodes, num_nodes)))
    return ret

def create_filter_old(laplacian, step, order=2):
    part1 = torch.diag(torch.ones(laplacian.shape[0], device=device) * math.pow(2, 1 / step - 1))
    part2 = (laplacian - torch.diag(torch.ones(laplacian.shape[0], device=device)) * torch.arange(
        0, 2.1, step, device=device
    ).view(
        -1, 1, 1
    )).matrix_power(4)
    part3 = torch.eye(laplacian.shape[0], device=device)
    return (part1.matmul(part2) + part3).matrix_power(-2)


def check_symmetric(a):
    return check_equality(a, a.T)


def check_equality(a, b, rtol=1e-05, atol=1e-03):
    return np.allclose(a, b, rtol=rtol, atol=atol)


def get_adjacency(edge_index):
    return torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1])).to_dense()


def symmetric(X):
    return X.triu() + X.triu(1).transpose(-1, -2)


def get_class_idx(num_classes, idx, y):
    y = y if len(y.shape) > 1 else y.view(-1, 1)
    class_idx = ([(y[:, j].view(-1) == i).nonzero().view(-1).tolist() for i in range(num_classes)] for j in
                 range(y.shape[1]))
    class_idx = [[list(set(i).intersection(set(idx.tolist()))) for i in class_i] for class_i in class_idx]
    class_idx = filter(lambda class_i: len(class_i[0]) != 0 and len(class_i[1]) != 0, class_idx)
    return list(class_idx)


def sample_negative_graphs(num_classes, idx, y):
    class_idx = get_class_idx(num_classes, idx, y)
    return [list(product(*class_i)) for class_i in class_idx]


def sample_positive_graphs(num_classes, idx, y):
    return get_class_idx(num_classes, idx, y)


def sample_positive_nodes_nce(mask, y):
    positive_samples = []
    for idx in mask.nonzero().view(-1):
        positive_mask = mask.logical_and(y == y[idx])
        if positive_mask.count_nonzero() > 1:
            positive_mask[idx] = False
        positive_indices = positive_mask.nonzero().view(-1)
        n = np.random.choice(positive_indices.cpu())
        positive_samples.append(n)
    return torch.tensor(positive_samples, device=device)


def sample_negative_nodes_nce(mask, y, K):
    negative_samples = []
    for idx in mask.nonzero().view(-1):
        negative_mask = mask.logical_and(y != y[idx])
        negative_indices = negative_mask.nonzero().view(-1)
        n = np.random.choice(negative_indices.cpu(), K)
        negative_samples.append(n)
    return torch.tensor(negative_samples, device=device)


def sample_positive_nodes_cont(mask, y, K):
    positive_samples = []
    for idx in mask.nonzero().view(-1):
        positive_mask = mask.logical_and(y == y[idx])
        if positive_mask.count_nonzero() > 1:
            positive_mask[idx] = False
        positive_indices = positive_mask.nonzero().view(-1)
        n = np.random.choice(positive_indices.cpu(), K)
        positive_samples.append(n)
    return torch.tensor(positive_samples, device=device)


def sample_negative_nodes_cont(mask, y, K):
    negative_samples = []
    for idx in mask.nonzero().view(-1):
        negative_mask = mask.logical_and(y != y[idx])
        negative_indices = negative_mask.nonzero().view(-1)
        n = np.random.choice(negative_indices.cpu(), K)
        negative_samples.append(n)
    return torch.tensor(negative_samples, device=device)


# Warning: this method returns indices from the masked list, not the original list
def sample_positive_nodes_dict(mask, y, K):
    positive_samples = []
    masked_y = torch.masked_select(y, mask)
    for masked_idx in range(mask.count_nonzero()):
        positive_mask = masked_y == masked_y[masked_idx]
        if positive_mask.count_nonzero() > 1:
            positive_mask[masked_idx] = False
        positive_indices = positive_mask.nonzero().view(-1)
        n = np.random.choice(positive_indices.cpu(), K)
        positive_samples.append(n)
    return torch.tensor(positive_samples, device=device)


# Warning: this method returns indices from the masked list, not the original list
def sample_negative_nodes_dict(mask, y, K):
    negative_samples = []
    masked_y = torch.masked_select(y, mask)
    for masked_idx in range(mask.count_nonzero()):
        negative_mask = masked_y != masked_y[masked_idx]
        negative_indices = negative_mask.nonzero().view(-1)
        n = np.random.choice(negative_indices.cpu(), K)
        negative_samples.append(n)
    return torch.tensor(negative_samples, device=device)


def sample_negative_nodes_naive(mask, y):
    negative_samples = []
    selectedNodes = set()
    for _ in range(len(mask.nonzero(as_tuple=True)[0])):
        nodes = []
        for i in range(y.max().item()):
            if y[mask].bincount()[i] <= len(negative_samples):
                continue
            iter = 0
            while True:
                iter += 1
                n = np.random.choice((y.cpu() == i).logical_and(mask.cpu()).nonzero(as_tuple=True)[0])
                if n not in selectedNodes:
                    selectedNodes.add(n)
                    nodes.append(n)
                    break
                if iter > 100:
                    break
        negative_samples.append(nodes)
    return list(filter(lambda s: len(s) > 1, negative_samples))


def sample_negative_nodes_naive(mask, y):
    positive_nodes = sample_positive_nodes_naive(mask, y)
    return positive_nodes.T


def sample_positive_nodes_naive(mask, y):
    y_masked = y[mask]
    positive_sample_mask = y_masked.view(1, -1) == torch.arange(y.max() + 1).view(-1, 1)
    positive_samples = list(map(lambda s:s.nonzero().view(-1), positive_sample_mask))
    min_len = min(map(len, positive_samples))
    positive_samples = [np.random.choice(s.cpu(), min_len) for s in positive_samples]
    return torch.tensor(positive_samples, device=device)


def sample_negative_nodes_naive_2(mask, y):
    negative_samples = []
    selectedNodes = set()
    for _ in range(len(mask.nonzero(as_tuple=True)[0])):
        nodes = []
        for i in range(y.max().item() + 1):
            if y[mask].bincount()[i] <= len(negative_samples):
                continue
            iter = 0
            while True:
                iter += 1
                n = np.random.choice((y.cpu() == i).logical_and(mask.cpu()).nonzero(as_tuple=True)[0])
                if n not in selectedNodes:
                    selectedNodes.add(n)
                    nodes.append(n)
                    break
                if iter > 100:
                    break
        negative_samples.append(torch.tensor(nodes, device=device))
    return list(filter(lambda s: len(s) > 1, negative_samples))


def sample_positive_nodes_naive_2(mask, y):
    positive_samples = []
    for i in range(y.max().item() + 1):
        if ((y == i).logical_and(mask)).any():
            positive_samples.append((y == i).logical_and(mask).nonzero().view(-1))
    return positive_samples


def rewire_graph(model, dataset, keep_num_edges=False, threshold=None):
    C = model['C']
    step = 2.1 / C.shape[0]

    dictionary = {}
    for i in range(len(dataset)):
        L_index, L_weight = get_laplacian(dataset[i].edge_index, normalization='sym')
        L = torch.zeros(dataset[i].num_nodes, dataset[i].num_nodes, device=L_index.device)
        L = L.index_put((L_index[0], L_index[1]), L_weight).to(device)
        D = create_filter_sparse(L, step).permute(1, 2, 0)
        dictionary[i] = D

    C = torch.nn.functional.normalize(C, dim=0, p=2)

    for i in range(len(dataset)):
        D = dictionary[i]
        L = D.matmul(C).squeeze()

        A_hat = torch.eye(L.shape[0]).to(device) - L
        A_hat = torch.nn.functional.normalize(A_hat, dim=(0, 1), p=1)
        if keep_num_edges:
            num_edges = dataset[i].num_edges
            indices = A_hat.view(-1).topk(num_edges)[1]
            rows = indices // A_hat.shape[0]
            cols = indices % A_hat.shape[0]
            edge_index = torch.stack([rows, cols], dim=0)
        else:
            A_hat = A_hat.abs() >= threshold
            edge_index = A_hat.nonzero().T
        dataset._data_list[i].edge_index = edge_index
    return dataset


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
    return to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes).squeeze()


def get_adj(edge_index, edge_weight, num_nodes):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    return to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes).squeeze()


def get_random_signals(num_nodes, size=None, epsilon=0.25):
    with torch.no_grad():
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


def create_label_sim_matrix(data):
    community = torch.zeros(data.num_nodes, data.num_nodes, device=device)
    for c in range(data.y.max() + 1):
        community = community + (data.y ==c).float().view(-1, 1).mm((data.y ==c).float().view(1, -1))
    community.fill_diagonal_(0)
    return community


def get_distance_diff_indices(community, num_samples=5):
    indices = []
    for i in range(community.shape[0]):
        intra_class = community[i].bool().nonzero().view(-1).tolist()
        inverse_comm = community[i].bool().logical_not()
        inverse_comm[i] = False
        inter_class = inverse_comm.nonzero().view(-1).tolist()
        if len(inter_class) > 0 and len(intra_class)> 0:
            intra_class = np.random.choice(intra_class, num_samples)
            inter_class = np.random.choice(inter_class, num_samples)
            indices += product(product([i], intra_class), product([i], inter_class))
    return torch.tensor(indices)

def get_distance_diff_indices_sparse(index, y, num_samples=5):
    indices = []
    for id, i in enumerate(index.tolist()):
        intra_class = (y[index] == y[i]).nonzero().view(-1).cpu()
        inter_class = (y[index] != y[i]).nonzero().view(-1).cpu()
        if len(inter_class) > 0 and len(intra_class) > 0:
            intra_class = np.random.choice(intra_class, num_samples)
            inter_class = np.random.choice(inter_class, num_samples)
            indices += product(product([id], intra_class), product([id], inter_class))
    return torch.tensor(indices)


def get_masked_edges(adj, mask):
    subgraph_adj = adj[mask][:,mask]
    return subgraph_adj.nonzero().T


if __name__ == '__main__':
    test()


def dot_product(a):
    eps = 1e-8
    if len(a.shape) == 1:
        a = a.view(-1, 1)
    if len(a.shape) == 2:
        a = a.view(1, a.shape[0], a.shape[1])
    # a_n, b_n = a.norm(dim=2)[:, None], a.norm(dim=2)[:, None]
    # a_norm = a / torch.clamp(a_n, min=eps).view(a.shape[0], a.shape[1], 1)
    # b_norm = a / torch.clamp(b_n, min=eps).view(a.shape[0], a.shape[1], 1)
    # sim_mt = a_norm.matmul(b_norm.permute(0, 2, 1))
    sim_mt = a.matmul(a.permute(0, 2, 1))
    return sim_mt


def find_optimal_edges(num_nodes, dist, mask, step=None, sparse=False):
    if step:
        edge_step = step
    else:
        edge_step = int(num_nodes)
    best_homo = 0
    best_edges = torch.tensor([], device=device)
    max_edges = dist.values().shape[0] if sparse else dist.numel()
    for num_edges in range(edge_step, max_edges, edge_step):
        if sparse:
            _, idx = torch.topk(dist.values(), int(num_edges), largest=False)
            edges = dist.indices()[:, idx]
        else:
            _, idx = torch.topk(dist, int(num_edges), largest=False)
            triu_indices = torch.triu_indices(num_nodes, num_nodes, 1, device=device)
            edges = to_undirected(triu_indices[:, idx])
        edges = to_undirected(edges)
        edges, _ = add_self_loops(edges, num_nodes=num_nodes)
        homo = our_homophily_measure(edges, mask).item()
        if homo >= best_homo:
            best_homo = homo
            best_edges = edges
        else:
            break
    return best_edges
