import torch
from utils import compat_matrix_edge_idx, our_homophily_measure
import numpy as np
from config import DEVICE
from  dataset.datasets import get_dataset
from tqdm import tqdm

device=DEVICE

def generate_graph_edges(y, h_den, min_density=None, start_edges=None):
    A = torch.zeros(y.shape[0], y.shape[0], device=device)
    if start_edges:
        A[start_edges[0], start_edges[1]] = 1

    pbar = tqdm(range(0, 999999999999999))
    for _ in pbar:
        for idx, i in enumerate(y):
            edge_index = A.nonzero().T
            cur_homo = our_homophily_measure(edge_index, y).item()
            pbar.set_description('h_den: ' + str(cur_homo))
            if np.abs(cur_homo - h_den) < 0.0001:
                if min_density is not None:
                    if A.count_nonzero() / A.numel() >= min_density:
                        return edge_index, cur_homo
                else:
                    return edge_index, cur_homo
            if cur_homo <= h_den:
                intra = True
            else:
                intra = False

            if intra is True:
                n = np.random.choice((y == i).logical_and(A[idx] == 0).nonzero().view(-1).cpu())
            else:
                n = np.random.choice((y != i).logical_and(A[idx] == 0).nonzero().view(-1).cpu())
            A[idx, n] = 1
            A[n, idx] = 1


dataset = get_dataset('Cora')
data = dataset.data
target_hen = 0.1
print('Creating a neutral graph')
edge_index, h_den = generate_graph_edges(data.y, 0.5, min_density=0.0002)
print('Adding edges for h_den='+str(target_hen))
edge_index, h_den = generate_graph_edges(data.y, target_hen, edge_index)
torch.save(edge_index, 'sync_{}_h_den_{}_edges.pt'.format(dataset.name, h_den))
