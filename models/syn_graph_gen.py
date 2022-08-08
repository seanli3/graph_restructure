import torch
from utils import compat_matrix_edge_idx, our_homophily_measure
import numpy as np
from config import DEVICE

device=DEVICE

def generate_graph_edges(y, h_den):
    num_class = y.max().item() + 1
    h_hat = 2*h_den - 1
    counts = y.unique(return_counts=True)[1].float()
    counts = counts.view(-1, 1).mm(counts.view(1, -1))

    if h_hat >= 0:
        h_intra = np.random.rand() * (1-h_hat) + h_hat
        h_inter = h_intra - h_hat
    else:
        h_inter = np.random.rand() * (1+h_hat) - h_hat
        h_intra = h_inter + h_hat

    intra_pro = h_intra * counts.diag().sum()
    inter_pro = h_inter * ((counts.sum() - counts.diag().sum())/(num_class-1))
    scaled_intra_pro = intra_pro / (inter_pro + intra_pro)
    scaled_inter_pro = inter_pro / (inter_pro + intra_pro)

    edges = []
    cur_homo = None
    while cur_homo is None or np.abs(cur_homo - h_den) > 0.0001:
        draw = np.random.uniform()
        if draw < scaled_intra_pro:
            intra = True
        else:
            intra = False
        for idx, i in enumerate(y):
            if intra is True:
                n = np.random.choice((y == i).nonzero().view(-1).cpu())
            else:
                n = np.random.choice((y != i).nonzero().view(-1).cpu())
            e = (idx, n)
            if e not in edges:
                edges.append(e)
                if e[1] != e[0]:
                    edges.append((e[1], e[0]))
        edge_index = torch.tensor(edges, device=device).T
        cur_homo = our_homophily_measure(edge_index, y).item()
        H = compat_matrix_edge_idx(edge_index, y)
        if torch.allclose(H, counts):
            break
        print(cur_homo)
    print(edges)


generate_graph_edges(torch.tensor([0,1,0,1,2,2,0,0,1,1,2], device=device), 0.7083)