from torch_scatter import segment_csr
import torch

def bincount(a):
    device = a.device
    num_bins = a.max() + 1
    return (torch.arange(num_bins, device=device).repeat(a.shape[0]).view(a.shape[0], -1) == a.view(-1, 1)).sum(dim=0)


def global_mean_pool_deterministic(x, batch) :
    ptr = index_to_ptr(batch)
    return segment_csr(x, ptr, reduce="mean")


def scatter_add_deterministic(x, col) :
    ptr = index_to_ptr(col)
    return segment_csr(x, ptr, reduce="sum")


def index_to_ptr(index):
    device = index.device
    bins = bincount(index)
    size = bins.shape[0]
    m = bins.repeat(size).view(size, -1)
    triu_indices = torch.triu_indices(row=size, col=size, offset=1, device=device)
    m[triu_indices[0], triu_indices[1]] = 0
    ptr = torch.cat([torch.tensor([0], device=device), m.sum(dim=1)])
    return ptr