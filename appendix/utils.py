from torch_geometric.data import InMemoryDataset
import torch
from torch_geometric.data.data import Data
import scipy.io as sio
import numpy as np
import math

class TwoDGrid(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TwoDGrid, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["2Dgrid.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A']
        # list of output
        F=a['F']
        F=F.astype(np.float32)
        Y=a['Y']
        Y=Y.astype(np.float32)
        M=a['mask']
        M=M.astype(np.float32)


        data_list = []
        E=np.where(A>0)
        edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
        x=torch.tensor(F)
        y=torch.tensor(Y)   
        m=torch.tensor(M)     
        data_list.append(Data(edge_index=edge_index, x=x, y=y,m=m))
        
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class BandClassDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,contfeat=False):
        self.contfeat=contfeat
        super(BandClassDataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["bandclass.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) 
        # list of adjacency matrix
        A=a['A']
        F=a['F']
        Y=a['Y']
        F=np.expand_dims(F,2)

        data_list = []
        for i in range(len(A)):
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.tensor(F[i,:,:]) 
            y=torch.tensor(Y[i,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class DegreeMaxEigTransform(object):   

    def __init__(self,adddegree=True,maxdeg=40,addposition=False):
        self.adddegree=adddegree
        self.maxdeg=maxdeg
        self.addposition=addposition

    def __call__(self, data):

        n=data.x.shape[0] 
        A=np.zeros((n,n),dtype=np.float32)        
        A[data.edge_index[0],data.edge_index[1]]=1         
        if self.adddegree:
            data.x=torch.cat([data.x,torch.tensor(1/self.maxdeg*A.sum(0)).unsqueeze(-1)],1)
        if self.addposition:
            data.x=torch.cat([data.x,data.pos],1)


        d = A.sum(axis=0) 
        # normalized Laplacian matrix.
        dis=1/np.sqrt(d)
        dis[np.isinf(dis)]=0
        dis[np.isnan(dis)]=0
        D=np.diag(dis)
        nL=np.eye(D.shape[0])-(A.dot(D)).T.dot(D)
        V,U = np.linalg.eigh(nL)               
        vmax=np.abs(V).max()
        # keep maximum eigenvalue for Chebnet if it is needed
        data.lmax=vmax.astype(np.float32)        
        return data


def neumann_inv(m, k):
    I = torch.diag_embed(torch.ones(m.shape[0], m.shape[1], device='cuda'))
    ret = I
    for l in range(k):
        ret = ret.matmul(I + (I-m).matrix_power(int(math.pow(2,l))))
    if ret.isnan().any() or ret.isinf().any():
        raise RuntimeError('NaN or Inf in Neumann approximation, try reduce order')
    return ret


# This new rational approximator does not show any advatanges so far
def create_filter(laplacian, step, order=3, neumann_order=10):
    num_nodes = laplacian.shape[0]
    a = torch.arange(-step/2, 2-step/2, step, device='cuda').view(-1, 1, 1)
    s = 4/step
    m = order*2
    e = 2*math.pow(s, 2*m)/(math.pow(s, 2*m)-1)-2 + 1e-6

    if len(laplacian.shape) > 1:
        I = torch.eye(num_nodes, device='cuda')
        B = ((laplacian - a * I) / (2 + e)).matrix_power(m) + I / math.pow(s, m)
        ret = neumann_inv(B, neumann_order)
        # ret = B.matrix_power(-1)
        ret = (I / math.pow(s, m)).matmul(ret)
    else:
        ret = 1/s.pow(m)*(((laplacian - a)/(2+e)).pow(m) + 1/s.pow(m)).pow(-1)
    return ret.float()


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
            device='cuda', generator=torch.Generator('cuda').manual_seed(729)
        )
        return random_signal
