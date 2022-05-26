# Modified from https://github.com/balcilar/gnn-spectral-expressive-power/blob/main/appendix_I.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,GINConv,ARMAConv,
                                global_mean_pool,GATConv,ChebConv,GCNConv)
from utils import TwoDGrid
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib
from utils import BandClassDataset, create_filter, get_random_signals
from torch_geometric.utils import get_laplacian
import math
from copy import deepcopy



# read dataset
dataset = TwoDGrid(root='dataset/2Dgrid', pre_transform=None)

# it consists of just one graph
train_loader = DataLoader(dataset, batch_size=10, shuffle=False)

# ntask bandpass:0,  lowpass:1, highpass:2  
ntask=2


class SpectralSlicer(torch.nn.Module):
    def __init__(self, step=0.2):
        super(SpectralSlicer, self).__init__()
        self.step = step

        self.random_signals = get_random_signals(dataset[0].num_nodes, 2000)

        self.windows = math.ceil(2/self.step)

        self.layers = nn.Sequential(
            nn.Linear(self.windows, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, 1, bias=False),
        )
        self.layers.to(device)
        self.fc2 = torch.nn.Linear(self.random_signals.shape[1]+1, 1)
        # self.fc2 = torch.nn.Linear(64, 1)

        data = dataset[0]
        L_index, L_weight = get_laplacian(data.edge_index, normalization='sym')
        L = torch.sparse_coo_tensor(L_index, L_weight, device=device, size=(data.num_nodes, data.num_nodes)).to_dense()
        self.D = create_filter(L, self.step).permute(1, 2, 0)

    def forward(self, data):
        x_D = self.D
        L_hat = self.layers(x_D).squeeze()
        x = torch.cat((data.x, self.random_signals), dim=1)
        x = L_hat.matmul(x)
        # x = self.fc1(x)
        x = self.fc2(x)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                torch.nn.init.normal_(layer.weight)
        self.fc2.reset_parameters()
        self.fc1.reset_parameters()
        self.layers.to(device)



class GinNet(nn.Module):
    def __init__(self):
        super(GinNet, self).__init__()

        nn1 = Sequential(Linear(1, 64), ReLU(), Linear(64, 64))
        self.conv1 = GINConv(nn1,train_eps=True)
        self.bn1 = torch.nn.BatchNorm1d(64)

        nn2 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv2 = GINConv(nn2,train_eps=True)
        self.bn2 = torch.nn.BatchNorm1d(64)


        nn3 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv3 = GINConv(nn3,train_eps=True)
        self.bn3 = torch.nn.BatchNorm1d(64)

        nn4 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv4 = GINConv(nn4,train_eps=True)
        self.bn4 = torch.nn.BatchNorm1d(64)        
        
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, data):

        x=data.x            
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x=self.bn1(x)

        x = F.relu(self.conv2(x, edge_index))
        x=self.bn2(x)

        x = F.relu(self.conv3(x, edge_index))
        x=self.bn3(x)

        x = F.relu(self.conv4(x, edge_index))
        x=self.bn4(x)        
        
        return self.fc2(x) 


class GcnNet(nn.Module):
    def __init__(self):
        super(GcnNet, self).__init__()

        self.conv1 = GCNConv(1, 64*5, cached=False)
        self.conv2 = GCNConv(64*5, 64*5, cached=False)
        self.conv3 = GCNConv(64*5, 64, cached=False)  
        
        self.fc2 = torch.nn.Linear(64, 1) 

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))  
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))        
        return self.fc2(x) 

class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()

        self.conv1 = torch.nn.Linear(1, 64)   
        self.conv2 = torch.nn.Linear(64, 64) 
        self.conv3 = torch.nn.Linear(64, 64)      
        self.fc2 = torch.nn.Linear(64, 1) 

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))       
        return self.fc2(x) 

class ChebNet(nn.Module):
    def __init__(self,S=5):
        super(ChebNet, self).__init__()
        
        self.conv1 = ChebConv(1, 64,S)    
        self.conv2 = ChebConv(64, 64,S) 
        self.conv3 = ChebConv(64, 64,S)
        self.fc2 = torch.nn.Linear(64, 1) 

    def forward(self, data):
        x=data.x
              
        edge_index=data.edge_index        
        x = F.relu(self.conv1(x, edge_index))   
        x = F.relu(self.conv2(x, edge_index)) 
        x = F.relu(self.conv3(x, edge_index))   
           
        return self.fc2(x) 



class GatNet(nn.Module):
    def __init__(self):
        super(GatNet, self).__init__()
        self.conv1 = GATConv(1, 8, heads=8,concat=True, dropout=0.0)  
        self.conv2 = GATConv(64, 8, heads=8,concat=True, dropout=0.0) 
        self.conv3 = GATConv(64, 8, heads=8,concat=True, dropout=0.0)  
        
        self.fc2 = torch.nn.Linear(64, 1) 

    def forward(self, data):
        x=data.x
          
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.elu(self.conv2(x, data.edge_index))
        x = F.elu(self.conv3(x, data.edge_index))  
        
        return self.fc2(x) 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MlpNet().to(device)   # GatNet  ChebNet  GcnNet  GinNet  MlpNet
# model = SpectralSlicer().to(device)   # GatNet  ChebNet  GcnNet  GinNet  MlpNet
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def visualize(tensor):
    if torch.is_tensor(tensor):
        y=tensor.detach().cpu().numpy()
    else:
        y = tensor
    y=np.reshape(y,(95,95))

    plt.figure(dpi=200)
    fig, ax = plt.subplots()
    fig.set_figwidth(6)
    fig.set_figheight(5)
    ax.imshow(y.T)
    # plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    fig.tight_layout()


def train(epoch):
    model.train()
    ns=0
    L=0
    correct=0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        pre=model(data)        
        lss= torch.square(data.m*(pre- data.y[:, ntask:ntask+1])).sum()
        
        lss.backward()
        optimizer.step()        

        L+=lss.item()

        a=pre[data.m==1]    
        b=data.y[:,ntask:ntask+1] 
        b=b[data.m==1] 
        r2=r2_score(b.cpu().detach().numpy(),a.cpu().detach().numpy())


        # if you want to see the image that GNN  produce
        # visualize(pre)
    return L,r2


# matplotlib.use('Agg')
# visualize(dataset[0].x*dataset[0].m)
# # plt.savefig('./given_img.eps', bbox_inches='tight', format='eps')
# visualize(dataset[0].y[:, 1]*dataset[0].m.view(-1))
# # plt.savefig('./low_pass_img.eps', bbox_inches='tight', format='eps')
# visualize(dataset[0].y[:, 0]*dataset[0].m.view(-1))
# # plt.savefig('./band_pass_img.eps', bbox_inches='tight', format='eps')
# visualize(dataset[0].y[:, 2]*dataset[0].m.view(-1))
# # plt.savefig('./high_pass_img.eps', bbox_inches='tight', format='eps')

best_loss = torch.inf
best_r2 = 0
patience = 0
best_parameters = None
for epoch in range(1, 3001):
    trloss,r2=train(epoch)
    if trloss < best_loss:
        best_loss = trloss
        best_r2 = r2
        best_parameters = deepcopy(model.state_dict())
        patience = 0
    else:
        patience += 1
    if patience >= 100:
        break
    print('Epoch: {:02d}, loss: {:.4f}, R2: {:.4f}'.format(epoch,trloss,r2))

print('Best: loss: {:.4f}, R2: {:.4f}'.format(best_loss, best_r2))

for data in train_loader:
    data = data.to(device)
    model.load_state_dict(best_parameters)
    pre=model(data)
    visualize(pre*data.m)
    plt.show()
    # plt.savefig('./high_pass_mlp.eps', bbox_inches='tight', format='eps')
