import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot


class Net(torch.nn.Module):
    def __init__(self, dataset, L, dropout, hidden):
        super(Net, self).__init__()
        self.W1 = torch.nn.Parameter(torch.Tensor(dataset.num_features, hidden))
        self.W2 = torch.nn.Parameter(torch.Tensor(hidden, dataset.num_classes))
        self.I = torch.eye(dataset[0].num_nodes)
        self.L = L
        self.dropout = dropout

    def reset_parameters(self):
        glorot(self.W1)
        glorot(self.W2)

    def forward(self, data):
        L_hat = self.I - self.L
        x = F.relu(L_hat.mm(data.x).mm(self.W1))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = L_hat.mm(x).mm(self.W2)

        return F.log_softmax(x, dim=1), x
