import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, JumpingKnowledge
from kernel.utils import global_mean_pool_deterministic


class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, adj_t, batch = data.x, data.adj_t, data.batch
        x = F.relu(self.conv1(x, adj_t))
        for conv in self.convs:
            x = F.relu(conv(x, adj_t))
        x = global_mean_pool_deterministic(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GraphSAGEWithJK(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, mode='cat'):
        super(GraphSAGEWithJK, self).__init__()
        self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, adj_t, batch = data.x, data.adj_t, data.batch
        x = F.relu(self.conv1(x, adj_t))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, adj_t))
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool_deterministic(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
