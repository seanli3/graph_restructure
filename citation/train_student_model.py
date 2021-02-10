from model.layers import GraphSpectralFilterLayer
import argparse
from model.spectral_filter import Graph
import torch
import torch.nn.functional as F
from torch import nn
from random import seed as rseed
from numpy.random import seed as nseed
from citation import get_dataset, random_planetoid_splits, run
from citation.train_eval import evaluate
import numpy as np

# --dataset=PubMed --model_path=citation/model/best_PubMed_gpu.pkl
# --model_path=citation/model/best_PubMed_gpu.pkl --dataset=PubMed --student_layers=2 --student_hidden=1056
# --model_path=citation/model/best_Cora_lt_zero_gpu.pkl --dataset=Cora --student_layers=2 --student_hidden=672 --hidden=56
# --model_path=citation/model/best_Cora_lt_zero_gpu.pkl --dataset=Cora --hidden=56
# --model_path=citation/model/best_Citeseer_lt_zero.pkl --dataset=CiteSeer --hidden=32 --heads=14
# --model_path=citation/model/best_Citeseer_lt_zero.pkl --dataset=CiteSeer --hidden=32 --heads=14 --student_layers=2 --student_hidden=448

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--hidden', type=int, default=88)
parser.add_argument('--heads', type=int, default=12)
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--model_path', type=str, default='citation/model/best_PubMed_gpu.pkl')

parser.add_argument('--student_layers', type=int, default=1)
parser.add_argument('--student_dropout', type=int, default=0)
parser.add_argument('--student_heads', type=int, default=1)
parser.add_argument('--student_hidden', type=int, default=16)
args = parser.parse_args()
print(args)

dataset_name = args.dataset

random_splits = False
runs = args.runs
epochs = 2000
alpha = 0.2
seed = 729
weight_decay = 0.00012376256876336363
hidden = args.hidden
lr = args.lr
dropout = 0.7
heads = args.heads
output_heads = 10
normalize_features = True
pre_training = False
cuda = True
order = 16
edge_dropout = 0
node_feature_dropout = 0
filter_name = 'analysis'

rseed(seed)
nseed(seed)
torch.manual_seed(seed)

cuda = cuda and torch.cuda.is_available()

if cuda:
    torch.cuda.manual_seed(seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


#%%

class Net(torch.nn.Module):
    def __init__(self, dataset, hidden, heads):
        super(Net, self).__init__()
        data = dataset.data
        adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.num_edges))
        self.G = Graph(adj)
        self.G.estimate_lmax()

        self.analysis = GraphSpectralFilterLayer(self.G, dataset.num_node_features, hidden,
                                                 dropout=dropout, out_channels=heads, filter=filter_name,
                                                 pre_training=False, device='cuda' if cuda else 'cpu',
                                                 alpha=alpha, order=order, concat=True)
        self.synthesis = GraphSpectralFilterLayer(self.G, hidden * heads, dataset.num_classes, filter=filter_name,
                                                  device='cuda' if cuda else 'cpu', dropout=dropout,
                                                  out_channels=1, alpha=alpha, pre_training=False,
                                                  order=order, concat=False)

    def reset_parameters(self):
        self.analysis.reset_parameters()
        self.synthesis.reset_parameters()

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=dropout, training=self.training)
        x, att1 = self.analysis(x)

        layer1 = x

        x = F.dropout(x, p=dropout, training=self.training)
        x, att2 = self.synthesis(x)

        layer2 = x
        return F.log_softmax(x, dim=1), layer2, layer1

dataset = get_dataset(dataset_name, normalize_features, edge_dropout=edge_dropout,
                     cuda=cuda, node_feature_dropout=node_feature_dropout)


#%%

class StudentNet(torch.nn.Module):
    def __init__(self, dataset, heads, hidden, dropout, layers):
        super(StudentNet, self).__init__()
        data = dataset.data
        adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.num_edges))
        self.G = Graph(adj)
        self.G.estimate_lmax()

        self.analysis = GraphSpectralFilterLayer(self.G, dataset.num_node_features,
                                                 hidden if layers == 2 else dataset.num_classes,
                                                 dropout=dropout, out_channels=heads, filter=filter_name,
                                                 pre_training=False, device='cuda' if cuda else 'cpu',
                                                 alpha=alpha, order=order,
                                                 concat=True if layers == 2 else False)
        if layers == 2:
            self.synthesis = GraphSpectralFilterLayer(self.G, hidden * heads, dataset.num_classes, filter=filter_name,
                                                  device='cuda' if cuda else 'cpu', dropout=dropout,
                                                  out_channels=1, alpha=alpha, pre_training=False,
                                                  order=order, concat=False)
    def reset_parameters(self):
        self.analysis.reset_parameters()
        if hasattr(self, 'synthesis'):
            self.synthesis.reset_parameters()

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=dropout, training=self.training)
        x, _ = self.analysis(x)
        layer1 = layer2 = x
        if hasattr(self, 'synthesis'):
            x = F.dropout(x, p=dropout, training=self.training)
            x, _ = self.synthesis(x)
            layer2 = x
        x = F.elu(x)
        return F.log_softmax(x, dim=1), layer2, layer1

#%%

model = Net(dataset, hidden, heads)
if cuda:
    model.to('cuda')
model.load_state_dict(torch.load('{}'.format(args.model_path),  map_location={'cuda:0': 'cpu'} if not cuda else None))

# filter_kernel = model.analysis.filter_kernel

# model_correct_indices = get_correctly_predicted_node_idx(model, 'test', dataset)
eval_info = evaluate(model, dataset[0])
print('saved model', eval_info)

#%%

with torch.no_grad():
    _, target_2, target_1 = model(dataset[0])

#%%

from torch.optim import Adam
from sklearn.metrics import f1_score


def train_student(target_1, target_2=None, lr=lr):
    for _ in range(runs):
        print('Runs', _)
        student = StudentNet(dataset, args.student_heads, args.student_hidden, args.student_dropout, args.student_layers)
        if cuda:
            student.to('cuda')
        student.reset_parameters()
        optimizer = Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
        data = dataset.data
        best_val_loss = float('inf')
        eval_info_early_model = None
        bad_counter = 0

        for epoch in range(1, epochs + 1):
            student.train()
            optimizer.zero_grad()
            logits, out_2, out_1 = student(data)
            loss = F.mse_loss(out_1[dataset[0].train_mask], target_1[dataset[0].train_mask])
            if target_2 is not None:
                loss += F.mse_loss(out_2[dataset[0].train_mask], target_2[dataset[0].train_mask])
            loss.backward()
            optimizer.step()

            # eval_info['epoch'] = epoch
            student.eval()
            outs = {}
            outs['epoch'] = epoch
            for key in ['train', 'val', 'test']:
                mask = data['{}_mask'.format(key)]
                loss = F.mse_loss(out_1[mask], target_1[mask])
                if target_2 is not None:
                    loss += F.mse_loss(out_2[mask], target_2[mask])
                outs['{}_loss'.format(key)] = loss.item()

                pred = logits[mask].max(1)[1]
                outs['{}_micro_f1'.format(key)] = f1_score(data.y[mask].cpu(), logits[mask].max(1)[1].cpu(),
                                                           average='micro')

            if outs['val_loss'] <= best_val_loss:
                eval_info_early_model = outs
                best_val_loss = outs['val_loss']
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter == args.patience:
                    break

            if epoch%100 == 0:
                print(outs)

        print(eval_info_early_model)


#%%
if args.student_layers == 1:
    train_student(target_2)
else:
    train_student(target_1, target_2)
