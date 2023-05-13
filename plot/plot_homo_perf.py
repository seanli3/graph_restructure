# NCVoter
from matplotlib import cm

import math
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import pandas as pd

gcn_chame_edges = pd.read_csv('gcn_chame_edges.csv', delimiter=' ')
gcn_squirrel_edges = pd.read_csv('gcn_squirrel_edges.csv', delimiter=' ')
# sgc_chame_edges = pd.read_csv('sgc_chame_edges.csv', delimiter=' ')
# sgc_squirrel_edges = pd.read_csv('sgc_squirrel_edges.csv', delimiter=' ')

chame_edges = 36101
squirrel_edges = 217073
gcn_chame_ori_acc = 0.598
gcn_squirrel_ori_acc = 0.369
# sgc_chame_ori_acc = 0.335
# sgc_squirrel_ori_acc = 0.469

color_map = {
    'train': 'tab:blue',
    'val': 'tab:orange',
    'test': 'tab:green',
}

fig = plt.figure(figsize=(10,5))
lgd = None

def plot_chame(csv, origainl_edges, ori_acc):
    plt.figure(dpi=200)
    lns = []
    ax1 = fig.add_subplot(121)
    # for stage in ['train', 'val', 'test']:
    for stage in ['val', 'test']:
        csv[stage+'_acc_lower'] = csv[stage+'_acc'] - csv[stage+'_acc_std']
        csv[stage+'_acc_upper'] = csv[stage+'_acc'] + csv[stage+'_acc_std']
        ln = ax1.plot(csv['edges'], csv[stage+'_acc'], label=stage + ' accuracy ({})', color=color_map[stage])
        lns += ln
        ax1.plot(csv['edges'], csv[stage+'_acc_lower'], alpha=0.1, color=color_map[stage])
        ax1.plot(csv['edges'], csv[stage+'_acc_upper'], alpha=0.1, color=color_map[stage])
        ax1.fill_between(csv['edges'], csv[stage+'_acc_lower'], csv[stage+'_acc_upper'], alpha=0.2, color=color_map[stage])


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # for stage in ['train', 'val', 'test']:
    for stage in ['val', 'test']:
        lns3=ax2.plot(csv['edges'], csv['h_den_'+stage], linestyle='--', label=r'h_{den}_'+stage, color=color_map[stage])
        lns += lns3


    ax1.set_ylim(0.4, 1)
    ax2.set_ylim(0.5015, 0.507)


    ax1.set_xlim(2000, 130000)
    ax1.set_xticks([25000, 125000])
    ax1.set_xticklabels([25000, 125000], fontsize=22)
    ax1.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax1.set_yticklabels([0.5, 0.6, 0.7, 0.8, 0.9], fontsize=22)

    ax2.set_yticks([0.502, 0.503, 0.504, 0.505, 0.506])
    ax2.set_yticklabels([0.502, 0.503, 0.504, 0.505, 0.506], fontsize=22)

    # lns2=ax1.plot(csv['edges'], csv['h_norm'], color='tab:brown', marker='x', label=r'h_{norm}')
    # lns += lns2

    ax2.scatter([48002], [0.5068625807762146], marker='*', s=200, color="tab:red", alpha=1)
    # ax2.annotate('Chosen number of edges', (5500, 0.5064), fontsize=17)
    # ax1.axvline(x=48002, linestyle='-', label="Optimal edges", color='tab:pink')

    # lns4=ax1.axvline(x=origainl_edges, linestyle=':', label="Original edges", color='tab:purple')
    # ax2.annotate('Original edges', (origainl_edges-30000, 0.5045), fontsize=16)
    # lns5=ax1.axhline(y=ori_acc, color='tab:gray', linestyle='--', label="Original accuracy")


    ax1.set_xlabel(r'Edges', fontsize=22)
    ax1.set_ylabel(r'Accuracy', fontsize=22)
    # ax2.set_ylabel(r'$h_{den}$', fontsize=20)

    # fig.tight_layout()
    # legends = [s + ' acc.' for s in ['Train.', 'Val.', 'Test']]
    # legends += [s + r' $h_{den}$' for s in ['Train.', 'Val.', 'Test']]

    legends = [s + ' acc.' for s in ['Val.', 'Test']]
    legends += [s + r' $h_{den}$' for s in ['Val.', 'Test']]
    # legends += [r'$h_{norm}$']
    # lgd = ax1.legend(
    #     lns,
    #     legends,
    #     # loc='upper right',
    #     prop={'size': 20},
    #     # bbox_to_anchor=(1.6, 1.02),
    #     ncol=1,
    #     fancybox=True
    # )
    # plt.savefig('./chameleon_gcn_h_den.pdf', bbox_inches='tight', format='pdf')
    # plt.clf()
    ax1.set_title('CHAMELEON', fontdict={'fontsize': 22}, y=-0.4)

def plot_squirrel(csv, origainl_edges, ori_acc):
    plt.figure(dpi=200)
    lns = []
    ax1 = fig.add_subplot(122)
    # for stage in ['train', 'val', 'test']:
    for stage in ['val', 'test']:
        csv[stage+'_acc_lower'] = csv[stage+'_acc'] - csv[stage+'_acc_std']
        csv[stage+'_acc_upper'] = csv[stage+'_acc'] + csv[stage+'_acc_std']
        ln = ax1.plot(csv['edges'], csv[stage+'_acc'], label=stage + ' accuracy ({})', color=color_map[stage])
        lns += ln
        ax1.plot(csv['edges'], csv[stage+'_acc_lower'], alpha=0.1, color=color_map[stage])
        ax1.plot(csv['edges'], csv[stage+'_acc_upper'], alpha=0.1, color=color_map[stage])
        ax1.fill_between(csv['edges'], csv[stage+'_acc_lower'], csv[stage+'_acc_upper'], alpha=0.2, color=color_map[stage])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # for stage in ['train', 'val', 'test']:
    for stage in ['val', 'test']:
        lns3=ax2.plot(csv['edges'], csv['h_den_'+stage], linestyle='--', label=r'h_{den}_'+stage, color=color_map[stage])
        lns += lns3

    # lns2=ax1.plot(csv['edges'], csv['h_norm'], color='tab:brown', marker='x', label=r'h_{norm}')
    # lns += lns2

    # lns4=ax1.axvline(x=origainl_edges, linestyle=':', label="Original edges", color='tab:purple')
    # ax2.annotate('Original edges', (origainl_edges-60000, 0.5008), fontsize=16)
    # lns5=ax1.axhline(y=ori_acc, color='tab:gray', linestyle='--', label="Original accuracy")

    ax1.set_xlim(2000, 270000)
    ax1.set_ylim(0.4, 1)
    ax2.set_ylim(0.5, 0.501)

    ax1.set_xticks([50000, 250000])
    ax1.set_xticklabels([50000, 250000], fontsize=22)
    ax1.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax1.set_yticklabels([0.5, 0.6, 0.7, 0.8, 0.9], fontsize=22)

    ax2.set_yticks([0.5002, 0.5004, 0.5006, 0.5008])
    ax2.set_yticklabels([0.5002, 0.5004, 0.5006, 0.5008], fontsize=22)

    ax2.scatter([26002], [0.5007653832435608], marker='*', s=200, color="tab:red", alpha=1)
    # ax2.annotate('Chosen number of edges', (6002, 0.5008), fontsize=17)
    # ax1.axvline(x=26002, linestyle='-', label="Optimal edges", color='tab:pink')


    ax1.set_xlabel(r'Edges', fontsize=22)
    # ax1.set_ylabel(r'accuracy/$h_{norm}$', fontsize=20)
    ax2.set_ylabel(r'$h_{den}$', fontsize=22)

    # fig.tight_layout()
    legends = [s + ' acc.' for s in ['Val.', 'Test']]
    legends += [s + r' $h_{den}$' for s in ['Val.', 'Test']]
    # legends += [r'$h_{norm}$']
    lgd = fig.legend(
        lns,
        legends,
        columnspacing=0.4,
        loc=(0.03, 0.88),
        prop={'size':22},
        ncol=4,
        fancybox=True
    )
    ax1.set_title('SQUIRREL', fontdict={'fontsize':22}, y=-0.4)

    # plt.savefig('./squirrel_gcn_h_den.pdf', bbox_inches='tight', bbox_extra_artists=(lgd,), format='pdf')
    # plt.clf()




plot_chame(gcn_chame_edges, chame_edges, gcn_chame_ori_acc)
plot_squirrel(gcn_squirrel_edges, squirrel_edges, gcn_squirrel_ori_acc)

fig.tight_layout(w_pad=3)
fig.subplots_adjust(top=0.85, bottom=0.25)
fig.savefig('./gcn_h_den.pdf', bbox_extra_artists=(lgd,), format='pdf')
# plt.show()
plt.clf()
# plot(sgc_chame_edges, chame_edges, sgc_chame_ori_acc)

# data = sgc_chame_edges.join(gcn_chame_edges.set_index(['edges']), on = ['edges'], lsuffix="_gcn", rsuffix="_sgc", how="inner")
# plot(data, chame_edges, sgc_chame_ori_acc)
# data = sgc_squirrel_edges.join(gcn_squirrel_edges.set_index(['edges']), on = ['edges'], lsuffix="_gcn", rsuffix="_sgc", how="inner")
# plot(data, squirrel_edges, sgc_squirrel_ori_acc)