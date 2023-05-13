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

syn_gcn = pd.read_csv('syn_cora_gcn.csv', delimiter='\t')
syn_sgc = pd.read_csv('syn_cora_sgc.csv', delimiter='\t')

color_map = {
    'h_den_rewired': 'tab:blue',
    'h_den': 'tab:pink',
    'test': 'tab:orange',
    'test_rewired': 'tab:green',
}


def plot_gcn(csv):
    plt.figure(dpi=200)
    lns = []
    fig, ax1 = plt.subplots()
    fig.set_figwidth(5)
    fig.set_figheight(4)
    for stage in ['test', 'test_rewired']:
        csv[stage+'_acc_lower'] = csv[stage+'_acc'] - csv[stage+'_acc_std']
        csv[stage+'_acc_upper'] = csv[stage+'_acc'] + csv[stage+'_acc_std']
        ln = ax1.plot(csv['h_den'], csv[stage+'_acc'], label=stage + ' accuracy ({})', color=color_map[stage])
        lns += ln
        ax1.plot(csv['h_den'], csv[stage+'_acc_lower'], alpha=0.1, color=color_map[stage])
        ax1.plot(csv['h_den'], csv[stage+'_acc_upper'], alpha=0.1, color=color_map[stage])
        ax1.fill_between(csv['h_den'], csv[stage+'_acc_lower'], csv[stage+'_acc_upper'], alpha=0.2, color=color_map[stage])

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # lns3=ax2.plot(csv['h_den'], csv['h_den_rewired'], linestyle='--', label=r'h_{den} rewired', color=color_map['h_den_rewired'])
    # lns += lns3


    # ax1.set_ylim(0.3, 1.05)
    # ax2.set_ylim(0.3, 1.05)

    ax1.set_xlim(0, 0.55)
    ax1.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax1.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=22)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax1.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1], fontsize=22)

    # ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    # ax2.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1], fontsize=18)

    ax1.set_xlabel(r'Homophily $h_{den}$', fontsize=22)
    ax1.set_ylabel(r'Test accuracy', fontsize=22)
    # ax2.set_ylabel(r'Rewired $h_{den}$', fontsize=16)

    fig.tight_layout()
    legends = [s + '' for s in ['Original', 'Restructured']]
    legends += [r'Rewired $h_{den}$']

    lgd = ax1.legend(
        lns,
        legends,
        loc='lower left',
        prop={'size': 22},
        # bbox_to_anchor=(1.6, 1.02),
        ncol=1,
        fancybox=True
    )

    # plt.show()
    plt.savefig('./syn_cora_sgc.pdf', bbox_inches='tight', format='pdf')
    plt.clf()


def plot_h_den(csv):
    plt.figure(dpi=200)
    lns = []
    fig, ax1 = plt.subplots()
    fig.set_figwidth(6)
    fig.set_figheight(5)
    ln1 = ax1.plot(csv['h_den'], csv['h_den_rewired'], label='accuracy ({})', color=color_map['h_den_rewired'])
    ln2 = ax1.plot(csv['h_den'], csv['h_den'], label='accuracy ({})', color=color_map['h_den'], linestyle="--")

    ax1.set_ylim(0.0, 1.02)

    ax1.set_xlim(0, 0.9)
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax1.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8], fontsize=18)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=18)

    ax1.set_xlabel(r'$h_{den}$', fontsize=18)
    ax1.set_ylabel(r'$h_{den}$', fontsize=18)

    fig.tight_layout()
    legends = ['Restructured $h_{den}$', 'Original $h_{den}$']

    lgd = ax1.legend(
        ln1+ln2,
        legends,
        loc='lower left',
        prop={'size': 14},
        # bbox_to_anchor=(1.6, 1.02),
        ncol=1,
        fancybox=True
    )

    plt.show()
    # plt.savefig('./syn_cora_h_den.pdf', bbox_inches='tight', format='pdf')
    # plt.clf()

# plot_gcn(syn_gcn)
plot_gcn(syn_sgc)
# plot_h_den(syn_sgc)
