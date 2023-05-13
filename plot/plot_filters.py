# NCVoter
from matplotlib import cm

import math
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import torch
import pandas as pd

x = torch.arange(-0.2, 2.2, 0.01)
y_low = torch.exp(-100*torch.pow(x, 2))
y_band = torch.exp(-1000*torch.pow(x-0.5, 2))
y_high = 1-torch.exp(-10*torch.pow(x, 2))


def slicer(x, m=12, s=20):
    a = torch.arange(0, 2 + 2/s, 4/s).view(-1, 1)
    eps = 1e-1
    return 1/math.pow(s, 2*m)*torch.pow(torch.pow((x-a)/(2+eps), 2*m) + 1/math.pow(s, 2*m), -1)

def plot(x, y):
    plt.figure(dpi=200)
    lns = []
    fig, ax = plt.subplots()
    fig.set_figwidth(6)
    fig.set_figheight(5)
    if len(y.shape) == 2:
        for i in range(y.shape[0]):
            ax.plot(x, y[i])
    else:
        ax.plot(x, y)
    # ax.set_ylim(0.4, 1)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels([0.0, 0.5, 1.0, 1.5, 2.0], fontsize=26)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=26)

    fig.tight_layout()
    plt.show()

plot(x, slicer(x))
plt.savefig('./slicer.pdf', bbox_inches='tight', format='pdf')
# plot(x, y_low)
# plt.savefig('./low_pass.eps', bbox_inches='tight', format='eps')
# plt.clf()
# plot(x, y_band)
# plt.savefig('./band_pass.eps', bbox_inches='tight', format='eps')
# plt.clf()
# plot(x, y_high)
# plt.savefig('./high_pass.eps', bbox_inches='tight', format='eps')
# plt.clf()
