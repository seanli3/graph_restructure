import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('Agg')


# cora_data = pd.read_csv('cora.csv', delimiter='\t')
# citeseer_data = pd.read_csv('citeseer.csv', delimiter='\t')
actor_data = pd.read_csv('actor.csv', delimiter='\t')
squirrel_data = pd.read_csv('squirrel.csv', delimiter='\t')
chameleon_data = pd.read_csv('chameleon.csv', delimiter='\t')
wisconsin_data = pd.read_csv('wisconsin.csv', delimiter='\t')
cornell_data = pd.read_csv('cornell.csv', delimiter='\t')
texas_data = pd.read_csv('texas.csv', delimiter='\t')

matplotlib.rcParams.update({'font.size': 16})

def grouped_barplot(df, cat, subcat, val , err, ylim, legend=False, filename=''):
    # u = df[cat].unique()
    u = df[df['Method'] == 'Original'][cat]
    x = np.arange(len(u))
    # subx = df[subcat].unique()
    subx = ['Original', 'GDC', 'Ours']
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean() - 0.02
    axs = []
    for i,gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        ax = plt.bar(x+offsets[i], dfg[val].values, width=width,
                label="{} {}".format(subcat, gr), yerr=dfg[err].values)
        axs.append(ax)
    # plt.xlabel(cat)
    baselines = df[df['Method'] == 'Baseline'][cat]
    shapes = ['--', '-.', ':']
    colors = ['pink', 'orange', 'brown']
    for i, b in enumerate(baselines):
        ax = plt.axhline(y=df[df[cat] == b][val].item(), linestyle=shapes[i], color=colors[i])
        axs.append(ax)
        # plt.axhspan(
        #     (df[df[cat] == b][val] - df[df[cat] == b][err]).item(),
        #     (df[df[cat] == b][val] + df[df[cat] == b][err]).item(),
        #     color=colors[i], alpha=0.2)
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.xticks(x, u, fontsize=16)
    import math
    ymin = math.floor((df[val].values -df[err].values).min())
    ymax = math.ceil((df[val].values+df[err].values).max())
    ystep = math.floor((ymax - ymin)/4)
    yrange = list(range(ymin, ymax, ystep))
    plt.yticks(yrange, yrange, fontsize=16)
    plt.ylim(ymin, ymax)
    if legend:
        plt.legend(axs, ['Original', 'GDC', 'Ours'] + baselines.values.tolist() )
    # plt.show()
    plt.savefig('./{}.pdf'.format(filename), format="pdf", bbox_inches='tight', prop={'size': 16})
    plt.clf()


cat = "GNN"
subcat = "Method"
val = "Acc"
err = "Std"

# call the function with df from the question
# grouped_barplot(cora_data, cat, subcat, val, err, ylim=[70, 90], legend=True, filename='hist_cora')
# grouped_barplot(citeseer_data, cat, subcat, val, err , ylim=[65, 75], filename='hist_citeseer')
grouped_barplot(actor_data, cat, subcat, val, err , ylim=[20, 40], filename='his_actor')
grouped_barplot(chameleon_data, cat, subcat, val, err , ylim=[28, 72], filename='his_chame')
grouped_barplot(squirrel_data, cat, subcat, val, err , ylim=[28, 60], filename='his_squir')
grouped_barplot(wisconsin_data, cat, subcat, val, err , ylim=[45, 90], filename='his_wisconsin')
grouped_barplot(cornell_data, cat, subcat, val, err , ylim=[50, 90], filename='his_cornell', legend=True)
grouped_barplot(texas_data, cat, subcat, val, err , ylim=[50, 97], filename='his_texas')

