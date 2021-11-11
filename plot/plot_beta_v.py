# NCVoter
from matplotlib import cm

import math
import sys
import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

citeseer = torch.load('./citeseer_beta_acc.pkl')
pubmed = torch.load('./pubmed_beta_acc.pkl')

# =================================================================================================
# Generate plot 
#---------------------------------------------------
# Citeseer


x_axis_values_c = citeseer['x'][:5]
x_tick_labels_c = ['{:.1f}'.format(i) for i in x_axis_values_c]

y_axis_values_c = [ acc.tolist()[:5] for acc in citeseer['accs'] ]
del y_axis_values_c[-2]

label_list_c = citeseer['labels']
del label_list_c[-2]
label_list_c[-1] = 'Our method'
label_list_c[-2] = 'MV'


#-------------------------------------------------------------------------------
# PubMed

x_axis_values_p = pubmed['x'][:5]
x_tick_labels_p = ['{:.1f}'.format(i) for i in x_axis_values_c]

y_axis_values_p = [ acc.tolist()[:5] for acc in pubmed['accs'] ]
del y_axis_values_p[-2]

label_list_p = pubmed['labels']
del label_list_p[-2]
label_list_p[-1] = 'Our method'
label_list_p[-2] = "MV"


# color_list = ['r', 'b', 'g', 'purple', 'pink', ]
line_style = ['o-', 's--', 'v-.', '^:', '*--', 'd:', '--', 'X-']
print('Generate accuracy plot - against budget')



#w,h = plt.figaspect(0.5)
plt.figure(figsize=(30,10))


#----------------------------------------------------------
# plot citeseer
plt.subplot(1,2,1)

plt.title("(a) CiteSeer", fontsize=25)
plt.xlabel(r'$\beta_v$', fontsize=25)
plt.ylabel("Micro-F1 (accuracy)", fontsize=25)

plt.xlim(xmin=0.08, xmax=0.52)
plt.ylim(ymin = 0.00, ymax = 1.05)
locs, labels = plt.xticks(x_axis_values_c,x_tick_labels_c)
plt.setp(labels)


for i in range( len(y_axis_values_c) ):
  plt.plot(x_axis_values_c, y_axis_values_c[i], line_style[i], label = label_list_c[i], color = cm.Set2(i), linewidth = 3, markersize=15)

plt.legend(loc= 'upper left', prop={'size':18})
locs, labels = plt.xticks(fontsize=18)
locs, labels = plt.yticks(fontsize=18)
#----------------------------------------------------------
# plot restaurant
plt.subplot(1,2,2)

plt.title("(b) PubMed", fontsize=25)
plt.xlabel(r'$\beta_v$', fontsize=25)

plt.xlim(xmin=0.08, xmax=0.52)
plt.ylim(ymin = 0.00, ymax = 1.05)
locs, labels = plt.xticks(x_axis_values_p,x_tick_labels_p)
plt.setp(labels)

for i in range( len(y_axis_values_p) ):
  plt.plot(x_axis_values_p, y_axis_values_p[i], line_style[i], label = label_list_p[i], color = cm.Set2(i), linewidth = 3, markersize=15)

plt.legend(loc= 'upper left', prop={'size':18})
locs, labels = plt.xticks(fontsize=18)
locs, labels = plt.yticks(fontsize=18)
# plt.show()

plt.savefig('./beta_v.pdf', format='pdf')

plt.savefig('./beta_v.eps', bbox_inches='tight')
plt.clf()

