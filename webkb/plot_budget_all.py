# NCVoter

import math
import sys
import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# =================================================================================================
# Generate plot 
#---------------------------------------------------
# Cora


threshold_list_c = [0.4, 0.6, 0.8]

x_axis_values_c = [20, 50, 100, 150, 200, 500]  # budget
x_tick_labels_c = ['20', '50', '100', '150', '200', '500'] 

y_axis_values_c = [[0.2, 0.3, 0.5, 0.7, 0.9, 1.0], # threshold 0.4
                 [0.2, 0.4, 0.6, 0.9, 0.9, 1.0], # threshold 0.6
                 [0.3, 0.2, 0.4, 0.7, 0.8, 1.0], # threshold 0.8
                 [0.2, 0.2, 0.3, 0.2, 0.3, 0.2] # random
                ]

label_list_c = [r'$\epsilon = 0.6$', r'$\epsilon = 0.4$', r'$\epsilon = 0.2$', 'RSL']


#-------------------------------------------------------------------------------
# Restaurant

threshold_list_r = [0.6, 0.8, 0.9]

x_axis_values_r = [20, 50, 100, 150, 200, 500]  # this is the rule value
x_tick_labels_r = ['20', '50', '100', '150', '200', '500']   # this is the label

y_axis_values_r = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # threshold 0.6
                 [0.8, 0.9, 1.0, 1.0, 1.0, 1.0], # threshold 0.8
                 [0.9, 0.9, 0.9, 1.0, 0.9, 1.0], # threshold 0.9
                 [0.01, 0.01, 0.01, 0.1, 0.1, 0.2] # random
                ]


label_list_r = [r'$\epsilon = 0.4$', r'$\epsilon = 0.2$', r'$\epsilon = 0.1$', 'RSL']

#-------------------------------------------------------------------------------
# DBLP-ACM


threshold_list_d = [0.4, 0.6, 0.9]

x_axis_values_d = [30, 50, 100, 150, 200]  # budget
x_tick_labels_d = ['30', '50', '100', '150', '200'] 

y_axis_values_d = [[0.7, 0.8, 0.9, 0.9, 1.0], # threshold 0.6
                 [0.9, 0.9, 0.9, 1.0, 1.0], # threshold 0.85
                 [0.5, 0.6, 0.7, 0.7, 0.8], # threshold 0.9
                 [0.1, 0.1, 0.1, 0.1, 0.2] # random
                ]

label_list_d = [r'$\epsilon = 0.4$', r'$\epsilon = 0.15$', r'$\epsilon = 0.1$', 'RSL']



#-------------------------------------------------------------------------------
# NCVoter


threshold_list_n = [0.4, 0.6, 0.9]

x_axis_values_n = [50, 100, 150, 200, 250]  # budget
x_tick_labels_n = ['50', '100', '150', '200', '250'] 

y_axis_values_n = [[0.7, 0.8, 0.8, 0.9, 0.9], # threshold 0.4
                 [0.8, 0.9, 0.9, 1.0, 1.0], # threshold 0.6
                 [0.8, 0.9, 1.0, 1.0, 1.0], # threshold 0.9
                 [0.01, 0.01, 0.01, 0.02, 0.02] # random
                ]

label_list_n = [r'$\epsilon = 0.6$', r'$\epsilon = 0.4$', r'$\epsilon = 0.1$', 'RSL']




color_list = ['r', 'b', 'g', 'purple']
line_style = ['o-', 's--', 'v-.', '^:']
print('Generate accuracy plot - against budget')






#w,h = plt.figaspect(0.5)
plt.figure(figsize=(30,6))


#----------------------------------------------------------
# plot cora
plt.subplot(1,4,1)

plt.title("(a) Cora", fontsize=25)
plt.xlabel('Sample size', fontsize=25)
plt.ylabel("Constraint satisfaction", fontsize=25)

plt.xlim(xmin=20, xmax=250)
plt.ylim(ymin = 0.00, ymax = 1.05)
locs, labels = plt.xticks(x_axis_values_c,x_tick_labels_c)
plt.setp(labels)


for i in range( len(y_axis_values_c) ):
  plt.plot(x_axis_values_c, y_axis_values_c[i], line_style[i], label = label_list_c[i], color = color_list[i], linewidth = 3)

plt.legend(loc= 'center right', prop={'size':18})
locs, labels = plt.xticks(fontsize=18)
locs, labels = plt.yticks(fontsize=18)
#----------------------------------------------------------
# plot restaurant
plt.subplot(1,4,2)

plt.title("(b) Restaurant", fontsize=25)
plt.xlabel('Sample size', fontsize=25)

plt.xlim(xmin=20, xmax=250)
plt.ylim(ymin = 0.00, ymax = 1.05)
locs, labels = plt.xticks(x_axis_values_r,x_tick_labels_r)
plt.setp(labels)


for i in range( len(y_axis_values_r) ):
  plt.plot(x_axis_values_r, y_axis_values_r[i], line_style[i], label = label_list_r[i], color = color_list[i], linewidth = 3)

plt.legend(loc= 'center right', prop={'size':18})
locs, labels = plt.xticks(fontsize=18)
locs, labels = plt.yticks(fontsize=18)
#----------------------------------------------------------
# plot DBLP-ACM

plt.subplot(1,4,3)

plt.title("(c) DBLP-ACM", fontsize=25)
plt.xlabel('Sample size', fontsize=25)

plt.xlim(xmin=30, xmax=200)
plt.ylim(ymin = 0.00, ymax = 1.05)
locs, labels = plt.xticks(x_axis_values_d,x_tick_labels_d)
plt.setp(labels)


for i in range( len(y_axis_values_d) ):
  plt.plot(x_axis_values_d, y_axis_values_d[i], line_style[i], label = label_list_d[i], color = color_list[i], linewidth = 3)

plt.legend(loc= 'center right', prop={'size':18})
locs, labels = plt.xticks(fontsize=18)
locs, labels = plt.yticks(fontsize=18)


#----------------------------------------------------------
# plot NCVoter

plt.subplot(1,4,4)

plt.title("(d) NCVR", fontsize=25)
plt.xlabel('Sample size', fontsize=25)

plt.xlim(xmin=50, xmax=250)
plt.ylim(ymin = 0.00, ymax = 1.05)
locs, labels = plt.xticks(x_axis_values_n,x_tick_labels_n)
plt.setp(labels)


for i in range( len(y_axis_values_n) ):
  plt.plot(x_axis_values_n, y_axis_values_n[i], line_style[i], label = label_list_n[i], color = color_list[i], linewidth = 3)

plt.legend(loc= 'center right', prop={'size':18})
locs, labels = plt.xticks(fontsize=18)
locs, labels = plt.yticks(fontsize=18)

plt.savefig('./All_budget.pdf', format='pdf')

plt.savefig('./All_budget.eps', bbox_inches='tight')
plt.clf()

