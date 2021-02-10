# NCVoter

import math
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# =================================================================================================
# Generate plot 

width = 5    # the width of the bars


font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 18}

plt.rc('font', **font)

print('Generate bar Plot')

#w,h = plt.figaspect(plot_aspect)


#w,h = plt.figaspect(0.2)
plt.figure(figsize=(13,5))
# plt.subplots_adjust(hspace=0.6)

plt.xscale("linear")
plt.yscale("linear")

plt.xlim(xmin=0, xmax = 1430-(310*width)+160)
plt.ylim(ymin= 0.6, ymax = 0.9)


#----------------------------------------------------------
# plot RR
plt.subplot(1,2,1)

plt.title("(a) 2-layer GNAN", fontsize=20)
# plt.xlabel("Ablated frequency range", fontsize=25)
plt.ylabel("Micro-F1 (accuracy)", fontsize=18)
plt.xlabel("Ablated frequency range", fontsize=18)
plt.ylim(ymin= 0.6, ymax = 0.9)

x_axis_values_by_1 = [1430-(400*width)+50, 1430-(340*width) + 50]
x_axis_values2_by_1 = [1430-(400*width)+95, 1430-(340*width)+95]
x_axis_values3_by_1 = [1430-(400*width)+140, 1430-(340*width)+140]

Cornell_axis_values_by_1 = np.array([0.8648648648648649, 0.7297297297297297])
Texas_axis_values_by_1 = np.array([0.8648648648648649, 0.7297297297297297])
Wisconsin_axis_values_by_1 = np.array([0.8235294117647058, 0.6078431372549019])

bar1 = plt.bar(x_axis_values_by_1, Cornell_axis_values_by_1, 35, color='r', capsize=4, ecolor='k', label='Cornell')
bar2 = plt.bar(x_axis_values2_by_1, Texas_axis_values_by_1, 35, color='c', hatch='/', capsize=4, ecolor='k', label='Texas')
bar3 = plt.bar(x_axis_values3_by_1, Wisconsin_axis_values_by_1, 35, color='m', hatch='*', capsize=4, ecolor='k', label='Wisconsin')

x_tick_labels_by_1 = [r'0.0 - 1.0', r'1.0 - 2.0']
locs, labels = plt.xticks(x_axis_values2_by_1, x_tick_labels_by_1, fontsize=18)
# plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
locs, labels = plt.yticks(fontsize=18)
plt.setp(labels)

plt.legend(loc= 'upper right', prop={'size': 18})

#----------------------------------------------------------
# plot PC
# plt.subplot(2,3,2)
# plt.title("(b) 2-layer GNAN, $step = 0.5$", fontsize=17)
# # plt.xlabel("Ablated frequency range", fontsize=25)
# plt.ylim(ymin= 0.6, ymax = 0.95)
#
# x_axis_values_by_05 = [1430-(400*width), 1430-(360*width), 1430-(320*width), 1430-(280*width)]
# x_axis_values2_by_05 = [1430-(400*width)+35, 1430-(360*width)+35, 1430-(320*width)+35, 1430-(280*width)+35]
# x_axis_values3_by_05 = [1430-(400*width)+70, 1430-(360*width)+70, 1430-(320*width)+70, 1430-(280*width)+70]
#
# Cornell_axis_values_by_05 = np.array([0.8648648648648649, 0.8648648648648649, 0.7297297297297297, 0.8378378378378378])
# Texas_axis_values_by_05 = np.array([0.8648648648648649, 0.8648648648648649, 0.7297297297297297, 0.8378378378378378])
# Wisconsin_axis_values_by_05 = np.array([0.8431372549019607, 0.8431372549019607, 0.6274509803921569, 0.8431372549019607])
#
# bar1 = plt.bar(x_axis_values_by_05, Cornell_axis_values_by_05, 25, color='r', capsize=4, ecolor='k', label='Cornell')
# bar2 = plt.bar(x_axis_values2_by_05, Texas_axis_values_by_05, 25, color='c', hatch='/', capsize=4, ecolor='k', label='Texas')
# bar3 = plt.bar(x_axis_values3_by_05, Wisconsin_axis_values_by_05, 25, color='m', hatch='*', capsize=4, ecolor='k', label='Wisconsin')
#
# x_tick_labels_by_05 = [r'0.0 - 0.5', r'0.5 - 1.0', r'1.0 - 1.5', r'1.5 - 2']
#
# locs, labels = plt.xticks(x_axis_values3_by_05,x_tick_labels_by_05, fontsize=15)
# plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
# locs, labels = plt.yticks(fontsize=18)
# plt.setp(labels)
#----------------------------------------------------------
# plot PQ

# plt.subplot(2,3,3)
# plt.title("(c) 2-layer GNAN, $ step = 0.25$", fontsize=17)
# # plt.xlabel("Ablated frequency range", fontsize=25)
# plt.ylim(ymin= 0.6, ymax = 0.95)
#
# x_axis_values_by_025 = [1430-(400*width), 1430-(380*width), 1430-(360*width), 1430-(340*width), 1430-(320*width), 1430-(300*width), 1430-(280*width), 1430-(260*width)]
# x_axis_values2_by_025 = [1430-(400*width)+25, 1430-(380*width)+25, 1430-(360*width)+25, 1430-(340*width)+25, 1430-(320*width)+25, 1430-(300*width)+25, 1430-(280*width)+25, 1430-(260*width)+25]
# x_axis_values3_by_025 = [1430-(400*width)+50, 1430-(380*width)+50, 1430-(360*width)+50, 1430-(340*width)+50, 1430-(320*width)+50, 1430-(300*width)+50, 1430-(280*width)+50, 1430-(260*width)+50]
#
#
# Cornell_axis_values_by_025 = np.array([0.8648648648648649, 0.8648648648648649, 0.8648648648648649, 0.8648648648648649, 0.7297297297297297, 0.8648648648648649, 0.8378378378378378, 0.8378378378378378])
# Texas_axis_values_by_025 = np.array([0.8648648648648649, 0.8648648648648649, 0.8648648648648649, 0.7837837837837838, 0.8648648648648649, 0.8648648648648649, 0.8648648648648649, 0.8648648648648649])
# Wisconsin_axis_values_by_025 = np.array([0.8431372549019607, 0.8431372549019607, 0.8431372549019607, 0.8431372549019607, 0.6274509803921569, 0.8431372549019607, 0.8431372549019607, 0.8431372549019607])
#
# x_tick_labels_by_025 = [r'0.0 - 0.25', r'0.25 - 0.5', r'0.5 - 0.75', r'0.75 - 1.0', r'1.0 - 1.25', r'1.25 - 1.5', r'1.5 - 1.75', r'1.75 - 2.0']
#
# bar1 = plt.bar(x_axis_values_by_025, Cornell_axis_values_by_025, 16, color='r', capsize=4, ecolor='k', label='ASL')
# bar2 = plt.bar(x_axis_values2_by_025, Texas_axis_values_by_025, 16, color='c', hatch='/', capsize=4, ecolor='k', label='Fisher')
# bar3 = plt.bar(x_axis_values3_by_025, Wisconsin_axis_values_by_025, 16, color='m', hatch='*', capsize=4, ecolor='k', label='TBlo')
#
# locs, labels = plt.xticks(x_axis_values3_by_025,x_tick_labels_by_025, fontsize=15)
# plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
# locs, labels = plt.yticks(fontsize=18)
# plt.setp(labels)

#----------------------------------------------------------
# plot RR
plt.subplot(1,2,2)
plt.title("(b) 1-layer GNAN", fontsize=20)
# plt.xlabel("Ablated frequency range", fontsize=25)
# plt.ylabel("Micro-F1 (accuracy)", fontsize=18)
plt.xlabel("Ablated frequency range", fontsize=18)
plt.ylim(ymin= 0.6, ymax = 0.9)

x_axis_values_by_1 = [1430-(400*width)+50, 1430-(340*width) + 50]
x_axis_values2_by_1 = [1430-(400*width)+95, 1430-(340*width)+95]
x_axis_values3_by_1 = [1430-(400*width)+140, 1430-(340*width)+140]

Cornell_axis_values_by_1 = np.array([0.8108108108108109, 0.7027027027027027])
Texas_axis_values_by_1 = np.array([0.8648648648648649, 0.8378378378378378])
Wisconsin_axis_values_by_1 = np.array([0.8431372549019607, 0.6470588235294118])

bar1 = plt.bar(x_axis_values_by_1, Cornell_axis_values_by_1, 35, color='r', capsize=4, ecolor='k', label='Cornell')
bar2 = plt.bar(x_axis_values2_by_1, Texas_axis_values_by_1, 35, color='c', hatch='/', capsize=4, ecolor='k', label='Texas')
bar3 = plt.bar(x_axis_values3_by_1, Wisconsin_axis_values_by_1, 35, color='m', hatch='*', capsize=4, ecolor='k', label='Wisconsin')

x_tick_labels_by_1 = [r'0.0 - 1.0', r'1.0 - 2.0']
locs, labels = plt.xticks(x_axis_values2_by_1, x_tick_labels_by_1, fontsize=18)
# plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
locs, labels = plt.yticks(fontsize=18)
plt.setp(labels)

# #----------------------------------------------------------
# # plot PC
# plt.subplot(2,3,5)
# plt.title("(e) 1-layer GNAN, $step = 0.5$", fontsize=17)
# # plt.xlabel("Ablated frequency range", fontsize=25)
# plt.ylim(ymin= 0.6, ymax = 0.95)
#
# x_axis_values_by_05 = [1430-(400*width), 1430-(360*width), 1430-(320*width), 1430-(280*width)]
# x_axis_values2_by_05 = [1430-(400*width)+35, 1430-(360*width)+35, 1430-(320*width)+35, 1430-(280*width)+35]
# x_axis_values3_by_05 = [1430-(400*width)+70, 1430-(360*width)+70, 1430-(320*width)+70, 1430-(280*width)+70]
#
# Cornell_axis_values_by_05 = np.array([0.8108108108108109, 0.8108108108108109, 0.7567567567567567, 0.8108108108108109])
# Texas_axis_values_by_05 = np.array([0.8648648648648649, 0.7837837837837838, 0.8648648648648649, 0.8648648648648649])
# Wisconsin_axis_values_by_05 = np.array([0.8627450980392157, 0.8627450980392157, 0.6470588235294118, 0.8627450980392157])
#
# bar1 = plt.bar(x_axis_values_by_05, Cornell_axis_values_by_05, 25, color='r', capsize=4, ecolor='k', label='Cornell')
# bar2 = plt.bar(x_axis_values2_by_05, Texas_axis_values_by_05, 25, color='c', hatch='/', capsize=4, ecolor='k', label='Texas')
# bar3 = plt.bar(x_axis_values3_by_05, Wisconsin_axis_values_by_05, 25, color='m', hatch='*', capsize=4, ecolor='k', label='Wisconsin')
#
# x_tick_labels_by_05 = [r'0.0 - 0.5', r'0.5 - 1.0', r'1.0 - 1.5', r'1.5 - 2']
#
# locs, labels = plt.xticks(x_axis_values3_by_05,x_tick_labels_by_05, fontsize=15)
# plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
# locs, labels = plt.yticks(fontsize=18)
# plt.setp(labels)
# #----------------------------------------------------------
# # plot PQ
#
# plt.subplot(2,3,6)
# plt.title("(f) 1-layer GNAN, $step = 0.25$", fontsize=17)
# # plt.xlabel("Ablated frequency range", fontsize=25)
# plt.ylim(ymin= 0.6, ymax = 0.95)
#
# x_axis_values_by_025 = [1430-(400*width), 1430-(380*width), 1430-(360*width), 1430-(340*width), 1430-(320*width), 1430-(300*width), 1430-(280*width), 1430-(260*width)]
# x_axis_values2_by_025 = [1430-(400*width)+25, 1430-(380*width)+25, 1430-(360*width)+25, 1430-(340*width)+20, 1430-(320*width)+25, 1430-(300*width)+25, 1430-(280*width)+25, 1430-(260*width)+25]
# x_axis_values3_by_025 = [1430-(400*width)+50, 1430-(380*width)+50, 1430-(360*width)+50, 1430-(340*width)+40, 1430-(320*width)+50, 1430-(300*width)+50, 1430-(280*width)+50, 1430-(260*width)+50]
#
#
# Cornell_axis_values_by_025 = np.array([0.8108108108108109, 0.8108108108108109, 0.8108108108108109, 0.8108108108108109, 0.7567567567567567, 0.8108108108108109, 0.8108108108108109, 0.8108108108108109])
# Texas_axis_values_by_025 = np.array([0.8648648648648649, 0.8648648648648649, 0.8648648648648649, 0.7837837837837838, 0.8648648648648649, 0.8648648648648649, 0.8648648648648649, 0.8648648648648649])
# Wisconsin_axis_values_by_025 = np.array([0.8627450980392157, 0.8627450980392157, 0.8627450980392157, 0.8627450980392157, 0.6470588235294118, 0.8627450980392157, 0.8627450980392157, 0.8627450980392157])
#
# x_tick_labels_by_025 = [r'0.0 - 0.25', r'0.25 - 0.5', r'0.5 - 0.75', r'0.75 - 1.0', r'1.0 - 1.25', r'1.25 - 1.5', r'1.5 - 1.75', r'1.75 - 2.0']
#
# bar1 = plt.bar(x_axis_values_by_025, Cornell_axis_values_by_025, 16, color='r', capsize=4, ecolor='k', label='ASL')
# bar2 = plt.bar(x_axis_values2_by_025, Texas_axis_values_by_025, 16, color='c', hatch='/', capsize=4, ecolor='k', label='Fisher')
# bar3 = plt.bar(x_axis_values3_by_025, Wisconsin_axis_values_by_025, 16, color='m', hatch='*', capsize=4, ecolor='k', label='TBlo')
#
# locs, labels = plt.xticks(x_axis_values3_by_025,x_tick_labels_by_025, fontsize=15)
# plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
# locs, labels = plt.yticks(fontsize=18)
# plt.setp(labels)
#
# plt.ylim(ymin= 0.6, ymax = 0.95)



# plt.show()

plt.savefig('./disassortative_frequenc_ablation.pdf', format='pdf')
#
plt.savefig('./disassortative_frequenc_ablation.eps', bbox_inches='tight')
plt.clf()

