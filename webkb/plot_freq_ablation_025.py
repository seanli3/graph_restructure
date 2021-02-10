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
plt.figure(figsize=(15,8))
plt.subplots_adjust(hspace=1)

plt.xscale("linear")
plt.yscale("linear")

plt.xlim(xmin=0, xmax = 1430-(310*width)+160)
plt.ylim(ymin= 0.5, ymax = 0.94)

#----------------------------------------------------------
# plot PQ

plt.subplot(2,1,1)
plt.title("(a) 2-layer GNAN, $ step = 0.25$", fontsize=20)
plt.xlabel("Ablated frequency range", fontsize=20)
plt.ylabel("Micro-F1 (accuracy)", fontsize=20)
plt.ylim(ymin= 0.5, ymax = 0.94)

x_axis_values_by_025 = [1430-(400*width), 1430-(375*width), 1430-(350*width), 1430-(325*width), 1430-(300*width), 1430-(275*width), 1430-(250*width), 1430-(225*width)]
x_axis_values2_by_025 = [1430-(400*width)+20, 1430-(375*width)+20, 1430-(350*width)+20, 1430-(325*width)+20, 1430-(300*width)+20, 1430-(275*width)+20, 1430-(250*width)+20, 1430-(225*width)+20]
x_axis_values3_by_025 = [1430-(400*width)+40, 1430-(375*width)+40, 1430-(350*width)+40, 1430-(325*width)+40, 1430-(300*width)+40, 1430-(275*width)+40, 1430-(250*width)+40, 1430-(225*width)+40]


Cornell_axis_values_by_025_2 = np.array([0.827027027027027, 0.827027027027027, 0.827027027027027, 0.8027027027027026, 0.7648648648648648, 0.827027027027027, 0.827027027027027, 0.827027027027027])
Texas_axis_values_by_025_2 = np.array([0.854054054054054, 0.854054054054054, 0.8513513513513514, 0.7594594594594595, 0.8405405405405407, 0.8486486486486486, 0.8486486486486486, 0.8432432432432432])
Wisconsin_axis_values_by_025_2 = np.array([0.8588235294117647, 0.8588235294117647, 0.8588235294117647, 0.8588235294117647, 0.6176470588235294, 0.8588235294117647, 0.8607843137254901, 0.8607843137254901])

Cornell_axis_errors_by_025_2 = np.array([0.08276529065121335, 0.08276529065121335, 0.08276529065121335, 0.09366770256266692, 0.10587023095800066, 0.08276529065121335, 0.08276529065121335, 0.08276529065121335])
Texas_axis_errors_by_025_2 = np.array([0.05726215705250706, 0.05726215705250706, 0.0544281350161918, 0.05761541245160175, 0.07368686503307581, 0.0613671580754105, 0.0613671580754105, 0.06344798869066312])
Wisconsin_axis_errors_by_025_2 = np.array([0.03788595227761946, 0.03788595227761946, 0.03788595227761946, 0.03788595227761946, 0.06552267207765823, 0.03788595227761946, 0.041799416876652264, 0.041799416876652264])

x_tick_labels_by_025 = [r'0.0 - 0.25', r'0.25 - 0.5', r'0.5 - 0.75', r'0.75 - 1.0', r'1.0 - 1.25', r'1.25 - 1.5', r'1.5 - 1.75', r'1.75 - 2.0']

bar1 = plt.bar(x_axis_values_by_025, Cornell_axis_values_by_025_2, 15, color='r', capsize=4, ecolor='k', label='ASL', yerr=Cornell_axis_errors_by_025_2)
bar2 = plt.bar(x_axis_values2_by_025, Texas_axis_values_by_025_2, 15, color='c', hatch='/', capsize=4, ecolor='k', label='Fisher', yerr=Texas_axis_errors_by_025_2)
bar3 = plt.bar(x_axis_values3_by_025, Wisconsin_axis_values_by_025_2, 15, color='m', hatch='*', capsize=4, ecolor='k', label='TBlo', yerr=Wisconsin_axis_errors_by_025_2)

locs, labels = plt.xticks(x_axis_values3_by_025,x_tick_labels_by_025, fontsize=20)
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
locs, labels = plt.yticks(fontsize=20)
plt.setp(labels)

handles = [bar1, bar2, bar3]
label_list_p = ['Cornell', 'Texas', 'Wisconsin']
# plt.legend(loc= 'upper right', prop={'size': 16})
legend = plt.legend(handles, label_list_p, bbox_to_anchor=(1.0, 1.04), prop={'size':20}, title = "Legends", ncol=1)

plt.subplot(2,1,2)
plt.title("(b) 1-layer GNAN, $step = 0.25$", fontsize=20)
plt.xlabel("Ablated frequency range", fontsize=20)
plt.ylabel("Micro-F1 (accuracy)", fontsize=20)
plt.ylim(ymin= 0.5, ymax = 0.94)

x_axis_values_by_025 = [1430-(400*width), 1430-(375*width), 1430-(350*width), 1430-(325*width), 1430-(300*width), 1430-(275*width), 1430-(250*width), 1430-(225*width)]
x_axis_values2_by_025 = [1430-(400*width)+20, 1430-(375*width)+20, 1430-(350*width)+20, 1430-(325*width)+20, 1430-(300*width)+20, 1430-(275*width)+20, 1430-(250*width)+20, 1430-(225*width)+20]
x_axis_values3_by_025 = [1430-(400*width)+40, 1430-(375*width)+40, 1430-(350*width)+40, 1430-(325*width)+40, 1430-(300*width)+40, 1430-(275*width)+40, 1430-(250*width)+40, 1430-(225*width)+40]


Cornell_axis_values_by_025_1 = np.array([0.8054054054054054, 0.8054054054054054, 0.8054054054054054, 0.7972972972972973, 0.7432432432432433, 0.8054054054054054, 0.8054054054054054, 0.8081081081081081])
Texas_axis_values_by_025_1 = np.array([0.827027027027027, 0.827027027027027, 0.8243243243243243, 0.7378378378378379, 0.827027027027027, 0.827027027027027, 0.8243243243243243, 0.818918918918919])
Wisconsin_axis_values_by_025_1 = np.array([0.8431372549019607, 0.8431372549019607, 0.8431372549019607, 0.8431372549019607, 0.6058823529411764, 0.8431372549019607, 0.8470588235294118, 0.8470588235294118])


Cornell_axis_errors_by_025_1 = np.array([0.05524219717749873, 0.05524219717749873, 0.05524219717749873, 0.05442813501619177, 0.048094947081222125, 0.05524219717749873, 0.05524219717749873, 0.05761541245160176])
Texas_axis_errors_by_025_1 = np.array([0.057262157052507055, 0.057262157052507055, 0.054428135016191787, 0.05104221792510015, 0.057262157052507055, 0.057262157052507055, 0.054428135016191787, 0.054129077065945254])
Wisconsin_axis_errors_by_025_1 = np.array([0.05228758169934639, 0.05228758169934639, 0.05228758169934639, 0.05228758169934639, 0.06757676876111285, 0.05228758169934639, 0.04785099403326307, 0.04785099403326307])

x_tick_labels_by_025 = [r'0.0 - 0.25', r'0.25 - 0.5', r'0.5 - 0.75', r'0.75 - 1.0', r'1.0 - 1.25', r'1.25 - 1.5', r'1.5 - 1.75', r'1.75 - 2.0']

bar1 = plt.bar(x_axis_values_by_025, Cornell_axis_values_by_025_1, 15, color='r', capsize=4, ecolor='k', label='ASL', yerr=Cornell_axis_errors_by_025_1)
bar2 = plt.bar(x_axis_values2_by_025, Texas_axis_values_by_025_1, 15, color='c', hatch='/', capsize=4, ecolor='k', label='Fisher', yerr=Texas_axis_errors_by_025_1)
bar3 = plt.bar(x_axis_values3_by_025, Wisconsin_axis_values_by_025_1, 15, color='m', hatch='*', capsize=4, ecolor='k', label='TBlo', yerr=Wisconsin_axis_errors_by_025_1)

locs, labels = plt.xticks(x_axis_values3_by_025,x_tick_labels_by_025, fontsize=20)
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
locs, labels = plt.yticks(fontsize=20)
plt.setp(labels)

plt.ylim(ymin= 0.5, ymax = 0.94)


# plt.show()

plt.savefig('./disassortative_frequenc_ablation_025.pdf', format='pdf')

plt.savefig('./disassortative_frequenc_ablation_025.eps', bbox_inches='tight')
plt.clf()

