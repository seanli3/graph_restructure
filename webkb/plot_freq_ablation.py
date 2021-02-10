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
fig = plt.figure(figsize=(15,8))
plt.subplots_adjust(hspace=1.25, wspace=0.2, bottom=0.2, top=0.85)
# fig.tight_layout(pad=0, w_pad=-0, h_pad=0 )

plt.xscale("linear")
plt.yscale("linear")

plt.xlim(xmin=0, xmax = 1430-(310*width)+160)
plt.ylim(ymin= 0.5, ymax = 0.94)


#----------------------------------------------------------
# plot RR
plt.subplot(2,3,1)

plt.title("(a) 2-layer GNAN, $step = 1.0$", fontsize=19)
# plt.xlabel("Ablated frequency range", fontsize=18)
plt.ylabel("Micro-F1", fontsize=18)
plt.ylim(ymin= 0.5, ymax = 0.94)

x_axis_values_by_1 = [1430-(400*width), 1430-(360*width), 1430-(320*width)]
x_axis_values2_by_1 = [1430-(400*width)+30, 1430-(360*width)+30, 1430-(320*width) + 30]
x_axis_values3_by_1 = [1430-(400*width)+60, 1430-(360*width)+60, 1430-(320*width) + 60]

Cornell_axis_values_by_1_2 = np.array([0.827027027027027, 0.827027027027027, 0.818918918918919])
Texas_axis_values_by_1_2 = np.array([0.8513513513513514 , 0.7999999999999999, 0.7297297297297297])
Wisconsin_axis_values_by_1_2 = np.array([0.8588235294117647, 0.8313725490196078, 0.5862745098039216])

Cornell_axis_errors_by_1_2 = np.array([0.08276529065121335, 0.08276529065121335, 0.07755077195191835])
Texas_axis_errors_by_1_2 = np.array([0.057332982258368755, 0.05283920107609074, 0.07099419208953352])
Wisconsin_axis_errors_by_1_2 = np.array([0.03788595227761946, 0.0425590080943783, 0.07005975004372086])


bar1 = plt.bar(x_axis_values_by_1, Cornell_axis_values_by_1_2, 17, color='r', capsize=4, ecolor='k', label='Cornell', yerr=Cornell_axis_errors_by_1_2)
bar2 = plt.bar(x_axis_values2_by_1, Texas_axis_values_by_1_2, 17, color='c', hatch='/', capsize=4, ecolor='k', label='Texas', yerr=Texas_axis_errors_by_1_2)
bar3 = plt.bar(x_axis_values3_by_1, Wisconsin_axis_values_by_1_2, 17, color='m', hatch='*', capsize=4, ecolor='k', label='Wisconsin', yerr=Wisconsin_axis_errors_by_1_2)

x_tick_labels_by_1 = [r'no ablation', r'0.0 - 1.0', r'1.0 - 2.0']
locs, labels = plt.xticks(x_axis_values2_by_1, x_tick_labels_by_1, fontsize=18)
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
locs, labels = plt.yticks(fontsize=18)
plt.setp(labels)


#----------------------------------------------------------
# plot PC
plt.subplot(2,3,2)
plt.title("(b) 2-layer GNAN, $step = 0.5$", fontsize=19)
plt.xlabel("Ablated frequency range", fontsize=19)
plt.ylim(ymin= 0.5, ymax = 0.94)

x_axis_values_by_05 = [1430-(400*width), 1430-(360*width), 1430-(320*width), 1430-(280*width)]
x_axis_values2_by_05 = [1430-(400*width)+30, 1430-(360*width)+30, 1430-(320*width)+30, 1430-(280*width)+30]
x_axis_values3_by_05 = [1430-(400*width)+60, 1430-(360*width)+60, 1430-(320*width)+60, 1430-(280*width)+60]

Cornell_axis_values_by_05_2 = np.array([0.827027027027027, 0.8081081081081081, 0.7324324324324325, 0.8243243243243242])
Texas_axis_values_by_05_2 = np.array([0.8513513513513514, 0.7405405405405404, 0.8405405405405407, 0.8405405405405405])
Wisconsin_axis_values_by_05_2 = np.array([0.8588235294117647, 0.8588235294117647, 0.6098039215686274, 0.8568627450980392])

Cornell_axis_errors_by_05_2 = np.array([0.08276529065121335, 0.09905814059389882, 0.08868270055729523, 0.0857040432001822])
Texas_axis_errors_by_05_2 = np.array([0.0544281350161918, 0.054353524789183275, 0.07368686503307581, 0.0779682757600678])
Wisconsin_axis_errors_by_05_2 = np.array([0.03788595227761946, 0.03788595227761946, 0.06630041648583403, 0.040343266305911035])

bar1 = plt.bar(x_axis_values_by_05, Cornell_axis_values_by_05_2, 22, color='r', capsize=4, ecolor='k', label='Cornell', yerr=Cornell_axis_errors_by_05_2)
bar2 = plt.bar(x_axis_values2_by_05, Texas_axis_values_by_05_2, 22, color='c', hatch='/', capsize=4, ecolor='k', label='Texas', yerr=Texas_axis_errors_by_05_2)
bar3 = plt.bar(x_axis_values3_by_05, Wisconsin_axis_values_by_05_2, 22, color='m', hatch='*', capsize=4, ecolor='k', label='Wisconsin', yerr=Wisconsin_axis_errors_by_05_2)


x_tick_labels_by_05 = [r'0.0 - 0.5', r'0.5 - 1.0', r'1.0 - 1.5', r'1.5 - 2']

locs, labels = plt.xticks(x_axis_values2_by_05,x_tick_labels_by_05, fontsize=18)
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
locs, labels = plt.yticks(fontsize=18)
plt.setp(labels)


handles = [bar1, bar2, bar3]
label_list_p = ['Cornell', 'Texas', 'Wisconsin']
# plt.legend(loc= 'upper right', prop={'size': 16})
legend = plt.legend(handles, label_list_p, loc='lower left', bbox_to_anchor=(-0.5, 1.25), prop={'size':18}, ncol=3)
for handle in legend.get_patches():
    handle.set_alpha(1)

#----------------------------------------------------------
# plot PQ

plt.subplot(2,3,3)
plt.title("(c) 2-layer GNAN, $ step = 0.25$", fontsize=19)
# plt.xlabel("Ablated frequency range", fontsize=18)
plt.ylim(ymin= 0.5, ymax = 0.94)

x_axis_values_by_025 = [1430-(400*width), 1430-(375*width), 1430-(350*width), 1430-(325*width), 1430-(300*width), 1430-(275*width), 1430-(250*width), 1430-(225*width)]
x_axis_values2_by_025 = [1430-(400*width)+38, 1430-(375*width)+38, 1430-(350*width)+38, 1430-(325*width)+38, 1430-(300*width)+38, 1430-(275*width)+38, 1430-(250*width)+38, 1430-(225*width)+38]
x_axis_values3_by_025 = [1430-(400*width)+76, 1430-(375*width)+76, 1430-(350*width)+76, 1430-(325*width)+76, 1430-(300*width)+76, 1430-(275*width)+76, 1430-(250*width)+76, 1430-(225*width)+76]


Cornell_axis_values_by_025_2 = np.array([0.827027027027027, 0.827027027027027, 0.827027027027027, 0.8027027027027026, 0.7648648648648648, 0.827027027027027, 0.827027027027027, 0.827027027027027])
Texas_axis_values_by_025_2 = np.array([0.854054054054054, 0.854054054054054, 0.8513513513513514, 0.7594594594594595, 0.8405405405405407, 0.8486486486486486, 0.8486486486486486, 0.8432432432432432])
Wisconsin_axis_values_by_025_2 = np.array([0.8588235294117647, 0.8588235294117647, 0.8588235294117647, 0.8588235294117647, 0.6176470588235294, 0.8588235294117647, 0.8607843137254901, 0.8607843137254901])

Cornell_axis_errors_by_025_2 = np.array([0.08276529065121335, 0.08276529065121335, 0.08276529065121335, 0.09366770256266692, 0.10587023095800066, 0.08276529065121335, 0.08276529065121335, 0.08276529065121335])
Texas_axis_errors_by_025_2 = np.array([0.05726215705250706, 0.05726215705250706, 0.0544281350161918, 0.05761541245160175, 0.07368686503307581, 0.0613671580754105, 0.0613671580754105, 0.06344798869066312])
Wisconsin_axis_errors_by_025_2 = np.array([0.03788595227761946, 0.03788595227761946, 0.03788595227761946, 0.03788595227761946, 0.06552267207765823, 0.03788595227761946, 0.041799416876652264, 0.041799416876652264])

x_tick_labels_by_025 = [r'0.0 - 0.25', r'0.25 - 0.5', r'0.5 - 0.75', r'0.75 - 1.0', r'1.0 - 1.25', r'1.25 - 1.5', r'1.5 - 1.75', r'1.75 - 2.0']

bar1_right_1 = plt.bar(x_axis_values_by_025, Cornell_axis_values_by_025_2, 29, color='r', capsize=4, ecolor='k', label='ASL', yerr=Cornell_axis_errors_by_025_2)
bar2_right_1 = plt.bar(x_axis_values2_by_025, Texas_axis_values_by_025_2, 29, color='c', hatch='/', capsize=4, ecolor='k', label='Fisher', yerr=Texas_axis_errors_by_025_2)
bar3_right_1 = plt.bar(x_axis_values3_by_025, Wisconsin_axis_values_by_025_2, 29, color='m', hatch='*', capsize=4, ecolor='k', label='TBlo', yerr=Wisconsin_axis_errors_by_025_2)



locs, labels = plt.xticks(x_axis_values2_by_025,x_tick_labels_by_025, fontsize=18)
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
locs, labels = plt.yticks(fontsize=18)
plt.setp(labels)


#----------------------------------------------------------
# plot RR
plt.subplot(2,3,4)
plt.title("(d) 1-layer GNAN, $step = 1.0$", fontsize=19)
# plt.xlabel("Ablated frequency range", fontsize=18)
plt.ylabel("Micro-F1", fontsize=18)
plt.ylim(ymin= 0.5, ymax = 0.94)

x_axis_values_by_1 = [1430-(400*width), 1430-(360*width), 1430-(320*width)]
x_axis_values2_by_1 = [1430-(400*width)+30, 1430-(360*width)+30, 1430-(320*width) + 30]
x_axis_values3_by_1 = [1430-(400*width)+60, 1430-(360*width)+60, 1430-(320*width) + 60]

Cornell_axis_values_by_1_1 = np.array([0.8054054054054054, 0.8, 0.7162162162162162])
Texas_axis_values_by_1_1 = np.array([0.827027027027027, 0.7621621621621621, 0.8027027027027028])
Wisconsin_axis_values_by_1_1 = np.array([0.8431372549019607, 0.8235294117647058, 0.5725490196078431])

Cornell_axis_errors_by_1_1 = np.array([0.05524219717749873,  0.05435352478918329, 0.0929719254078842])
Texas_axis_errors_by_1_1 = np.array([0.057262157052507055, 0.047329461393022354, 0.05260829304741061])
Wisconsin_axis_errors_by_1_1 = np.array([0.05228758169934639, 0.05697907115739442, 0.07087812659725991])

bar1 = plt.bar(x_axis_values_by_1, Cornell_axis_values_by_1_1, 17, color='r', capsize=4, ecolor='k', label='Cornell', yerr=Cornell_axis_errors_by_1_1)
bar2 = plt.bar(x_axis_values2_by_1, Texas_axis_values_by_1_1, 17, color='c', hatch='/', capsize=4, ecolor='k', label='Texas', yerr=Texas_axis_errors_by_1_1)
bar3 = plt.bar(x_axis_values3_by_1, Wisconsin_axis_values_by_1_1, 17, color='m', hatch='*', capsize=4, ecolor='k', label='Wisconsin', yerr=Wisconsin_axis_errors_by_1_1)

x_tick_labels_by_1 = [r'no ablation', r'0.0 - 1.0', r'1.0 - 2.0']
locs, labels = plt.xticks(x_axis_values2_by_1, x_tick_labels_by_1, fontsize=18)
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
locs, labels = plt.yticks(fontsize=18)
plt.setp(labels)

#----------------------------------------------------------
# plot PC
plt.subplot(2,3,5)
plt.title("(e) 1-layer GNAN, $step = 0.5$", fontsize=19)
plt.xlabel("Ablated frequency range", fontsize=19)
plt.ylim(ymin= 0.5, ymax = 0.94)

x_axis_values_by_05 = [1430-(400*width), 1430-(360*width), 1430-(320*width), 1430-(280*width)]
x_axis_values2_by_05 = [1430-(400*width)+30, 1430-(360*width)+30, 1430-(320*width)+30, 1430-(280*width)+30]
x_axis_values3_by_05 = [1430-(400*width)+60, 1430-(360*width)+60, 1430-(320*width)+60, 1430-(280*width)+60]

Cornell_axis_values_by_05_1 = np.array([0.8027027027027028, 0.7729729729729728, 0.754054054054054, 0.8027027027027028])
Texas_axis_values_by_05_1 = np.array([0.8216216216216218, 0.7405405405405405, 0.827027027027027, 0.8243243243243243])
Wisconsin_axis_values_by_05_1 = np.array([0.8431372549019607, 0.8431372549019607, 0.596078431372549, 0.8392156862745097])

Cornell_axis_errors_by_05_1 = np.array([0.05260829304741058, 0.06136715807541046, 0.0736868650330758, 0.054129077065945254])
Texas_axis_errors_by_05_1 = np.array([0.05283920107609076, 0.05283920107609073, 0.057262157052507055, 0.054428135016191787])
Wisconsin_axis_errors_by_05_1 = np.array([0.05228758169934639, 0.05228758169934639, 0.07463582966598795, 0.053738051741423445])

bar1 = plt.bar(x_axis_values_by_05, Cornell_axis_values_by_05_1, 22, color='r', capsize=4, ecolor='k', label='Cornell', yerr=Cornell_axis_errors_by_05_1)
bar2 = plt.bar(x_axis_values2_by_05, Texas_axis_values_by_05_1, 22, color='c', hatch='/', capsize=4, ecolor='k', label='Texas', yerr=Texas_axis_errors_by_05_1)
bar3 = plt.bar(x_axis_values3_by_05, Wisconsin_axis_values_by_05_1, 22, color='m', hatch='*', capsize=4, ecolor='k', label='Wisconsin', yerr=Wisconsin_axis_errors_by_05_1)

x_tick_labels_by_05 = [r'0.0 - 0.5', r'0.5 - 1.0', r'1.0 - 1.5', r'1.5 - 2']

locs, labels = plt.xticks(x_axis_values2_by_05,x_tick_labels_by_05, fontsize=18)
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
locs, labels = plt.yticks(fontsize=18)
plt.setp(labels)
#----------------------------------------------------------
# plot PQ

plt.subplot(2,3,6)
plt.title("(f) 1-layer GNAN, $step = 0.25$", fontsize=19)
# plt.xlabel("Ablated frequency range", fontsize=18)
plt.ylim(ymin= 0.5, ymax = 0.94)

x_axis_values_by_025 = [1430-(400*width), 1430-(375*width), 1430-(350*width), 1430-(325*width), 1430-(300*width), 1430-(275*width), 1430-(250*width), 1430-(225*width)]
x_axis_values2_by_025 = [1430-(400*width)+38, 1430-(375*width)+38, 1430-(350*width)+38, 1430-(325*width)+38, 1430-(300*width)+38, 1430-(275*width)+38, 1430-(250*width)+38, 1430-(225*width)+38]
x_axis_values3_by_025 = [1430-(400*width)+76, 1430-(375*width)+76, 1430-(350*width)+76, 1430-(325*width)+76, 1430-(300*width)+76, 1430-(275*width)+76, 1430-(250*width)+76, 1430-(225*width)+76]


Cornell_axis_values_by_025_1 = np.array([0.8054054054054054, 0.8054054054054054, 0.8054054054054054, 0.7972972972972973, 0.7432432432432433, 0.8054054054054054, 0.8054054054054054, 0.8081081081081081])
Texas_axis_values_by_025_1 = np.array([0.827027027027027, 0.827027027027027, 0.8243243243243243, 0.7378378378378379, 0.827027027027027, 0.827027027027027, 0.8243243243243243, 0.818918918918919])
Wisconsin_axis_values_by_025_1 = np.array([0.8431372549019607, 0.8431372549019607, 0.8431372549019607, 0.8431372549019607, 0.6058823529411764, 0.8431372549019607, 0.8470588235294118, 0.8470588235294118])


Cornell_axis_errors_by_025_1 = np.array([0.05524219717749873, 0.05524219717749873, 0.05524219717749873, 0.05442813501619177, 0.048094947081222125, 0.05524219717749873, 0.05524219717749873, 0.05761541245160176])
Texas_axis_errors_by_025_1 = np.array([0.057262157052507055, 0.057262157052507055, 0.054428135016191787, 0.05104221792510015, 0.057262157052507055, 0.057262157052507055, 0.054428135016191787, 0.054129077065945254])
Wisconsin_axis_errors_by_025_1 = np.array([0.05228758169934639, 0.05228758169934639, 0.05228758169934639, 0.05228758169934639, 0.06757676876111285, 0.05228758169934639, 0.04785099403326307, 0.04785099403326307])

x_tick_labels_by_025 = [r'0.0 - 0.25', r'0.25 - 0.5', r'0.5 - 0.75', r'0.75 - 1.0', r'1.0 - 1.25', r'1.25 - 1.5', r'1.5 - 1.75', r'1.75 - 2.0']

bar1_right_2 = plt.bar(x_axis_values_by_025, Cornell_axis_values_by_025_1, 29, color='r', capsize=4, ecolor='k', label='ASL', yerr=Cornell_axis_errors_by_025_1)
bar2_right_2 = plt.bar(x_axis_values2_by_025, Texas_axis_values_by_025_1, 29, color='c', hatch='/', capsize=4, ecolor='k', label='Fisher', yerr=Texas_axis_errors_by_025_1)
bar3_right_2 = plt.bar(x_axis_values3_by_025, Wisconsin_axis_values_by_025_1, 29, color='m', hatch='*', capsize=4, ecolor='k', label='TBlo', yerr=Wisconsin_axis_errors_by_025_1)

locs, labels = plt.xticks(x_axis_values2_by_025,x_tick_labels_by_025, fontsize=18)
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
locs, labels = plt.yticks(fontsize=18)
plt.setp(labels)

plt.ylim(ymin= 0.5, ymax = 0.94)


idx = [0, 1, 6, 7]
for i in idx:
    bar1_right_1.patches[i].set_alpha(0.2)
    bar2_right_1.patches[i].set_alpha(0.2)
    bar3_right_1.patches[i].set_alpha(0.2)
    bar1_right_2.patches[i].set_alpha(0.2)
    bar2_right_2.patches[i].set_alpha(0.2)
    bar3_right_2.patches[i].set_alpha(0.2)


# plt.show()

plt.savefig('./disassortative_frequenc_ablation.pdf', format='pdf')

# plt.savefig('./disassortative_frequenc_ablation.eps', bbox_inches='tight')
plt.clf()

