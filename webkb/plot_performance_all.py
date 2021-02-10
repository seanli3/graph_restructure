# NCVoter

import math
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# =================================================================================================
# Generate plot 

width = 3.5    # the width of the bars
x_tick_labels = ['Cora', 'Restaurant', 'DBLP-ACM', 'NCVR']

x_axis_values = [1430-(400*width), 1430-(360*width), 1430-(320*width), 1430-(280*width)] 
x_axis_values2 = [1430-(400*width)+25, 1430-(360*width)+25, 1430-(320*width)+25, 1430-(280*width)+25]
x_axis_values3 = [1430-(400*width)+50, 1430-(360*width)+50, 1430-(320*width)+50, 1430-(280*width)+50]
x_axis_values4 = [1430-(400*width)+75, 1430-(360*width)+75, 1430-(320*width)+75, 1430-(280*width)+75]

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 18}

plt.rc('font', **font)

print('Generate bar Plot')

#w,h = plt.figaspect(plot_aspect)


#---------------------------------------------------
# RR


ASL_axis_values_rr = np.array([0.9602, 0.9997, 0.9998, 0.9802]) 
Base_axis_values_rr = np.array([0.9173, 0.9975, 0.9885, 0.9940])  
TBlo_axis_values_rr = np.array([0.9900, 0.9677, 0.9935, 0.9853]) 
RSL_axis_values_rr = np.array([0.9558, 0.9883, 0.9939, 0.9865])


#-------------------------------------------------------------------------------
# PC

ASL_axis_values_pc = np.array([0.8644, 0.8929, 0.4335, 0.9993]) 
Base_axis_values_pc = np.array([0.9249, 0.9375, 0.9928, 1.0000])  
TBlo_axis_values_pc = np.array([0.3296, 0.9911, 0.4388, 1.0000]) 
RSL_axis_values_pc = np.array([(ASL_axis_values_pc[0]+Base_axis_values_pc[0]+TBlo_axis_values_pc[0])/3, (ASL_axis_values_pc[1]+Base_axis_values_pc[1]+TBlo_axis_values_pc[1])/3, (ASL_axis_values_pc[2]+Base_axis_values_pc[2]+TBlo_axis_values_pc[2])/3, (ASL_axis_values_pc[3]+Base_axis_values_pc[3]+TBlo_axis_values_pc[3])/3])
#-------------------------------------------------------------------------------
# PQ

ASL_axis_values_pq = np.array([0.4455, 0.8000, 0.7309, 0.7409]) 
Base_axis_values_pq = np.array([0.2294, 0.1116, 0.0320, 0.0403])  
TBlo_axis_values_pq = np.array([0.6758, 0.0920, 0.0564, 0.0164])  
RSL_axis_values_pq = np.array([(ASL_axis_values_pq[0]+Base_axis_values_pq[0]+TBlo_axis_values_pq[0])/3, (ASL_axis_values_pq[1]+Base_axis_values_pq[1]+TBlo_axis_values_pq[1])/3, (ASL_axis_values_pq[2]+Base_axis_values_pq[2]+TBlo_axis_values_pq[2])/3, (ASL_axis_values_pq[3]+Base_axis_values_pq[3]+TBlo_axis_values_pq[3])/3])


#-------------------------------------------------------------------------------
# FM

ASL_axis_values_fm = np.array([(0.8644 * 0.4455 * 2)/ (0.8644 + 0.4455), (0.8929 * 0.8000 * 2)/ (0.8929 + 0.8000), (0.4335 * 0.7309 * 2)/ (0.4335 + 0.7309), (0.9993 * 0.7409 * 2)/ (0.9993 + 0.7409)]) 
Base_axis_values_fm = np.array([(0.9249 * 0.2294 * 2)/ (0.9249 + 0.2294), (0.9375 * 0.1116 * 2)/ (0.9375 + 0.1116), (0.9928 * 0.0320 * 2)/ (0.9928 + 0.0320), (1.0000 * 0.0403 * 2)/ (1.0000+ 0.0403)])  
TBlo_axis_values_fm = np.array([(0.3296 * 0.6758 * 2)/ (0.3296 + 0.6758), (0.9911 * 0.0920 * 2)/ (0.9911 + 0.0920), (0.4388 * 0.0564 * 2)/ (0.4388 + 0.0564), (1.0000 * 0.0164 * 2)/ (1.0000 + 0.0164)])  
RSL_axis_values_fm = np.array([(RSL_axis_values_pc[0] * RSL_axis_values_pq[0] * 2)/ (RSL_axis_values_pc[0] + RSL_axis_values_pq[0]), (RSL_axis_values_pc[1] * RSL_axis_values_pq[1] * 2)/ (RSL_axis_values_pc[1] + RSL_axis_values_pq[1]), (RSL_axis_values_pc[2] * RSL_axis_values_pq[2] * 2)/ (RSL_axis_values_pc[2] + RSL_axis_values_pq[2]), (RSL_axis_values_pc[3] * RSL_axis_values_pq[3] * 2)/ (RSL_axis_values_pc[3] + RSL_axis_values_pq[3])]) 


print('Generate accuracy plot - against budget')




#w,h = plt.figaspect(0.2)
plt.figure(figsize=(30,6)) 
plt.xscale("linear")
plt.yscale("linear")

plt.xlim(xmin=0, xmax = 1430-(310*width)+160)
plt.ylim(ymin= 0.0, ymax = 1.0) 


#----------------------------------------------------------
# plot RR
plt.subplot(1,4,1)
plt.title("(a) RR", fontsize=25)
plt.xlabel("Datasets", fontsize=25)
plt.ylabel("Reduction ratio", fontsize=25)
bar1 = plt.bar(x_axis_values, ASL_axis_values_rr, [11.0, 11.0, 11.0, 11.0], color='r', capsize=4, ecolor='k', label='ASL')  
bar2 = plt.bar(x_axis_values2, Base_axis_values_rr, [11.0, 11.0, 11.0, 11.0], color='c', hatch='/', capsize=4, ecolor='k', label='Fisher')  
bar3 = plt.bar(x_axis_values3, TBlo_axis_values_rr, [11.0, 11.0, 11.0, 11.0], color='LightGreen', hatch='o', capsize=4, ecolor='k', label='TBlo')  
bar4 = plt.bar(x_axis_values4, RSL_axis_values_rr, [11.0, 11.0, 11.0, 11.0], color='m', hatch='*', capsize=4, ecolor='k', label='RSL') 

locs, labels = plt.xticks(x_axis_values2,x_tick_labels, fontsize=15)
locs, labels = plt.yticks(fontsize=18)
plt.setp(labels)

plt.legend(loc= 'lower left', prop={'size':18})
#----------------------------------------------------------
# plot PC
plt.subplot(1,4,2)
plt.title("(b) PC", fontsize=25)
plt.xlabel("Datasets", fontsize=25)
plt.ylabel("Pairs completeness", fontsize=25)

bar1 = plt.bar(x_axis_values, ASL_axis_values_pc, [11.0, 11.0, 11.0, 11.0], color='r', capsize=4, ecolor='k', label='ASL')  
bar2 = plt.bar(x_axis_values2, Base_axis_values_pc, [11.0, 11.0, 11.0, 11.0], color='c', hatch='/', capsize=4, ecolor='k', label='Fisher')  
bar3 = plt.bar(x_axis_values3, TBlo_axis_values_pc, [11.0, 11.0, 11.0, 11.0], color='LightGreen', hatch='o', capsize=4, ecolor='k', label='TBlo')  
bar4 = plt.bar(x_axis_values4, RSL_axis_values_pc, [11.0, 11.0, 11.0, 11.0], color='m', hatch='*', capsize=4, ecolor='k', label='RSL')

locs, labels = plt.xticks(x_axis_values2,x_tick_labels, fontsize=15)
locs, labels = plt.yticks(fontsize=18)
plt.setp(labels)
#----------------------------------------------------------
# plot PQ

plt.subplot(1,4,3)
plt.title("(c) PQ", fontsize=25)
plt.xlabel("Datasets", fontsize=25)
plt.ylabel("Pairs quality", fontsize=25)

bar1 = plt.bar(x_axis_values, ASL_axis_values_pq, [11.0, 11.0, 11.0, 11.0], color='r', capsize=4, ecolor='k', label='ASL')  
bar2 = plt.bar(x_axis_values2, Base_axis_values_pq, [11.0, 11.0, 11.0, 11.0], color='c', hatch='/', capsize=4, ecolor='k', label='Fisher')  
bar3 = plt.bar(x_axis_values3, TBlo_axis_values_pq, [11.0, 11.0, 11.0, 11.0], color='LightGreen', hatch='o', capsize=4, ecolor='k', label='TBlo')  
bar4 = plt.bar(x_axis_values4, RSL_axis_values_pq, [11.0, 11.0, 11.0, 11.0], color='m', hatch='*', capsize=4, ecolor='k', label='RSL')

locs, labels = plt.xticks(x_axis_values2,x_tick_labels, fontsize=15)
locs, labels = plt.yticks(fontsize=18)
plt.setp(labels)

plt.ylim(ymin= 0.0, ymax = 1.0) 

#----------------------------------------------------------
# plot FM

plt.subplot(1,4,4)
plt.title("(d) FM", fontsize=25)
plt.xlabel("Datasets", fontsize=25)
plt.ylabel("F-measure", fontsize=25)

bar1 = plt.bar(x_axis_values, ASL_axis_values_fm, [11.0, 11.0, 11.0, 11.0], color='r', capsize=4, ecolor='k', label='ASL')  
bar2 = plt.bar(x_axis_values2, Base_axis_values_fm, [11.0, 11.0, 11.0, 11.0], color='c', hatch='/', capsize=4, ecolor='k', label='Fisher')  
bar3 = plt.bar(x_axis_values3, TBlo_axis_values_fm, [11.0, 11.0, 11.0, 11.0], color='LightGreen', hatch='o', capsize=4, ecolor='k', label='TBlo')  
bar4 = plt.bar(x_axis_values4, RSL_axis_values_fm, [11.0, 11.0, 11.0, 11.0], color='m', hatch='*', capsize=4, ecolor='k', label='RSL')

locs, labels = plt.xticks(x_axis_values2,x_tick_labels, fontsize=15)
locs, labels = plt.yticks(fontsize=18)
plt.setp(labels)
plt.ylim(ymin= 0.0, ymax = 1.0) 






plt.savefig('./All_performance.pdf', format='pdf')

plt.savefig('./All_performance.eps', bbox_inches='tight')
plt.clf()

