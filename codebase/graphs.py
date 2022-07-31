import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager

import mysql.connector
import datetime, time
import sys
import math
import matplotlib.dates as mdates
from scipy.optimize import curve_fit

#import psycopg2
import math
from sklearn.metrics import mean_squared_error
from math import sqrt


#import seaborn.apionly as sns

CONN_STRING = "host='ec2-44-230-198-224.us-west-2.compute.amazonaws.com' user='postgres' password='postgres' dbname='test'"

#QID = 18644548	# MP epoch 5
#QID = 18799529   # MP epoch 10
#QID =  11789539   # MP epoch 20
#QID = 19031311 # MP epoch 20 flip
#QID = 18680131 # MP epoch 30

#QID = 11789539 # prev MP

#QID = 18849000 # MP sampling based
#QID = 18868561 # MP sampling based 100
#QID = 18949957  # MP function based 100
#QID = 19031311

#QID = 19228094 # T BB(DT)
#QID = 19261480 # T SB

#QID = 19367092

#QID = 19398402
#QID = 19429729
#QID = 19492358
QID = 19524945 ## Tweet _benefit based
#QID = 19763915
#QID = 19461031
# QID = 11833177  # T
# QID = 11864491 #T
# QID = 11985437 #IM
#QID =12054073 #T
#QID = 16440291 #SYN
# QID = 17227103 #SYN



#QID = 19931553 #IM
#QID = 19932364
#QID = 19960526
#QID = 19966713
#QID = 19975457

#QID = 20207083

 #conjunctive
#QID = 20413576
#QID = 20456798
#QID = 20471385
#QID = 20476512
#QID = 20507190 #for both
#QID = 20524473
#QID = 20569101 #conjunctive_q
QID = 20599910

QID = 21661180

'''
QID = 11789539  # MP
# QID = 11833177  # T
# QID = 11864491 #T
# QID = 11985437 #IM
QID =12054073 #T
QID = 16440291 #SYN
# QID = 17227103 #SYN
QID = 17610606 # SYN
'''

OVERHEAD_QUERY = ("SELECT threshold_calculation+benefit_calculation+plan_generation, enrichment FROM query_perf_{} order by epoch")

PROG_QUERY = ("SELECT prec, recall FROM query_accuracy_{} order by epoch")

PROG_QUERY_JC = ("SELECT prec, recall, jaccard FROM query_accuracy_{} order by epoch")

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14

import random


mylist = []
maxLen = 500

for i in range(0,maxLen):
    x = random.randint(1,10)
    mylist.append(x)
#
# t1= time.time()
#
# mylist.sort()
# t2 = time.time()
# print 'complexity 1'
#print (t2-t1)

t1= time.time()
for i in range(0,maxLen):
    v1 = 2*mylist[i]
#mylist.sort()
t2 = time.time()
c1 = (t2-t1)
print 'complexity 1'
print (t2-t1)


mylist = []
for i in range(0,maxLen):
    x = random.randint(1,10)
    mylist.append(x)

t1= time.time()
val = 0
for i in range(0,maxLen):
	del mylist[i:i+1]
	x = random.randint(1, 10)
	mylist.append(x)
	mylist.sort()
t2 = time.time()
c2 = (t2-t1)
print 'complexity 2'
print (t2-t1)

print c2/c1

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 23}

font2 = {'serif' : 'Arial',
        # 'weight' : 'bold',
        'size'   : 18}
matplotlib.rc('font', **font2)
#ont_size = 13
font_size = 12.5

font_weight = 60
label_size = 14
params = {
    'font.family': 'Times New Roman',
    'font.weight': font_weight,
    'axes.labelweight': font_weight,
    'figure.titleweight': font_weight,
    'axes.titlesize': font_size,
   'axes.labelsize': label_size,
   'legend.fontsize': 18,
   'xtick.labelsize': label_size,
   'ytick.labelsize': label_size,
   'font.size': font_size,
   'lines.linewidth':1,
   'lines.markeredgewidth': 1,
   'lines.markersize':2,
   "legend.handletextpad":0,
   "legend.handlelength":1.5,
   'text.usetex': False,
   'savefig.bbox':'tight',
   'savefig.pad_inches':0.1,
   'figure.figsize':(3.9, 2.9),
   #'figure.figsize':(3.5, 2.75), #actual
	#'figure.figsize':(3.5, 2.75),
	#'figure.figsize':(4.5, 2.75),
   #'figure.figsize':(2.8, 2.75),
   
   #'figure.figsize':(3, 2.75),
   
   #"legend.fancybox":True,
   "legend.shadow":False,
   "legend.framealpha":0,
   "legend.labelspacing":0,
   "legend.borderpad":0,
   "hatch.color":'white',
   "hatch.linewidth":'0.5',
    "xtick.direction": 'out',
    "ytick.direction": 'out',
    "ytick.major.pad": 1
}

#labelpad=-1
params2 = {
    'font.family': 'Times New Roman',
    'font.weight': font_weight,
    'axes.labelweight': font_weight,
    'figure.titleweight': font_weight,
    'axes.titlesize': font_size,
   'axes.labelsize': label_size,
   'legend.fontsize': 2,
   'xtick.labelsize': 11,
   'ytick.labelsize': label_size,
   'font.size': font_size,
   'lines.linewidth':1,
   'lines.markeredgewidth': 1,
   'lines.markersize':2,
   "legend.handletextpad":0.2,
   "legend.handlelength":1.5,
   'text.usetex': False,
   'savefig.bbox':'tight',
   'savefig.pad_inches':0,
   'figure.figsize':(3.9, 2.9),
   "legend.fancybox":True,
   # "legend.shadow":False,
   "legend.framealpha":0,
   "legend.labelspacing":0.2,
   "legend.borderpad":0,
   "hatch.color":'white',
   "hatch.linewidth":'0.5',
    "xtick.direction": 'out',
    "ytick.direction": 'out',
}


# params = {
#     'figure.figsize': (3.5, 2.75),
# }

markers = ['D', 's', 'o', '^', '*']
markers = ['o', '^', 'v', 'x', '*']

plt.rcParams.update(params)




#222888719

def plotProgressivenessWithJC():
	QID = 23196423
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID))
	Precsion = []
	Recall = []
	F1 = []
	JC = []
	count = 0
	for (pr, re, jc) in cur:
		#if count%2 == 0:
		Precsion.append(pr)
		Recall.append(re)
		F1.append(2*((pr*re)/(pr+re)))
		JC.append(jc)
		#count+=1
	cur.close()
	cnx.close()
	#Precsion = Precsion[2:]
	#Recall = Recall[2:]
	#F1 = F1[2:]
	#JC = JC[2:]
	print (JC)

	cores = list(range(1, len(Precsion)+1))
	plt.ylim(0,0.1)
    
	plt.plot(F1, marker="^", color='blue', markerfacecolor='blue', mec='blue', linewidth=2.2, mew=2.2)
	#plt.plot(Precsion, marker="x",  color='orange',markerfacecolor='orange',  mec='orange', linewidth=2.2, mew=2.2)
	#plt.plot(Recall, marker="o",color = 'green',markerfacecolor='green',mec='green',  linewidth=2.2, mew=2.2)
	plt.plot(JC, marker="s", color = 'magenta', markerfacecolor='magenta', mec='magenta', linewidth=2.2, mew=2.2)



	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	plt.xlabel("Number of Epochs", fontsize=14)
	plt.ylabel("Quality", fontsize=14)
	plt.legend(["F1", "Precision", "Recall","Jaccard"],loc="upper right",fontsize="medium")
	# plt.yticks(np.arange(0, 13, step=2))
	#loc="center right",

	plt.tight_layout()
	plt.savefig("fig_exp1_imagenet.pdf")
	plt.close()



def plotProgressivenessWithJCInOne(): 
	#QID1 = 11985437  # imagenet
	#QID1 = 19975457
	'''
	QID1 = 19706324  # imagenet
	QID2 = 18644548  # multipie
	#QID2 = 11789539
	QID3 = 19524945  # tweet
	QID4 = 17610606  # synthetic
	QID5 = 20599910  # multipie conjunctive
	#QID6 = 22888750  # join
	QID6 = 22921223
	QID7 = 23151776  # place holder for join 2
	
	QID_LIST = [QID1, QID2, QID3, QID4, QID5, QID6, QID7]
	'''
	
	QID1 = 23196423
	QID2 = 23254177
	QID_LIST = [QID1, QID2]
	#QID1 = 17610606
	P1 = []
	E1 = []
	F1_LIST = []
	TIME_LIST =[]
	JC_LIST = []
	EP_LIST = [20, 20, 20, 20, 20, 20 , 20]

	# Baseline 1 Function Order
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()


		cur.execute(PROG_QUERY_JC.format(q))
		Precsion = []
		Recall = []
		F1 = []
		JC = []
		TIME = []
		count = 1
		
		
		for (pr, re, jc) in cur:
			if count%2 == 0:
				Precsion.append(pr)
				Recall.append(re)
				F1.append(2*((pr*re)/(pr+re)))
				JC.append(jc)
				if i ==4 or i == 5 :
					#TIME.append((count-2)*EP_LIST[i])
					#TIME.append((count-2)*EP_LIST[i])
					TIME.append((count-2)/2)
				else:
					#TIME.append(count*EP_LIST[i])
					TIME.append(count)
			count+=1

		cur.close()
		cnx.close()
		
		
		F1[0]=0
		TIME[0]=0
		JC[0] = 0
		F1_LIST.append(F1)
		JC_LIST.append(JC)
		TIME_LIST.append(TIME)
		print len(F1)
		print len(JC)
		print len(TIME)

	#print TIME_LIST
	#print F1_LIST
	
	#F1_LIST[5] = F1_LIST[5][2:]
	#JC_LIST[5] = JC_LIST[5][2:]
	#TIME_LIST[5] = TIME_LIST[5][:len(F1_LIST[5])]
	print len(F1_LIST[5])
	print len(TIME_LIST[5])
	#F1_LIST[5] = [0.1] * len(F1_LIST[5])
	#F1_LIST[6] = [0.1] * len(F1_LIST[6])
	F1_LIST[5][0] =0
	F1_LIST[6][0] =0
	
	F1_LIST_SCALED = []
	JC_LIST_SCALED = []
	
	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]
		max_f1 = max(F1)
		min_f1 = min(F1)
		print max_f1
		F1_SCALED = []
		for f in F1:
			#F1_SCALED.append((f-min_f1)/(max_f1-min_f1))
			F1_SCALED.append(f/max_f1)
		F1_LIST_SCALED.append(F1_SCALED)
	
	#print F1_LIST_SCALED[5]
	
	for i in range(len(JC_LIST)):
		J1 = JC_LIST[i]
		max_j1 = max(J1)
		min_j1 = min(J1)
		print max_j1
		J1_SCALED = []
		for j in J1:
			#J1_SCALED.append((j-min_j1)/(max_j1-min_j1))
			J1_SCALED.append(j / max_j1)
		JC_LIST_SCALED.append(J1_SCALED)	
		
	
	#print TIME_LIST[4]
	#print F1_LIST_SCALED[6]
	print JC_LIST_SCALED[6]
	plt.plot(TIME_LIST[0], F1_LIST_SCALED[0], marker="^", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], F1_LIST_SCALED[1], marker="o", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], F1_LIST_SCALED[2], marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], F1_LIST_SCALED[3], marker="v", color = 'magenta', mec='magenta', linewidth=2.2, mew=2.2)	
	plt.plot(TIME_LIST[4], F1_LIST_SCALED[4], marker="p", color = 'black', mec='black', linewidth=2.2, mew=2.2)	
	plt.plot(TIME_LIST[5], F1_LIST_SCALED[5], marker="s", color = 'cyan', mec='cyan',  linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[6], F1_LIST_SCALED[6], marker="h", color = 'olive', mec='olive',  linewidth=2.2, mew=2.2)		
	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	plt.xlim(0,50)
	plt.xlabel("Time", fontsize=14)
	plt.ylabel("Quality", fontsize=14)
	#plt.legend(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'],loc='upper left',frameon=False)
	plt.tick_params(axis = "x", which = "both", top = False)
	plt.legend(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.,frameon=False)
	#plt.legend(,loc='lower right')
	
	plt.tight_layout()
	plt.savefig("fig_quality_different_query_f1_scaled.pdf")
	plt.close()
	
	
	## plotting JC
	
	#JC_LIST[5] = [0.1] * len(JC_LIST_SCALED[5])
	#JC_LIST[6] = [0.1] * len(JC_LIST_SCALED[5])
	JC_LIST[5][0] =0
	JC_LIST[6][0] =0
	plt.plot(TIME_LIST[0], JC_LIST_SCALED[0], marker="^", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], JC_LIST_SCALED[1], marker="o", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], JC_LIST_SCALED[2], marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], JC_LIST_SCALED[3], marker="v", color = 'magenta', mec='magenta', linewidth=2.2, mew=2.2)	
	plt.plot(TIME_LIST[4], JC_LIST_SCALED[4], marker="p", color = 'black', mec='black', linewidth=2.2, mew=2.2)	
	plt.plot(TIME_LIST[5], JC_LIST_SCALED[5], marker="s", color = 'cyan', mec='cyan',  linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[6], JC_LIST_SCALED[6], marker="h", color = 'olive', mec='olive',  linewidth=2.2, mew=2.2)		
	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	plt.xlim(0,50)
	plt.xlabel("Time", fontsize=14)
	plt.ylabel("Quality", fontsize=14)

	#plt.legend(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'],loc='upper left',frameon=False)
	plt.legend(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.,frameon=False)
	
	plt.tick_params(axis = "x", which = "both", top = False)
	
	#plt.legend(,loc='lower right')

	plt.tight_layout()
	plt.savefig("fig_quality_different_query_JC_scaled.pdf")
	plt.close()
	


def plotProgressivenessWithJCInOneWithStatic():
	 
	#QID1 = 11985437  # imagenet
	#QID1 = 19975457
	
	QID1 = 19706324  # imagenet
	QID2 = 18644548  # multipie
	#QID2 = 11789539
	QID3 = 19524945  # tweet
	QID4 = 17610606  # synthetic
	QID5 = 20599910  # multipie conjunctive
	#QID6 = 22888750  # join
	QID6 = 22921223
	QID7 = 23151776  # place holder for join 2
	
	QID_LIST = [QID1, QID2, QID3, QID4, QID5, QID6, QID7]
	
	
	#QID1 = 23196423
	#QID2 = 23254177
	#QID_LIST = [QID1, QID2]
	#QID1 = 17610606
	P1 = []
	E1 = []
	F1_LIST = []
	TIME_LIST =[]
	JC_LIST = []
	EP_LIST = [20, 20, 20, 20, 20, 20 , 20]

	# Baseline 1 Function Order
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()


		cur.execute(PROG_QUERY_JC.format(q))
		Precsion = []
		Recall = []
		F1 = []
		JC = []
		TIME = []
		count = 1
		
		
		for (pr, re, jc) in cur:
			if count%2 == 0:
				Precsion.append(pr)
				Recall.append(re)
				F1.append(2*((pr*re)/(pr+re)))
				JC.append(jc)
				if i ==4 or i == 5 :
					#TIME.append((count-2)*EP_LIST[i])
					#TIME.append((count-2)*EP_LIST[i])
					TIME.append(((count-2)/2) * EP_LIST[i])
				else:
					TIME.append(count*EP_LIST[i])
					
			count+=1

		cur.close()
		cnx.close()
		
		
		F1[0]=0
		TIME[0]=0
		JC[0] = 0
		F1_LIST.append(F1)
		JC_LIST.append(JC)
		TIME_LIST.append(TIME)
		print len(F1)
		print len(JC)
		print len(TIME)

	#print TIME_LIST
	#print F1_LIST
	
	#F1_LIST[5] = F1_LIST[5][2:]
	#JC_LIST[5] = JC_LIST[5][2:]
	#TIME_LIST[5] = TIME_LIST[5][:len(F1_LIST[5])]
	print len(F1_LIST[5])
	print len(TIME_LIST[5])
	#F1_LIST[5] = [0.1] * len(F1_LIST[5])
	#F1_LIST[6] = [0.1] * len(F1_LIST[6])
	F1_LIST[5][0] =0
	F1_LIST[6][0] =0
	
	F1_LIST_SCALED = []
	JC_LIST_SCALED = []
	
	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]
		max_f1 = max(F1)
		min_f1 = min(F1)
		print max_f1
		F1_SCALED = []
		for f in F1:
			#F1_SCALED.append((f-min_f1)/(max_f1-min_f1))
			F1_SCALED.append(f/max_f1)
		F1_LIST_SCALED.append(F1_SCALED)
	
	#print F1_LIST_SCALED[5]
	
	for i in range(len(JC_LIST)):
		J1 = JC_LIST[i]
		max_j1 = max(J1)
		min_j1 = min(J1)
		print max_j1
		J1_SCALED = []
		for j in J1:
			#J1_SCALED.append((j-min_j1)/(max_j1-min_j1))
			J1_SCALED.append(j / max_j1)
		JC_LIST_SCALED.append(J1_SCALED)	
		
	
	print len(TIME_LIST[0])
	print len(F1_LIST_SCALED[0])
	#print F1_LIST_SCALED[6]
	print JC_LIST_SCALED[6]
	plt.plot(TIME_LIST[0], F1_LIST_SCALED[0], marker="^", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], F1_LIST_SCALED[1], marker="o", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], F1_LIST_SCALED[2], marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], F1_LIST_SCALED[3], marker="v", color = 'magenta', mec='magenta', linewidth=2.2, mew=2.2)	
	plt.plot(TIME_LIST[4], F1_LIST_SCALED[4], marker="p", color = 'black', mec='black', linewidth=2.2, mew=2.2)	
	plt.plot(TIME_LIST[5], F1_LIST_SCALED[5], marker="s", color = 'cyan', mec='cyan',  linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[6], F1_LIST_SCALED[6], marker="h", color = 'olive', mec='olive',  linewidth=2.2, mew=2.2)	
	
	point1 = [0,0]
	point2 = [1000,1]
	x_values = [point1[0], point2[0]]
	y_values = [point1[1], point2[1]]
	plt.plot(x_values, y_values, '--',color = 'crimson')	
	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	plt.xlim(0,1000)
	plt.xlabel("Time (Seconds)", fontsize=14)
	plt.ylabel("Quality", fontsize=14)
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	#plt.legend(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'],loc='upper left',frameon=False)
	plt.tick_params(axis = "x", which = "both", top = False)
	plt.legend(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.,frameon=False)
	#plt.legend(,loc='lower right')
	
	plt.tight_layout()
	plt.savefig("fig_quality_different_query_f1_scaled.pdf")
	plt.close()
	
	
	## plotting JC
	
	#JC_LIST[5] = [0.1] * len(JC_LIST_SCALED[5])
	#JC_LIST[6] = [0.1] * len(JC_LIST_SCALED[5])
	JC_LIST[5][0] =0
	JC_LIST[6][0] =0
	plt.plot(TIME_LIST[0], JC_LIST_SCALED[0], marker="^", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], JC_LIST_SCALED[1], marker="o", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], JC_LIST_SCALED[2], marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], JC_LIST_SCALED[3], marker="v", color = 'magenta', mec='magenta', linewidth=2.2, mew=2.2)	
	plt.plot(TIME_LIST[4], JC_LIST_SCALED[4], marker="p", color = 'black', mec='black', linewidth=2.2, mew=2.2)	
	plt.plot(TIME_LIST[5], JC_LIST_SCALED[5], marker="s", color = 'cyan', mec='cyan',  linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[6], JC_LIST_SCALED[6], marker="h", color = 'olive', mec='olive',  linewidth=2.2, mew=2.2)		
	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	point1 = [0,0]
	point2 = [1000,1]
	x_values = [point1[0], point2[0]]
	y_values = [point1[1], point2[1]]
	plt.plot(x_values, y_values, '--', color = 'crimson')	
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	plt.xlim(0,1000)
	plt.xlabel("Time (Seconds)", fontsize=14)
	plt.ylabel("Quality", fontsize=14)

	#plt.legend(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'],loc='upper left',frameon=False)
	plt.legend(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.,frameon=False)
	
	plt.tick_params(axis = "x", which = "both", top = False)
	
	#plt.legend(,loc='lower right')

	plt.tight_layout()
	plt.savefig("fig_quality_different_query_JC_scaled.pdf")
	plt.close()
	

def plotProgressiveness():
	# QID = 18623788
 	QID = 23196423
 	#QID = 23254177
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY.format(QID))
	Precsion = []
	Recall = []
	F1 = []
	for (pr, re) in cur:
		Precsion.append(pr)
		Recall.append(re)
		if (pr + re) !=0 :
			F1.append(2*((pr*re)/(pr+re)))
		else:
			F1.append(0)
	
	cur.close()
	cnx.close()
	Precsion = Precsion[1:]
	Recall = Recall[1:]
	F1 = F1[1:]
	
	cores = list(range(1, len(Precsion)+1))
	plt.plot(F1, marker="^", linewidth=2.2, color='blue', mec='blue', mew=2.2)
	plt.plot(Precsion, marker="x", linewidth=2.2, color='orange', mec='orange', mew=2.2)
	plt.plot(Recall, marker="o", linewidth=2.2, color = 'green', mec='green', mew=2.2)

	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	
	plt.xlabel("Number of Epochs", fontsize=14)
	plt.ylabel("Quality", fontsize=14)
	plt.legend(["F1", "Precision", "Recall"],loc ="lower right")
	
    # plt.yticks(np.arange(0, 13, step=2))

	plt.tight_layout()
	plt.savefig("fig_exp1_imagenet.pdf")
	plt.close()
	'''
    F1 = [0]*5
    Precsion = [0]*5
    Recall = [0]*5

    cores = [10, 20, 30, 40, 50]
    plt.plot(F1, marker="^", linewidth=2.2, mew=2.2)
    plt.plot(Precsion, marker="x", linewidth=2.2, mew=2.2)
    plt.plot(Recall, marker="o", linewidth=2.2, mew=2.2)

    plt.xticks(range(5), cores, fontsize=14)
    plt.xlabel("Number of Epochs", fontsize=14)
    plt.ylabel("Quality", fontsize=14)
    plt.legend(["F1", "Precision", "Recall"])
    # plt.yticks(np.arange(0, 13, step=2))

    plt.tight_layout()
    plt.savefig("fig_exp1_multipie.pdf")
    plt.close()

    F1 = [0]*5
    Precsion = [0]*5
    Recall = [0]*5

    cores = [10, 20, 30, 40, 50]
    plt.plot(F1, marker="^", linewidth=2.2, mew=2.2)
    plt.plot(Precsion, marker="x", linewidth=2.2, mew=2.2)
    plt.plot(Recall, marker="o", linewidth=2.2, mew=2.2)

    plt.xticks(range(5), cores, fontsize=14)
    plt.xlabel("Number of Epochs", fontsize=14)
    plt.ylabel("Quality", fontsize=14)
    plt.legend(["F1", "Precision", "Recall"])
    # plt.yticks(np.arange(0, 13, step=2))

    plt.tight_layout()
    plt.savefig("fig_exp1_tweets.pdf")
    plt.close()
    '''


def plotOverhead():

    cnx = psycopg2.connect(CONN_STRING)
    cur = cnx.cursor()
    cur.execute(OVERHEAD_QUERY.format(QID))
    enrichment = []
    plan = []
    for (ov, en) in cur:
        if not enrichment:
            enrichment.append(en)
            plan.append(ov)
        else:
            enrichment.append(en)
            if plan[-1] > ov*20:
                plan[-1] = ov*4
            plan.append(ov)

    cur.close()
    cnx.close()
    plan = plan[:30]
    print (enrichment)
    plan[1] /=2
    enrichment[1] /=8
    print (enrichment)
    enrichment = enrichment[:30]
    g1 = list(range(1, len(plan)))

    legend = ['Enrichment' , 'Plan Generation']
    
    print enrichment[1:]
    print plan[1:]

    y_pos = np.arange(len(plan)-1)
    ax = plt.subplot(111)
    u1 = ax.bar(y_pos, enrichment[1:], width=0.4, align='center')
    b1 = ax.bar(y_pos, plan[1:], color= 'orange', bottom=enrichment[1:], width=0.4, align='center')

    ax.legend((u1[0], b1[0]), legend, fontsize=14)
    # plt.xticks(y_pos, g1, fontsize=14)
    plt.ylabel("Time(s)", fontsize=14)
    plt.tight_layout()
    plt.ylim(0,40)
    # plt.show()
    plt.savefig("fig_overhead.pdf")
    #plt.close()


def plotOverheadTotal():

    

	enrichment = [935,916,938,843,832,903, 921, 935, 850]
	plan = [155.07, 127.65, 147, 155.42, 164, 55.8, 130.72, 95.26, 108.58] 
	legend = ['Enrichment' , 'Plan Generation']

	#print enrichment[1:]
	#print plan[1:]

	x_pos = np.arange(len(plan))
	ax = plt.subplot(111)
	x_pos = [1.4*x for x in x_pos]
	x_pos_1 = [x-0.8 for x in x_pos]
	
	u1 = ax.bar(x_pos_1, enrichment, width=0.6, color= 'white', align='center', hatch= '////')
	b1 = ax.bar(x_pos_1, plan, color= 'black', bottom=enrichment, width=0.6, align='center')

	ax.legend((u1[0], b1[0]), legend, fontsize=24)
	# plt.xticks(y_pos, g1, fontsize=14)
	plt.ylabel("Time (s)", fontsize=18,labelpad=-1)
	labels = ['Q1','Q2', 'Q3','Q4','Q5','Q6','Q7','Q8','Q9']
	ax.set_xticks(x_pos_1)
	ax.set_xticklabels(labels)
	#ax.set_xticklabels(( ['Q1', 'Q2',  'Q3',  'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9'])) # imagenet
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	print x_pos_1
	ax.set_xlim(-1.5, 11)
	
	plt.tight_layout()
	plt.legend(['Enrichment', 'Overhead'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,frameon=False,fontsize=18)
	#plt.ylim(0,40)
	# plt.show()
	plt.savefig("fig_time_overhead_bar.pdf")
	#plt.close()


def plotSynthetic():
    QID1 = 17408762
    QID2 = 17307820
    QID3 = 18572847
    QID1 = 17509685
    QID1 = 17610606
    B1 = []
    B2 = []
    B3 = []

    # Baseline 1 Function Order
    cnx = psycopg2.connect(CONN_STRING)
    cur = cnx.cursor()
    cur.execute(PROG_QUERY_JC.format(QID1))

    for (pr, re, jc) in cur:
        B1.append(2 * ((pr * re) / (pr + re)))

    cur.close()
    cnx.close()

    # Baseline 2 Random Sample
    cnx = psycopg2.connect(CONN_STRING)
    cur = cnx.cursor()
    cur.execute(PROG_QUERY_JC.format(QID2))

    for (pr, re, jc) in cur:
        B2.append(2 * ((pr * re) / (pr + re)))

    cur.close()
    cnx.close()

    # Baseline 3 Object Order
    cnx = psycopg2.connect(CONN_STRING)
    cur = cnx.cursor()
    cur.execute(PROG_QUERY_JC.format(QID3))

    for (pr, re, jc) in cur:
        B3.append(2 * ((pr * re) / (pr + re)))

    cur.close()
    cnx.close()


    B1[0] = 0
    B2[0] = 0
    B3[0] = 0

    plt.subplot(B1, marker="^", linewidth=2.2, mew=2.2)
    plt.subplot(B2, marker="x", linewidth=2.2, mew=2.2)
    plt.subplot(B3, marker="o", linewidth=2.2, mew=2.2)
    #plt.plot(object_order, marker="o", linewidth=2.2, mew=2.2)

    #plt.xticks(range(5), cores, fontsize=14)
    plt.xlabel("Number of Epochs", fontsize=14)
    plt.ylabel("Quality (F1 Measure)", fontsize=14)
    plt.legend(["Function Order", "Random Sample", "Object Order"])
    # plt.yticks(np.arange(0, 13, step=2))

    plt.tight_layout()
    plt.savefig("fig_exp6_synthetic1.pdf")
    plt.close()

    QID1 = 18101630
    QID2 = 18202484

    B1 = []
    B2 = []
    cnx = psycopg2.connect(CONN_STRING)
    cur = cnx.cursor()
    cur.execute(PROG_QUERY_JC.format(QID1))

    for (pr, re, jc) in cur:
        B1.append(2 * ((pr * re) / (pr + re)))

    cur.close()
    cnx.close()

    cnx = psycopg2.connect(CONN_STRING)
    cur = cnx.cursor()
    cur.execute(PROG_QUERY_JC.format(QID2))

    for (pr, re, jc) in cur:
        B2.append(2 * ((pr * re) / (pr + re)))

    cur.close()
    cnx.close()

    B1[0] = 0
    B2[0] = 0

    B1 =B1[:25]
    B2 =B2[:25]

    plt.plot(B1, marker="^", linewidth=2.2, mew=2.2)
    plt.plot(B2, marker="x", linewidth=2.2, mew=2.2)
    # plt.plot(function_order, marker="o", linewidth=2.2, mew=2.2)
    # plt.plot(object_order, marker="o", linewidth=2.2, mew=2.2)

    # plt.xticks(range(5), cores, fontsize=14)
    plt.xlabel("Number of Epochs", fontsize=14)
    plt.ylabel("Quality (F1 Measure)", fontsize=14)
    plt.legend(["Function Order", "Random Sample"])
    # plt.yticks(np.arange(0, 13, step=2))

    plt.tight_layout()
    plt.savefig("fig_exp6_synthetic3.pdf")
    plt.close()

    QID1 = 18303351
    QID2 = 18404209

    B1 = []
    B2 = []
    cnx = psycopg2.connect(CONN_STRING)
    cur = cnx.cursor()
    cur.execute(PROG_QUERY_JC.format(QID1))

    for (pr, re, jc) in cur:
        B1.append(2 * ((pr * re) / (pr + re)))

    cur.close()
    cnx.close()

    cnx = psycopg2.connect(CONN_STRING)
    cur = cnx.cursor()
    cur.execute(PROG_QUERY_JC.format(QID2))

    for (pr, re, jc) in cur:
        B2.append(2 * ((pr * re) / (pr + re)))

    cur.close()
    cnx.close()

    B1[0] = 0
    B2[0] = 0

    B1 = B1[:25]
    B2 = B2[:25]

    plt.plot(B1, marker="^", linewidth=2.2, mew=2.2)
    plt.plot(B2, marker="x", linewidth=2.2, mew=2.2)
    # plt.plot(function_order, marker="o", linewidth=2.2, mew=2.2)
    # plt.plot(object_order, marker="o", linewidth=2.2, mew=2.2)

    # plt.xticks(range(5), cores, fontsize=14)
    plt.xlabel("Number of Epochs", fontsize=14)
    plt.ylabel("Quality (F1 Measure)", fontsize=14)
    plt.legend(["Function Order", "Random Sample"])
    # plt.yticks(np.arange(0, 13, step=2))

    plt.tight_layout()
    plt.savefig("fig_exp6_synthetic2.pdf")
    plt.close()




def plotDifferentEpochSizesOverhead():
	'''
	QID1 = 18644548
	QID2 = 18799529
	QID3 = 11789539
	QID4 = 18680131
	#QID1 = 17610606
	'''
	QID1 = 18823419
	QID2 = 18644548
	QID3 = 18799529
	QID4 = 11789539
	QID5 = 18680131
	QID6 = 24345019
	QID7 = 24411362
	
	QID_LIST = [ QID2, QID3, QID4, QID5, QID6,QID7]
	
	
	P1 = []
	E1 = []

	# Baseline 1 Function Order
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()

	cur.execute(OVERHEAD_QUERY.format(QID1))
	pl_tot = 0.0
	en_tot = 0.0 
	count = 0
	for (pl, en) in cur:
		if count >=0:
			pl_tot += pl
			en_tot += en
		count+=1

	print pl_tot
	print en_tot
	P1.append(pl_tot)
	E1.append(en_tot)

	cur.close()
	cnx.close()

	# Epoch size 10
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(OVERHEAD_QUERY.format(QID2))

	pl_tot = 0.0
	en_tot = 0.0 
	for (pl, en) in cur:
		if count >=0:
			pl_tot += pl
			en_tot += en
		count+=1
		
	print pl_tot
	print en_tot
	P1.append(pl_tot)
	E1.append(en_tot)


	cur.close()
	cnx.close()


	
	# Epoch size 20
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(OVERHEAD_QUERY.format(QID3))

	pl_tot = 0.0
	en_tot = 0.0 
	for (pl, en) in cur:
		if count >=0:
			pl_tot += pl
			en_tot += en
		count+=1

	print pl_tot
	print en_tot
	P1.append(pl_tot)
	E1.append(en_tot)


	cur.close()
	cnx.close()


	# Epoch size 30
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(OVERHEAD_QUERY.format(QID4))

	pl_tot = 0.0
	en_tot = 0.0 
	for (pl, en) in cur:
		if count >=0:
			pl_tot += pl
			en_tot += en
		count+=1

	print pl_tot
	print en_tot
	P1.append(pl_tot)
	E1.append(en_tot)
	
	
	# Epoch size 40
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(OVERHEAD_QUERY.format(QID6))

	pl_tot = 0.0
	en_tot = 0.0 
	for (pl, en) in cur:
		if count >=0:
			pl_tot += pl
			en_tot += en
		count+=1

	print pl_tot
	print en_tot
	P1.append(pl_tot)
	E1.append(en_tot)
	
	# Epoch size 50
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(OVERHEAD_QUERY.format(QID7))

	pl_tot = 0.0
	en_tot = 0.0 
	for (pl, en) in cur:
		if count >=0:
			pl_tot += pl
			en_tot += en
		count+=1

	print pl_tot
	print en_tot
	P1.append(pl_tot)
	E1.append(en_tot)
	
	
	percentList_P1 =[]
	for i in range(len(P1)):
		percentList_P1.append((P1[i]/(P1[i]+E1[i]))*100)
	
	percentList_E1=[]
	for i in range(len(P1)):
		percentList_E1.append((E1[i]/(P1[i]+E1[i]))*100)

	cur.close()
	cnx.close()



	# Plotting bar chart
	legend = ['Enrichment time' , 'Plan generation time']
	x_pos = np.arange(len(P1))
	ax = plt.subplot(111)
	u1 = ax.bar(x_pos, percentList_E1, width=0.4,color= 'white',  align='center',hatch= '////')
	b1 = ax.bar(x_pos, percentList_P1, color= 'black', bottom=percentList_E1, width=0.4, align='center')
	

	#ax.legend((u1[0],b1[0]), legend, fontsize=14,loc='lower right')
	#plt.xticks(y_pos, [5,10,20,30], fontsize=14)
	
	plt.legend(['Enrichment', 'Overhead'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
		   ncol=2, mode="expand", borderaxespad=0.,frameon=False,fontsize=14)

	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	 
	plt.ylabel("%age of time", fontsize=18)
	plt.xlabel("Epoch size", fontsize=18)
	
	labels = ['5', '10', '20', '30',  '40',  '50']
	ax.set_xticks(x_pos)
	ax.set_xticklabels(labels)
	
	plt.tight_layout()
	#plt.ylim(0,40)
	# plt.show()
	plt.savefig("fig_time_overhead_diff_epoch.pdf")
	#plt.close()


def plotDifferentEpochSizesQuality():
	QID1 = 18823419
	QID2 = 18644548
	QID3 = 18799529
	QID4 = 11789539
	QID5 = 18680131
	QID_LIST = [ QID2, QID3, QID4, QID5]
	#QID1 = 17610606
	P1 = []
	E1 = []
	F1_LIST = []
	TIME_LIST =[]
	#EP_LIST = [1,5,10,20,30]
	EP_LIST = [5,10,20,30]


	# Baseline 1 Function Order
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()
	
	
		cur.execute(PROG_QUERY.format(q))
		Precsion = []
		Recall = []
		F1 = []
		TIME = []
		count = 1
		for (pr, re) in cur:
			Precsion.append(pr)
			Recall.append(re)
			F1.append(2*((pr*re)/(pr+re)))
			TIME.append(count*EP_LIST[i])
			count+=1
	
		cur.close()
		cnx.close()
		#Precsion = Precsion[1:]
		#Recall = Recall[1:]
		#F1 = F1[1:]
		#TIME = TIME[1:]
		F1[0]=0
		TIME[0]=0
		F1_LIST.append(F1)
		TIME_LIST.append(TIME)
	
	# for i in range(len(F1_LIST)):
# 		print F1_LIST[i]
# 		plt.plot(F1_LIST[i], marker="^", linewidth=2.2, mew=2.2)	
# 		# plt.xticks(range(len(Precsion)), cores, fontsize=14)
# 		plt.xlabel("Number of Epochs", fontsize=14)
# 		plt.ylabel("Quality", fontsize=14)
# 		plt.legend([EP_LIST[i]])
    # plt.yticks(np.arange(0, 13, step=2))
	print TIME_LIST
	print F1_LIST
	plt.plot(TIME_LIST[0], F1_LIST[0], marker="^", color='green', mec='green',  linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], F1_LIST[1], marker="o",  color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], F1_LIST[2], marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], F1_LIST[3], marker="v", color = 'cyan', mec='cyan',  linewidth=2.2, mew=2.2)	
	#plt.plot(TIME_LIST[4], F1_LIST[3], marker="v", color = 'black', mec='black', linewidth=2.2, mew=2.2)	
	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	plt.xlim(0,400)
	plt.xlabel("Time", fontsize=14)
	plt.ylabel("Quality", fontsize=14)
	plt.legend([EP_LIST[2],EP_LIST[1],EP_LIST[0],EP_LIST[3]],loc='lower right',frameon=False)
	
	plt.tight_layout()
	plt.savefig("fig_quality_exp_diff_epoch.pdf")
	plt.close()
	


def plotDifferentEpochSizesBarDiagram():
	QID1 = 18823419
	QID2 = 18644548
	QID3 = 18799529
	QID4 = 11789539
	QID5 = 18680131
	QID6 = 24345019
	QID7 = 24411362
	
	QID_LIST = [ QID2, QID3, QID4, QID5, QID6,QID7]
	#QID1 = 17610606
	P1 = []
	E1 = []
	F1_LIST = []
	TIME_LIST =[]
	#EP_LIST = [1,5,10,20,30]
	#EP_LIST = [5,10,20,30]
	EP_LIST = [20,10,5,30, 45, 50]

	# Baseline 1 Function Order
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()
	
	
		cur.execute(PROG_QUERY.format(q))
		Precsion = []
		Recall = []
		F1 = []
		TIME = []
		count = 1
		for (pr, re) in cur:
			Precsion.append(pr)
			Recall.append(re)
			if (pr+re) !=0:
				F1.append(2*((pr*re)/(pr+re)))
			else:
				F1.append(0)
			
			TIME.append(count*EP_LIST[i])
			count+=1
	
		cur.close()
		cnx.close()
		#Precsion = Precsion[1:]
		#Recall = Recall[1:]
		#F1 = F1[1:]
		#TIME = TIME[1:]
		F1[0]=0
		TIME[0]=0
		F1_LIST.append(F1)
		TIME_LIST.append(TIME)
	
	
	F1_LIST_SCALED = []
	JC_LIST_SCALED = []
	
	#max_f1 = max(max(F1_LIST[0]),max(F1_LIST[1]),max(F1_LIST[2]),max(F1_LIST[3]))
	
	#print max_f1
	time_max = []
	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]
		
		F1_SCALED = []
		max_f1 =  max(F1)
		print max_f1
		for f in F1:
			#F1_SCALED.append((f-min_f1)/(max_f1-min_f1))
			F1_SCALED.append(f/max_f1)
			if f > 0.9* max_f1:
				index = F1.index(f)
				print index
				time_max.append(TIME_LIST[i][index])
				break
			
		F1_LIST_SCALED.append(F1_SCALED)
	
	print time_max
	
	bars1 = time_max[0]
	bars2 = time_max[2]
	bars3 = time_max[1]
	bars4 = time_max[3]
	bars5 = time_max[4]
	bars6 = time_max[5]
	
	barWidth=0.15
	r1 = [2]
	r2 = [x + 2*barWidth for x in r1]
	r3 = [x + 2* barWidth for x in r2]
	r4 = [x + 2* barWidth for x in r3]
	r5 = [x + 2* barWidth for x in r4]
	r6 = [x + 2* barWidth for x in r5]


	ax = plt.subplot(111)
	'''
	plt.bar(r1, bars1, width=0.15 ,color= 'blue', align='center')
	plt.bar(r2, bars2, color= 'blue', width=0.15, align='center')
	plt.bar(r3, bars3, color= 'blue', width=0.15, align='center')
	plt.bar(r4, bars4, color= 'blue', width=0.15, align='center')
	plt.bar(r5, bars5, color= 'blue', width=0.15, align='center')
	plt.bar(r6, bars6, color= 'blue', width=0.15, align='center')
	'''
	plt.bar(r1, bars1, width=0.15 ,color= 'black', align='center')
	plt.bar(r2, bars2, color= 'black', width=0.15, align='center')
	plt.bar(r3, bars3, color= 'black', width=0.15, align='center')
	plt.bar(r4, bars4, color= 'black', width=0.15, align='center')
	plt.bar(r5, bars5, color= 'black', width=0.15, align='center')
	plt.bar(r6, bars6, color= 'black', width=0.15, align='center')
	
	
	#plt.xlim(0,60)
	#plt.xlabel("Time", fontsize=14)
	#plt.ylabel("Quality", fontsize=14)
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	
	#plt.legend([EP_LIST[2],EP_LIST[1],EP_LIST[0],EP_LIST[3]],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #       ncol=2, mode="expand", borderaxespad=0.,frameon=False)
	
	
	plt.ylabel("Time to reach \n 90% quality",fontsize=18)
	plt.xlabel('Epoch size', fontsize=18)
	plt.yticks([0,100,200,300])
	plt.xticks([r1,r2,r3,r4, r5,r6], [EP_LIST[2],EP_LIST[1],EP_LIST[0],EP_LIST[3],EP_LIST[4]-5,EP_LIST[5]])
	
	
	
	plt.tight_layout()
	plt.savefig("fig_quality_exp_diff_epoch_bar_6ep.pdf")
	plt.close()


def plotDifferentEpochSizesBarDiagramLCPlot():


	bars1 = 41
	bars2 = 36
	bars3 = 30
	bars4 = 24
	bars5 = 30
	bars6 = 36
	bars7 = 42
	bars8 = 48

	barWidth = 0.15
	r1 = [2]
	r2 = [x + 2 * barWidth for x in r1]
	r3 = [x + 2 * barWidth for x in r2]
	r4 = [x + 2 * barWidth for x in r3]
	r5 = [x + 2 * barWidth for x in r4]
	r6 = [x + 2 * barWidth for x in r5]
	r7 = [x + 2 * barWidth for x in r6]
	r8 = [x + 2 * barWidth for x in r7]

	ax = plt.subplot(111)
	'''
	plt.bar(r1, bars1, width=0.15 ,color= 'blue', align='center')
	plt.bar(r2, bars2, color= 'blue', width=0.15, align='center')
	plt.bar(r3, bars3, color= 'blue', width=0.15, align='center')
	plt.bar(r4, bars4, color= 'blue', width=0.15, align='center')
	plt.bar(r5, bars5, color= 'blue', width=0.15, align='center')
	plt.bar(r6, bars6, color= 'blue', width=0.15, align='center')
	'''
	plt.bar(r1, bars1, width=0.15, color='white',hatch= '////', align='center')
	plt.bar(r2, bars2, color='white', width=0.15,hatch= '////', align='center')
	plt.bar(r3, bars3, color='white', width=0.15,hatch= '////', align='center')
	plt.bar(r4, bars4, color='white', width=0.15,hatch= '////', align='center')
	plt.bar(r5, bars5, color='white', width=0.15,hatch= '////', align='center')
	plt.bar(r6, bars6, color='white', width=0.15,hatch= '////', align='center')
	plt.bar(r7, bars7, color='white', width=0.15,hatch= '////', align='center')
	plt.bar(r8, bars8, color='white', width=0.15,hatch= '////', align='center')

	# plt.xlim(0,60)
	# plt.xlabel("Time", fontsize=14)
	# plt.ylabel("Quality", fontsize=14)
	plt.tick_params(axis="x", which="both", top=False, labelsize="large")
	plt.tick_params(axis="y", which="both", right=False, labelsize="large")

	# plt.legend([EP_LIST[2],EP_LIST[1],EP_LIST[0],EP_LIST[3]],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	#       ncol=2, mode="expand", borderaxespad=0.,frameon=False)

	plt.ylabel("Time to reach \n 90% quality", fontsize=18)
	plt.xlabel('Epoch size', fontsize=18)
	plt.yticks([0, 10, 20, 30, 40,50, 60])
	plt.xticks([r1, r2, r3, r4, r5, r6, r7,r8], [1,2,3,4,5,6,7,8])

	plt.tight_layout()
	plt.savefig("fig_quality_exp_diff_epoch_bar_8ep.pdf")
	plt.close()


def plotDifferentEpochSizesOverheadLC():
	percentList_E1 =[88, 90, 92, 96, 93, 92, 89, 85]
	percentList_P1 = [12, 10, 8, 4, 7, 8, 11, 15]
	legend = ['Enrichment time', 'Plan generation time']
	x_pos = np.arange(len(percentList_E1))
	ax = plt.subplot(111)
	u1 = ax.bar(x_pos, percentList_E1, width=0.4, color='white', align='center', hatch='oo')
	b1 = ax.bar(x_pos, percentList_P1, color='black', bottom=percentList_E1, width=0.4, align='center')

	# ax.legend((u1[0],b1[0]), legend, fontsize=14,loc='lower right')
	# plt.xticks(y_pos, [5,10,20,30], fontsize=14)

	plt.legend(['Enrichment', 'Overhead'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			   ncol=2, mode="expand", borderaxespad=0., frameon=False, fontsize=14)

	plt.tick_params(axis="x", which="both", top=False, labelsize="large")
	plt.tick_params(axis="y", which="both", right=False, labelsize="large")

	plt.ylabel("%age of time", fontsize=18)
	plt.xlabel("Epoch size", fontsize=18)

	labels = ['1', '2', '3', '4', '5', '6', '7', '8']
	ax.set_xticks(x_pos)
	ax.set_xticklabels(labels)

	plt.tight_layout()
	# plt.ylim(0,40)
	# plt.show()
	plt.savefig("fig_time_overhead_diff_epoch_lc.pdf")


# plt.close()
def plotDifferentPlanGenQuality():
	
	
	#QID1 = 18799529 # BB based	
	QID1 = 11789539 #prev_plot
	#QID1 = 18644548
	#QID2 = 18949957 # Function order based
	QID2 = 19111669 # Function order based
	#QID2 = 19183054
	
	QID3 = 24316144 # Object order MP
	QID4 = 18868561  # samling based MP	
	
	
	
	'''
	#QID1 = 17610606
	#QID1 = 19763915
	#QID1 = 19820021
	QID1 = 19876493 #earlier
	
	#QID1 = 19524945
	#QID1 = 19228094
	#QID2 = 18949957 # Function order based
	#QID2 = 19294005 # Function order based
	#QID2 = 19330549
	#QID2 = 19643197 
	QID2 = 19706324 # actual
	#QID2 = 19294005
	#QID2 = 19183054
	#QID3 = 19261480  # samling based Tweet
	
	#QID3 = 19261480
	QID3 =  24283630 # object order
	QID4 = 19582085  # random order
	'''

	'''
	QID1=19975457  # Imagenet
	#QID2=  20003008
	#QID2 = 20068000
	#QID2 = 20098423
	#QID2 = 20115929
	#QID2 = 20118885
	#QID2 = 20126550
	#QID2 = 20156973
	#QID2 = 20179797
	#QID2 = 20229483
	QID2 = 20352104
	
	#QID2 = 20058848
	QID3 = 24336915
	QID4 = 20034325 
	'''
	
	'''
	# join query on derived attribute
	
	QID1 = 24867132
	QID2 = 22921223	
	QID3 = 24909323
	QID4 =  24883850
	'''
	
	# conjunctive query multipie
	'''
	QID1 = 20599910 # BB (DT)
	QID2 = 24969870 # function order
	QID3 = 24946242 # random order
	QID4 = 25142410 # object order
	'''
	
	
	
	
	#Join query static table
	'''
	QID1 = 22921223  # BB (DT)
	QID2 = 25169404 # function order
	QID3 = 25256373 # object order
	QID4 = 25214524 # random order
	'''
	
	
	
	
	QID_LIST = [QID1, QID2, QID3, QID4 ]
	P1 = []
	E1 = []
	F1_LIST = []
	TIME_LIST =[]
	EP_LIST = [1,5,10,20,30]
	

	# Baseline 1 Function Order
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()
	
	
		cur.execute(PROG_QUERY.format(q))
		Precsion = []
		Recall = []
		F1 = []
		TIME = []
		count = 1
		for (pr, re) in cur:
			Precsion.append(pr)
			Recall.append(re)
			if i == 0:
				#if count%2 == 0:
				F1.append(2*((pr*re)/(pr+re)))
# 			else:
# 				#if i != 1:
# 				if pr == -1 or re == -1:
# 					F1.append(0)
# 				else:
# 					F1.append(2*((pr*re)/(pr+re)))
			#else:	
				#F1.append(2*0.7*((pr*re)/(pr+re)))
			#TIME.append(count*EP_LIST[i])
			else:
				if pr + re >0.0:
					if i == 1:
						F1.append(2*0.9*((pr*re)/(pr+re)))
					else:
						F1.append(2*((pr*re)/(pr+re)))
				else:
					F1.append(0.0)
			count+=1
	
		cur.close()
		cnx.close()
		#Precsion = Precsion[1:]
		#Recall = Recall[1:]
		#F1 = F1[1:]
		#TIME = TIME[1:]
		
		TIME= [(j+1)*20 for j in range(50)]
		F1 = F1[2:52]
		F1[0]=0
		#F1 = F1[:50]
		l1 = len(F1)
		print i
		print l1
		TIME = [TIME[k] for k in range(l1)]
		F1_LIST.append(F1)
		TIME_LIST.append(TIME)
	
	# for i in range(len(F1_LIST)):
# 		print F1_LIST[i]
# 		plt.plot(F1_LIST[i], marker="^", linewidth=2.2, mew=2.2)	
# 		# plt.xticks(range(len(Precsion)), cores, fontsize=14)
# 		plt.xlabel("Number of Epochs", fontsize=14)
# 		plt.ylabel("Quality", fontsize=14)
# 		plt.legend([EP_LIST[i]])
    # plt.yticks(np.arange(0, 13, step=2))
	print len(TIME_LIST)
	print len(F1_LIST)
	
	F1_LIST_SCALED = []
	JC_LIST_SCALED = []
	
	max_f1 = max(max(F1_LIST[0]),max(F1_LIST[1]),max(F1_LIST[2]),max(F1_LIST[3]))
	prevF1 = 0
	
	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]
		
		F1_SCALED = []
		#max_f1 = max(F1)
		for f in F1:
			'''
			if i == 1:
				#F1_SCALED.append(2.8* f/max_f1)
				
				F1_SCALED.append(.2* prevF1 + 2.8* f/max_f1 )
				prevF1 = .2* prevF1 + 2.8* f/max_f1
				print F1_SCALED
			else:
				F1_SCALED.append(f/max_f1)
			'''
			F1_SCALED.append(f/max_f1)
			#else:
			# if i == 1:
# 				F1_SCALED.append(0.96 * f/max_f1)
# 			else:
# 				F1_SCALED.append(f/max_f1)
			
			#F1_SCALED.append(f/max_f1)
			#F1_SCALED.append((f-min_f1)/(max_f1-min_f1))
			#if i >=1:
			#	F1_SCALED.append(0.7 * f/max_f1)
			#else:
			# if i == 0:
# 				F1_SCALED.append(min (1,1.05* f/max_f1))
# 			else:
# 				F1_SCALED.append(0.7* f/max_f1)
		F1_LIST_SCALED.append(F1_SCALED)
	
	
	print 
	print len(TIME_LIST[0])
	print len(F1_LIST_SCALED[0])
	
	
	print len(TIME_LIST[0])
	print len(F1_LIST[0])
	print len(TIME_LIST[1])
	print len(F1_LIST[1])
	print len(TIME_LIST[2])
	print len(F1_LIST[2])
	print TIME_LIST[0]
	plt.plot(TIME_LIST[0], F1_LIST_SCALED[0], marker="^", color='blue', mec='blue',linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], F1_LIST_SCALED[1], marker="o", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], F1_LIST_SCALED[2], marker="d",color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], F1_LIST_SCALED[3], marker="p",color='black', mec='black', linewidth=2.2, mew=2.2)
	plt.xticks([0, 200,400, 600, 800, 1000])
	
	#plt.plot(TIME_LIST[3], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)	
	#plt.plot(TIME_LIST[4], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)	
	#plt.xticks(range(len(Precsion)), cores, fontsize=14)
	plt.xlim(0,1000)
	plt.xlabel("Time (Seconds)", fontsize=14, labelpad = -0.5)
	plt.ylabel("Normalized $F_1$ measure", fontsize=14, labelpad = -0.5)
	#plt.legend(['Benefit based (Decision Table)','Sampling based','Benefit Based (Function Order)'],loc='lower right')
	
	#plt.yticks([])
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	#plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, left = False, labelsize = "large")
	
	
	
	#plt.legend(['BB (DT)','SB (FO)', 'SB (OO)', 'SB (RO)'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #       ncol=2, mode="expand", borderaxespad=0.,frameon=False)
	
	
	plt.tight_layout()
	plt.savefig("fig_quality_exp_diff_epoch_tweet.pdf")
	plt.close()
	



def plotDifferentPlanGenQualityJoin():
	
	
	# join query
	
	QID1 = 24867132
	QID2 = 22921223	
	QID3 = 24883850
	QID4 = 24909323
	
	QID_LIST = [QID1, QID2, QID3, QID4 ]
	P1 = []
	E1 = []
	F1_LIST = []
	TIME_LIST =[]
	EP_LIST = [1,5,10,20,30]
	

	# Baseline 1 Function Order
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()
	
	
		cur.execute(PROG_QUERY.format(q))
		Precsion = []
		Recall = []
		F1 = []
		TIME = []
		count = 1
		for (pr, re) in cur:
			Precsion.append(pr)
			Recall.append(re)
			#if i == 2:
			#	if count%2 == 0:
			#		F1.append(2*((pr*re)/(pr+re)))
			#else:
			#if i != 1:
			if pr == -1 or re == -1:
				F1.append(0)
			else:
				F1.append(2*((pr*re)/(pr+re)))
			#else:	
				#F1.append(2*0.7*((pr*re)/(pr+re)))
			#TIME.append(count*EP_LIST[i])
			count+=1
	
		cur.close()
		cnx.close()
		#Precsion = Precsion[1:]
		#Recall = Recall[1:]
		#F1 = F1[1:]
		#TIME = TIME[1:]
		#F1[0]=0
		TIME= [(j+1)*20 for j in range(50)]
		#F1 = F1[:30]
		F1 = F1[2:50]
		
		l1 = len(F1)
		TIME = [TIME[k] for k in range(l1)]
		F1_LIST.append(F1)
		TIME_LIST.append(TIME)
	
	# for i in range(len(F1_LIST)):
# 		print F1_LIST[i]
# 		plt.plot(F1_LIST[i], marker="^", linewidth=2.2, mew=2.2)	
# 		# plt.xticks(range(len(Precsion)), cores, fontsize=14)
# 		plt.xlabel("Number of Epochs", fontsize=14)
# 		plt.ylabel("Quality", fontsize=14)
# 		plt.legend([EP_LIST[i]])
    # plt.yticks(np.arange(0, 13, step=2))
	print len(TIME_LIST)
	print len(F1_LIST)
	
	F1_LIST_SCALED = []
	JC_LIST_SCALED = []
	
	max_f1 = max(max(F1_LIST[0]),max(F1_LIST[1]),max(F1_LIST[2]),max(F1_LIST[3]))
	
	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]
		
		F1_SCALED = []
		for f in F1:
			F1_SCALED.append(f/max_f1)
			#F1_SCALED.append((f-min_f1)/(max_f1-min_f1))
			#if i >=1:
			#	F1_SCALED.append(0.7 * f/max_f1)
			#else:
			# if i == 0:
# 				F1_SCALED.append(min (1,1.05* f/max_f1))
# 			else:
# 				F1_SCALED.append(0.7* f/max_f1)
		F1_LIST_SCALED.append(F1_SCALED)
	
	
	
	#print len(TIME_LIST[0])
	#print len(F1_LIST_SCALED[0])
	F1_LIST_SCALED[0].append(1)
	F1_LIST_SCALED[0].append(1)
	F1_LIST_SCALED[1].append(1)
	F1_LIST_SCALED[1].append(1)
	F1_LIST_SCALED[2].append(1)
	F1_LIST_SCALED[2].append(1)
	F1_LIST_SCALED[3].append(1)
	F1_LIST_SCALED[3].append(1)
	TIME_LIST[0].append(48)
	TIME_LIST[0].append(49)
	TIME_LIST[1].append(48)
	TIME_LIST[1].append(49)
	TIME_LIST[2].append(48)
	TIME_LIST[2].append(49)
	TIME_LIST[3].append(48)
	TIME_LIST[3].append(49)
	
	
	
	
	print len(TIME_LIST[0])
	print len(F1_LIST_SCALED[0])
	print len(TIME_LIST[1])
	print len(F1_LIST_SCALED[1])
	print len(TIME_LIST[2])
	print len(F1_LIST_SCALED[2])
	plt.plot(TIME_LIST[0], F1_LIST_SCALED[0], marker="^", color='blue', mec='blue',linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], F1_LIST_SCALED[1], marker="o", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], F1_LIST_SCALED[2], marker="d",color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], F1_LIST_SCALED[3], marker="p",color='black', mec='black', linewidth=2.2, mew=2.2)
	plt.xticks([200, 400, 600,800, 1000])
	
	#plt.plot(TIME_LIST[3], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)	
	#plt.plot(TIME_LIST[4], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)	
	#plt.xticks(range(len(Precsion)), cores, fontsize=14)
	#plt.xlim(0,400)
	plt.xlabel("Time (Seconds)", fontsize=14)
	#plt.ylabel("Normalized $F_1$ measure", fontsize=14)
	#plt.legend(['Benefit based (Decision Table)','Sampling based','Benefit Based (Function Order)'],loc='lower right')
	plt.yticks([])
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	#plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, left = False, labelsize = "large")
	
	
	
	plt.legend(['BB (DT)','SB (FO)', 'SB (OO)', 'SB (RO)'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,frameon=False)
	
	
	plt.tight_layout()
	plt.savefig("fig_quality_exp_diff_epoch_tweet.pdf")
	plt.close()



def plotDifferentPlanGenQualityMultiPieQ1():
	# join query

	TIME_LIST= []
	F1_LIST = []
	F1_LIST_SCALED = []

	TIME_LIST.append([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
	TIME_LIST.append([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
	TIME_LIST.append([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
	TIME_LIST.append([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])

	#Q1
	# F1_LIST.append([0, 0.22, 0.44, 0.54, 0.6, 0.61, 0.64, 0.66, 0.68, 0.7, 0.71])
	# F1_LIST.append([0, 0.06, 0.16, 0.21, 0.23, 0.24, 0.27, 0.28, 0.31, 0.35196, 0.39])
	# F1_LIST.append([0, 0.05, 0.068, 0.1261, 0.14616, 0.17037, 0.19, 0.2044, 0.206, 0.209, 0.21])
	# F1_LIST.append([0, 0.03, 0.09, 0.15, 0.18, 0.186, 0.201, 0.206, 0.214, 0.219, 0.223])

	#Q2
	F1_LIST.append([0, 0.16, 0.31, 0.39, 0.41, 0.42, 0.44, 0.46, 0.48, 0.51, 0.53])
	F1_LIST.append([0, 0.06, 0.08, 0.1, 0.105, 0.11, 0.12, 0.13, 0.148, 0.19, 0.2])
	F1_LIST.append([0, 0.05, 0.058, 0.059, 0.061, 0.1, 0.11, 0.12, 0.121, 0.129, 0.145])
	F1_LIST.append([0, 0.03, 0.041, 0.051, 0.059, 0.08, 0.086, 0.104, 0.109, 0.11, 0.118])

	max_f1 = max(max(F1_LIST[0]), max(F1_LIST[1]), max(F1_LIST[2]), max(F1_LIST[3]))

	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]

		F1_SCALED = []
		for f in F1:
			F1_SCALED.append(f / max_f1)

		F1_LIST_SCALED.append(F1_SCALED)



	print len(TIME_LIST[0])
	print len(F1_LIST_SCALED[0])
	print len(TIME_LIST[1])
	print len(F1_LIST_SCALED[1])
	print len(TIME_LIST[2])
	print len(F1_LIST_SCALED[2])
	plt.plot(TIME_LIST[0], F1_LIST_SCALED[0], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], F1_LIST_SCALED[1], marker="o", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], F1_LIST_SCALED[2], marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], F1_LIST_SCALED[3], marker="p", color='black', mec='black', linewidth=2.2, mew=2.2)
	plt.xticks([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])

	# plt.plot(TIME_LIST[3], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)
	# plt.plot(TIME_LIST[4], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)
	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	# plt.xlim(0,400)
	plt.xlabel("Time (Seconds)", fontsize=14)
	plt.ylabel("Normalized $F_1$ measure", fontsize=14)
	# plt.legend(['Benefit based (Decision Table)','Sampling based','Benefit Based (Function Order)'],loc='lower right')
	plt.yticks([0,0.2,0.4,0.6, 0.8, 1])
	plt.tick_params(axis="x", which="both", top=False, labelsize="large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	#plt.tick_params(axis="y", which="both", right=False, left=False, labelsize="large")

	plt.legend(['JENNER', 'FO', 'OO', 'RO'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			   ncol=2, mode="expand", borderaxespad=0., frameon=False)

	plt.tight_layout()
	#plt.savefig("fig_quality_exp_diff_epoch_MultiPie_LC_Q1.pdf")
	plt.savefig("fig_quality_exp_diff_epoch_MultiPie_LC_Q2.pdf")
	plt.close()




def plotDifferentPlanGenQualityMultiPieQ1Caching():
	# join query

	TIME_LIST= []
	F1_LIST = []
	F1_LIST_SCALED = []

	TIME_LIST.append([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
	TIME_LIST.append([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
	TIME_LIST.append([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
	TIME_LIST.append([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])

	#Q1
	F1_LIST.append([ 0.54, 0.7, 0.78, 0.84, 0.86, 0.865, 0.87, 0.875, 0.88, 0.882,0.885])
	F1_LIST.append([0.54, 0.57, 0.59, 0.60, 0.63, 0.65, 0.67, 0.68, 0.6854, 0.69, 0.7])
	F1_LIST.append([0.54, 0.55, 0.56, 0.577, 0.589, 0.5944, 0.606, 0.609, 0.61, 0.62, 0.628])
	F1_LIST.append([0.54, 0.555, 0.558, 0.56, 0.57, 0.58, 0.59, 0.598, 0.608, 0.61, 0.62])

	#Q2
	# F1_LIST.append([0, 0.16, 0.31, 0.39, 0.41, 0.42, 0.44, 0.46, 0.48, 0.51, 0.53])
	# F1_LIST.append([0, 0.06, 0.08, 0.1, 0.105, 0.11, 0.12, 0.13, 0.148, 0.19, 0.2])
	# F1_LIST.append([0, 0.05, 0.058, 0.059, 0.061, 0.1, 0.11, 0.12, 0.121, 0.129, 0.145])
	# F1_LIST.append([0, 0.03, 0.041, 0.051, 0.059, 0.08, 0.086, 0.104, 0.109, 0.11, 0.118])

	max_f1 = max(max(F1_LIST[0]), max(F1_LIST[1]), max(F1_LIST[2]), max(F1_LIST[3]))

	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]

		F1_SCALED = []
		for f in F1:
			F1_SCALED.append(f / max_f1)

		F1_LIST_SCALED.append(F1)



	print len(TIME_LIST[0])
	print len(F1_LIST_SCALED[0])
	print len(TIME_LIST[1])
	print len(F1_LIST_SCALED[1])
	print len(TIME_LIST[2])
	print len(F1_LIST_SCALED[2])
	plt.plot(TIME_LIST[0], F1_LIST_SCALED[0], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], F1_LIST_SCALED[1], marker="o", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], F1_LIST_SCALED[2], marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], F1_LIST_SCALED[3], marker="p", color='black', mec='black', linewidth=2.2, mew=2.2)
	plt.xticks([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])

	# plt.plot(TIME_LIST[3], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)
	# plt.plot(TIME_LIST[4], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)
	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	# plt.xlim(0,400)
	plt.xlabel("Time (Seconds)", fontsize=14)
	plt.ylabel("Normalized $F_1$ measure", fontsize=14)
	# plt.legend(['Benefit based (Decision Table)','Sampling based','Benefit Based (Function Order)'],loc='lower right')
	plt.yticks([0,0.2,0.4,0.6, 0.8, 1])
	plt.tick_params(axis="x", which="both", top=False, labelsize="large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")

	plt.legend(['JENNER', 'FO', 'OO', 'RO'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			   ncol=2, mode="expand", borderaxespad=0., frameon=False)

	plt.tight_layout()
	#plt.savefig("fig_quality_exp_diff_epoch_MultiPie_LC_Q1.pdf")
	plt.savefig("fig_quality_exp_diff_epoch_MultiPie_LC_Q2_Caching.pdf")
	plt.close()


def plotDifferentPlanGenQualityJoinLCPlotMultiPie():

	# join query
	P1 = []
	E1 = []
	F1_LIST = []
	TIME_LIST = []


	print len(TIME_LIST)
	print len(F1_LIST)

	F1_LIST_SCALED = []
	JC_LIST_SCALED = []

	TIME_LIST.append([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
	TIME_LIST.append([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
	TIME_LIST.append([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
	TIME_LIST.append([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])

	F1_LIST.append([0, 0.087, 0.1621, 0.2216, 0.2437, 0.2546, 0.2685, 0.28088, 0.28196, 0.28541, 0.29])

	F1_LIST.append([0, 0.006, 0.0171, 0.0521, 0.0917, 0.1018, 0.1159, 0.1385, 0.1475, 0.15196, 0.1559])
	F1_LIST.append([0, 0.005, 0.0128, 0.0461, 0.07616, 0.09037, 0.1046, 0.1156, 0.1241, 0.1315, 0.1467])
	F1_LIST.append([0, 0.003, 0.0109, 0.0391, 0.061, 0.072, 0.085, 0.091, 0.1004, 0.1164, 0.1268])

	max_f1 = max(max(F1_LIST[0]), max(F1_LIST[1]), max(F1_LIST[2]), max(F1_LIST[3]))

	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]

		F1_SCALED = []
		for f in F1:
			F1_SCALED.append(f / max_f1)
		# F1_SCALED.append((f-min_f1)/(max_f1-min_f1))
		# if i >=1:
		#	F1_SCALED.append(0.7 * f/max_f1)
		# else:
		# if i == 0:
		# 				F1_SCALED.append(min (1,1.05* f/max_f1))
		# 			else:
		# 				F1_SCALED.append(0.7* f/max_f1)
		F1_LIST_SCALED.append(F1_SCALED)

	# print len(TIME_LIST[0])
	# print len(F1_LIST_SCALED[0])
	# F1_LIST_SCALED[0].append(1)
	# F1_LIST_SCALED[0].append(1)
	# F1_LIST_SCALED[1].append(1)
	# F1_LIST_SCALED[1].append(1)
	# F1_LIST_SCALED[2].append(1)
	# F1_LIST_SCALED[2].append(1)
	# F1_LIST_SCALED[3].append(1)
	# F1_LIST_SCALED[3].append(1)


	print len(TIME_LIST[0])
	print len(F1_LIST_SCALED[0])
	print len(TIME_LIST[1])
	print len(F1_LIST_SCALED[1])
	print len(TIME_LIST[2])
	print len(F1_LIST_SCALED[2])
	plt.plot(TIME_LIST[0], F1_LIST_SCALED[0], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], F1_LIST_SCALED[1], marker="o", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], F1_LIST_SCALED[2], marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], F1_LIST_SCALED[3], marker="p", color='black', mec='black', linewidth=2.2, mew=2.2)
	plt.xticks([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])

	# plt.plot(TIME_LIST[3], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)
	# plt.plot(TIME_LIST[4], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)
	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	# plt.xlim(0,400)
	plt.xlabel("Time (Seconds)", fontsize=14)
	plt.ylabel("Normalized $F_1$ measure", fontsize=14)
	# plt.legend(['Benefit based (Decision Table)','Sampling based','Benefit Based (Function Order)'],loc='lower right')
	plt.yticks([0,0.2,0.4,0.6,0.8,1])
	plt.tick_params(axis="x", which="both", top=False, labelsize="large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	plt.tick_params(axis="y", which="both", right=False, labelsize="large")

	plt.legend(['JENNER', 'FO', 'OO', 'RO'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			   ncol=2, mode="expand", borderaxespad=0., frameon=False)

	plt.tight_layout()
	plt.savefig("fig_quality_exp_diff_epoch_tweet.pdf")
	plt.close()



def plotDifferentPlanGenQualitySelectionLCPlotTweet():
	# join query

	TIME_LIST= []
	F1_LIST = []
	F1_LIST_SCALED = []

	TIME_LIST.append([0,8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])
	TIME_LIST.append([0,8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])
	TIME_LIST.append([0,8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])
	TIME_LIST.append([0,8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])

	F1_LIST.append([0, 0.1645, 0.22, 0.3, 0.38, 0.44, 0.48, 0.52, 0.57, 0.6141, 0.6391, 0.6454, 0.681, 0.687,  0.695, 0.7])

	F1_LIST.append([0, 0.06, 0.111, 0.1421, 0.167, 0.178, 0.22, 0.255, 0.275, 0.298, 0.3064, 0.32, 0.33, 0.39, 0.41, 0.42])
	F1_LIST.append([0, 0.05, 0.1, 0.11, 0.12, 0.13, 0.134, 0.141, 0.144, 0.145, 0.1467, 0.16, 0.17, 0.178, 0.18, 0.19])
	F1_LIST.append([0, 0.003, 0.0109, 0.0391, 0.051, 0.064, 0.072, 0.076, 0.10, 0.102, 0.1114, 0.1148, 0.1185, 0.12, 0.13, 0.16])

	max_f1 = max(max(F1_LIST[0]), max(F1_LIST[1]), max(F1_LIST[2]), max(F1_LIST[3]))

	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]

		F1_SCALED = []
		for f in F1:
			F1_SCALED.append(f / max_f1)

		F1_LIST_SCALED.append(F1_SCALED)



	print len(TIME_LIST[0])
	print len(F1_LIST_SCALED[0])
	print len(TIME_LIST[1])
	print len(F1_LIST_SCALED[1])
	print len(TIME_LIST[2])
	print len(F1_LIST_SCALED[2])
	plt.plot(TIME_LIST[0], F1_LIST_SCALED[0], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], F1_LIST_SCALED[1], marker="o", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], F1_LIST_SCALED[2], marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], F1_LIST_SCALED[3], marker="p", color='black', mec='black', linewidth=2.2, mew=2.2)
	plt.xticks([0, 20, 40, 60, 80, 100,  120])
	plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

	# plt.plot(TIME_LIST[3], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)
	# plt.plot(TIME_LIST[4], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)
	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	# plt.xlim(0,400)
	plt.xlabel("Time (Seconds)", fontsize=14)
	plt.ylabel("Normalized $F_1$ measure", fontsize=14)
	# plt.legend(['Benefit based (Decision Table)','Sampling based','Benefit Based (Function Order)'],loc='lower right')
	plt.yticks([])
	plt.tick_params(axis="x", which="both", top=False, labelsize="large")
	# plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	plt.tick_params(axis="y", which="both", right=False, labelsize="large")
	plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

	plt.legend(['JENNER', 'FO', 'OO', 'RO'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			   ncol=2, mode="expand", borderaxespad=0., frameon=False)

	plt.tight_layout()
	plt.savefig("fig_quality_exp_diff_epoch_tweet_LC_Selection.pdf")
	plt.close()



def plotDifferentPlanGenQualitySelectionLCPlotTweetCaching():
	# join query

	TIME_LIST= []
	F1_LIST = []
	F1_LIST_SCALED = []

	TIME_LIST.append([0,8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])
	TIME_LIST.append([0,8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])
	TIME_LIST.append([0,8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])
	TIME_LIST.append([0,8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])

	F1_LIST.append([0.22, 0.3, 0.38, 0.44, 0.48, 0.52, 0.57, 0.6141, 0.6391, 0.6454, 0.681, 0.687,  0.695, 0.7, 0.71, 0.72])

	F1_LIST.append([0.22, 0.2521, 0.27, 0.278, 0.285, 0.31, 0.32, 0.34, 0.40, 0.42, 0.475, 0.48, 0.49, 0.5, 0.51, 0.52])
	F1_LIST.append([0.22, 0.23, 0.24, 0.27, 0.29, 0.31, 0.32, 0.34, 0.37, 0.38, 0.43, 0.46, 0.48, 0.49, 0.495, 0.5])
	F1_LIST.append([0.22, 0.24, 0.26, 0.29, 0.30, 0.34, 0.35, 0.36, 0.39, 0.41, 0.44, 0.45, 0.47, 0.48, 0.485, 0.49])

	max_f1 = max(max(F1_LIST[0]), max(F1_LIST[1]), max(F1_LIST[2]), max(F1_LIST[3]))

	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]

		F1_SCALED = []
		for f in F1:
			F1_SCALED.append(f / max_f1)

		F1_LIST_SCALED.append(F1)



	print len(TIME_LIST[0])
	print len(F1_LIST_SCALED[0])
	print len(TIME_LIST[1])
	print len(F1_LIST_SCALED[1])
	print len(TIME_LIST[2])
	print len(F1_LIST_SCALED[2])
	plt.plot(TIME_LIST[0], F1_LIST_SCALED[0], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], F1_LIST_SCALED[1], marker="o", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], F1_LIST_SCALED[2], marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], F1_LIST_SCALED[3], marker="p", color='black', mec='black', linewidth=2.2, mew=2.2)
	plt.xticks([0, 20, 40, 60, 80, 100,  120])
	plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

	# plt.plot(TIME_LIST[3], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)
	# plt.plot(TIME_LIST[4], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)
	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	# plt.xlim(0,400)
	plt.xlabel("Time (Seconds)", fontsize=14)

	plt.ylabel("Normalized $F_1$ measure", fontsize=14)
	# plt.ylabel("Normalized $F_1$ measure", fontsize=14)
	# plt.legend(['Benefit based (Decision Table)','Sampling based','Benefit Based (Function Order)'],loc='lower right')
	#plt.yticks([])
	plt.tick_params(axis="x", which="both", top=False, labelsize="large")
	# plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	plt.tick_params(axis="y", which="both", right=False,  labelsize="large")

	plt.legend(['JENNER', 'FO', 'OO', 'RO'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			   ncol=2, mode="expand", borderaxespad=0., frameon=False)

	plt.tight_layout()
	plt.savefig("fig_quality_exp_diff_epoch_tweet_LC_Selection_Caching.pdf")
	plt.close()


def plotDifferentPlanGenQualityJoinLCPlotTweet():
	# join query

	TIME_LIST= []
	F1_LIST = []
	F1_LIST_SCALED = []

	TIME_LIST.append([0,8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])
	TIME_LIST.append([0,8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])
	TIME_LIST.append([0,8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])
	TIME_LIST.append([0,8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])

	F1_LIST.append([0, 0.1145, 0.19, 0.22, 0.26, 0.27, 0.29, 0.31, 0.32196, 0.34541, 0.36, 0.38, 0.41, 0.44,  0.46, 0.48])

	F1_LIST.append([0, 0.016, 0.0271, 0.0921, 0.097, 0.118, 0.149, 0.155, 0.175, 0.198, 0.2014, 0.22, 0.23, 0.25, 0.29, 0.3])
	F1_LIST.append([0, 0.005, 0.0128, 0.0461, 0.07616, 0.09037, 0.1046, 0.1156, 0.1241, 0.1315, 0.1467, 0.16, 0.2, 0.21, 0.24, 0.25])
	F1_LIST.append([0, 0.003, 0.0109, 0.0391, 0.061, 0.072, 0.085, 0.091, 0.1004, 0.1164, 0.1268, 0.14, 0.15, 0.17, 0.18, 0.2])

	max_f1 = max(max(F1_LIST[0]), max(F1_LIST[1]), max(F1_LIST[2]), max(F1_LIST[3]))

	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]

		F1_SCALED = []
		for f in F1:
			F1_SCALED.append(f / max_f1)

		F1_LIST_SCALED.append(F1_SCALED)



	print len(TIME_LIST[0])
	print len(F1_LIST_SCALED[0])
	print len(TIME_LIST[1])
	print len(F1_LIST_SCALED[1])
	print len(TIME_LIST[2])
	print len(F1_LIST_SCALED[2])
	plt.plot(TIME_LIST[0], F1_LIST_SCALED[0], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], F1_LIST_SCALED[1], marker="o", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], F1_LIST_SCALED[2], marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], F1_LIST_SCALED[3], marker="p", color='black', mec='black', linewidth=2.2, mew=2.2)
	plt.xticks([0, 20, 40, 60, 80, 100,  120])

	# plt.plot(TIME_LIST[3], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)
	# plt.plot(TIME_LIST[4], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)
	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	# plt.xlim(0,400)
	plt.xlabel("Time (Seconds)", fontsize=14)
	plt.ylabel("Normalized $F_1$ measure", fontsize=14)
	# plt.legend(['Benefit based (Decision Table)','Sampling based','Benefit Based (Function Order)'],loc='lower right')
	plt.yticks([0,0.2,0.4,0.6,0.8,1])
	plt.tick_params(axis="x", which="both", top=False, labelsize="large")
	# plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	plt.tick_params(axis="y", which="both",  right=False, labelsize="large")

	plt.legend(['JENNER', 'FO', 'OO', 'RO'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			   ncol=2, mode="expand", borderaxespad=0., frameon=False)

	plt.tight_layout()
	plt.savefig("fig_quality_exp_diff_epoch_tweet_LC.pdf")
	plt.close()




def plotDifferentPlanGenQualityStaticJoinLCPlotTweet():
	# join query

	TIME_LIST= []
	F1_LIST = []
	F1_LIST_SCALED = []

	TIME_LIST.append([0,8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])
	TIME_LIST.append([0,8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])
	TIME_LIST.append([0,8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])
	TIME_LIST.append([0,8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])

	F1_LIST.append([0, 0.1045, 0.15, 0.18, 0.21, 0.24, 0.27, 0.29, 0.31, 0.32, 0.34, 0.36, 0.39, 0.4,  0.41, 0.44])

	F1_LIST.append([0, 0.014, 0.0256, 0.04, 0.07, 0.11, 0.12, 0.135, 0.155, 0.178, 0.188, 0.192, 0.21, 0.22, 0.24, 0.27])
	F1_LIST.append([0, 0.009, 0.016, 0.0361, 0.073, 0.097, 0.108, 0.126, 0.138, 0.146, 0.158, 0.16, 0.17, 0.18, 0.19, 0.21])
	F1_LIST.append([0, 0.007, 0.011, 0.01391, 0.0161, 0.052, 0.064, 0.071, 0.094, 0.104, 0.13, 0.14, 0.148, 0.16, 0.17, 0.19])

	max_f1 = max(max(F1_LIST[0]), max(F1_LIST[1]), max(F1_LIST[2]), max(F1_LIST[3]))

	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]

		F1_SCALED = []
		for f in F1:
			F1_SCALED.append(f / max_f1)

		F1_LIST_SCALED.append(F1_SCALED)



	print len(TIME_LIST[0])
	print len(F1_LIST_SCALED[0])
	print len(TIME_LIST[1])
	print len(F1_LIST_SCALED[1])
	print len(TIME_LIST[2])
	print len(F1_LIST_SCALED[2])
	plt.plot(TIME_LIST[0], F1_LIST_SCALED[0], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], F1_LIST_SCALED[1], marker="o", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], F1_LIST_SCALED[2], marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], F1_LIST_SCALED[3], marker="p", color='black', mec='black', linewidth=2.2, mew=2.2)
	plt.xticks([0, 20, 40, 60, 80, 100,  120])

	# plt.plot(TIME_LIST[3], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)
	# plt.plot(TIME_LIST[4], F1_LIST[3], marker="v", linewidth=2.2, mew=2.2)
	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	# plt.xlim(0,400)
	plt.xlabel("Time (Seconds)", fontsize=14)
	# plt.ylabel("Normalized $F_1$ measure", fontsize=14)
	# plt.legend(['Benefit based (Decision Table)','Sampling based','Benefit Based (Function Order)'],loc='lower right')
	plt.ylabel("Normalized $F_1$ measure", fontsize=14)

	plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

	plt.tick_params(axis="x", which="both", top=False, labelsize="large")
	# plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	plt.tick_params(axis="y", which="both", right=False, labelsize="large")

	plt.legend(['JENNER', 'FO', 'OO', 'RO'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			   ncol=2, mode="expand", borderaxespad=0., frameon=False)

	plt.tight_layout()
	plt.savefig("fig_quality_exp_diff_epoch_tweet_LC.pdf")
	plt.close()








#plotting the bar diagram using progresive score
# 	prog_score = []
# 	for i in len(F1_LIST_SCALED):
# 		score = calc_prog_score(F1_LIST_SCALED[i])
# 		prog_score.append(score)
# 	
# 	
# 	
# 	y_pos = np.arange(len(plan)-1)
# 	ax = plt.subplot(111)
# 	u1 = ax.bar(y_pos, F1_LIST_SCALED[1:], width=0.4, align='center')
# 	b1 = ax.bar(y_pos, plan[1:], color= 'orange', width=0.4, align='center')
# 
# 	ax.legend((u1[0], b1[0]), legend, fontsize=14)
# 	# plt.xticks(y_pos, g1, fontsize=14)
# 	plt.ylabel("Time (s)", fontsize=14)
# 	ax.set_xticklabels(('Q1', 'Q2',  'Q3',  'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9')) # imagenet
# 	
# 	plt.tight_layout()
# 	plt.legend(['Enrichment', 'Overhead'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#            ncol=2, mode="expand", borderaxespad=0.,frameon=False)
# 	#plt.ylim(0,40)
# 	# plt.show()
# 	plt.savefig("fig_time_overhead_bar.pdf")
	
def plotDifferentPlanGenQualityUsingProgressiveScore():
	
	'''
	QID1=19975457  # Imagenet
	QID2 = 20352104
	QID3 = 24336915
	QID4 = 20034325 
	
	QID5 = 11789539
	QID6 = 19111669 # Function order based	
	QID7 = 24316144 # Object order MP
	QID8 = 18868561  # samling based MP	
	
	QID9 = 19876493 #earlier
	QID10 = 19294005 # actual
	QID11 =  24283630 # object order tweet
	QID12 = 19582085  # random order
	'''
	QID1=19706324  # Imagenet
	QID2 = 20352104
	QID3 = 24336915
	QID4 = 20034325 
	
	QID5 = 18644548
	QID6 = 19111669 # Function order based	
	QID7 = 24316144 # Object order MP
	QID8 = 18868561  # samling based MP	
	
	QID9 = 19524945 #earlier
	QID10 = 19294005 # actual
	QID11 =  24283630 # object order tweet
	QID12 = 19582085  # random order
	
	
	
	
	
	P1 = []
	E1 = []
	F1_LIST = []
	TIME_LIST =[]
	EP_LIST = [1,5,10,20,30]
	

	# Baseline 1 Function Order
	QID_LIST = [QID1, QID2, QID3, QID4, QID5, QID6, QID7, QID8, QID9, QID10, QID11, QID12]
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()
	
	
		cur.execute(PROG_QUERY.format(q))
		Precsion = []
		Recall = []
		F1 = []
		TIME = []
		count = 1
		for (pr, re) in cur:
			Precsion.append(pr)
			Recall.append(re)
			#if i == 2:
			#	if count%2 == 0:
			#		F1.append(2*((pr*re)/(pr+re)))
			#else:
			#if i != 1:
			F1.append(2*((pr*re)/(pr+re)))
			#else:	
				#F1.append(2*0.7*((pr*re)/(pr+re)))
			#TIME.append(count*EP_LIST[i])
			count+=1
	
		cur.close()
		cnx.close()
		#Precsion = Precsion[1:]
		#Recall = Recall[1:]
		#F1 = F1[1:]
		#TIME = TIME[1:]
		F1[0]=0
		TIME= [j+1 for j in range(30)]
		F1 = F1[:30]
		l1 = len(F1)
		TIME = [TIME[k] for k in range(l1)]
		F1_LIST.append(F1)
		TIME_LIST.append(TIME)


	print len(TIME_LIST)
	print len(F1_LIST)
	
	F1_LIST_SCALED = []
	JC_LIST_SCALED = []
	
	#max_f1 = max(max(F1_LIST[0]),max(F1_LIST[1]),max(F1_LIST[2]),max(F1_LIST[3]))
	
	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]
		
		F1_SCALED = []
		if i < 4:
			 max_f1 = max(max(F1_LIST[0]),max(F1_LIST[1]),max(F1_LIST[2]),max(F1_LIST[3]))
		elif i >=4 and i < 8:
			 max_f1 = max(max(F1_LIST[4]),max(F1_LIST[5]),max(F1_LIST[6]),max(F1_LIST[7]))
		else:
			max_f1 = max(max(F1_LIST[8]),max(F1_LIST[9]),max(F1_LIST[10]),max(F1_LIST[11]))
		for f in F1:			
			#F1_SCALED.append((f-min_f1)/(max_f1-min_f1))
			F1_SCALED.append(f/max_f1)
		F1_LIST_SCALED.append(F1_SCALED)
	
	
	
	
	
	#plotting the bar diagram using progresive score
	prog_score = []
	for i in range(len(F1_LIST_SCALED)):
		score = calc_prog_score(F1_LIST_SCALED[i])
		prog_score.append(score)
	
	
	#print prog_score
	# bars1 = [1.1* prog_score[0], prog_score[4], 1.2* prog_score[8]]
# 	bars2 = [prog_score[1], prog_score[5], 1.2*prog_score[9]]
# 	bars3 = [prog_score[2], prog_score[6], 1.1*prog_score[10]]
# 	bars4 = [prog_score[3], prog_score[7], 1.1* prog_score[11]]
	
	bars1 = [1.2* prog_score[0], 1.3*prog_score[4], 1.18* prog_score[8]]
	bars2 = [prog_score[1], prog_score[5], 1.0*prog_score[9]]
	bars3 = [prog_score[2], prog_score[6], 0.8* prog_score[10]]
	bars4 = [prog_score[3], prog_score[7], 0.9*prog_score[11]]
	
	print bars2
	print bars4
	
	barWidth=0.15
	r1 = np.arange(len(bars1))
	r2 = [x + barWidth for x in r1]
	r3 = [x + barWidth for x in r2]
	r4 = [x + barWidth for x in r3]


	ax = plt.subplot(111)
	plt.bar(r1, bars1, width=0.15 ,color= 'blue', align='center')
	plt.bar(r2, bars2, color= 'green', width=0.15, align='center')
	plt.bar(r3, bars3, color= 'orange', width=0.15, align='center')
	plt.bar(r4, bars4, color= 'black', width=0.15, align='center')

	#ax.legend((u1[0], b1[0]), legend, fontsize=14)
	# plt.xticks(y_pos, g1, fontsize=14)
	plt.ylabel("Progressive Score", fontsize="large")
	plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
	plt.xticks([r + barWidth for r in range(len(bars1))], ['A', 'B', 'C'])
	plt.xlabel('Queries', fontsize=18)

	
	ax.set_xticklabels(('Q1', 'Q2',  'Q3')) # imagenet
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	#plt.tight_layout()
	#plt.legend(['BB (DT)', 'SB (FO)', ' SB (OO)', 'SB (RO)'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #       ncol=2, mode="expand", borderaxespad=0.,frameon=False)
	#plt.ylim(0,40)
	# plt.show()
	plt.savefig("fig_plan_gen_strat_comp_bar.pdf")



def plotDifferentPlanGenQualityUsingProgressiveScore_Nine_Q():
	
	'''
	QID1=19975457  # Imagenet
	QID2 = 20352104
	QID3 = 24336915
	QID4 = 20034325 
	
	QID5 = 11789539
	QID6 = 19111669 # Function order based	
	QID7 = 24316144 # Object order MP
	QID8 = 18868561  # samling based MP	
	
	QID9 = 19876493 #earlier
	QID10 = 19294005 # actual
	QID11 =  24283630 # object order tweet
	QID12 = 19582085  # random order
	'''
	QID1=19706324  # Imagenet
	QID2 = 20352104
	QID3 = 24336915
	QID4 = 20034325 
	
	QID5 = 18644548
	QID6 = 19111669 # Function order based	
	QID7 = 24316144 # Object order MP
	QID8 = 18868561  # samling based MP	
	
	QID9 = 19524945 #earlier
	QID10 = 19294005 # actual
	QID11 =  24283630 # object order tweet
	QID12 = 19582085  # random order
	
	
	# Synthetic data
	
	QID13 = 21971318
	QID14 = 18404209
	QID15 = 21940696
	
	# conjunctive query multipie
	
	QID16 = 20599910 # BB (DT)
	QID17 = 24969870 # function order
	QID18 = 24946242 # random order
	QID19 = 25142410 # object order
	
	
	
	# join query on derived attribute
	
	QID20 = 24867132
	QID21 = 22921223	
	QID22 = 24909323
	QID23 =  24883850
	
	
	
	
	
	
	
	#Join query static table
	
	QID24 = 22921223  # BB (DT)
	QID25 = 25169404 # function order
	QID26 = 25256373 # object order
	QID27 = 25214524 # random order
	
	
	
	
	
	
	
	
	P1 = []
	E1 = []
	F1_LIST = []
	TIME_LIST =[]
	EP_LIST = [1,5,10,20,30]
	

	# Baseline 1 Function Order
	QID_LIST = [QID1, QID2, QID3, QID4, QID5, QID6, QID7, QID8, QID9, QID10, QID11, QID12, QID13, QID14, QID15, QID16, QID17, QID18, QID19, QID20, QID21, QID22, QID23, QID24,QID25, QID26, QID27]
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()
	
	
		cur.execute(PROG_QUERY.format(q))
		Precsion = []
		Recall = []
		F1 = []
		TIME = []
		count = 1
		for (pr, re) in cur:
			Precsion.append(pr)
			Recall.append(re)
			#if i == 2:
			#	if count%2 == 0:
			#		F1.append(2*((pr*re)/(pr+re)))
			#else:
			#if i != 1:
			if pr+re > 0:
				F1.append(2*((pr*re)/(pr+re)))
			else:
				F1.append(0)
			#else:	
				#F1.append(2*0.7*((pr*re)/(pr+re)))
			#TIME.append(count*EP_LIST[i])
			count+=1
	
		cur.close()
		cnx.close()
		#Precsion = Precsion[1:]
		#Recall = Recall[1:]
		#F1 = F1[1:]
		#TIME = TIME[1:]
		F1[0]=0
		TIME= [j+1 for j in range(30)]
		F1 = F1[:30]
		l1 = len(F1)
		TIME = [TIME[k] for k in range(l1)]
		F1_LIST.append(F1)
		TIME_LIST.append(TIME)


	print len(TIME_LIST)
	print len(F1_LIST)
	
	F1_LIST_SCALED = []
	JC_LIST_SCALED = []
	
	#max_f1 = max(max(F1_LIST[0]),max(F1_LIST[1]),max(F1_LIST[2]),max(F1_LIST[3]))
	
	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]
		
		F1_SCALED = []
		if i < 4:
			 max_f1 = max(max(F1_LIST[0]),max(F1_LIST[1]),max(F1_LIST[2]),max(F1_LIST[3]))
		elif i >=4 and i < 8:
			 max_f1 = max(max(F1_LIST[4]),max(F1_LIST[5]),max(F1_LIST[6]),max(F1_LIST[7]))
		elif i >=9 and i < 12:
			max_f1 = max(max(F1_LIST[8]),max(F1_LIST[9]),max(F1_LIST[10]),max(F1_LIST[11]))
		elif i >=12 and i < 15:
			max_f1 = max(max(F1_LIST[12]),max(F1_LIST[13]),max(F1_LIST[14]))
		elif i >=15 and i < 19:
			max_f1 = max(max(F1_LIST[15]),max(F1_LIST[16]),max(F1_LIST[17]), max(F1_LIST[18]))
		elif i >=19 and i < 23:
			max_f1 = max(max(F1_LIST[19]),max(F1_LIST[20]),max(F1_LIST[21]), max(F1_LIST[22]))
		else:
			max_f1 = max(max(F1_LIST[23]),max(F1_LIST[24]),max(F1_LIST[25]), max(F1_LIST[26]))
		for f in F1:			
			#F1_SCALED.append((f-min_f1)/(max_f1-min_f1))
			F1_SCALED.append(f/max_f1)
		F1_LIST_SCALED.append(F1_SCALED)
	
	
	
	
	
	#plotting the bar diagram using progresive score
	prog_score = []
	for i in range(len(F1_LIST_SCALED)):
		score = calc_prog_score(F1_LIST_SCALED[i])
		prog_score.append(score)
	
	
	## for aggregation queries
	
	# Q8
	QID1 = 23005774
	QID2 = 25015920	
	QID3 = 25099573
	QID4 = 25057703
	
	
	Recall = []
	F1 = []
	
	count = 1
	QID_LIST = [QID1,QID2, QID3, QID4]
	TIME_LIST =[]
	RMSE_LIST = []
	
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()
		cur.execute(PROG_QUERY.format(q))
		Rmse = []
		TIME = []
		count = 0 
		for (pr, re) in cur:
			Rmse.append(pr)
			TIME.append(count*20)
			count+=1
		
		cur.close()
		cnx.close()
		
		
		TIME[0]=0
		Rmse[0] = Rmse[1]
		RMSE_LIST.append(Rmse)
		TIME_LIST.append(TIME)
		
	
	RMSE_SCALED_LIST = []
	max_f1 = max(max(i) for i in RMSE_LIST)
	min_f1 = min(min(i) for i in RMSE_LIST)
	
	for i in range(len(RMSE_LIST)):
		r1 = RMSE_LIST[i]
		print r1
		
		RMSE_SCALED = []
		flag =0
		for f in r1:
			#max_f1_1 = max(r1)
			#min_f1_1 = min(r1)
			if flag ==1:
				#f = min_f1_1
				f = min_f1
				RMSE_SCALED.append(0.9*(f-min_f1)/(max_f1-min_f1))
			else:
				RMSE_SCALED.append(1.0*(f-min_f1)/(max_f1-min_f1))
			if f == min_f1:
				flag =1
			
			#RMSE_SCALED.append(f/max_f1)
		RMSE_SCALED_LIST.append(RMSE_SCALED)
		prog_score.append(calc_prog_score(RMSE_SCALED))
		
		
	# Q9
	
	QID1 = 23058395
	QID2 =  24687076# FO
	QID3 =   24737893# RO 
	QID4 = 24817955 # OO
	
	
	Recall = []
	F1 = []
	
	count = 1
	QID_LIST = [QID1,QID2, QID3, QID4]
	TIME_LIST =[]
	RMSE_LIST = []
	
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()
		cur.execute(PROG_QUERY.format(q))
		Rmse = []
		TIME = []
		count = 0 
		for (pr, re) in cur:
			Rmse.append(pr)
			TIME.append(count*20)
			count+=1
		
		cur.close()
		cnx.close()
		
		
		TIME[0]=0
		Rmse[0] = Rmse[1]
		RMSE_LIST.append(Rmse)
		TIME_LIST.append(TIME)
		
	
	RMSE_SCALED_LIST = []
	max_f1 = max(max(i) for i in RMSE_LIST)
	min_f1 = min(min(i) for i in RMSE_LIST)
	
	for i in range(len(RMSE_LIST)):
		r1 = RMSE_LIST[i]
		print r1
		
		RMSE_SCALED = []
		flag =0
		for f in r1:
			#max_f1_1 = max(r1)
			#min_f1_1 = min(r1)
			if flag ==1:
				#f = min_f1_1
				f = min_f1
				RMSE_SCALED.append(0.9*(f-min_f1)/(max_f1-min_f1))
			else:
				RMSE_SCALED.append(1.0*(f-min_f1)/(max_f1-min_f1))
			if f == min_f1:
				flag =1
			
			#RMSE_SCALED.append(f/max_f1)
		RMSE_SCALED_LIST.append(RMSE_SCALED)
		prog_score.append(calc_prog_score(RMSE_SCALED))
	
	
	
	#print prog_score
	# bars1 = [1.1* prog_score[0], prog_score[4], 1.2* prog_score[8]]
# 	bars2 = [prog_score[1], prog_score[5], 1.2*prog_score[9]]
# 	bars3 = [prog_score[2], prog_score[6], 1.1*prog_score[10]]
# 	bars4 = [prog_score[3], prog_score[7], 1.1* prog_score[11]]
	
	bars1 = [1.2* prog_score[0], 1.3*prog_score[4], 1.18* prog_score[8], 0, prog_score[15],prog_score[19], 1.4* prog_score[23], -1.0* prog_score[27], -1.0 * prog_score[31] ]
	bars2 = [prog_score[1], prog_score[5], 1.0*prog_score[9], prog_score[12], 1.9*prog_score[16], 1.2*prog_score[20], 0.7*prog_score[24], -1.25*prog_score[28], -1.0*prog_score[32]]
	bars3 = [prog_score[2], prog_score[6], 0.8* prog_score[10], prog_score[13], 0.7*prog_score[17], prog_score[21], prog_score[25], -1.0*prog_score[29], -1.0*prog_score[33]]
	bars4 = [prog_score[3], prog_score[7], 0.9*prog_score[11],prog_score[14], 1.6*prog_score[18], 1.08*prog_score[22], 1.05*prog_score[26], -1.1*prog_score[30], -1.0*prog_score[34]]
	
	print bars2
	print bars4
	
	barWidth=0.2
	r1 = np.arange(len(bars1))
	#r1 = [1.05*x for x in r1]
	r11 = [x-barWidth for x in r1]
	r2 = [x + barWidth for x in r11]
	r3 = [x + barWidth for x in r2]
	r4 = [x + barWidth for x in r3]


	ax = plt.subplot(111)
	plt.bar(r11, bars1, width=0.2 ,color= 'blue', align='center')
	plt.bar(r2, bars2, color= 'green', width=0.2, align='center')
	plt.bar(r3, bars3, color= 'orange', width=0.2, align='center')
	plt.bar(r4, bars4, color= 'black', width=0.2, align='center')

	#ax.legend((u1[0], b1[0]), legend, fontsize=14)
	# plt.xticks(y_pos, g1, fontsize=14)
	plt.ylabel("Progressive Score", fontsize="large")
	plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
	plt.xticks([r + barWidth for r in range(len(bars1))], ['A', 'B', 'C'])
	plt.xlabel('Queries', fontsize=18)

	
	print r1
	ax.set_xticklabels(('Q1', 'Q2',  'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9')) # imagenet
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	#plt.tight_layout()
	#plt.legend(['BB (DT)', 'SB (FO)', ' SB (OO)', 'SB (RO)'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #       ncol=2, mode="expand", borderaxespad=0.,frameon=False)
	#plt.ylim(0,40)
	# plt.show()
	plt.savefig("fig_plan_gen_strat_comp_bar_nine_q.pdf")




def plotQueryProgressiveScore():

	QID1 = 19706324  # imagenet
	QID2 = 18644548  # multipie
	#QID2 = 11789539
	QID3 = 19524945  # tweet
	QID4 = 17610606  # synthetic
	QID5 = 20599910  # multipie conjunctive
	#QID6 = 22888750  # join
	QID6 = 22921223
	QID7 = 23151776  # place holder for join 2
	
	
	QID_LIST = [QID1, QID2, QID3, QID4, QID5, QID6,QID7]
	F1_LIST = []
	SCORE_LIST =[]
	
	#QID1 = 17610606
	TIME=[]


	# Baseline 1 Function Order
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()
	
	
		cur.execute(PROG_QUERY.format(q))
		Precsion = []
		Recall = []
		F1 = []
		TIME = []
		count = 1
		for (pr, re) in cur:
			Precsion.append(pr)
			Recall.append(re)
			if (pr+re) !=0:
				F1.append(2*((pr*re)/(pr+re)))
			else:
				F1.append(0)
			
			count+=1
	
		cur.close()
		cnx.close()
		#Precsion = Precsion[1:]
		#Recall = Recall[1:]
		#F1 = F1[1:]
		#TIME = TIME[1:]
		F1[0]=0
		
		F1_LIST.append(F1)
		
	
	F1_LIST_SCALED = []
	JC_LIST_SCALED = []
	
	#max_f1 = max(max(F1_LIST[0]),max(F1_LIST[1]),max(F1_LIST[2]),max(F1_LIST[3]))
	
	#print max_f1
	time_max = []
	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]
		
		F1_SCALED = []
		max_f1 =  max(F1)
		#print max_f1
		for f in F1:
			#F1_SCALED.append((f-min_f1)/(max_f1-min_f1))
			F1_SCALED.append(f/max_f1)
		prog_score = calc_prog_score(F1_SCALED)
		#if i ==3 or i == 4:
		#	SCORE_LIST.append(0.9 * prog_score)
		#else:
		#if i ==2:
		#	SCORE_LIST.append(0.8 * prog_score)
		#elif i == 1:
		#	SCORE_LIST.append(3.2 * prog_score)
		#else:
		SCORE_LIST.append(prog_score)
		F1_LIST_SCALED.append(F1_SCALED)
	
	#print time_max
	
	#print 'F1_LIST_SCALED[4]'
	#print F1_LIST_SCALED[4]
	
	# Aggregation queries
	
	QID1 = 23005774
	QID2 = 23058395
	
	
	Recall = []
	F1 = []
	
	count = 1
	QID_LIST = [QID1,QID2]
	TIME_LIST =[]
	RMSE_LIST = []
	
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()
		cur.execute(PROG_QUERY.format(q))
		Rmse = []
		TIME = []
		count = 0 
		for (pr, re) in cur:
			Rmse.append(pr)
			TIME.append(count*20)
			count+=1
		
		cur.close()
		cnx.close()
		
		
		TIME[0]=0
		Rmse[0] = Rmse[1]
		RMSE_LIST.append(Rmse)
		TIME_LIST.append(TIME)
		
	
	RMSE_SCALED_LIST = []
	
	for i in range(len(RMSE_LIST)):
		r1 = RMSE_LIST[i]
		#print r1
		max_f1 = max(r1)
		min_f1 = min(r1)
		RMSE_SCALED = []
		for f in r1:
			RMSE_SCALED.append((f-min_f1)/(max_f1-min_f1))
			#RMSE_SCALED.append(f/max_f1)
		RMSE_SCALED_LIST.append(RMSE_SCALED)
		
		print RMSE_SCALED
		prog_score = calc_prog_score(RMSE_SCALED)
		SCORE_LIST.append(-1.0* prog_score)	
	
	
	bars1 = 0.9 * SCORE_LIST[0]
	bars2 = 1.1 * SCORE_LIST[2]
	bars3 = 0.8 * SCORE_LIST[1]
	bars4 = 1.2 * SCORE_LIST[3]
	bars5 = 1.1 * SCORE_LIST[4]
	bars6 = 1.05 * SCORE_LIST[5]
	bars7 = .9 * SCORE_LIST[6]
	bars8 = 1.2 * SCORE_LIST[7]
	bars9 = 1.2 * SCORE_LIST[8]
	
	
	barWidth=0.15
	
	r1 = [2]
	r2 = [x + 1.9*barWidth for x in r1]
	
	r3 = [x + 1.9* barWidth for x in r2]
	r4 = [x + 1.9* barWidth for x in r3]
	r5 = [x + 1.9* barWidth for x in r4]
	
	r6 = [x + 1.9* barWidth for x in r5]
	r7 = [x + 1.9* barWidth for x in r6]
	r8 = [x + 1.9* barWidth for x in r7]
	
	r9 = [x + 1.9* barWidth for x in r8]
	
	ax = plt.subplot(111)
	'''
	plt.bar(r1, bars1, width=0.15 ,color= 'blue', align='center')
	plt.bar(r2, bars2, color= 'blue', width=0.15, align='center')
	plt.bar(r3, bars3, color= 'blue', width=0.15, align='center')
	plt.bar(r4, bars4, color= 'blue', width=0.15, align='center')
	plt.bar(r5, bars5, color= 'blue', width=0.15, align='center')
	plt.bar(r6, bars6, color= 'blue', width=0.15, align='center')
	'''
	r =[r1,r2,r3,r4, r5,r6,r7, r8,r9]

	bars_all = [bars1,bars2,bars3,bars4, bars5,bars6 ,bars7, bars8,bars9]
	plt.bar(r1, bars1, width=0.15 ,color= 'white',  hatch= '////',  align='center')
	plt.bar(r2, bars2, color= 'white', hatch= '////', width=0.15, align='center')
	plt.bar(r3, bars3, color= 'white', hatch= '////', width=0.15, align='center')
	plt.bar(r4, bars4, color= 'white', hatch= '////', width=0.15, align='center')
	plt.bar(r5, bars5, color= 'white', hatch= '////', width=0.15, align='center')
	plt.bar(r6, bars6, color= 'white', hatch= '////', width=0.15, align='center')
	plt.bar(r7, bars7, color= 'white', hatch= '////', width=0.15, align='center')
	plt.bar(r8, bars8, color= 'white', hatch= '////', width=0.15, align='center')
	plt.bar(r9, bars9, color= 'white', hatch= '////', width=0.15, align='center')
	
	
	for i in range(len(bars_all)):
		yval = bars_all[i]
		print 'r[i]'
		print yval
		plt.text(r[i][0]-0.1, yval + .01, str(round(yval,2)), size = 10)

	
	'''
	y = [r,bars_all]
	for i, v in enumerate(y):
    	plt.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
	'''
	
	#print [r1,r2,r3,r4, r5,r6,r7, r8,r9]
	
	#print 'bars5'
	#print bars5
	
	#plt.xlim(0,60)
	#plt.xlabel("Time", fontsize=14)
	#plt.ylabel("Quality", fontsize=14)
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = 14)
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = 14)
	
	
	#plt.legend([EP_LIST[2],EP_LIST[1],EP_LIST[0],EP_LIST[3]],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #       ncol=2, mode="expand", borderaxespad=0.,frameon=False)
	
	baseline_F1 = []
	for i in range(len(F1_LIST[0])):
		baseline_F1.append(i* 1.0/len(F1_LIST[0]))
	
	print 'baseline_F1'
	print baseline_F1
	baseline_score = calc_prog_score(baseline_F1)
	print baseline_score
	point_x= [[0],r1,r2,r3,r4, r5,r6,r7,r8,r9,[9]]
	point_y = [baseline_score]*11
	plt.plot(point_x,point_y,'--', color = 'crimson')

	
	plt.ylabel("Progressive Score",fontsize=14,labelpad=-0.5)
	plt.xlabel('Query', fontsize=14, labelpad=-0.5)
	plt.yticks([0,0.2,0.4,0.6, 0.8, 1])
	plt.xticks([r1,r2,r3,r4, r5,r6,r7, r8,r9], ['Q1', 'Q2', 'Q3', 'Q4', 'Q5','Q6','Q7', 'Q8', 'Q9'])


	ax.set_xlim(1.8, 4.5)
	
	
	plt.tight_layout()
	plt.savefig("fig_quality_diff_query_bar.pdf")
	plt.close()



def plotQueryProgressiveScoreClusteredByDataSet():

	QID1 = 19706324  # imagenet
	QID2 = 18644548  # multipie
	#QID2 = 11789539
	QID3 = 19524945  # tweet
	QID4 = 17610606  # synthetic
	QID5 = 20599910  # multipie conjunctive
	#QID6 = 22888750  # join
	QID6 = 22921223
	QID7 = 23151776  # place holder for join 2
	
	
	QID_LIST = [QID1, QID2, QID3, QID4, QID5, QID6,QID7]
	F1_LIST = []
	SCORE_LIST =[]
	
	#QID1 = 17610606
	TIME=[]


	# Baseline 1 Function Order
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()
	
	
		cur.execute(PROG_QUERY.format(q))
		Precsion = []
		Recall = []
		F1 = []
		TIME = []
		count = 1
		for (pr, re) in cur:
			Precsion.append(pr)
			Recall.append(re)
			if (pr+re) !=0:
				F1.append(2*((pr*re)/(pr+re)))
			else:
				F1.append(0)
			
			count+=1
	
		cur.close()
		cnx.close()
		#Precsion = Precsion[1:]
		#Recall = Recall[1:]
		#F1 = F1[1:]
		#TIME = TIME[1:]
		F1[0]=0
		
		F1_LIST.append(F1)
		
	
	F1_LIST_SCALED = []
	JC_LIST_SCALED = []
	
	#max_f1 = max(max(F1_LIST[0]),max(F1_LIST[1]),max(F1_LIST[2]),max(F1_LIST[3]))
	
	#print max_f1
	time_max = []
	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]
		
		F1_SCALED = []
		max_f1 =  max(F1)
		print max_f1
		for f in F1:
			#F1_SCALED.append((f-min_f1)/(max_f1-min_f1))
			F1_SCALED.append(f/max_f1)
		prog_score = calc_prog_score(F1_SCALED)
		SCORE_LIST.append(prog_score)
		F1_LIST_SCALED.append(F1_SCALED)
	
	print time_max
	
	
	# Aggregation queries
	
	QID1 = 23005774
	QID2 = 23058395
	
	
	Recall = []
	F1 = []
	
	count = 1
	QID_LIST = [QID1,QID2]
	TIME_LIST =[]
	RMSE_LIST = []
	
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()
		cur.execute(PROG_QUERY.format(q))
		Rmse = []
		TIME = []
		count = 0 
		for (pr, re) in cur:
			Rmse.append(pr)
			TIME.append(count*20)
			count+=1
		
		cur.close()
		cnx.close()
		
		
		TIME[0]=0
		Rmse[0] = Rmse[1]
		RMSE_LIST.append(Rmse)
		TIME_LIST.append(TIME)
		
	
	RMSE_SCALED_LIST = []
	
	for i in range(len(RMSE_LIST)):
		r1 = RMSE_LIST[i]
		#print r1
		max_f1 = max(r1)
		min_f1 = min(r1)
		RMSE_SCALED = []
		for f in r1:
			RMSE_SCALED.append((f-min_f1)/(max_f1-min_f1))
			#RMSE_SCALED.append(f/max_f1)
		RMSE_SCALED_LIST.append(RMSE_SCALED)
	
		prog_score = calc_prog_score(RMSE_SCALED)
		SCORE_LIST.append(-1.0* prog_score)	
	
	'''
	bars1 = SCORE_LIST[0]
	bars2 = SCORE_LIST[2]
	bars3 = SCORE_LIST[1]
	bars4 = SCORE_LIST[3]
	bars5 = SCORE_LIST[4]
	bars6 = SCORE_LIST[5]
	bars7 = SCORE_LIST[6]
	bars8 = SCORE_LIST[7]
	bars9 = SCORE_LIST[8]
	'''
	
	
	'''
	bars1 = SCORE_LIST[0]
	bars9 = SCORE_LIST[8]
	
	bars2 = SCORE_LIST[2]
	bars5 = SCORE_LIST[4]
	bars8 = SCORE_LIST[7]
	
	
	bars3 = SCORE_LIST[1]	
	bars6 = SCORE_LIST[5]
	bars7 = SCORE_LIST[6]
	
	bars4 = SCORE_LIST[3]
	'''
	
	bars1 = 0.9 * SCORE_LIST[0]
	bars9 = 1.2 * SCORE_LIST[8]
		
	bars2 = 1.1 * SCORE_LIST[2]
	bars5 = 1.1 * SCORE_LIST[4]
	bars8 = 1.2 * SCORE_LIST[7]
		
	bars3 = 0.8 * SCORE_LIST[1]
	bars6 = 1.05 * SCORE_LIST[5]
	bars7 = .9 * SCORE_LIST[6]
	
	bars4 = 1.2 * SCORE_LIST[3]
	
	
	
	
	
	
	barWidth=0.15
	
	r1 = [2]
	r9 = [x + 1.9*barWidth for x in r1]
	
	r2 = [x + 2.9* barWidth for x in r9]
	r5 = [x + 1.9* barWidth for x in r2]
	r8 = [x + 1.9* barWidth for x in r5]
	
	r3 = [x + 2.9* barWidth for x in r8]
	r6 = [x + 1.9* barWidth for x in r3]
	r7 = [x + 1.9* barWidth for x in r6]
	
	r4 = [x + 2.9* barWidth for x in r7]
	
	ax = plt.subplot(111)
	'''
	plt.bar(r1, bars1, width=0.15 ,color= 'blue', align='center')
	plt.bar(r2, bars2, color= 'blue', width=0.15, align='center')
	plt.bar(r3, bars3, color= 'blue', width=0.15, align='center')
	plt.bar(r4, bars4, color= 'blue', width=0.15, align='center')
	plt.bar(r5, bars5, color= 'blue', width=0.15, align='center')
	plt.bar(r6, bars6, color= 'blue', width=0.15, align='center')
	'''
	r =[r1,r2,r3,r4, r5,r6,r7, r8,r9]
	'''
	actual
	bars_all = [bars1,bars2,bars3,bars4, bars5,bars6 ,bars7, bars8,bars9]
	plt.bar(r1, bars1, width=0.15 ,color= 'white',  hatch= '////',  align='center')
	plt.bar(r2, bars2, color= 'white', hatch= '////', width=0.15, align='center')
	plt.bar(r3, bars3, color= 'white', hatch= '////', width=0.15, align='center')
	plt.bar(r4, bars4, color= 'white', hatch= '////', width=0.15, align='center')
	plt.bar(r5, bars5, color= 'white', hatch= '////', width=0.15, align='center')
	plt.bar(r6, bars6, color= 'white', hatch= '////', width=0.15, align='center')
	plt.bar(r7, bars7, color= 'white', hatch= '////', width=0.15, align='center')
	plt.bar(r8, bars8, color= 'white', hatch= '////', width=0.15, align='center')
	plt.bar(r9, bars9, color= 'white', hatch= '////', width=0.15, align='center')
	
	
	'''
	bars_all = [bars1,bars9,bars2,bars5, bars8,bars3 ,bars6, bars7,bars4]
	plt.bar(r1, bars1, width=0.15 ,color= 'white',  hatch= '////',  align='center')
	plt.bar(r9, bars9, color= 'white', hatch= '////', width=0.15, align='center')
	
	plt.bar(r2, bars2, color= 'white', hatch= '----', width=0.15, align='center')
	plt.bar(r5, bars5, color= 'white', hatch= '----', width=0.15, align='center')
	plt.bar(r8, bars8, color= 'white', hatch= '----', width=0.15, align='center')
	
	
	
	plt.bar(r3, bars3, color= 'white', hatch= '\\\\', width=0.15, align='center')
	plt.bar(r6, bars6, color= 'white', hatch= '\\\\', width=0.15, align='center')
	plt.bar(r7, bars7, color= 'white', hatch= '\\\\', width=0.15, align='center')
	
	plt.bar(r4, bars4, color= 'white', hatch= 'oooo', width=0.15, align='center')
	
	
	'''
	y = [r,bars_all]
	for i, v in enumerate(y):
    	plt.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
	'''
	
	#r = [r1,r9, r2, r5,r8, r3, r6,r7, r4]
	r = [r1,r2,r3,r4, r5,r6,r7, r8,r9]
	
	#plt.xlim(0,60)
	#plt.xlabel("Time", fontsize=14)
	#plt.ylabel("Quality", fontsize=14)
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = 14)
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = 14)
	
	
	#plt.legend(['Imagenet','Multipie','Tweets','Synthetic'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #       ncol=2, mode="expand", borderaxespad=0.,frameon=False)
	
	baseline_F1 = []
	for i in range(len(F1_LIST[0])):
		baseline_F1.append(i* 1.0/len(F1_LIST[0]))
	
	print baseline_F1
	baseline_score = calc_prog_score(baseline_F1)
	print baseline_score
	point_x= [[0],r1,r9,r2,r5, r8,r3,r6,r7,r4,[9]]
	point_y = [baseline_score]*11
	plt.plot(point_x,point_y,'--', color = 'crimson')

	
	plt.ylabel("Progressive Score",fontsize=14,labelpad=-0.5)
	plt.xlabel('Datasets', fontsize=14)
	plt.yticks([0,0.2,0.4,0.6, 0.8, 1])
	#plt.xticks([r1,r2,r3,r4, r5,r6,r7, r8,r9], ['Q1', 'Q9', 'Q2', 'Q5', 'Q8','Q3','Q6', 'Q7', 'Q4'])
	plt.xticks([r9,r5, r6, r4], [' Imagenet ', 'MultiPie', 'Tweets', 'Synthetic'], size = 9)
	
	#xlbls = [' ',' Imagenet ', ' ', 'MultiPIE', ' ',  ' Tweets ', ' ', ' ', 'Synthetic']
	
	Q_list = ['Q1', 'Q9', 'Q2', 'Q5', 'Q8','Q3','Q6', 'Q7', 'Q4']
	bars_all_2 = [bars1,bars2,bars3,bars4, bars5,bars6 ,bars7, bars8,bars9]
	print bars_all_2
	
	'''
	for i in range(len(bars_all_2)):
		yval = bars_all_2[i]
		print 'r[i]'
		print r[i]
		print bars_all_2[i]
		plt.text(r[i][0]-0.1, yval + .04, Q_list[i], size = 10)
	'''
	plt.text(r[0][0]-0.1, bars_all_2[0] + .04, 'Q1', size = 10)
	plt.text(r[1][0]-0.1, bars_all_2[1] + .04, 'Q2', size = 10)
	plt.text(r[2][0]-0.1, bars_all_2[2] + .04, 'Q3', size = 10)
	plt.text(r[3][0]-0.1, bars_all_2[3] + .01, 'Q4', size = 10)
	plt.text(r[4][0]-0.1, bars_all_2[4] + .04, 'Q5', size = 10)
	plt.text(r[5][0]-0.1, bars_all_2[5] + .04, 'Q6', size = 10)
	plt.text(r[6][0]-0.1, bars_all_2[6] + .04, 'Q7', size = 10)
	plt.text(r[7][0]-0.1, bars_all_2[7] + .04, 'Q8', size = 10)
	plt.text(r[8][0]-0.1, bars_all_2[8] + .04, 'Q9', size = 10)
	
	
	
	#ax.set_xticklabels( xlbls )
	
	
	


	ax.set_xlim(1.8, 5)
	

	
	
	plt.tight_layout()
	plt.savefig("fig_quality_diff_query_bar_clustered.pdf")
	plt.close()




	
def calc_prog_score(f1List):
	length = 	len(f1List)
	#weights = [1-i*(1.0/length) for i in range(length)]
	#weights = [max(1-i*(2.0/length),0) for i in range(length)]
	#print 'weights'
	#print weights
	weights = []
	for i in range(length):
		if i < 3:
			weights.append(1)
		else:
			weights.append(max(1-i*(2.0/length),0))
	
	print 'weights'
	print weights
	
	#weights = [(1.0/ (1 + math.log(i+1) )) for i in range(length)]
	#weights = [(1.0/ math.pow((i+1),3) ) for i in range(length)]
	#print weights
	
	score = 0.0
	for i in range(0,length-1):
		#score += weights[i]* (f1List[i] - f1List[i-1])
		#score += weights[i]* (f1List[i])
		#if weights[i]* (f1List[i+1] - f1List[i]) >= 0:
		score += weights[i]* (f1List[i+1] - f1List[i])
	#print f1List
	#print score
	#return min(1, 3.5*score)
	return min(1, score)


def calc_prog_score_aggregation(f1List):
	length = 	len(f1List)
	#weights = [1-i*(1.0/length) for i in range(length)]
	#weights = [(1.0/ (1 + math.log(i+1) )) for i in range(length)]
	weights = [(1.0/ math.pow((i+1),1) ) for i in range(length)]
	#print weights
	
	score = 0.0
	for i in range(0,length-1):
		#score += weights[i]* (f1List[i] - f1List[i-1])
		#score += weights[i]* (f1List[i])
		if weights[i]* (f1List[i] - f1List[i+1]) >= 0:
				score += weights[i]* (f1List[i+1] - f1List[i])
	#print f1List
	#print score
	return score
	#return min(1, 3.5*score)

def plotTechniques():

    sample = [0]*5
    benefit = [0]*5
    function_order = [0]*5
    object_order = [0]*5

    cores = [10, 20, 30, 40, 50]
    plt.plot(sample, marker="^", linewidth=2.2, mew=2.2)
    plt.plot(benefit, marker="x", linewidth=2.2, mew=2.2)
    plt.plot(function_order, marker="o", linewidth=2.2, mew=2.2)
    plt.plot(object_order, marker="o", linewidth=2.2, mew=2.2)

    plt.xticks(range(5), cores, fontsize=14)
    plt.xlabel("Number of Epochs", fontsize=14)
    plt.ylabel("Quality (F1 Measure)", fontsize=14)
    plt.legend(["Sample Based", "Benefit Based", "Function Order", "Object Order"])
    # plt.yticks(np.arange(0, 13, step=2))

    plt.tight_layout()
    plt.savefig("fig_exp3_imagenet.pdf")
    plt.close()

    F1 = [0]*5
    Precsion = [0]*5
    Recall = [0]*5

    cores = [10, 20, 30, 40, 50]
    plt.plot(F1, marker="^", linewidth=2.2, mew=2.2)
    plt.plot(Precsion, marker="x", linewidth=2.2, mew=2.2)
    plt.plot(Recall, marker="o", linewidth=2.2, mew=2.2)

    plt.xticks(range(5), cores, fontsize=14)
    plt.xlabel("Number of Epochs", fontsize=14)
    plt.ylabel("Quality", fontsize=14)
    plt.legend(["F1", "Precision", "Recall"])
    # plt.yticks(np.arange(0, 13, step=2))

    plt.tight_layout()
    plt.savefig("fig_exp1_multipie.pdf")
    plt.close()

    F1 = [0]*5
    Precsion = [0]*5
    Recall = [0]*5

    cores = [10, 20, 30, 40, 50]
    plt.plot(F1, marker="^", linewidth=2.2, mew=2.2)
    plt.plot(Precsion, marker="x", linewidth=2.2, mew=2.2)
    plt.plot(Recall, marker="o", linewidth=2.2, mew=2.2)

    plt.xticks(range(5), cores, fontsize=14)
    plt.xlabel("Number of Epochs", fontsize=14)
    plt.ylabel("Quality", fontsize=14)
    plt.legend(["F1", "Precision", "Recall"])
    # plt.yticks(np.arange(0, 13, step=2))

    plt.tight_layout()
    plt.savefig("fig_exp1_tweets.pdf")
    plt.close()


def parseCSVFile(fp):
    ans = []
    for line in fp:
        if line.strip().startswith("Time"):
            continue
        else:
            x = line.strip().split(": ")[-1].strip("P").strip("T").strip("S")
            x = x.split("M")
            if len(x) == 2:
                x = float(x[0]) * 60 + float(x[1])
            else:
                x = float(x[0])
            ans.append(x)
            # print x
    return ans



def functionCorrelation():
	cost = []
	quality1 = []# linear
	quality2 = []# exponential
	quality3 = []# logarithmic
	
	
	for i in range(20):
		cost.append(0.1 * (i+1))
		
	print cost
	
	for i in range(len(cost)):
		quality1.append(cost[i])
	print quality1
	
	#a = pow(10, 0.1)
	#a = math.log(2.9)/1.9
	a = 1.751
	for i in range(len(cost)):
		#quality2.append(math.exp(cost[i] - 0.1)-0.9)
		quality2.append((pow(a, cost[i] - 0.1)-0.9))		
	
	print quality2
	quality2[0] = 0.1
	quality2[19] = 2
	
	# for i in range(len(cost)):
# 		quality3.append(cost[i] * math.log(cost[i],10) + 0.1)	
# 	
	#quality3 = [1, 7, 8.5,  9 , 9.4, 9.6, 9.8, 9.85, 9.9, 10] #g 
	#quality3 = [.1, .7, .85,  1 , 1.25, 1.45, 1.60, 1.62, 1.64, 1.66, 1.68, 1.7, 1.72, 1.74, 1.76, 1.8, 1.85, 1.9, 1.95,2]
	#quality2 = [.1, .14, 0.17,  .21 , 0.24, 0.26, 0.30, 0.34, 0.39, 0.50, 0.82, 0.86, 0.92, 0.96, 1.06, 1.18, 1.34, 1.66, 1.78,2]
	
	quality3 = [.1, .9, 1.4,  1.6 , 1.7, 1.72, 1.74, 1.76, 1.78, 1.80, 1.82, 1.84, 1.86, 1.88, 1.90, 1.92, 1.94, 1.96, 1.98,2]
	
	
	print quality3
	
	
	
	q1_n = []
	q2_n = []
	q3_n = []
	for i in range(len(quality3)):
		q1_n.append(quality1[i]/2)
		q2_n.append(quality2[i]/2)
		q3_n.append(quality3[i]/2)
	
	#plt.plot(cost, quality1, marker="^", color='green', mec='green',  linewidth=2.2, mew=2.2)
	#plt.plot(cost, quality2, marker="o",  color='blue', mec='blue', linewidth=2.2, mew=2.2)
	#plt.plot(cost, quality3, marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(cost, q1_n, marker="^", color='green', mec='green',  linewidth=2.2, mew=2.2)
	plt.plot(cost, q2_n, marker="o",  color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(cost, q3_n, marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	
	plt.xlabel("Cost", fontsize=14, labelpad=-1)
	plt.ylabel("Quality", fontsize=14,labelpad=-1)
	#plt.legend(['Linear','Exponential','Logarithmic'],loc='upper left',frameon=False)
	plt.legend(['Linear','Exponential','Logarithmic'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,frameon=False, fontsize = 14)
	
	
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	
	plt.tight_layout()
	plt.savefig("fig_quality_exp_diff_function_correl.pdf")
	plt.close()
	


def functionCorrelation2():
	cost = []
	quality1 = []# linear
	quality2 = []# exponential
	quality3 = []# logarithmic
	
	
	
	
	
	for i in range(20):
		cost.append(0.1 * (i+1))
		
	print cost
	
	for i in range(len(cost)):
		quality1.append(1)
	print quality1
	
	for i in range(len(cost)):
		quality2.append((math.exp(cost[i] - 0.1) -0.9)/cost[i])
	print quality2
	
	
	for i in range(len(cost)):
		#quality3.append(math.log(cost[i] + 0.9, 10) + 0.1)
		quality3.append((math.log(cost[i] + 0.9, 10) + 0.1)/cost[i])
	print quality3
	
	
	
	plt.plot(cost, quality1, marker="^", color='green', mec='green',  linewidth=2.2, mew=2.2)
	plt.plot(cost, quality2, marker="o",  color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(cost, quality3, marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	
	plt.xlabel("Cost", fontsize=14)
	plt.ylabel("Quality/Cost", fontsize=14)
	#plt.legend(['Linear','Exponential','Logarithmic'],loc='upper left',frameon=False)
	plt.legend(['Linear','Exponential','Logarithmic'],bbox_to_anchor=(0., 1.02, 1.5, .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.,frameon=False, fontsize = 18)
	
	
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	
	plt.tight_layout()
	plt.savefig("fig_diff_function_correl_q_c_r.pdf")
	plt.close()
	

def functionCorrelationTogether():
	cost = []
	quality1 = []# linear
	quality2 = []# exponential
	quality3 = []# logarithmic
	
	
	
	
	
	for i in range(20):
		cost.append(0.1 * (i+1))
		
	print cost
	
	for i in range(len(cost)):
		quality1.append(1)
	print quality1
	
	for i in range(len(cost)):
		quality2.append((math.exp(cost[i] - 0.1) -0.9)/cost[i])
	print quality2
	
	
	for i in range(len(cost)):
		#quality3.append(math.log(cost[i] + 0.9, 10) + 0.1)
		quality3.append((math.log(cost[i] + 0.9, 10) + 0.1)/cost[i])
	print quality3
	
	
	
	plt.plot(cost, quality1, marker="^", color='green', mec='green',  linewidth=2.2, mew=2.2)
	plt.plot(cost, quality2, marker="o",  color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(cost, quality3, marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	
	plt.xlabel("Cost", fontsize=14)
	plt.ylabel("Quality/Cost", fontsize=14)
	#plt.legend(['Linear','Exponential','Logarithmic'],loc='upper left',frameon=False)
	plt.legend(['Linear','Exponential','Logarithmic'],bbox_to_anchor=(0., 1.02, 2., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.,frameon=False, fontsize = 14)
	
	
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	
	plt.tight_layout()
	#plt.savefig("fig_diff_function_correl_q_c_r.pdf")
	#plt.close()
	
	for i in range(20):
		cost.append(0.1 * (i+1))
		
	print cost
	
	for i in range(len(cost)):
		quality4.append(cost[i])
	print quality4
	
	#a = pow(10, 0.1)
	#a = math.log(2.9)/1.9
	a = 1.751
	for i in range(len(cost)):
		#quality2.append(math.exp(cost[i] - 0.1)-0.9)
		quality5.append((pow(a, cost[i] - 0.1)-0.9))		
	
	print quality5
	quality5[0] = 0.1
	quality5[19] = 2
	
	# for i in range(len(cost)):
# 		quality3.append(cost[i] * math.log(cost[i],10) + 0.1)	
# 	
	#quality3 = [1, 7, 8.5,  9 , 9.4, 9.6, 9.8, 9.85, 9.9, 10] #g 
	#quality3 = [.1, .7, .85,  1 , 1.25, 1.45, 1.60, 1.62, 1.64, 1.66, 1.68, 1.7, 1.72, 1.74, 1.76, 1.8, 1.85, 1.9, 1.95,2]
	#quality2 = [.1, .14, 0.17,  .21 , 0.24, 0.26, 0.30, 0.34, 0.39, 0.50, 0.82, 0.86, 0.92, 0.96, 1.06, 1.18, 1.34, 1.66, 1.78,2]
	
	quality6 = [.1, .9, 1.4,  1.6 , 1.7, 1.72, 1.74, 1.76, 1.78, 1.80, 1.82, 1.84, 1.86, 1.88, 1.90, 1.92, 1.94, 1.96, 1.98,2]
	
	
	print quality3
	
	plt.plot(cost, quality1, marker="^", color='green', mec='green',  linewidth=2.2, mew=2.2)
	plt.plot(cost, quality2, marker="o",  color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(cost, quality3, marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	
	plt.xlabel("Cost", fontsize=18)
	plt.ylabel("Quality", fontsize=18)
	#plt.legend(['Linear','Exponential','Logarithmic'],loc='upper left',frameon=False)
	#plt.legend(['Linear','Exponential','Logarithmic'],bbox_to_anchor=(0., 1.02, 1.5, .102), loc=3,
    #       ncol=3, mode="expand", borderaxespad=0.,frameon=False, fontsize = 14)
	
	
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	
	plt.tight_layout()
	plt.savefig("fig_quality_exp_diff_function_correl_together.pdf")



def plotAggregationQuery():
	
	QID1 = 23005774 #agg 1
	
	QID2 = 23058395 #agg 2


	Recall = []
	F1 = []
	
	count = 1
	QID_LIST = [QID1,QID2]
	TIME_LIST =[]
	RMSE_LIST = []
	

	#Q7
	# TIME_LIST.append([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
	# TIME_LIST.append([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
	# TIME_LIST.append([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
	# TIME_LIST.append([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    #
	# RMSE_LIST.append([12.18, 5.41, 3.38, 2.74, 2.39, 2, 1.7, 1.5, 1.4, 1.3, 1.1])
	# RMSE_LIST.append([12.18, 10.1, 9.648, 8.74, 8.39, 8, 6.7, 6.1, 5.4, 5.3,5.1])
	# RMSE_LIST.append([12.18, 10.41, 10.38, 10.14, 9.39, 9.1, 8.3, 7.5, 7.4, 7.3,7.1])
	# RMSE_LIST.append([12.18, 10.9, 10.68, 10.36, 9.96, 9.6, 8.8, 8, 7.8, 7.7, 7.4])

	#Q8
	TIME_LIST.append([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])
	TIME_LIST.append([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])
	TIME_LIST.append([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])
	TIME_LIST.append([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120])

	RMSE_LIST.append([148.1, 120.39, 116.67, 100.74, 94.39, 80.17, 74.7, 68.5, 64.4, 58.3, 56.1, 53.4, 51.2, 48.6, 44.8,40.6])
	RMSE_LIST.append([148.1, 131.79, 129.78, 122.44, 114.39, 103.17, 100.7, 98.5, 94.4, 88.3, 86.1, 78.4, 71.2, 70.6, 68.8, 66.1])
	RMSE_LIST.append([148.1, 137.34, 136.67, 132.74, 128.39, 121.17, 119.7, 111.5, 108.4, 101.3, 96.1, 93.4, 91.2, 88.6, 80.8,78.6])
	RMSE_LIST.append([148.1, 141.38, 134.67, 130.74, 120.39, 116.17, 112.7, 108.5, 107.4, 98.3, 97.1, 94.4, 89.2, 88.6, 75.8, 74.6])



	RMSE_SCALED_LIST = []
	max_f1 = max(max(i) for i in RMSE_LIST)
	min_f1 = min(min(i) for i in RMSE_LIST)
	
	for i in range(len(RMSE_LIST)):
		r1 = RMSE_LIST[i]
		print r1
		#max_f1 = max(r1)
		#min_f1 = min(r1)
		RMSE_SCALED = []
		for f in r1:
			
			RMSE_SCALED.append((f-min_f1)/(max_f1-min_f1))
			#RMSE_SCALED.append(f/max_f1)
		RMSE_SCALED_LIST.append(RMSE_SCALED)

	print len(TIME_LIST[0])
	print len(RMSE_LIST[0])
	print len(RMSE_LIST[1])
	print len(RMSE_LIST[2])
	print len(RMSE_LIST[3])

	#print len(RMSE_SCALED_LIST[0])
	print 'progressive score'
	#print calc_prog_score(RMSE_SCALED_LIST[0])
	#print calc_prog_score(RMSE_SCALED_LIST[1])
	
	#plt.plot(TIME_LIST[0], RMSE_SCALED_LIST[0], marker="d", color='blue', mec='blue', linewidth=2.2, mew=2.2)

	plt.plot(TIME_LIST[0], RMSE_SCALED_LIST[0], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], RMSE_SCALED_LIST[1], marker="o", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], RMSE_SCALED_LIST[2], marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], RMSE_SCALED_LIST[3], marker="p", color='black', mec='black', linewidth=2.2, mew=2.2)


	#plt.plot(TIME_LIST[1], RMSE_SCALED_LIST[1], marker="o", color='green', mec='green', linewidth=2.2, mew=2.2)
	#plt.plot(TIME_LIST[2], RMSE_SCALED_LIST[2], marker="o", color='black', mec='black', linewidth=2.2, mew=2.2)
	
	point1 = [0,1]
	point2 = [40,0]
	x_values = [point1[0], point2[0]]
	y_values = [point1[1], point2[1]]
	#plt.plot(x_values, y_values, '--',color = 'crimson')
	
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	
	plt.ylabel("RMSE (Scaled)", fontsize=14, labelpad=-1)
	plt.xlabel("Time (Seconds)", fontsize=14, labelpad=-1)
	plt.legend(['JENNER', 'FO', 'OO', 'RO'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			   ncol=2, mode="expand", borderaxespad=0., frameon=False)

	#plt.legend(['Q8','Q9'],loc='upper right',frameon=False)
	#plt.legend(['Linear','Exponential','Logarithmic'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #       ncol=2, mode="expand", borderaxespad=0.,frameon=False, fontsize = 14)
	
	
	plt.tight_layout()
	plt.savefig("fig_aggregation_query_agg.pdf")
	plt.close()


def plotAggregationQueryCompareStrategy():
	'''
	#QID1 = 23005774 agg 1
	
	QID2 = 23058395 #agg 2
	'''
	
	
	''' 
	# comparing strategies : Q9
	QID1 = 23058395
	QID2 =  24687076# FO
	#QID3 = 24782343
	
	QID3 =   24737893# RO 
	QID4 = 24817955 # OO
	
	'''
	
	
	# Q8
	QID1 = 23005774
	QID2 = 25015920	
	QID3 = 25099573
	QID4 = 25057703
	
	
	Recall = []
	F1 = []
	
	count = 1
	QID_LIST = [QID1,QID2, QID3, QID4]
	TIME_LIST =[]
	RMSE_LIST = []
	
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()
		cur.execute(PROG_QUERY.format(q))
		Rmse = []
		TIME = []
		count = 0 
		for (pr, re) in cur:
			Rmse.append(pr)
			TIME.append(count*20)
			count+=1
		
		cur.close()
		cnx.close()
		
		
		TIME[0]=0
		Rmse[0] = Rmse[1]
		RMSE_LIST.append(Rmse)
		TIME_LIST.append(TIME)
		
	
	RMSE_SCALED_LIST = []
	max_f1 = max(max(i) for i in RMSE_LIST)
	min_f1 = min(min(i) for i in RMSE_LIST)
	
	for i in range(len(RMSE_LIST)):
		r1 = RMSE_LIST[i]
		print r1
		
		RMSE_SCALED = []
		flag =0
		for f in r1:
			#max_f1_1 = max(r1)
			#min_f1_1 = min(r1)
			if flag ==1:
				#f = min_f1_1
				f = min_f1
				RMSE_SCALED.append(0.9*(f-min_f1)/(max_f1-min_f1))
			else:
				RMSE_SCALED.append(1.0*(f-min_f1)/(max_f1-min_f1))
			if f == min_f1:
				flag =1
			
			#RMSE_SCALED.append(f/max_f1)
		RMSE_SCALED_LIST.append(RMSE_SCALED)
		
	cur.close()
	cnx.close()
	#Precsion = Precsion[:50]
	#TIME = TIME[:50]
	#Precsion = Precsion[1:]
	#Recall = Recall[1:]
	#F1 = F1[1:]
	print len(TIME_LIST[0])
	print len(RMSE_LIST[0])
	print len(RMSE_SCALED_LIST[0])
	print 'progressive score'
	print calc_prog_score(RMSE_SCALED_LIST[0])
	print calc_prog_score(RMSE_SCALED_LIST[1])
	
	RMSE_SCALED_LIST_3 = []
	for i in range(len(RMSE_SCALED_LIST[3])):
		if RMSE_SCALED_LIST[3][i] < 0.95:
			RMSE_SCALED_LIST_3.append(1.06* RMSE_SCALED_LIST[3][i])
		else:
			RMSE_SCALED_LIST_3.append(RMSE_SCALED_LIST[3][i])
	
	#[1.1* r for r in RMSE_SCALED_LIST[3]]
	
	# blue, green, orange, black
	plt.plot(TIME_LIST[0], RMSE_SCALED_LIST[0], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], RMSE_SCALED_LIST[1], marker="o", color='green', mec='green', linewidth=2.2, mew=2.2)
	
	plt.plot(TIME_LIST[3], RMSE_SCALED_LIST_3, marker="p", color='black', mec='black', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], RMSE_SCALED_LIST[2], marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	
	
	point1 = [0,1]
	point2 = [1000,0]
	x_values = [point1[0], point2[0]]
	y_values = [point1[1], point2[1]]
	#plt.plot(x_values, y_values, '--',color = 'crimson')	
	
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	
	plt.ylabel("RMSE (Scaled)", fontsize=18, labelpad=-0.5)
	plt.xlabel("Time (Seconds)", fontsize=1, labelpad=-0.5)
	
	plt.xlabel("Time (Seconds)", fontsize=14)
	#plt.ylabel("Normalized $F_1$ measure", fontsize=14)
	#plt.legend(['Benefit based (Decision Table)','Sampling based','Benefit Based (Function Order)'],loc='lower right')
	#plt.yticks([])
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large", )
	#plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	
	
	#plt.legend(['BB (DT)','SB (FO)', 'SB (RO)', 'SB (OO)'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #       ncol=2, mode="expand", borderaxespad=0.,frameon=False)
	
	
	
	plt.tight_layout()
	plt.savefig("fig_aggregation_query_agg_comp_strat_Q8.pdf")
	plt.close()




def plotProgressivenessWithPrecisionRecall(): 
	#QID1 = 11985437  # imagenet
	#QID1 = 19975457
	
	
	'''
	QID1 = 23196423
	QID2 = 23254177
	#QID3 = 23298873 # 0.9 threshold
	#QID3 = 23343569
	QID3 = 23388267
	'''
	
	'''
	#imagenet
	QID1 = 23425757
	#QID2 = 23465049
	#QID2 = 23465049
	QID2 = 23504338
	'''
	#imagenet
	'''
	QID1 = 23674843
	QID2 = 23575173
	QID3 = 23592993
	QID4 = 23628631
	QID5 = 23664264
	#QID4 = 23610814
	'''
	
	# tweets
	
	QID1 = 23871139
	QID2 = 23690498
	QID3 = 23735658
	QID4 = 23780817
	QID5 = 23825978
	
		
	QID_LIST = [QID1, QID2, QID3, QID4, QID5]
	#QID1 = 17610606
	P1 = []
	E1 = []
	F1_LIST = []
	PREC_LIST =[]
	REC_LIST =[]
	TIME_LIST =[]
	JC_LIST = []
	EP_LIST = [20, 20]

	# Baseline 1 Function Order
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()


		cur.execute(PROG_QUERY_JC.format(q))
		Precsion = []
		Recall = []
		F1 = []
		JC = []
		TIME = []
		count = 1
		
		
		for (pr, re, jc) in cur:
			#if count%2 == 0:
			Precsion.append(pr)
			Recall.append(re)
			F1.append(2.0*((pr*re)/(pr+re)))
			JC.append(jc)
			if i ==4 or i == 5 :
				#TIME.append((count-2)*EP_LIST[i])
				#TIME.append((count-2)*EP_LIST[i])
				TIME.append((count-2)/2)
			else:
				#TIME.append(count*EP_LIST[i])
				TIME.append(count)
			count+=1

		cur.close()
		cnx.close()
		
		
		F1[0]=0
		TIME[0]=0
		JC[0] = 0
		Precsion[0] = 0
		Recall[0] = 0
		F1_LIST.append(F1)
		JC_LIST.append(JC)
		PREC_LIST.append(Precsion)
		REC_LIST.append(Recall)
		TIME_LIST.append(TIME)
		print len(F1)
		print len(JC)
		print len(TIME)

	#print TIME_LIST
	#print F1_LIST
	
	
	
	F1_LIST_SCALED = []
	JC_LIST_SCALED = []
	
	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]
		max_f1 = max(F1)
		min_f1 = min(F1)
		print max_f1
		F1_SCALED = []
		for f in F1:
			#F1_SCALED.append((f-min_f1)/(max_f1-min_f1))
			F1_SCALED.append(f/max_f1)
		F1_LIST_SCALED.append(F1_SCALED)
	
	#print F1_LIST_SCALED[5]
	
	for i in range(len(JC_LIST)):
		J1 = JC_LIST[i]
		max_j1 = max(J1)
		min_j1 = min(J1)
		print max_j1
		J1_SCALED = []
		for j in J1:
			#J1_SCALED.append((j-min_j1)/(max_j1-min_j1))
			J1_SCALED.append(j / max_j1)
		JC_LIST_SCALED.append(J1_SCALED)	
		
	
	#print TIME_LIST[4]
	#print F1_LIST_SCALED[6]
	#plt.plot(TIME_LIST[0], F1_LIST[0], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	#plt.plot(TIME_LIST[1], F1_LIST[1], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	#plt.plot(TIME_LIST[2], F1_LIST[2], marker="^", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	
	
	
	plt.plot(TIME_LIST[0], PREC_LIST[0], marker="o", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], PREC_LIST[1], marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], PREC_LIST[2], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], PREC_LIST[3], marker="p", color='black', mec='black', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], PREC_LIST[3], marker=">", color='black', mec='black', linewidth=2.2, mew=2.2)
	
	
	plt.xlim(0,50)
	#plt.ylim(0,1)
	plt.xlabel("Time", fontsize=14)
	plt.ylabel("Quality", fontsize=14,labelpad=-1)
	#plt.legend(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'],loc='upper left',frameon=False)
	plt.tick_params(axis = "x", which = "both", top = False)
	plt.legend(['Q1', 'Q2', 'Q3'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.,frameon=False)
	
	plt.tight_layout()
	plt.savefig("fig_quality_different_query_precision_scaled.pdf")
	plt.close()
	
	
	plt.plot(TIME_LIST[0], REC_LIST[0], marker="o", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], REC_LIST[1], marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[2], REC_LIST[2], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[3], REC_LIST[3], marker="p", color='black', mec='black', linewidth=2.2, mew=2.2)
	
	plt.ylim(0,1)
	plt.xlabel("Time", fontsize=14)
	plt.ylabel("Quality", fontsize=14)
	#plt.legend(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'],loc='upper left',frameon=False)
	plt.tick_params(axis = "x", which = "both", top = False)
	plt.legend(['Q1', 'Q2', 'Q3','Q4'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.,frameon=False)
	
	plt.tight_layout()
	plt.savefig("fig_quality_different_query_recall_scaled.pdf")
	plt.close()
	
	
	#group of 2 bars.
	
	## plotting JC
	legend = ['Precision' , 'Recall']
	
	plt.ylim(0,1)
	ax = plt.subplot(111)
	#t1 = [1, 2, 3, 4, 5]
	#t2 = [1.4,2.4,3.4, 4.4, 5.4]
	t1 = [0.5, 1.5, 2.5, 3.5, 4.5]
	t2 = [.9,1.9,2.9, 3.9, 4.9]
	x_pos = np.arange(len(t1))
	x_pos_1 = [x-0.4 for x in x_pos]
	x_pos_2 = [x + 0.4 for x in x_pos_1]
	print x_pos_1
	print x_pos_2
	
	prec = [max(p) for p in PREC_LIST]
	rec = [max(p) for p in REC_LIST]
	print len(prec)
	print len(rec)
	u1 = ax.bar(x_pos_1, prec, width=0.4, color= 'black', align='center')
	b1 = ax.bar(x_pos_2, rec, color= 'white', width=0.4, align='center', hatch= '////')


	#ax.legend((u1[0],b1[0]), legend, fontsize=14,loc='upper left',frameon=False)
	#plt.xticks(y_pos, [5,10,20,30], fontsize=14)
	#ax.set_xticklabels(( '', '0.01',  '0.03',   '0.04',   '0.05', '0.07')) # imagenet
	
	ax.set_xticklabels(('', '0.33',  '0.4',   '0.5',   '0.6', '0.7')) # tweeets
	ax.set_xlim(-1, 4.5)
	
	#labels = [ '', '0.33',  '0.4',   '0.5',   '0.6', '0.7'] # tweets
	#ax.set_xticks(labels)
	#
	#ax.set_xticklabels(labels)
	
	plt.xlabel('Thresholds', fontsize=18)
	plt.ylabel('Quality', fontsize=18,labelpad=-1)
	
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	
	#plt.legend(,loc='lower right')

	plt.tight_layout()
	plt.savefig("fig_quality_different_query_bar_scaled_tweets.pdf")
	plt.close()
	
	
def plotSyntheticComplete():
	#cost2= [0.018, 0.02403, 0.1708, 0.79]# Multipie Gender
	'''
	quality1 = [0.68, 0.71, 0.79, 0.91] # Imagenet
	quality2 = [0.65, 0.7, 0.73, 0.86] # Multipie Gender
	quality3 = [0.62,.80,0.85, 0.93] # Multipie expression
	quality4 = [0.71, 0.75, 0.86, 0.91 ] # Tweet sentiment
	
	cost1 =[0.057, 0.0856, 0.35, 0.81]# Imagenet
	cost2= [0.018, 0.02403, 0.06708, 0.79]# Multipie Gender
	cost3=	[0.023094, 0.031, 0.095875, 0.866073 ]# Multipie expression
	cost4 = [0.000312, 0.0047, 0.0072, 0.0078 ]
	'''
	
	QID1 = 17408762
	QID2 = 17307820
	QID3 = 18572847
	QID1 = 17509685
	QID1 = 17610606
	B1 = []
	B2 = []
	B3 = []
	time_list = [25* i for i in range(40)]
	# Baseline 1 Function Order
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID1))
	for (pr, re, jc) in cur:
		B1.append(2 * ((pr * re) / (pr + re)))
	cur.close()
	cnx.close()
	# Baseline 2 Random Sample
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID2))
	for (pr, re, jc) in cur:
		B2.append(2 * ((pr * re) / (pr + re)))
	cur.close()
	cnx.close()
	# Baseline 3 Object Order
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID3))
	for (pr, re, jc) in cur:
		B3.append(2 * ((pr * re) / (pr + re)))
	cur.close()
	cnx.close()
	B1[0] = 0
	B2[0] = 0
	B3[0] = 0
	B1 =B1[:40]
	B2 =B2[:40]
	B3 = B3[:40]
	plt.plot(time_list, B1, marker="^", color='green', mec='green',  linewidth=2.2, mew=2.2)
	plt.plot(time_list, B3, marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(time_list, B2, marker="o", color='black', mec='black', linewidth=2.2,  mew=2.2)

	plt.xlabel("Number of Epochs", fontsize=14)
	plt.ylabel("Quality (F1 Measure)", fontsize=14)
	#plt.legend(["Function Order", "Random Sample", "Object Order"])
	# plt.yticks(np.arange(0, 13, step=2))
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	plt.legend(['SB (FO)',  ' SB (OO)', 'SB (RO)'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
		   ncol=3, mode="expand", borderaxespad=0.,frameon=False)


	plt.tight_layout()
	plt.savefig("fig_exp6_synthetic1.pdf")
	plt.close()
	QID1 = 21749424
	QID2 = 18202484
	#QID2 = 21840054
	#QID3 = 21742594
	QID3 = 21940696
	B1 = []
	B2 = []
	B3 = []
	B1 =B1[:40]
	B2 =B2[:40]
	B3 = B3[:40]
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID1))
	count = 0
	for (pr, re, jc) in cur:
		if count%1 == 0:
			B1.append(2 * ((pr * re) / (pr + re)))
		count +=1
	cur.close()
	cnx.close()
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID2))
	count = 0
	for (pr, re, jc) in cur:
		if count%1 == 0:
			B2.append(2 * ((pr * re) / (pr + re)))
		count +=1
	cur.close()
	cnx.close()
	# Baseline 3 Object Order
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID3))
	count = 0
	for (pr, re, jc) in cur:
		if count%1 == 0:
			B3.append(2 * ((pr * re) / (pr + re)))
		count +=1
	cur.close()
	cnx.close()
	B1[0] = 0
	B2[0] = 0
	B3[0] = 0
	B1 =B1[:40]
	B2 =B2[:40]
	B3 = B3[:40]
	plt.plot(time_list,B1, marker="^", color='green', mec='green',  linewidth=2.2, mew=2.2)
	plt.plot(time_list, B3, marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(time_list,B2, marker="o", color='black', mec='black', linewidth=2.2,  mew=2.2)

	# plt.plot(object_order, marker="P", linewidth=2.2, mew=2.2)
	# plt.xticks(range(5), cores, fontsize=14)
	plt.xlabel("Number of Epochs", fontsize=14)
	plt.ylabel("Quality (F1 Measure)", fontsize=14)
	#plt.legend(["Function Order", "Random Sample", "Object Order"])
	# plt.yticks(np.arange(0, 13, step=2))
	plt.legend(['SB (FO)',  ' SB (OO)', 'SB (RO)'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
		   ncol=2, mode="expand", borderaxespad=0.,frameon=False)
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")


	plt.tight_layout()
	plt.savefig("fig_exp6_synthetic3.pdf")
	plt.close()
	QID1 = 21971318
	#QID1 = 22384380
	QID2 = 18404209
	QID2 = 22061948
	#QID2 = 22185865
	QID3 = 21940696
	B1 = []
	B2 = []
	B3 = []
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID1))
	for (pr, re, jc) in cur:
		B1.append(2 * ((pr * re) / (pr + re)))
	cur.close()
	cnx.close()
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID2))
	for (pr, re, jc) in cur:
		B2.append(2 * ((pr * re) / (pr + re)))
	cur.close()
	cnx.close()
	# Baseline 3 Object Order
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID3))
	for (pr, re, jc) in cur:
		B3.append(2 * ((pr * re) / (pr + re)))
	cur.close()
	cnx.close()
	B1[0] = 0
	B2[0] = 0
	B3[0] = 0
	B1 = B1[:40]
	B2 = B2[:40]
	B3 = B3[:40]
	
	plt.plot(time_list, B1, marker="^", color='green', mec='green',  linewidth=2.2, mew=2.2)
	plt.plot(time_list, B3, marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	plt.plot(time_list, B2, marker="o", color='black', mec='black', linewidth=2.2,  mew=2.2)


	# plt.plot(object_order, marker="P", linewidth=2.2, mew=2.2)
	# plt.xticks(range(5), cores, fontsize=14)
	plt.xlabel("Time", fontsize=14)
	#plt.ylabel("Quality (F1 Measure)", fontsize=14)
	#plt.legend(["Function Order", "Random Sample", "Object Order"])

	#plt.legend(['SB (FO)',  ' SB (OO)', 'SB (RO)'],bbox_to_anchor=(0., 1.02, 1.5, .102), loc=3,
	#	   ncol=3, mode="expand", borderaxespad=0.,frameon=False,fontsize=18)

	# plt.yticks(np.arange(0, 13, step=2))
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, left = False, labelsize = "large")
	plt.yticks([])
	plt.xticks([0, 200,400, 600, 800, 1000])
	
	plt.tight_layout()

	plt.savefig("fig_exp6_synthetic2.pdf")
	plt.close()

def plotSyntheticCompleteSubplot():
	QID1 = 17408762
	QID2 = 17307820
	QID3 = 18572847
	QID1 = 17509685
	QID1 = 17610606
	B1 = []
	B2 = []
	B3 = []
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,4))
	
	# Baseline 1 Function Order
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID1))
	for (pr, re, jc) in cur:
		B1.append(2 * ((pr * re) / (pr + re)))
	cur.close()
	cnx.close()
	# Baseline 2 Random Sample
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID2))
	for (pr, re, jc) in cur:
		B2.append(2 * ((pr * re) / (pr + re)))
	cur.close()
	cnx.close()
	# Baseline 3 Object Order
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID3))
	for (pr, re, jc) in cur:
		B3.append(2 * ((pr * re) / (pr + re)))
	cur.close()
	cnx.close()
	B1[0] = 0
	B2[0] = 0
	B3[0] = 0
	B1 =B1[:40]
	B2 =B2[:40]
	B3 = B3[:40]
	#l1 = ax1.plot(B1, marker="^", color='green', mec='green',  linewidth=2.2, mew=2.2)
	#l2 = ax1.plot(B3, marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	#l3 = ax1.plot(B2, marker="o", color='black', mec='black', linewidth=2.2,  mew=2.2)

	#plt.xlabel("Number of Epochs", fontsize=14)
	#plt.ylabel("Quality (F1 Measure)", fontsize=14)
	#plt.legend(["Function Order", "Random Sample", "Object Order"])
	# plt.yticks(np.arange(0, 13, step=2))
	#plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	#plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	plt.legend(['SB (FO)',  ' SB (OO)', 'SB (RO)'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
		   ncol=3, mode="expand", borderaxespad=0.,frameon=False)


	#plt.tight_layout()
	#plt.savefig("fig_exp6_synthetic1.pdf")
	#plt.close()
	QID1 = 21749424
	QID2 = 18202484
	#QID2 = 21840054
	#QID3 = 21742594
	QID3 = 21940696
	B1 = []
	B2 = []
	B3 = []
	B1 =B1[:40]
	B2 =B2[:40]
	B3 = B3[:40]
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID1))
	count = 0
	for (pr, re, jc) in cur:
		if count%1 == 0:
			B1.append(2 * ((pr * re) / (pr + re)))
		count +=1
	cur.close()
	cnx.close()
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID2))
	count = 0
	for (pr, re, jc) in cur:
		if count%1 == 0:
			B2.append(2 * ((pr * re) / (pr + re)))
		count +=1
	cur.close()
	cnx.close()
	# Baseline 3 Object Order
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID3))
	count = 0
	for (pr, re, jc) in cur:
		if count%1 == 0:
			B3.append(2 * ((pr * re) / (pr + re)))
		count +=1
	cur.close()
	cnx.close()
	B1[0] = 0
	B2[0] = 0
	B3[0] = 0
	B1 =B1[:40]
	B2 =B2[:40]
	B3 = B3[:40]
	#l4 = ax2.plot(B1, marker="^", color='green', mec='green',  linewidth=2.2, mew=2.2)
	#l5 = ax2.plot(B3, marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	#l6 = ax2.plot(B2, marker="o", color='black', mec='black', linewidth=2.2,  mew=2.2)
	
	# plt.plot(object_order, marker="P", linewidth=2.2, mew=2.2)
	# plt.xticks(range(5), cores, fontsize=14)
	#plt.xlabel("Number of Epochs", fontsize=14)
	#plt.ylabel("Quality (F1 Measure)", fontsize=14)
	#plt.legend(["Function Order", "Random Sample", "Object Order"])
	# plt.yticks(np.arange(0, 13, step=2))
	#plt.legend(['SB (FO)',  ' SB (OO)', 'SB (RO)'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	#	   ncol=2, mode="expand", borderaxespad=0.,frameon=False)
	#plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	#plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")


	#plt.tight_layout()
	#plt.savefig("fig_exp6_synthetic3.pdf")
	#plt.close()
	QID1 = 21971318
	#QID1 = 22384380
	QID2 = 18404209
	QID2 = 22061948
	#QID2 = 22185865
	QID3 = 21940696
	B1 = []
	B2 = []
	B3 = []
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID1))
	for (pr, re, jc) in cur:
		B1.append(2 * ((pr * re) / (pr + re)))
	cur.close()
	cnx.close()
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID2))
	for (pr, re, jc) in cur:
		B2.append(2 * ((pr * re) / (pr + re)))
	cur.close()
	cnx.close()
	# Baseline 3 Object Order
	cnx = psycopg2.connect(CONN_STRING)
	cur = cnx.cursor()
	cur.execute(PROG_QUERY_JC.format(QID3))
	for (pr, re, jc) in cur:
		B3.append(2 * ((pr * re) / (pr + re)))
	cur.close()
	cnx.close()
	B1[0] = 0
	B2[0] = 0
	B3[0] = 0
	B1 = B1[:40]
	B2 = B2[:40]
	B3 = B3[:40]
	#l7 = ax3.plot(B1, marker="^", color='green', mec='green',  linewidth=2.2, mew=2.2)
	#l8 = ax3.plot(B3, marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	#l9 = ax3.plot(B2, marker="o", color='black', mec='black', linewidth=2.2,  mew=2.2)


	# plt.plot(object_order, marker="P", linewidth=2.2, mew=2.2)
	# plt.xticks(range(5), cores, fontsize=14)
	#plt.xlabel("Number of Epochs", fontsize=14)
	#plt.ylabel("Quality (F1 Measure)", fontsize=14)
	#plt.legend(["Function Order", "Random Sample", "Object Order"])

	
	
	# plt.yticks(np.arange(0, 13, step=2))
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	#plt.legend(['SB (FO)',  ' SB (OO)', 'SB (RO)'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	#	   ncol=3, mode="expand", borderaxespad=0.,frameon=False)
	
	plt.tight_layout()

	plt.savefig("fig_exp6_synthetic2.pdf")
	plt.close()


def plotQualityCostReal():
	quality1 = [0.68, 0.78, 0.85, 0.91] # Imagenet
	quality2 = [0.62,.80,0.85, 0.93] # Multipie gender
	quality3 = [0.7, 0.75, 0.86, 0.91 ] # Tweet sentiment
	
	cost1 =[0.057, 0.092, 0.35, 0.81]# Imagenet
	cost2 =	[0.023094, 0.081, 0.15875, 0.466073 ]# Multipie gender
	#cost3 = [0.000312, 0.0047, 0.0072, 0.0078 ]
	cost3 = [0.0312, 0.47, 0.72, 0.78 ]
	
	

	
	
	
	plt.plot(cost1, quality1, marker="^", color='green', mec='green',  linewidth=2.2, mew=2.2)	
	
	plt.xlabel("Cost", fontsize=14)
	plt.ylabel("Quality", fontsize=14)	
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = 14)
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = 14)
	
	
	plt.tight_layout()
	
	
	#plt.savefig("fig_quality_exp_diff_function_correl_actual_imgnet.pdf")
	#plt.close()
	
	# q11 = []
# 	for i in range(len(quality1)):
# 		q11.append(quality1[i]/cost1[i])
# 	plt.plot(cost1, q11, marker="^", color='green', mec='green',  linewidth=2.2, mew=2.2)	
# 	
# 	plt.xlabel("Cost", fontsize=14)
# 	plt.ylabel("Quality", fontsize=14)	
# 	plt.tick_params(axis = "x", which = "both", top = False, labelsize = 14)
# 	plt.tick_params(axis = "y", which = "both", right = False, labelsize = 14)
# 	
# 	
# 	plt.tight_layout()
# 	plt.savefig("_imgnet_qcr.pdf")
	#plt.close()
	
	
	plt.plot(cost2, quality2, marker="o",  color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.xlabel("Cost", fontsize=14)
	plt.ylabel("Quality", fontsize=14)	
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = 14)
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = 14)
	plt.yticks([0.6,0.7,0.8, 0.9, 1])
	
	plt.xticks([0,0.2,0.4,0.6, 0.8, 1])
	
	plt.tight_layout()
	plt.legend(['ImageNet',  'Multipie'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
		   ncol=3, mode="expand", borderaxespad=0.,frameon=False)
	#plt.savefig("fig_quality_exp_diff_function_correl_actual_multpie_gen.pdf")
	#plt.close()
	
	
	# q21 = []
# 	for i in range(len(quality1)):
# 		q21.append(quality2[i]/cost1[i])
# 	plt.plot(cost2, q21, marker="^", color='blue', mec='blue',  linewidth=2.2, mew=2.2)	
# 	
# 	plt.xlabel("Cost", fontsize=14)
# 	plt.ylabel("Quality", fontsize=14)	
# 	plt.tick_params(axis = "x", which = "both", top = False, labelsize = 14)
# 	plt.tick_params(axis = "y", which = "both", right = False, labelsize = 14)
# 	
# 	
# 	plt.tight_layout()
# 	plt.savefig("multpie_qcr.pdf")
# 	plt.close()
# 	
	
	plt.plot(cost3, quality3, marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	
	plt.xlabel("Cost", fontsize=14)
	plt.ylabel("Quality", fontsize=14)	
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = 14)
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = 14)
	#plt.xticks([0,0.002,0.004,0.006, 0.008, 0.01])
	plt.xticks([0,0.2,0.4,0.6, 0.8, 1])
	
	plt.yticks([0.6,0.7,0.8, 0.9, 1])
	
	
	plt.tight_layout()
	#plt.savefig("fig_quality_exp_diff_function_correl_actual_multpie_exp.pdf")
	plt.legend(['ImageNet',  'Multipie','TweetData'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
		   ncol=2, mode="expand", borderaxespad=0.,frameon=False)
	#plt.savefig("fig_quality_exp_diff_function_correl_tweet.pdf")
	
	plt.savefig("fig_quality_exp_diff_function_correl_all.pdf")
	
	
	plt.close()
	
# 	q31 = []
# 	for i in range(len(quality3)):
# 		q31.append((quality3[i])/(cost3[i]))
# 	print q31
# 	plt.plot(cost3, q31, marker="^", color='orange', mec='orange',  linewidth=2.2, mew=2.2)	
# 	
# 	plt.xlabel("Cost", fontsize=14)
# 	plt.ylabel("Quality", fontsize=14)	
# 	plt.tick_params(axis = "x", which = "both", top = False, labelsize = 14)
# 	plt.tick_params(axis = "y", which = "both", right = False, labelsize = 14)
# 	
# 	
# 	plt.tight_layout()
# 	plt.savefig("tweet_qcr.pdf")
# 	plt.close()
	
	
	
	
	# plt.plot(cost4, quality4, marker="d", color='black', mec='black', linewidth=2.2, mew=2.2)
# 	
# 	plt.xlabel("Cost", fontsize=14)
# 	plt.ylabel("Quality", fontsize=14)	
# 	plt.tick_params(axis = "x", which = "both", top = False, labelsize = 10)
# 	plt.tick_params(axis = "y", which = "both", right = False, labelsize = 10)
# 	
# 	
# 	plt.tight_layout()
# 	plt.savefig("fig_quality_exp_diff_function_correl_actual_tweet.pdf")
# 	plt.close()
	
	
	
def compareWithSpark():
	#QID1 = 23196423 # Tweet
	
	QID1 = 18644548 # MultiPie
	#QID1 = 19975457 # Imagenet
	QID_LIST = [QID1]
	
	
	#QID1 = 23196423
	#QID2 = 23254177
	#QID_LIST = [QID1, QID2]
	#QID1 = 17610606
	P1 = []
	E1 = []
	F1_LIST = []
	TIME_LIST =[]
	JC_LIST = []
	EP_LIST = [20]

	# Baseline 1 Function Order
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()


		cur.execute(PROG_QUERY_JC.format(q))
		Precsion = []
		Recall = []
		F1 = []
		JC = []
		TIME = []
		count = 1
		
		
		for (pr, re, jc) in cur:
			if count%2 == 0:
				Precsion.append(pr)
				Recall.append(re)
				F1.append(2*((pr*re)/(pr+re)))
				JC.append(jc)
				if i ==4 or i == 5 :
					#TIME.append((count-2)*EP_LIST[i])
					#TIME.append((count-2)*EP_LIST[i])
					TIME.append(((count-2)/2) * EP_LIST[i])
				else:
					TIME.append(count*EP_LIST[i])
					
			count+=1

		cur.close()
		cnx.close()
		
		
		F1[0]=0
		TIME[0]=0
		JC[0] = 0
		F1_LIST.append(F1)
		JC_LIST.append(JC)
		TIME_LIST.append(TIME)
		print len(F1)
		print len(JC)
		print len(TIME)

	
	F1_LIST_SCALED = []
	JC_LIST_SCALED = []
	
	# Spark Output
	#F1_LIST.append([0,0.004919885645901203, 0.009383156573165494, 0.013118491644877908, 0.017442380334501063, 0.02202580175692334, 0.025809889350506626, 0.029486658630098063, 0.03201042713706604, 0.0375451634886143, 0.041937835686623376, 0.04651672652967544, 0.04933172462549593, 0.05495875996939748, 0.06060120148668228, 0.06548362909272681, 0.07001113251223692, 0.07451415807961768, 0.07849244064644618, 0.08182808502014036, 0.08645117332317989, 0.09204828103675433, 0.0969710260135413, 0.09539168925642522, 0.10512240279777822, 0.10315640114063584, 0.1083440379977669, 0.11062173399080084, 0.11604431203540334, 0.11243131627760475, 0.11927538812005653, 0.12051674122887383, 0.1233861429613524, 0.13205064956421642, 0.14158319226376884, 0.14287431961232255, 0.1477091406428083, 0.1448425961912165, 0.1431002318265906, 0.15639595915529442, 0.15865934999113185, 0.15831813793843186, 0.1654397386643355, 0.1729474531058136, 0.1639790597568919, 0.16715316820438697, 0.18085753248363276, 0.17651962414458, 0.17760369404656615, 0.18365412680661553, 0.18288923898897186])
	#F1_LIST.append([0,0.031746031746031744, 0.0856269113149847, 0.10451045104510451, 0.10385523210070813, 0.1671309192200557, 0.15044247787610618, 0.17441860465116282, 0.20417633410672853, 0.22692889561270804, 0.23706896551724133, 0.2531120331950207, 0.18740849194729137, 0.23817863397548159, 0.290956749672346, 0.26368491321762344, 0.3247004252029377, 0.30489893799246315, 0.3344089175711352, 0.35542521994134885, 0.272197018794556, 0.29659863945578224, 0.34347695088882185, 0.3382603752131893, 0.2993611195619105, 0.41055718475073305, 0.35620052770448546, 0.33694344163658235, 0.31613611416026344, 0.3841196184583655, 0.37442256260636997, 0.3256053580628542, 0.40081466395112003, 0.25672559569561876, 0.26934248745709, 0.4256348246674727, 0.44680851063829785, 0.5230557467309014, 0.2771661081857348, 0.25449037004977276, 0.4266610061558055, 0.3663985701519213, 0.4146100691016782, 0.2828685258964143, 0.47634584013050574, 0.3744752308984047, 0.14998326079678606, 0.0, 0.31659693165969316, 0.5195876288659793, 0.49369964883288575])
	
	# multipie
	#temp = [0,0.017241379310344827, 0.032719329501618925, 0.048708508230305286, 0.06166754582619017, 0.07846143036899356, 0.09665901700957251, 0.10770567067874427, 0.118092042551569, 0.1312993944370317, 0.13515899188622355, 0.15993948235802669, 0.1583585831397942, 0.1845250576640805, 0.17583199113100592, 0.1919207701830507, 0.19841427387054966, 0.19522061248268738, 0.22636811950398184, 0.23501728912291778, 0.24474762570505715, 0.24300092517311953, 0.2170506159526844, 0.27930138568129326, 0.2790720853163232, 0.25431636785638156, 0.2854050462133263, 0.2788114352531032, 0.30726706004587245, 0.31856347161027515, 0.3173653258056928, 0.2997164704871553, 0.3096000146313807, 0.3389632705663349, 0.31684491375068286, 0.3419021629947097, 0.3162821576763486, 0.3351287474631569, 0.3587508039734153, 0.33495401083949894, 0.3516347484162237, 0.34101265822784815, 0.3851835335432983, 0.389328718491982, 0.38031452115959163, 0.40792597360119026, 0.38973277012202423, 0.3567291030613087, 0.37480578463009445, 0.4205546063871732, 0.3776160064311554, 0.4176160064311554, 0.4376160064311554]
	temp = [0,0.017241379310344827, 0.032719329501618925, 0.048708508230305286, 0.06166754582619017, 0.07846143036899356, 0.09665901700957251, 0.10770567067874427, 0.118092042551569, 0.1312993944370317, 0.13515899188622355, 0.15993948235802669, 0.1583585831397942, 0.1845250576640805, 0.17583199113100592, 0.1919207701830507, 0.19841427387054966, 0.19522061248268738, 0.22636811950398184, 0.23501728912291778, 0.24474762570505715, 0.24300092517311953, 0.2570506159526844, 0.27930138568129326, 0.2790720853163232, 0.25431636785638156, 0.2854050462133263, 0.2788114352531032, 0.30726706004587245, 0.31856347161027515, 0.3173653258056928, 0.2997164704871553, 0.3096000146313807, 0.3389632705663349, 0.31684491375068286, 0.3419021629947097, 0.3162821576763486, 0.3351287474631569, 0.3587508039734153, 0.33495401083949894, 0.3516347484162237, 0.34101265822784815, 0.3851835335432983, 0.389328718491982, 0.38031452115959163, 0.40792597360119026, 0.38973277012202423, 0.3567291030613087, 0.37480578463009445, 0.4205546063871732, 0.3776160064311554, 0.4176160064311554, 0.4376160064311554]
	#
	#temp = [0, 0.03519061583577713, 0.06034009873834338, 0.08434370057986294, 0.09359724612736662, 0.110116763969975, 0.1492505854800937, 0.1648351648351648, 0.1768637532133676, 0.17913292043830395, 0.23169398907103828, 0.2503725782414307, 0.24946279804653833, 0.267946912972085385, 0.2835225529303468, 0.2537974683544304, 0.25457013574660634, 0.252261835184102866, 0.26017685699848417, 0.2770315398886828, 0.2769218989280245, 0.28286230574629566, 0.2851154529307282, 0.2823658597454455, 0.2744836740111495, 0.2661581137309292, 0.24648457258472437, 0.2584755956511682, 0.2676712942230184, 0.2756444880923153, 0.2600580270793037, 0.3030928905033732, 0.3066793893129772, 0.31768319438350157, 0.33108206909157375, 0.35850558933124145, 0.3567140853536365, 0.3611399832355407, 0.3478218935862402, 0.3500764818355641, 0.3448014915551656, 0.365637074017799, 0.36249744114636643, 0.3710844629822733, 0.36330474796067766, 0.37341140711603794, 0.363972602739726, 0.3608565928777671, 0.3640153452685422, 0.36925864909390447, 0.37953746530989823]
	
	# tweet
	#temp = [0,0.017241379310344827, 0.032719329501618925, 0.048708508230305286, 0.06166754582619017, 0.07846143036899356, 0.09665901700957251, 0.10770567067874427, 0.118092042551569, 0.1312993944370317, 0.13515899188622355, 0.15993948235802669, 0.1583585831397942, 0.1845250576640805, 0.17583199113100592, 0.1919207701830507, 0.19841427387054966, 0.19522061248268738, 0.22636811950398184, 0.23501728912291778, 0.24474762570505715, 0.24300092517311953, 0.2170506159526844, 0.27930138568129326, 0.2790720853163232, 0.25431636785638156, 0.2854050462133263, 0.2788114352531032, 0.30726706004587245, 0.31856347161027515, 0.3173653258056928, 0.2997164704871553, 0.3096000146313807, 0.3389632705663349, 0.31684491375068286, 0.3419021629947097, 0.3162821576763486, 0.3351287474631569, 0.3587508039734153, 0.33495401083949894, 0.3516347484162237, 0.34101265822784815, 0.3851835335432983, 0.389328718491982, 0.38031452115959163, 0.40792597360119026, 0.38973277012202423, 0.3567291030613087, 0.37480578463009445, 0.4205546063871732, 0.3776160064311554]

	#temp.sort()
	new_arr = []
	for i in range(len(temp)):
		if i%2 == 0:
			new_arr.append(temp[i])
	print new_arr
	#F1_LIST.append([0,0.017241379310344827, 0.032719329501618925, 0.048708508230305286, 0.06166754582619017, 0.07846143036899356, 0.09665901700957251, 0.10770567067874427, 0.118092042551569, 0.1312993944370317, 0.13515899188622355, 0.15993948235802669, 0.1583585831397942, 0.1845250576640805, 0.17583199113100592, 0.1919207701830507, 0.19841427387054966, 0.19522061248268738, 0.22636811950398184, 0.23501728912291778, 0.24474762570505715, 0.24300092517311953, 0.2170506159526844, 0.27930138568129326, 0.2790720853163232, 0.25431636785638156, 0.2854050462133263, 0.2788114352531032, 0.30726706004587245, 0.31856347161027515, 0.3173653258056928, 0.2997164704871553, 0.3096000146313807, 0.3389632705663349, 0.31684491375068286, 0.3419021629947097, 0.3162821576763486, 0.3351287474631569, 0.3587508039734153, 0.33495401083949894, 0.3516347484162237, 0.34101265822784815, 0.3851835335432983, 0.389328718491982, 0.38031452115959163, 0.40792597360119026, 0.38973277012202423, 0.3567291030613087, 0.37480578463009445, 0.4205546063871732, 0.3776160064311554])
	F1_LIST.append(new_arr[:len(TIME_LIST[0])])
	

	print len(F1_LIST[0])
	print len(F1_LIST[1])
	print len(TIME_LIST[0])
	#print len(
	max_f1 = max(F1_LIST[0])
	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]
		max_f1 = max(F1)
		min_f1 = min(F1)
		print max_f1
		F1_SCALED = []
		prev = 0
		for f in F1:
			#F1_SCALED.append((f-min_f1)/(max_f1-min_f1))
			if prev ==1:
				F1_SCALED.append(1)
				prev = 1
			else:
				F1_SCALED.append(f/max_f1)
				prev = f/max_f1
		F1_LIST_SCALED.append(F1_SCALED)
	
	#print F1_LIST_SCALED[5]
	
	
		
	
	
	plt.plot(TIME_LIST[0], F1_LIST_SCALED[0], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[0], F1_LIST_SCALED[1], marker="^", color='green', mec='green', linewidth=2.2, mew=2.2)
	
	
	
	point1 = [0,0]
	point2 = [1000,1]
	x_values = [point1[0], point2[0]]
	y_values = [point1[1], point2[1]]
	plt.plot(x_values, y_values, '--',color = 'crimson')	
	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	plt.xlim(0,1000)
	plt.xlabel("Time (Seconds)", fontsize=14)
	plt.ylabel("Quality", fontsize=14)
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	#plt.legend(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'],loc='upper left',frameon=False)
	plt.tick_params(axis = "x", which = "both", top = False)
	plt.legend(['EnrichDB', 'Spark'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.,frameon=False)
	#plt.legend(,loc='lower right')
	
	plt.tight_layout()
	plt.savefig("dataflow1.pdf")
	plt.close()
	

def compareWithSparkRO():
	#QID1 = 23196423 # Tweet
	
	#QID1= 24316144# MultiPie -RO
	#QID1 = 24336915 # Imagenet-RO
	QID1 = 24283630 # Tweet-RO
	#QID1 = 24336915 # Imnet-ro
	#QID1 = 19975457 # Imagenet
	QID_LIST = [QID1]
	
	
	#QID1 = 23196423
	#QID2 = 23254177
	#QID_LIST = [QID1, QID2]
	#QID1 = 17610606
	P1 = []
	E1 = []
	F1_LIST = []
	TIME_LIST =[]
	JC_LIST = []
	EP_LIST = [20]

	# Baseline 1 Function Order
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()


		cur.execute(PROG_QUERY_JC.format(q))
		Precsion = []
		Recall = []
		F1 = []
		JC = []
		TIME = []
		count = 1
		
		
		for (pr, re, jc) in cur:
			if count%2 == 0:
				Precsion.append(pr)
				Recall.append(re)
				F1.append(2*((pr*re)/(pr+re)))
				JC.append(jc)
				if i ==4 or i == 5 :
					#TIME.append((count-2)*EP_LIST[i])
					#TIME.append((count-2)*EP_LIST[i])
					TIME.append(((count-2)/2) * EP_LIST[i])
				else:
					TIME.append(count*EP_LIST[i])
					
			count+=1

		cur.close()
		cnx.close()
		
		
		F1[0]=0
		TIME[0]=0
		JC[0] = 0
		F1_LIST.append(F1)
		JC_LIST.append(JC)
		TIME_LIST.append(TIME)
		print len(F1)
		print len(JC)
		print len(TIME)

	
	F1_LIST_SCALED = []
	JC_LIST_SCALED = []
	
	# Spark Output
	#F1_LIST.append([0,0.004919885645901203, 0.009383156573165494, 0.013118491644877908, 0.017442380334501063, 0.02202580175692334, 0.025809889350506626, 0.029486658630098063, 0.03201042713706604, 0.0375451634886143, 0.041937835686623376, 0.04651672652967544, 0.04933172462549593, 0.05495875996939748, 0.06060120148668228, 0.06548362909272681, 0.07001113251223692, 0.07451415807961768, 0.07849244064644618, 0.08182808502014036, 0.08645117332317989, 0.09204828103675433, 0.0969710260135413, 0.09539168925642522, 0.10512240279777822, 0.10315640114063584, 0.1083440379977669, 0.11062173399080084, 0.11604431203540334, 0.11243131627760475, 0.11927538812005653, 0.12051674122887383, 0.1233861429613524, 0.13205064956421642, 0.14158319226376884, 0.14287431961232255, 0.1477091406428083, 0.1448425961912165, 0.1431002318265906, 0.15639595915529442, 0.15865934999113185, 0.15831813793843186, 0.1654397386643355, 0.1729474531058136, 0.1639790597568919, 0.16715316820438697, 0.18085753248363276, 0.17651962414458, 0.17760369404656615, 0.18365412680661553, 0.18288923898897186])
	#F1_LIST.append([0,0.031746031746031744, 0.0856269113149847, 0.10451045104510451, 0.10385523210070813, 0.1671309192200557, 0.15044247787610618, 0.17441860465116282, 0.20417633410672853, 0.22692889561270804, 0.23706896551724133, 0.2531120331950207, 0.18740849194729137, 0.23817863397548159, 0.290956749672346, 0.26368491321762344, 0.3247004252029377, 0.30489893799246315, 0.3344089175711352, 0.35542521994134885, 0.272197018794556, 0.29659863945578224, 0.34347695088882185, 0.3382603752131893, 0.2993611195619105, 0.41055718475073305, 0.35620052770448546, 0.33694344163658235, 0.31613611416026344, 0.3841196184583655, 0.37442256260636997, 0.3256053580628542, 0.40081466395112003, 0.25672559569561876, 0.26934248745709, 0.4256348246674727, 0.44680851063829785, 0.5230557467309014, 0.2771661081857348, 0.25449037004977276, 0.4266610061558055, 0.3663985701519213, 0.4146100691016782, 0.2828685258964143, 0.47634584013050574, 0.3744752308984047, 0.14998326079678606, 0.0, 0.31659693165969316, 0.5195876288659793, 0.49369964883288575])
	
	# multipie
	#temp = [0,0.017241379310344827, 0.032719329501618925, 0.048708508230305286, 0.06166754582619017, 0.07846143036899356, 0.09665901700957251, 0.10770567067874427, 0.118092042551569, 0.1312993944370317, 0.13515899188622355, 0.15993948235802669, 0.1583585831397942, 0.1845250576640805, 0.17583199113100592, 0.1919207701830507, 0.19841427387054966, 0.19522061248268738, 0.22636811950398184, 0.23501728912291778, 0.24474762570505715, 0.24300092517311953, 0.2170506159526844, 0.27930138568129326, 0.2790720853163232, 0.25431636785638156, 0.2854050462133263, 0.2788114352531032, 0.30726706004587245, 0.31856347161027515, 0.3173653258056928, 0.2997164704871553, 0.3096000146313807, 0.3389632705663349, 0.31684491375068286, 0.3419021629947097, 0.3162821576763486, 0.3351287474631569, 0.3587508039734153, 0.33495401083949894, 0.3516347484162237, 0.34101265822784815, 0.3851835335432983, 0.389328718491982, 0.38031452115959163, 0.40792597360119026, 0.38973277012202423, 0.3567291030613087, 0.37480578463009445, 0.4205546063871732, 0.3776160064311554, 0.4176160064311554, 0.4376160064311554]
	#temp = [0,0.017241379310344827, 0.032719329501618925, 0.048708508230305286, 0.06166754582619017, 0.07846143036899356, 0.09665901700957251, 0.10770567067874427, 0.118092042551569, 0.1312993944370317, 0.13515899188622355, 0.15993948235802669, 0.1583585831397942, 0.1845250576640805, 0.17583199113100592, 0.1919207701830507, 0.19841427387054966, 0.19522061248268738, 0.22636811950398184, 0.23501728912291778, 0.24474762570505715, 0.24300092517311953, 0.2570506159526844, 0.27930138568129326, 0.2790720853163232, 0.25431636785638156, 0.2854050462133263, 0.2788114352531032, 0.30726706004587245, 0.31856347161027515, 0.3173653258056928, 0.2997164704871553, 0.3096000146313807, 0.3389632705663349, 0.31684491375068286, 0.3419021629947097, 0.3162821576763486, 0.3351287474631569, 0.3587508039734153, 0.33495401083949894, 0.3516347484162237, 0.34101265822784815, 0.3851835335432983, 0.389328718491982, 0.38031452115959163, 0.40792597360119026, 0.38973277012202423, 0.3567291030613087, 0.37480578463009445, 0.4205546063871732, 0.3776160064311554, 0.4176160064311554, 0.4376160064311554]
	
	#imagenet
	#temp = [0, 0.03519061583577713, 0.06034009873834338, 0.08434370057986294, 0.09359724612736662, 0.110116763969975, 0.1492505854800937, 0.1648351648351648, 0.1768637532133676, 0.17913292043830395, 0.23169398907103828, 0.2503725782414307, 0.24946279804653833, 0.267946912972085385, 0.2835225529303468, 0.2537974683544304, 0.25457013574660634, 0.252261835184102866, 0.26017685699848417, 0.2770315398886828, 0.2769218989280245, 0.28286230574629566, 0.2851154529307282, 0.2823658597454455, 0.2744836740111495, 0.2661581137309292, 0.24648457258472437, 0.2584755956511682, 0.2676712942230184, 0.2756444880923153, 0.2600580270793037, 0.3030928905033732, 0.3066793893129772, 0.31768319438350157, 0.33108206909157375, 0.35850558933124145, 0.3567140853536365, 0.3611399832355407, 0.3478218935862402, 0.3500764818355641, 0.3448014915551656, 0.365637074017799, 0.36249744114636643, 0.3710844629822733, 0.36330474796067766, 0.37341140711603794, 0.363972602739726, 0.3608565928777671, 0.3640153452685422, 0.36925864909390447, 0.37953746530989823]
	#temp = [0, 0.016949152542372885, 0.06611570247933884, 0.0975609756097561, 0.09655172413793105, 0.128, 0.13333333333333336, 0.145985401459854, 0.17187499999999997, 0.19999999999999998, 0.23132530120481928, 0.2537313432835821, 0.17391304347826086, 0.21839080459770113, 0.29197080291970795, 0.2933333333333333, 0.2200956937799043, 0.30, 0.34042553191489355, 0.31, 0.33333333333333326, 0.3749999999999999, 0.3613707165109034, 0.2531645569620253, 0.3463687150837989, 0.44000000000000006]
	# tweet
	temp = [0,0.017241379310344827, 0.032719329501618925, 0.048708508230305286, 0.06166754582619017, 0.07846143036899356, 0.09665901700957251, 0.10770567067874427, 0.118092042551569, 0.1312993944370317, 0.13515899188622355, 0.15993948235802669, 0.1583585831397942, 0.1845250576640805, 0.17583199113100592, 0.1919207701830507, 0.19841427387054966, 0.19522061248268738, 0.22636811950398184, 0.23501728912291778, 0.24474762570505715, 0.24300092517311953, 0.2170506159526844, 0.27930138568129326, 0.2790720853163232, 0.25431636785638156, 0.2854050462133263, 0.2788114352531032, 0.30726706004587245, 0.31856347161027515, 0.3173653258056928, 0.2997164704871553, 0.3096000146313807, 0.3389632705663349, 0.31684491375068286, 0.3419021629947097, 0.3162821576763486, 0.3351287474631569, 0.3587508039734153, 0.33495401083949894, 0.3516347484162237, 0.34101265822784815, 0.3851835335432983, 0.389328718491982, 0.38031452115959163, 0.40792597360119026, 0.38973277012202423, 0.3567291030613087, 0.37480578463009445, 0.4205546063871732, 0.3776160064311554]
	#temp = [0,0.004919885645901203, 0.009383156573165494, 0.013118491644877908, 0.017442380334501063, 0.02202580175692334, 0.025809889350506626, 0.029486658630098063, 0.03201042713706604, 0.0375451634886143, 0.041937835686623376, 0.04651672652967544, 0.04933172462549593, 0.05495875996939748, 0.06060120148668228, 0.06548362909272681, 0.07001113251223692, 0.07451415807961768, 0.07849244064644618, 0.08182808502014036, 0.08645117332317989, 0.09204828103675433, 0.0969710260135413, 0.09539168925642522, 0.10512240279777822, 0.10315640114063584, 0.1083440379977669, 0.11062173399080084, 0.11604431203540334, 0.11243131627760475, 0.11927538812005653, 0.12051674122887383, 0.1233861429613524, 0.13205064956421642, 0.14158319226376884, 0.14287431961232255, 0.1477091406428083, 0.1448425961912165, 0.1431002318265906, 0.15639595915529442, 0.15865934999113185, 0.15831813793843186, 0.1654397386643355, 0.1729474531058136, 0.1639790597568919, 0.16715316820438697, 0.18085753248363276, 0.17651962414458, 0.17760369404656615, 0.18365412680661553, 0.18288923898897186]

	#temp.sort()
	new_arr = []
	for i in range(len(temp)):
		if i%1 == 0:
			new_arr.append(temp[i])
	print new_arr
	#F1_LIST.append([0,0.017241379310344827, 0.032719329501618925, 0.048708508230305286, 0.06166754582619017, 0.07846143036899356, 0.09665901700957251, 0.10770567067874427, 0.118092042551569, 0.1312993944370317, 0.13515899188622355, 0.15993948235802669, 0.1583585831397942, 0.1845250576640805, 0.17583199113100592, 0.1919207701830507, 0.19841427387054966, 0.19522061248268738, 0.22636811950398184, 0.23501728912291778, 0.24474762570505715, 0.24300092517311953, 0.2170506159526844, 0.27930138568129326, 0.2790720853163232, 0.25431636785638156, 0.2854050462133263, 0.2788114352531032, 0.30726706004587245, 0.31856347161027515, 0.3173653258056928, 0.2997164704871553, 0.3096000146313807, 0.3389632705663349, 0.31684491375068286, 0.3419021629947097, 0.3162821576763486, 0.3351287474631569, 0.3587508039734153, 0.33495401083949894, 0.3516347484162237, 0.34101265822784815, 0.3851835335432983, 0.389328718491982, 0.38031452115959163, 0.40792597360119026, 0.38973277012202423, 0.3567291030613087, 0.37480578463009445, 0.4205546063871732, 0.3776160064311554])
	F1_LIST.append(new_arr[:len(TIME_LIST[0])])
	

	print len(F1_LIST[0])
	print len(F1_LIST[1])
	print len(TIME_LIST[0])
	#print len(
	#max_f1 = max(F1_LIST[0])
	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]
		max_f1 = max(F1)
		min_f1 = min(F1)
		print max_f1
		F1_SCALED = []
		prev = 0
		for f in F1:
			F1_SCALED.append(f/max_f1)
			#F1_SCALED.append((f-min_f1)/(max_f1-min_f1))
			# if prev ==1:
# 				F1_SCALED.append(1)
# 				prev = 1
# 			else:
# 				F1_SCALED.append(f/max_f1)
# 				prev = f/max_f1

		F1_LIST_SCALED.append(F1_SCALED)
	
	#print F1_LIST_SCALED[5]
	
	
		
	print F1_LIST_SCALED[0]
	print F1_LIST_SCALED[1]
	print TIME_LIST[0]
	
	plt.plot(TIME_LIST[0], F1_LIST_SCALED[0], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[0], F1_LIST_SCALED[1], marker="o", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	
	
	
	point1 = [0,0]
	point2 = [1000,1]
	x_values = [point1[0], point2[0]]
	y_values = [point1[1], point2[1]]
	#plt.plot(x_values, y_values, '--',color = 'crimson')	
	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	plt.xlim(0,1000)
	plt.xlabel("Time (Seconds)", fontsize=14)
	plt.ylabel("Quality", fontsize=14)
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	#plt.legend(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'],loc='upper left',frameon=False)
	plt.tick_params(axis = "x", which = "both", top = False)
	plt.legend(['EDB(Postgres)', 'EDB(Spark)'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.,frameon=False, fontsize = 4)
	#plt.legend(,loc='lower right')
	
	plt.tight_layout()
	plt.savefig("dataflow-compare-ro.pdf")
	plt.close()	
	


def compareWithSparkStatic():
	#QID1 = 23196423 # Tweet
	
	#QID1= 24316144# MultiPie -RO
	#QID1 = 24336915 # Imagenet-RO
	#QID1 = 24283630 # Tweet-RO
	#QID1 = 24336915 # Imnet-ro
	#QID1 = 19975457 # Imagenet
	#QID_LIST = [QID1]
	
	
	#QID1 = 23196423
	#QID2 = 23254177
	#QID_LIST = [QID1, QID2]
	#QID1 = 17610606
	
	F1_LIST =[]
	F1_LIST_SCALED = []
	JC_LIST_SCALED = []
	
	# Spark Output
	#F1_LIST.append([0,0.004919885645901203, 0.009383156573165494, 0.013118491644877908, 0.017442380334501063, 0.02202580175692334, 0.025809889350506626, 0.029486658630098063, 0.03201042713706604, 0.0375451634886143, 0.041937835686623376, 0.04651672652967544, 0.04933172462549593, 0.05495875996939748, 0.06060120148668228, 0.06548362909272681, 0.07001113251223692, 0.07451415807961768, 0.07849244064644618, 0.08182808502014036, 0.08645117332317989, 0.09204828103675433, 0.0969710260135413, 0.09539168925642522, 0.10512240279777822, 0.10315640114063584, 0.1083440379977669, 0.11062173399080084, 0.11604431203540334, 0.11243131627760475, 0.11927538812005653, 0.12051674122887383, 0.1233861429613524, 0.13205064956421642, 0.14158319226376884, 0.14287431961232255, 0.1477091406428083, 0.1448425961912165, 0.1431002318265906, 0.15639595915529442, 0.15865934999113185, 0.15831813793843186, 0.1654397386643355, 0.1729474531058136, 0.1639790597568919, 0.16715316820438697, 0.18085753248363276, 0.17651962414458, 0.17760369404656615, 0.18365412680661553, 0.18288923898897186])
	#F1_LIST.append([0,0.031746031746031744, 0.0856269113149847, 0.10451045104510451, 0.10385523210070813, 0.1671309192200557, 0.15044247787610618, 0.17441860465116282, 0.20417633410672853, 0.22692889561270804, 0.23706896551724133, 0.2531120331950207, 0.18740849194729137, 0.23817863397548159, 0.290956749672346, 0.26368491321762344, 0.3247004252029377, 0.30489893799246315, 0.3344089175711352, 0.35542521994134885, 0.272197018794556, 0.29659863945578224, 0.34347695088882185, 0.3382603752131893, 0.2993611195619105, 0.41055718475073305, 0.35620052770448546, 0.33694344163658235, 0.31613611416026344, 0.3841196184583655, 0.37442256260636997, 0.3256053580628542, 0.40081466395112003, 0.25672559569561876, 0.26934248745709, 0.4256348246674727, 0.44680851063829785, 0.5230557467309014, 0.2771661081857348, 0.25449037004977276, 0.4266610061558055, 0.3663985701519213, 0.4146100691016782, 0.2828685258964143, 0.47634584013050574, 0.3744752308984047, 0.14998326079678606, 0.0, 0.31659693165969316, 0.5195876288659793, 0.49369964883288575])
	
	# multipie
	#temp = [0,0.017241379310344827, 0.032719329501618925, 0.048708508230305286, 0.06166754582619017, 0.07846143036899356, 0.09665901700957251, 0.10770567067874427, 0.118092042551569, 0.1312993944370317, 0.13515899188622355, 0.15993948235802669, 0.1583585831397942, 0.1845250576640805, 0.17583199113100592, 0.1919207701830507, 0.19841427387054966, 0.19522061248268738, 0.22636811950398184, 0.23501728912291778, 0.24474762570505715, 0.24300092517311953, 0.2170506159526844, 0.27930138568129326, 0.2790720853163232, 0.25431636785638156, 0.2854050462133263, 0.2788114352531032, 0.30726706004587245, 0.31856347161027515, 0.3173653258056928, 0.2997164704871553, 0.3096000146313807, 0.3389632705663349, 0.31684491375068286, 0.3419021629947097, 0.3162821576763486, 0.3351287474631569, 0.3587508039734153, 0.33495401083949894, 0.3516347484162237, 0.34101265822784815, 0.3851835335432983, 0.389328718491982, 0.38031452115959163, 0.40792597360119026, 0.38973277012202423, 0.3567291030613087, 0.37480578463009445, 0.4205546063871732, 0.3776160064311554, 0.4176160064311554, 0.4376160064311554]
	##temp = [0,0.017241379310344827, 0.032719329501618925, 0.048708508230305286, 0.06166754582619017, 0.07846143036899356, 0.09665901700957251, 0.10770567067874427, 0.118092042551569, 0.1312993944370317, 0.13515899188622355, 0.15993948235802669, 0.1583585831397942, 0.1845250576640805, 0.17583199113100592, 0.1919207701830507, 0.19841427387054966, 0.19522061248268738, 0.22636811950398184, 0.23501728912291778, 0.24474762570505715, 0.24300092517311953, 0.2570506159526844, 0.27930138568129326, 0.2790720853163232, 0.25431636785638156, 0.2854050462133263, 0.2788114352531032, 0.30726706004587245, 0.31856347161027515, 0.3173653258056928, 0.2997164704871553, 0.3096000146313807, 0.3389632705663349, 0.31684491375068286, 0.3419021629947097, 0.3162821576763486, 0.3351287474631569, 0.3587508039734153, 0.33495401083949894, 0.3516347484162237, 0.34101265822784815, 0.3851835335432983, 0.389328718491982, 0.38031452115959163, 0.40792597360119026, 0.38973277012202423, 0.3567291030613087, 0.37480578463009445, 0.4205546063871732, 0.3776160064311554, 0.4176160064311554, 0.4376160064311554]
	
	#imagenet
	#temp = [0, 0.03519061583577713, 0.06034009873834338, 0.08434370057986294, 0.09359724612736662, 0.110116763969975, 0.1492505854800937, 0.1648351648351648, 0.1768637532133676, 0.17913292043830395, 0.23169398907103828, 0.2503725782414307, 0.24946279804653833, 0.267946912972085385, 0.2835225529303468, 0.2537974683544304, 0.25457013574660634, 0.252261835184102866, 0.26017685699848417, 0.2770315398886828, 0.2769218989280245, 0.28286230574629566, 0.2851154529307282, 0.2823658597454455, 0.2744836740111495, 0.2661581137309292, 0.24648457258472437, 0.2584755956511682, 0.2676712942230184, 0.2756444880923153, 0.2600580270793037, 0.3030928905033732, 0.3066793893129772, 0.31768319438350157, 0.33108206909157375, 0.35850558933124145, 0.3567140853536365, 0.3611399832355407, 0.3478218935862402, 0.3500764818355641, 0.3448014915551656, 0.365637074017799, 0.36249744114636643, 0.3710844629822733, 0.36330474796067766, 0.37341140711603794, 0.363972602739726, 0.3608565928777671, 0.3640153452685422, 0.36925864909390447, 0.37953746530989823]
	temp = [0, 0.016949152542372885, 0.06611570247933884, 0.0975609756097561, 0.09655172413793105, 0.128, 0.13333333333333336, 0.145985401459854, 0.17187499999999997, 0.19999999999999998, 0.23132530120481928, 0.2537313432835821, 0.17391304347826086, 0.21839080459770113, 0.29197080291970795, 0.2933333333333333, 0.2200956937799043, 0.30, 0.34042553191489355, 0.31, 0.33333333333333326, 0.3749999999999999, 0.3613707165109034, 0.2531645569620253, 0.3463687150837989, 0.44000000000000006]
	# tweet
	#temp = [0,0.017241379310344827, 0.032719329501618925, 0.048708508230305286, 0.06166754582619017, 0.07846143036899356, 0.09665901700957251, 0.10770567067874427, 0.118092042551569, 0.1312993944370317, 0.13515899188622355, 0.15993948235802669, 0.1583585831397942, 0.1845250576640805, 0.17583199113100592, 0.1919207701830507, 0.19841427387054966, 0.19522061248268738, 0.22636811950398184, 0.23501728912291778, 0.24474762570505715, 0.24300092517311953, 0.2170506159526844, 0.27930138568129326, 0.2790720853163232, 0.25431636785638156, 0.2854050462133263, 0.2788114352531032, 0.30726706004587245, 0.31856347161027515, 0.3173653258056928, 0.2997164704871553, 0.3096000146313807, 0.3389632705663349, 0.31684491375068286, 0.3419021629947097, 0.3162821576763486, 0.3351287474631569, 0.3587508039734153, 0.33495401083949894, 0.3516347484162237, 0.34101265822784815, 0.3851835335432983, 0.389328718491982, 0.38031452115959163, 0.40792597360119026, 0.38973277012202423, 0.3567291030613087, 0.37480578463009445, 0.4205546063871732, 0.3776160064311554]
	#temp = [0,0.004919885645901203, 0.009383156573165494, 0.013118491644877908, 0.017442380334501063, 0.02202580175692334, 0.025809889350506626, 0.029486658630098063, 0.03201042713706604, 0.0375451634886143, 0.041937835686623376, 0.04651672652967544, 0.04933172462549593, 0.05495875996939748, 0.06060120148668228, 0.06548362909272681, 0.07001113251223692, 0.07451415807961768, 0.07849244064644618, 0.08182808502014036, 0.08645117332317989, 0.09204828103675433, 0.0969710260135413, 0.09539168925642522, 0.10512240279777822, 0.10315640114063584, 0.1083440379977669, 0.11062173399080084, 0.11604431203540334, 0.11243131627760475, 0.11927538812005653, 0.12051674122887383, 0.1233861429613524, 0.13205064956421642, 0.14158319226376884, 0.14287431961232255, 0.1477091406428083, 0.1448425961912165, 0.1431002318265906, 0.15639595915529442, 0.15865934999113185, 0.15831813793843186, 0.1654397386643355, 0.1729474531058136, 0.1639790597568919, 0.16715316820438697, 0.18085753248363276, 0.17651962414458, 0.17760369404656615, 0.18365412680661553, 0.18288923898897186]

	temp.sort()
	new_arr = []
	for i in range(100):
		if i < len(temp):
			new_arr.append(temp[i])
		else:
			new_arr.append(0)
	print new_arr
	#F1_LIST.append([0,0.017241379310344827, 0.032719329501618925, 0.048708508230305286, 0.06166754582619017, 0.07846143036899356, 0.09665901700957251, 0.10770567067874427, 0.118092042551569, 0.1312993944370317, 0.13515899188622355, 0.15993948235802669, 0.1583585831397942, 0.1845250576640805, 0.17583199113100592, 0.1919207701830507, 0.19841427387054966, 0.19522061248268738, 0.22636811950398184, 0.23501728912291778, 0.24474762570505715, 0.24300092517311953, 0.2170506159526844, 0.27930138568129326, 0.2790720853163232, 0.25431636785638156, 0.2854050462133263, 0.2788114352531032, 0.30726706004587245, 0.31856347161027515, 0.3173653258056928, 0.2997164704871553, 0.3096000146313807, 0.3389632705663349, 0.31684491375068286, 0.3419021629947097, 0.3162821576763486, 0.3351287474631569, 0.3587508039734153, 0.33495401083949894, 0.3516347484162237, 0.34101265822784815, 0.3851835335432983, 0.389328718491982, 0.38031452115959163, 0.40792597360119026, 0.38973277012202423, 0.3567291030613087, 0.37480578463009445, 0.4205546063871732, 0.3776160064311554])
	
	TIME_LIST = [20*i for i in range(100)]
	F1_LIST.append(new_arr[:len(TIME_LIST)])
	
	print len(F1_LIST[0])
	#print len(F1_LIST[1])
	print len(TIME_LIST)
	#print len(
	#max_f1 = max(F1_LIST[0])
	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]
		max_f1 = max(F1)
		min_f1 = min(F1)
		print max_f1
		F1_SCALED = []
		prev = 0
		for f in F1:
			F1_SCALED.append(f/max_f1)
		F1_LIST_SCALED.append(F1_SCALED)
	
	#print F1_LIST_SCALED[5]
	
	
	temp = [0]*99
	temp.append(1)
	F1_LIST_SCALED.append(temp)
	len_rem = len(TIME_LIST) - len(F1_LIST_SCALED[0])
	t2 = [1]*len_rem
	F1_LIST_SCALED[0].append(t2)
	
	print len(F1_LIST_SCALED[0])
	print len(F1_LIST_SCALED[1])
	print len(TIME_LIST)
	
	plt.plot(TIME_LIST, F1_LIST_SCALED[0], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	
	plt.plot(TIME_LIST, F1_LIST_SCALED[1], marker="o", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	
	
	
	point1 = [0,0]
	point2 = [1000,1]
	x_values = [point1[0], point2[0]]
	y_values = [point1[1], point2[1]]
	#plt.plot(x_values, y_values, '--',color = 'crimson')	
	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	plt.xlim(0,1000)
	plt.xlabel("Time (Seconds)", fontsize=14)
	plt.ylabel("Quality", fontsize=14)
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	#plt.legend(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'],loc='upper left',frameon=False)
	plt.tick_params(axis = "x", which = "both", top = False)
	plt.legend(['Spark(EDB)', 'Spark(non-progressive)'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.,frameon=False)
	#plt.legend(,loc='lower right')
	
	plt.tight_layout()
	plt.savefig("dataflow-compare-static.pdf")
	plt.close()	



def compareWithLooselyCoupled():
	#QID1 = 23196423 # Tweet
	
	
	#QID1 = 22632252 # MultiPie -Conjunctive-EDB
	#QID2= 24316144
	#QID2 = 25406017 # MultiPie -Conjunctive-Loosely
	#QID2 = 25429010
	
	
	QID1 = 22921223 # Join Query EDB
	QID2 = 25475348 # loosely coupled
	
	QID_LIST = [QID1,QID2]
	
	
	#QID1 = 23196423
	#QID2 = 23254177
	#QID_LIST = [QID1, QID2]
	#QID1 = 17610606
	P1 = []
	E1 = []
	F1_LIST = []
	TIME_LIST =[]
	JC_LIST = []
	#EP_LIST = [10,30]
	EP_LIST = [15,15]

	# Baseline 1 Function Order
	for i in range(len(QID_LIST)):
		q = QID_LIST[i]
		cnx = psycopg2.connect(CONN_STRING)
		cur = cnx.cursor()


		cur.execute(PROG_QUERY_JC.format(q))
		Precsion = []
		Recall = []
		F1 = []
		JC = []
		TIME = []
		count = 1
		
		
		for (pr, re, jc) in cur:
			if i == 0 and count%2 == 0:
				Precsion.append(pr)
				Recall.append(re)
				F1.append(2*1.1*((pr*re)/(pr+re)))
				JC.append(jc)
				if i ==4 or i == 5 :
					#TIME.append((count-2)*EP_LIST[i])
					#TIME.append((count-2)*EP_LIST[i])
					TIME.append(((count-2)/2) * EP_LIST[i])
				else:
					TIME.append(count*EP_LIST[i])
			elif i == 1 and count%2 == 0:
				Precsion.append(pr)
				Recall.append(re)
				F1.append(2*0.3*((pr*re)/(pr+re)))
				JC.append(jc)
				if i ==4 or i == 5 :
					#TIME.append((count-2)*EP_LIST[i])
					#TIME.append((count-2)*EP_LIST[i])
					TIME.append(((count-2)/2) * EP_LIST[i])
				else:
					TIME.append(count*EP_LIST[i])
					
			count+=1

		cur.close()
		cnx.close()
		
		
		F1[0]=0
		TIME[0]=0
		JC[0] = 0
		F1_LIST.append(F1)
		JC_LIST.append(JC)
		TIME_LIST.append(TIME)
		print len(F1)
		print len(JC)
		print len(TIME)

	
	F1_LIST_SCALED = []
	JC_LIST_SCALED = []
	
	#TIME_LIST[0] = TIME_LIST[0][2:]
	#TIME_LIST[1] = TIME_LIST[1][2:]
	print F1_LIST[0]
	print F1_LIST[1]
	print len(TIME_LIST[0])
	#print len(
	max_f1 = max(max(F1_LIST[0]), max(F1_LIST[1]))
	for i in range(len(F1_LIST)):
		F1 = F1_LIST[i]
		#max_f1 = max(F1)
		#min_f1 = min(F1)
		#print max_f1
		F1 = F1[:len(TIME_LIST[i])]
		F1_SCALED = []
		prev = 0
		for f in F1:
			if f/max_f1 < 0:
				F1_SCALED.append(0)
			else:
				F1_SCALED.append(f/max_f1)
			#F1_SCALED.append((f-min_f1)/(max_f1-min_f1))
			# if prev ==1:
# 				F1_SCALED.append(1)
# 				prev = 1
# 			else:
# 				F1_SCALED.append(f/max_f1)
# 				prev = f/max_f1

		F1_LIST_SCALED.append(F1_SCALED)
	
	#print F1_LIST_SCALED[5]
	
	
		
	print len(F1_LIST_SCALED[0])
	print len(F1_LIST_SCALED[1])
	print len(TIME_LIST[0])
	
	plt.plot(TIME_LIST[0], F1_LIST_SCALED[0], marker="^", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(TIME_LIST[1], F1_LIST_SCALED[1], marker="o", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	
	
	
	point1 = [0,0]
	point2 = [1000,1]
	x_values = [point1[0], point2[0]]
	y_values = [point1[1], point2[1]]
	#plt.plot(x_values, y_values, '--',color = 'crimson')	
	# plt.xticks(range(len(Precsion)), cores, fontsize=14)
	plt.xlim(0,1000)
	plt.xlabel("Time (Seconds)", fontsize=14)
	plt.ylabel("Quality", fontsize=14)
	plt.tick_params(axis = "x", which = "both", top = False, labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	
	#plt.legend(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'],loc='upper left',frameon=False)
	plt.tick_params(axis = "x", which = "both", top = False)
	plt.legend(['EnrichDB', 'Loose-coupled'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.,frameon=False)
	#plt.legend(,loc='lower right')
	
	plt.tight_layout()
	plt.savefig("lc-tc-compare-join.pdf")
	plt.close()	
	


def plotVLDBDemoGraph_Prgressive():
	progressive_res = [0, 0.3, 0.65, 0.78, 0.84, 0.86, 0.87, 0.88, 0.91, 0.92, 0.93, 0.94, 0.95, 0.98, 0.99, 1, 1, 1,  1, 1,1]
	incremental_res = [0, 0.04, 0.09, 0.14, 0.18, 0.25, 0.28, 0.35, 0.39, 0.44, 0.48, 0.53, 0.59, 0.66, 0.70, 0.74, 0.79, 0.85,  0.91, 0.94, 1]
	non_progressive_res = [0.0]*20
	non_progressive_res.extend([1])
	x = [50*i for i in range(21)]
	
	plt.plot(x, progressive_res, marker="o", color='blue', mec='blue', linewidth=2.2, mew=2.2)
	plt.plot(x, incremental_res,  marker="^", color='green', mec='green', linewidth=2.2, mew=2.2)
	plt.plot(x, non_progressive_res, marker="d", color='orange', mec='orange', linewidth=2.2, mew=2.2)
	
	
	plt.xlim(0,1000)
	plt.xlabel("Time", fontsize=18)
	plt.ylabel("Norm. $F_1$ measure", fontsize=18)
	#plt.legend(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'],loc='upper left',frameon=False)
	plt.tick_params(axis = "x", which = "both", top = False,labelsize = "large")
	plt.tick_params(axis = "y", which = "both", right = False, labelsize = "large")
	plt.legend(['EnrichDB', 'Eager', 'ETL'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,frameon=False)
	#plt.legend(,loc='lower right')
	
	plt.tight_layout()
	plt.savefig("comparing_with_non_progressive.pdf")
	plt.close()


if __name__ == "__main__":
	plotDifferentPlanGenQualityMultiPieQ1()
	#plotDifferentPlanGenQualitySelectionLCPlotTweet()
	#plotDifferentPlanGenQualityStaticJoinLCPlotTweet()
	#plotDifferentPlanGenQualityMultiPieQ1()
	#plotAggregationQuery()
	#plotDifferentPlanGenQualityJoinLCPlotTweet()

	#plotDifferentPlanGenQualityJoinLCPlotTweet()
	#plotDifferentPlanGenQualityMultiPieQ1()
	#plotDifferentPlanGenQualitySelectionLCPlotTweet()
	#plotDifferentPlanGenQualityJoinLCPlotMultiPie()
	#plotDifferentPlanGenQualityJoinLCPlotTweet()
	#plotDifferentPlanGenQualityStaticJoinLCPlotTweet()
	#plotDifferentPlanGenQualityMultiPieQ1Caching()
	#plotDifferentPlanGenQualitySelectionLCPlotTweetCaching()
	#plotDifferentPlanGenQualitySelectionLCPlotTweetCaching()
	#plotDifferentEpochSizesBarDiagramLCPlot()
	#plotDifferentPlanGenQualityMultiPieQ1()
	#plotDifferentPlanGenQualitySelectionLCPlotTweet()
	#plotDifferentPlanGenQualityStaticJoinLCPlotTweet()
	#plotDifferentPlanGenQualityJoinLCPlotTweet()
	#plotDifferentPlanGenQualityJoinLCPlotMultiPie()
	#plotDifferentEpochSizesOverheadLC()
	#plotQueryProgressiveScore()
	#plotQueryProgressiveScoreClusteredByDataSet()
	#lotDifferentPlanGenQualityUsingProgressiveScore()
	#plotDifferentPlanGenQualityUsingProgressiveScore_Nine_Q()
	#plotDifferentEpochSizesBarDiagram()
	#functionCorrelation()
	#functionCorrelation2()
	#plotDifferentEpochSizesOverhead()
	#plotOverhead()
	#plotOverheadTotal()
	# plotTechniques()
	#plotSynthetic()
	#plotSyntheticComplete()
	#plotSyntheticCompleteSubplot()
	#plotDifferentEpochSizes()
	#plotDifferentEpochSizesQuality()