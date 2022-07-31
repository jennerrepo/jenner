import sys,re
import time,pickle
import numpy as np
import heapq
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
import random
from operator import truediv,mul,sub
import csv
import math
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import tree
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import copy
import operator
from operator import itemgetter
import matplotlib.pyplot as plt
#import tweet_features, tweet_pca


import warnings
warnings.filterwarnings("ignore")


#dl,nl = pickle.load(open('MuctTestGender2_XY.p','rb'))
#dl,nl = pickle.load(open('MuctTestGender6_XY.p','rb'))

sys.setrecursionlimit(1500)



sentiment_dt = joblib.load(open('Classifiers/Tweet/sentiment_decision_tree.p', 'rb'))
sentiment_rf = pickle.load(open('Classifiers/Tweet/sentiment_rf.p', 'rb'))
sentiment_knn = joblib.load(open('Classifiers/Tweet/sentiment_knn.p', 'rb'))
sentiment_svm = joblib.load(open('Classifiers/Tweet/sentiment_svm.p', 'rb'))
sentiment_et = pickle.load(open('Classifiers/Tweet/sentiment_et.p', 'rb'))


print 'loading clf finished'


rf_thresholds, gnb_thresholds, et_thresholds,  svm_thresholds = [], [], [] , []
rf_tprs, gnb_tprs, et_tprs,  svm_tprs = [], [] ,[], []
rf_fprs, gnb_fprs, et_fprs, svm_fprs  = [], [] ,[], []
rf_probabilities, gnb_probabilities, et_probabilities, svm_probabilities = [], [], [], []

dl = []
nl = []
listInitial,list0,list1,list2,list3,list01,list02,list03,list12,list13,list23,list012,list013,list023,list123=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]


dl2,nl2 = pickle.load(open('Data/Tweet/TweetDataSmall.p','rb'))

t_load_data_start = time.time()

#dl2,nl2 = pickle.load(open('StanfordTweetSentimentBigDataset8_XY.p','rb'))
print dl2[0]
#dl2= np.array(dl2)
#nl2= np.array(nl2)
t_load_data_end = time.time()

t_load_data = (t_load_data_end - t_load_data_start)
 


print 'loading data finished'


def genderPredicate1(rl):
	gProb = sentiment_gnb.prob_classify(rl)
	gProbSmile = gProb.prob('1')
	return gProbSmile

def genderPredicate3(rl):
	gProb = sentiment_rf.prob_classify(rl)
	gProbSmile = gProb.prob('1')
	#print gProbSmile
	return gProbSmile

def genderPredicate4(rl):
	gProb = sentiment_lr.prob_classify(rl)
	gProbSmile = gProb.prob('1')
	return gProbSmile

def genderPredicate6(rl):
	gProb = sentiment_dt.prob_classify(rl)
	
	gProbSmile = gProb.prob('1')
	return gProbSmile
	
def genderPredicate7(rl):
	gProb = sentiment_knn.prob_classify(rl)
	gProbSmile = gProb.prob('1')
	return gProbSmile

def genderPredicate10(rl):
	gProb = sentiment_svm.prob_classify(rl)
	gProbSmile = gProb.prob('1')
	return gProbSmile

	
def genderPredicate12(rl):
	gProb = sentiment_sgd.prob_classify(rl)
	gProbSmile = gProb.prob('1')
	return gProbSmile
	
def genderPredicate14(rl):
	gProb = sentiment_maxent.prob_classify(rl)
	gProbSmile = gProb.prob('1')
	return gProbSmile
	
def genderPredicate16(rl):
	gProb = sentiment_et.prob_classify(rl)
	gProbSmile = gProb.prob('1')
	return gProbSmile
	
def genderPredicate18(rl):
	gProb = sentiment_knn_reduced.prob_classify(rl)
	gProbSmile = gProb.prob('1')
	return gProbSmile
	
	
def setup():
	included_cols = [0]
	
	skipRow= 0
	
	
	with open('DecisionTableTweet/Feature1/listInitialDetails.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 1
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				#print content
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				listInitial.append(temp1)
			rowNum = rowNum+1
	
	
	with open('DecisionTableTweet/Feature1/list0Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				#print content
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list0.append(temp1)
			rowNum = rowNum+1
	
	with open('DecisionTableTweet/Feature1/list1Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list1.append(temp1)
			rowNum = rowNum+1
			
	with open('DecisionTableTweet/Feature1/list2Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list2.append(temp1)
			rowNum = rowNum+1
	
	with open('DecisionTableTweet/Feature1/list3Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list3.append(temp1)
			rowNum = rowNum+1
		
	with open('DecisionTableTweet/Feature1/list01Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list01.append(temp1)
			rowNum = rowNum+1
		
	with open('DecisionTableTweet/Feature1/list02Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list02.append(temp1)
			rowNum = rowNum+1
			
	with open('DecisionTableTweet/Feature1/list03Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list03.append(temp1)
			rowNum = rowNum+1
			
	with open('DecisionTableTweet/Feature1/list12Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list12.append(temp1)
			rowNum = rowNum+1
		
	with open('DecisionTableTweet/Feature1/list13Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list13.append(temp1)
			rowNum = rowNum+1
			
	with open('DecisionTableTweet/Feature1/list23Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list23.append(temp1)
			rowNum = rowNum+1
	
	with open('DecisionTableTweet/Feature1/list012Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list012.append(temp1)
			rowNum = rowNum+1
			
	with open('DecisionTableTweet/Feature1/list013Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list013.append(temp1)
			rowNum = rowNum+1
			
	with open('DecisionTableTweet/Feature1/list023Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list023.append(temp1)
			rowNum = rowNum+1
			
	with open('DecisionTableTweet/Feature1/list123Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list123.append(temp1)
			rowNum = rowNum+1
	
	
	
	
	
def chooseNextBest(prevClassifier,uncertainty):
	#print currentProbability
	#print prevClassifier
	#print uncertainty
	noOfClassifiers = len(prevClassifier)
	uncertaintyList = []
	
	#print prevClassifier
	nextClassifier = -1 
	
	# for objects gone through zero classifiers. This is the initialization stage.
	
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==0) :
		uncertaintyList = listInitial
	
	
	# for objects only gone through one classifiers
	
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==0) :
		uncertaintyList = list0
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==0) :
		uncertaintyList = list1
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==0) :
		uncertaintyList = list2
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==1) :
		uncertaintyList = list3
	
	# for objects gone through two classifiers
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==0) :
		uncertaintyList = list01
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==0) :
		uncertaintyList = list02
	if  prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==1 :
		uncertaintyList = list03
	if  prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==0 :
		uncertaintyList = list12
	if  prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==1 :
		uncertaintyList = list13
	if  prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==1 :
		uncertaintyList = list23
	
	# for objects gone through three classifiers
	
	if  prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==0 :
		uncertaintyList = list012
	if  prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==1 :
		uncertaintyList = list123
	if  prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==1 :
		uncertaintyList = list023
	if  prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==1 :
		uncertaintyList = list013
	
	if  prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==1 :
		return ['NA',0]
	#print 'uncertaintyList'
	#print uncertaintyList
	[nextClassifier,deltaU] = chooseBestBasedOnUncertainty(uncertaintyList, uncertainty)
		
			
	return [nextClassifier,deltaU]
	
def convertEntropyToProb(entropy):
	#print 'entropy: %f'%(entropy)
	for i in range(50):
		f= -0.01*i * np.log2(0.01*i) -(1-0.01*i)*np.log2(1-0.01*i)
		#print f
		if abs(f-entropy) < 0.02:
			#print 0.01*i
			break
	#print 'entropy found: %f'%(0.01*i)
	return 0.01*i
	
	
	

def chooseBestBasedOnUncertainty(uncertaintyList, uncertainty):
	bestClassifier = -1
	index = 0
	#print 'current uncertainty:%f'%(uncertainty)
	#print 'uncertaintyList'
	#print uncertaintyList
	for i in range(len(uncertaintyList)):
		element = uncertaintyList[i]
		if float(element[0]) >= float(uncertainty) :
			index = i
			break
	uncertaintyListElement =  uncertaintyList[index]
	bestClassifier = uncertaintyListElement[1]
	#print bestClassifier
	
	deltaUncertainty = uncertaintyListElement[2]
	#print deltaUncertainty
	
	return [bestClassifier,deltaUncertainty]
	
	

	
def calculateBlockSize(budget, thinkTime,thinkTimePercent):
	costClassifier = float(cost('GNB')+cost('ET')+cost('RF')+cost('SVM'))/4
	print 'costClassifier:%f'%(costClassifier)
	print 'budget:%f'%(budget)
	print 'thinkTime:%f'%(thinkTime)
	thinkBudget = thinkTimePercent * budget
	numIteration = math.floor(float(thinkBudget)/thinkTime)
	blockSize = (1-thinkTimePercent)*thinkTime/(thinkTimePercent*costClassifier)
	return int(blockSize)
	
def cost(classifier):
	cost=0
	'''
	Cost in Muct Dataset
	gnb,et,rf,svm
	[0.029360,0.018030,0.020180,0.790850]

	'''
	#costSet = [0.029360,0.018030,0.020180,0.790850]
	#print 'classifier'
	#print classifier
	if classifier =='LDA':
		cost = 0.018175
	if classifier =='DT':
		cost = 0.000277
	if classifier =='GNB':
		cost = 0.0002
	if classifier =='RF':
		#cost = 0.020180
		cost = 0.057491
	if classifier =='LR':
		cost = 0.000236
	if classifier =='KNN':
		cost = 0.003995
	if classifier =='SVM':
		cost =0.013686
		
	return cost
	
def combineProbability (probList):
	sumProb = 0
	countProb = 0
	flag = 0
	#print probList
	weights = determineWeights()
	
	for i in range(len(probList[0])):
		if probList[0][i]!=-1:
			sumProb = sumProb+weights[i]*probList[0][i]
			countProb = countProb+weights[i]
			flag = 1
	
	if flag ==1:
		return float(sumProb)/countProb
	else:
		return 0.5
		
	 

def convertToRocProb(prob,operator):
	#print 'In convertToRocProb method, %f'%prob
	#print operator
	clf_thresholds =[]
	clf_fpr =[]
	if operator.__name__== 'genderPredicate1' :
		clf_thresholds = lr_thresholds
		clf_fpr = lr_fprs
	if operator.__name__== 'genderPredicate2' :
		clf_thresholds = et_thresholds
		clf_fpr = et_fprs
	if operator.__name__== 'genderPredicate3' :
		clf_thresholds = rf_thresholds
		clf_fpr = rf_fprs
	if operator.__name__== 'genderPredicate4' :
		clf_thresholds = ab_thresholds
		clf_fpr = ab_fprs
	if operator.__name__== 'genderPredicate5' :
		clf_thresholds = svm_thresholds
		clf_fpr = svm_fprs
	
	thresholdIndex = (np.abs(clf_thresholds - prob)).argmin()
	rocProb = 1- clf_fpr[thresholdIndex]
	return rocProb
	
	
def findUncertainty(prob):
	if prob ==0 or prob == 1:
		return 0
	else :
		return (-prob* np.log2(prob) - (1- prob)* np.log2(1- prob))
	


def findQuality(currentProbability):
	probabilitySet = []
	probDictionary = {}
	#t1_q=time.time()
	for i in range(len(dl)):
		
		combinedProbability = combineProbability(currentProbability[i])
		probabilitySet.append(combinedProbability)
		
		#key = i
		value = combinedProbability
		probDictionary[i] = [value]
	
	#t2_q=time.time()
	#print 'time init 1: %f'%(t2_q - t1_q)
	
	#probabilitySet.sort(reverse=True)
	#t1_s=time.time()
	sortedProbSet = probabilitySet[:]
	sortedProbSet.sort(reverse=True)
	
	#t1_th=time.time()
	totalSum = sum(sortedProbSet[0:len(sortedProbSet)])
	prevF1 = 0
	precision =0
	recall = 0
	f1Value = 0
	probThreshold = 0
	sumOfProbability =0
	
	for i in range(len(sortedProbSet)):
		sizeOfAnswer = i
		sumOfProbability = sumOfProbability + sortedProbSet[i]
		
		if i>0:
			precision = float(sumOfProbability)/(i)
			if totalSum >0:
				recall = float(sumOfProbability)/totalSum
			else:
				recall = 0 
			if (precision+recall) >0 :
				f1Value = 2*precision*recall/(precision+recall)
			else:
				f1Value = 0
			#f1Value = 2*float(sumOfProbability)/(totalSum +i)
		
		if f1Value < prevF1 :
			break
		else:
			prevF1 = f1Value
	indexSorted = i
	#print sortedProbSet
	probThreshold = sortedProbSet[indexSorted]
	#print 'indexSorted value : %d'%(indexSorted)
	#print 'threshold probability value : %f'%(probThreshold)
	
	#t2_th=time.time()
	#print 'time threshold 1: %f'%(t2_th - t1_th)
	
	returnedImages = []
	outsideImages = []
	
	#t1_ret=time.time()
	for i in range(len(probabilitySet)):
		if probabilitySet[i] > probThreshold:
			returnedImages.append(i)
		else:
			outsideImages.append(i)
			
	
	return [prevF1,precision, recall, returnedImages, outsideImages]
	

def determineWeights():
	#set = [0.85,0.92,0.92,0.89]
	#set = [1,2,2,1]
	set = [1,4,4,4]
	
	sumValue = sum(set)
	weightValues = [float(x)/sumValue for x in set]
	return weightValues
	
def findRealF1(imageList):
	sizeAnswer = len(imageList)	
	sizeDataset = len(nl)
	num_ones = np.count_nonzero(nl == 4)
	
	
	temp_nl = list(nl)
	num_ones =  temp_nl.count(4)   # For Stanford Corpus it will be 4.
	
	
	count = 0
	for i in imageList:
		if nl[i]==4: # For Stanford Corpus it will be 4.
			count+=1
	#print 'size of answer: %f'%(sizeAnswer)
	#print 'number of correct ones:%f'%(count)
	if sizeAnswer > 0 :
		precision = float(count)/sizeAnswer
	else:
		precision = 0
	if num_ones > 0:
		recall = float(count)/num_ones
	else:
		recall = 0
	
	if precision !=0 or recall !=0:
		f1Measure = float(2*precision*recall)/(precision+recall)
	else:
		f1Measure = 0
	#print 'precision:%f, recall : %f, f1 measure: %f'%(precision,recall,f1Measure)
	return f1Measure
	
def findStates(outsideObjects,prevClassifier):
	stateCollection = []
	for i in range(len(outsideObjects)):
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = 'init'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '0'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '1'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '2'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '3'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '01'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '02'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '03'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '12'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '13'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '23'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '012'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '013'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '023'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '123'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = 'NA'
		
		stateCollection.append(state)
	
	
	return stateCollection
	


def runAllClassifiers():
	
	f1 = open('QueryExecutionResultAll.txt','w+')
	
	
	#Initialization step. 
	currentProbability = {}
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbability:
			currentProbability[key].append(value)
		else:
			currentProbability[key] = [value]	
			
	t1 = time.time()
	#lr,et,rf,ab
	#set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
	set = [genderPredicate6,genderPredicate1,genderPredicate4,genderPredicate10]
	
	
	for i in range(len(set)):
		t1  = time.time()
		operator = set[i]
		
		#rocProb = prob[0]
		for j in range(len(dl)):
			probValues = operator(dl[j])
			imageProb = probValues
			rocProb = imageProb
			averageProbability = 0;
			#print 'image:%d'%(j)
			#print("Roc Prob : {} ".format(rocProb))
				
			#index of classifier
			indexClf = i
			tempProb = currentProbability[j][0]
			tempProb[indexClf] = rocProb

		print 'round %d completed'%(i)
	
		t2 = time.time()
		timeElapsed = t2-t1
		print>>f1,'time taken: %f'%(timeElapsed)
	
	
	qualityOfAnswer = findQuality(currentProbability)
	print 'returned images'
	print qualityOfAnswer[3]
	print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
	if len(qualityOfAnswer[3]) > 0:
		realF1 = findRealF1(qualityOfAnswer[3])
	else: 
		realF1 = 0
	print>>f1,'real F1 : %f'%(realF1)
		
	print>>f1,'time taken: %f'%(timeElapsed)
		
	
def findNewQuality(currentProbability,index,newProbabilityValue2):
	# index is the index of the object whose prob is changed
	probabilitySet = []
	probDictionary = {}
	#print 'inside new quality function'
	for i in range(len(dl)):
		
		sumProb = 0
		countProb = 0		
		flag = 0
		
		if i != index:
			for p in currentProbability[i][0]:
			#print>>f1,'current probability: {}'.format(currentProbability[i][0])
				if p!=-1 :
					#productProbability = productProbability*(1-p)
					sumProb = sumProb+p
					countProb = countProb+1
					flag = 1
			if flag==0:
				combinedProbability = 0.5	
			else: 
				#combinedProbability = 1 - productProbability
				combinedProbability = float(sumProb)/countProb
		else:
			combinedProbability = newProbabilityValue2
		probabilitySet.append(combinedProbability)
		
		key = i
		value = combinedProbability
		probDictionary[key] = [value]
	sortedProbSet = probabilitySet[:]
	sortedProbSet.sort(reverse=True)
	sorted_x = sorted(probDictionary.items(), key=operator.itemgetter(1), reverse = True)
	
	
	totalSum = sum(sortedProbSet[0:len(sortedProbSet)])
	prevF1 = 0
	precision =0
	recall = 0
	f1Value = 0
	probThreshold = 0
	for i in range(len(sortedProbSet)):
		sizeOfAnswer = i
		sumOfProbability =0
		for j in range(i):
			#probThreshold = sorted_x.get(j)
			sumOfProbability = sumOfProbability + sortedProbSet[j]   #without dictionary
			#sumOfProbability = sumOfProbability + sorted_x.get(j)
		if i>0:
			precision = float(sumOfProbability)/(i)
			if totalSum >0:
				recall = float(sumOfProbability)/totalSum
			else:
				recall = 0
			if (precision+recall) >0:
				f1Value = 2*precision*recall/(precision+recall)
			else:
				f1Value = 0
			
		if f1Value < prevF1 :
			break
		else:
			prevF1 = f1Value
	indexSorted = i
	#print 'indexSorted value : %d'%(indexSorted)
	
	returnedImages = []
	
	
	for key in sorted_x[:indexSorted]:
		returnedImages.append(key[0])
	
	# this part is to eliminate objects which have not gone through any of the classifiers.
	eliminatedImage = []
	for k in range(len(returnedImages)):
		flag1 = 0
		for p in currentProbability[returnedImages[k]][0]:
				if p!=-1 :
					flag1 = 1
		if flag1==0:
			eliminatedImage.append(returnedImages[k])

	selectedImages = [x for x in returnedImages if x not in eliminatedImage]
			
	return [prevF1,precision, recall, selectedImages]
	
	

def runOneClassifier():
	
	f1 = open('ResultOneClfCompareTweet.txt','w+')
	
	
	#gnb,et,rf,svm
	#set = [genderPredicate4]
	#set = [genderPredicate6, genderPredicate1,  genderPredicate4, genderPredicate7, genderPredicate10]
	#set = [genderPredicate6,genderPredicate10]
	#set = [genderPredicate7,genderPredicate4,genderPredicate10]
	#set = [genderPredicate7,genderPredicate6,genderPredicate10]
	#set = [genderPredicate16]
	
	set = [genderPredicate7,genderPredicate3,genderPredicate16,genderPredicate10] #working set 2
	
	#set = [genderPredicate18]
	#set = [genderPredicate7]
	
	print 'in runOneClassifier'
	print nl
	
	#print [sentiment_gnb.prob_classify(dl[i]).prob('1') for i in range(len(dl))]
	for i in range(len(set)):
		#print i
		#Initialization step. 
		currentProbability = {}
		for k in range(len(dl)):
			key = k
			value = [-1,-1,-1,-1]
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]	
				
		t1 = time.time()
		operator = set[i]
		
		#rocProb = prob[0]
		for j in range(len(dl)):
			#probValues= operator(dl)
			#probValues = operator([dl[j]])
			#imageProb = probValues[j]
			
			rocProb = operator(dl[j])
			
			if rocProb < 0.8:
				rocProb = 0.1
				
			#index of classifier
			indexClf = i
			tempProb = currentProbability[j][0]
			tempProb[indexClf] = rocProb

		print 'round %d completed'%(i)
			
			
		t2 = time.time()
		timeElapsed = t2-t1
		qualityOfAnswer = findQuality(currentProbability)
		realF1 = findRealF1(qualityOfAnswer[3])
		print>>f1,operator
		print>>f1,'Time taken: %f'%(timeElapsed)
		print>>f1,"Current Answer set : {} ".format(qualityOfAnswer[3])
		probSet = [combineProbability(currentProbability[x]) for x in qualityOfAnswer[3]]	# storing probability values of the objects in answer set:
		print>>f1,"Probability values of objects in Answer set : {} ".format(probSet)
		print>>f1,'Length of answer set:%d'%(len(qualityOfAnswer[3]))
		print>>f1,"Actual F1 measure : %f "%(findRealF1(qualityOfAnswer[3]))
		
		currentProbability.clear()




def findUnprocessed(currentProbability):
	unprocessedImages = []
	for k in range(len(dl)):
		flag1 = 0
		for p in currentProbability[k][0]:
				if p!=-1 :
					flag1 = 1
		if flag1==0:
			unprocessedImages.append(k)

	return unprocessedImages
	





def adaptiveOrder10(timeBudget,epoch):
	#1:Gaussian Naive Bayes
	#2:Extra Tree
	#3:Random Forest
	#4:Adaptive Boosting
	
	f1 = open('queryTestSentimentTweet10.txt','w+')

	#lr,et,rf,ab
	
	#set = [genderPredicate6,genderPredicate1,genderPredicate7,genderPredicate10]
	#knn,rf,lr,svm
	
	#set = [genderPredicate7,genderPredicate3,genderPredicate6,genderPredicate10] #working
	set = [genderPredicate7,genderPredicate3,genderPredicate16,genderPredicate10] #working set 2
	#set = [genderPredicate18,genderPredicate3,genderPredicate16,genderPredicate10]
	
	print timeBudget
	outsideObjects=[]
	
	#blockList = [4000]
	blockList = [800]
	
	
	
	executionPerformed = 0
	thinkTimeList = []
	executionTimeList = []
	#for percent in thinkPercentList:
	realF1List = []
	for block in blockList:
		#totalAllowedExecution = 1000
		executionPerformed = 0
		# The dictionary currentProbability stores the information about the output of previously ran classifiers. Suppose image 20 has gone through c2 and c3 and the output probability was 0.5 and 0.6. 
		# Then the hashmap element of image 20 will be as follows:  20: [0,0.5,0.6,0]. 
		currentProbability = {}
		for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]			
		#print currentProbability
		
		
		# The dictionary prevClassifier stores the information about previously ran classifiers. Suppose image 20 has gone through c2 and c3. Then the hashmap element of image 20 will be as follows:  20: [0,1,1,0]. 
		# The bit vector corresponding to classifier 2 and classifier 3 are set.
		prevClassifier = {}
		for i in range(len(dl)):
			key = i
			value = [0,0,0,0]
			if key in prevClassifier:
				prevClassifier[key].append(value)
			else:
				prevClassifier[key] = [value]
				
		#print prevClassifier
		
		#currentUncertainty list stores the information of current uncertainty of all the images.
		
		currentUncertainty = [0.99]*len(dl)
		currentF1measure = 0
		#print currentUncertainty
		operator = set[0]
		count = 0
		totalExecutionTime = 0
		totalThinkTime = 0
		topKIndexes= [0]
		timeElapsed = 0
		timeList =[]
		f1List =[]
		blockSize = 800	
		executionTime = 0
		
		stepSize = epoch   #step size of 20 seconds. After every 20 seconds evaluate the quality
		currentTimeBound = epoch
	
		t11 = 0
		t12 = 0
		
		
		while True:		
			#t11 = time.time()
			#for i in topKIndexes:
		
			if count ==0:
				
				operator = set[0]
			
				for i1 in range(len(dl)):
					t1 = time.time()
					probValues = operator(dl[i1])
					#print>>f1,probValues
					indexClf = set.index(operator)
					
					# Adding the probability value of the object.
					tempProb = currentProbability[i1][0]
					'''					
					if probValues<0.8:
						probValues = 0
					'''
					
					if probValues>0.5:
						probValues = probValues + 0.2
					else:
						probValues = 0.1
				
					#print tempProb
					#print probValues
					tempProb[indexClf] = probValues
					
				
					
					# setting the bit for the corresponding classifier
					tempClf = prevClassifier[i1][0]
					tempClf[indexClf] = 1
					#print prevClassifier[i][0]
					
					# calculating the current cobined probability
					combinedProbability = combineProbability(currentProbability[i1])
					
					# using the combined probability value to calculate uncertainty
					uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
					currentUncertainty[i1] = uncertainty
					
			
				
				qualityOfAnswer = findQuality(currentProbability)
				#print 'size of answer set : %d'%(len(qualityOfAnswer[3]))
				#print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
				if len(qualityOfAnswer[3]) > 0:
					realF1 = findRealF1(qualityOfAnswer[3])
				else: 
					realF1 = 0
				#print 'f1 measure after initial classifier:%f'%(realF1)
				#print>>f1,'real F1 : %f'%(realF1)
				#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				timeList.append(0)
				f1List.append(f1measure)
				#currentTimeBound = currentTimeBound + stepSize
				
				#print>>f1,'time bound completed:%d'%(currentTimeBound)
			
				#print 'after running one classifier'
			
			#KNN,RF,LR,SVM
			#1,6,7,10
			
			#t11 = time.time()
			if count >0:
				for w in range(4):
					tempClfList = ['KNN','RF','ET','SVM']
					#print>>f1,"w = %d"%(w)	
					#print>>f1, tempClfList[w]					
					imageIndex = [item for item in topKIndexes if nextBestClassifier[item] == tempClfList[w]]
					if w!=4:
						operator = set[w]
					#print>>f1,operator
					images = [dl[k] for k in imageIndex]
					
					#print>>f1,"images to be run with this operator : {} ".format(imageIndex)
					if len(imageIndex) >0:
						#probValues = operator(images)
						######## Executing the function on all the objects ###########								
						for i1 in range(len(imageIndex)):
							t11 = time.time()
									
							probValues = operator(images[i1])
							rocProb = probValues
							
							indexClf = w						
							tempProb = currentProbability[imageIndex[i1]][0]
							tempProb[indexClf] = rocProb
							#print tempProb
							
							# setting the bit for the corresponding classifier
							tempClfList = prevClassifier[imageIndex[i1]][0]
							tempClfList[indexClf] = 1
							#print tempClf
							
							# calculating the current cobined probability
							combinedProbability = combineProbability(currentProbability[imageIndex[i1]])
							
							# using the combined probability value to calculate uncertainty
							uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
							currentUncertainty[imageIndex[i1]] = uncertainty
							
							
							t12 = time.time()
							totalExecutionTime = totalExecutionTime + (t12-t11)	
							timeElapsed = timeElapsed +(t12-t11)	
							
							if timeElapsed > currentTimeBound:
								qualityOfAnswer = findQuality(currentProbability)
				
								if len(qualityOfAnswer[3]) > 0:
									realF1 = findRealF1(qualityOfAnswer[3])
								else: 
									realF1 = 0
								#print>>f1,'real F1 : %f'%(realF1)
								#f1measure = qualityOfAnswer[0]
								f1measure = realF1
								timeList.append(timeElapsed)
								f1List.append(f1measure)
								print 'real f1:%f'%(realF1)
				
								#print 'time bound completed:%d'%(currentTimeBound)	
								print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
								#print 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				
								currentTimeBound = currentTimeBound + stepSize
								#break
								
							if timeElapsed > timeBudget:
								break
							
				
							
					imageIndex[:]=[]
					images[:]=[]
					#print>>f1,"Outside of inner for loop"
					#continue
							
				#print>>f1,"Finished executing four functions"
					#imageIndex[:]=[]
					#images[:] =[]
					#topKIndexes[:]=[]
					#probValues[:]=[]
			#t12 = time.time()
			#totalExecutionTime = totalExecutionTime + (t12-t11)	
			#executionTimeList.append(t12-t11)
			
			executionTimeList.append(totalExecutionTime)
					
				
			
			nextBestClassifier = [-1]*len(dl)
			deltaUncertainty = [0] *len(dl)
			benefitArray =[float(-10)]* len(dl) 
			topKIndexes = [0] * len(dl) # initial block size.
			
			#currentTempProbability = copy.deepcopy(currentProbability)
			newUncertaintyValue = 0 #initializing
			
			#### Think phase starts
			# calculating benefit of each objects. Benefit is measured in terms of improvement in f1 measure.
			t21 = time.time()
			# first determining the objects which are not in answer set
			qualityOfAnswer = findQuality(currentProbability)
			currentAnswerSet = qualityOfAnswer[3]
			allObjects = list(range(0,len(dl)))
			
			#outsideObjects = allObjects
			
			### Uncomment this part for choosing objects from outside of the answer set.
			if count !=0:
				outsideObjects = [x for x in allObjects if x not in currentAnswerSet]
			
			
			#print>>f1,"inside objects : {} ".format(currentAnswerSet)
			#print>>f1,"length of inside objects : %f"%len(currentAnswerSet)
			stateListInside =[]
			stateListInside = findStates(currentAnswerSet,prevClassifier)
			#print>>f1,"state of inside objects: {}".format(stateListInside)
			
			#print>>f1,"outsideObjects : {} ".format(outsideObjects)
			#print>>f1,"length of outsideObjects : %f"%len(outsideObjects)
			stateListOutside =[]
			stateListOutside = findStates(outsideObjects,prevClassifier)
			#print>>f1,"state of outside objects: {}".format(stateListOutside)
			
			
			if(len(outsideObjects)==0 and count !=0):
				break
			
			
			#print>>f1,"outsideObjects : {} ".format(outsideObjects)
			#print 'count=%d'%(count)
			
			for j in range(len(outsideObjects)):
				#print>>f1,'deciding for object %d'%(outsideObjects[j])
				#print>>f1,"currentUncertainty: {}".format(currentUncertainty)
				[nextBestClassifier[outsideObjects[j]],deltaUncertainty[outsideObjects[j]]] = chooseNextBest(prevClassifier.get(outsideObjects[j])[0],currentUncertainty[outsideObjects[j]])	
				newUncertaintyValue = currentUncertainty[outsideObjects[j]]  + float(deltaUncertainty[outsideObjects[j]])
				newProbabilityValue1 = convertEntropyToProb(newUncertaintyValue)
				#print 'newUncertaintyValue:%f'%(newUncertaintyValue)
				
				
				#finding index of classifier
				#indexTempProbClf = set.index(nextBestClassifier[j])
				if nextBestClassifier[outsideObjects[j]] == 'KNN':
					#nextBestClassifier[outsideObjects[j]] = 'NA'
					indexTempProbClf = 0					
				if nextBestClassifier[outsideObjects[j]] == 'RF':					
					indexTempProbClf = 1
				if nextBestClassifier[outsideObjects[j]] == 'ET':
					indexTempProbClf = 2
				if nextBestClassifier[outsideObjects[j]] == 'SVM':
					indexTempProbClf = 3
				
				# higher probability value	
				newProbabilityValue2 = 1 - newProbabilityValue1			
		
				#benefit is  (pi * pi_new)/cost(i) 
				probability_i = combineProbability(currentProbability[outsideObjects[j]])
				#print 'probability_i: %f, new probability : %f, cost : %f'%(probability_i,newProbabilityValue2,cost(nextBestClassifier[j]))
				if cost(nextBestClassifier[outsideObjects[j]]) != 0:
					benefit = float((probability_i*newProbabilityValue2)/float(cost(nextBestClassifier[outsideObjects[j]])))
					benefitArray[outsideObjects[j]] = benefit
				else:
					benefitArray[outsideObjects[j]] = -1
				
			#seq = sorted(benefitArray)
			#print benefitArray
			#Ordering the objects based on deltaUncertainty Value
			#order = [seq.index(v) for v in benefitArray]
			#topIndex= benefitArray.index(max(benefitArray))
			
			#topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			
			
			if len(outsideObjects) < blockSize :
				topKIndexes = outsideObjects
				#topKIndexes = heapq.nlargest(len(outsideObjects), range(len(outsideObjects)), benefitArray.__getitem__)
			else:
				topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			
			'''
			if len(outsideObjects) > (blockSize/4):
				topKIndexes = [x for x in topKIndexes if x not in currentAnswerSet]
			else:
				topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			'''
			#print 'top index:%d'%(topIndex)
			
			t22 = time.time()
			totalThinkTime = totalThinkTime + (t22-t21)
			thinkTimeList.append(t22-t21)
			
			'''
			if(all(element==0 or element==-1 for element in benefitArray) and count >20):
				break
			'''
			#i=topIndex #next image to be run
			t2 = time.time()
			
			#timeElapsed = totalExecutionTime + totalThinkTime
			timeElapsed = timeElapsed + totalThinkTime
			
			#timeList.append(timeElapsed)
			#print 'next images to be run'
			#print topKIndexes
			
			#print>>f1,'benefit array: {}'.format(benefitArray)
			#print>>f1,'next images to be run: {}'.format(topKIndexes)
			classifierSet = [nextBestClassifier[item2] for item2 in topKIndexes]
			'''
			if(all(element=='NA' for element in classifierSet) and count > 10):
				break
			'''
			allClassifierSet = [nextBestClassifier[item3] for item3 in allObjects]
			if(all(element=='NA' for element in classifierSet) and count > 10):
				topKIndexes = allObjects
				#break
			if(all(element=='NA' for element in allClassifierSet) and count > 10):
				break
			#print>>f1,'classifier set: {}'.format(classifierSet)
			benefitArray[:] =[]
			classifierSet[:] = []
			
			#print 'round %d completed'%(count)
			#print 'time taken %f'%(timeElapsed)
			
			# block size is determined in this part.
			if count ==0:
				blockSize = block
				topKIndexes[:]= []
				print 'blockSize: %d'%(blockSize)
			
			
			
			
			###### Time check########
			
			if timeElapsed > currentTimeBound:
				qualityOfAnswer = findQuality(currentProbability)
				
				if len(qualityOfAnswer[3]) > 0:
					realF1 = findRealF1(qualityOfAnswer[3])
				else: 
					realF1 = 0
				#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				timeList.append(timeElapsed)
				f1List.append(f1measure)
				
				#f1measure = qualityOfAnswer[0]
				#timeList.append(timeElapsed)
				#f1List.append(f1measure)
				
				#print 'time bound completed:%d'%(currentTimeBound)	
				print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				#print 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				
				currentTimeBound = currentTimeBound + stepSize
				
					
			
			
			if timeElapsed > timeBudget:
				break
			
			

			#if count >= 5000:
			#	break
			count=count+1
			
		plt.title('Quality vs Time Value')
		plt.ylabel('Quality')
		plt.xlabel('time')
		xValue = timeList
		yValue = f1List
	
		'''
		# uncomment this part for plotting
		plt.plot(xValue, yValue,'b')
		plt.ylim([0, 1])
		plt.legend(loc="upper left")
		plt.savefig('plotQualityAdaptive8'+str(block)+'.eps',format = 'eps')
		plt.title('Quality vs Time for block size = '+str(block))
		#plt.show()
		plt.close()
		'''
		#xValue = timeList
		#yValue = f1List
		
	
	'''
	plt.ylabel('Quality')
	plt.xlabel('time')
	xValue = timeList
	yValue = f1List
	print>>f1,"x value : {} ".format(xValue)
	print>>f1,"y value : {} ".format(yValue)
	print "x value : {} ".format(xValue)
	print "y value : {} ".format(yValue)
	
	#yValue = f1measure
	#yValue = f1measurePerAction
	#labelValue = 'Adaptive algorithm(Think='+str(percent)+'%)'
	#labelValue = 'Adaptive algorithm(Block size='+str(block)+')'
	
	# uncomment this part for plotting
	plt.plot(xValue, yValue,'g')
	plt.ylim([0, 1])
	plt.legend(loc="upper left")
	plt.savefig('plotQualityAdaptive8.png')
	#plt.show()
	plt.close()
	'''
	return [timeList,f1List]
	


def adaptiveOrder10Join(timeBudget, epoch):
    # 1:Gaussian Naive Bayes
    # 2:Extra Tree
    # 3:Random Forest
    # 4:Adaptive Boosting

    f1 = open('queryTestSentimentTweet10.txt', 'w+')

    # lr,et,rf,ab

    # set = [genderPredicate6,genderPredicate1,genderPredicate7,genderPredicate10]
    # knn,rf,lr,svm

    # set = [genderPredicate7,genderPredicate3,genderPredicate6,genderPredicate10] #working
    set = [genderPredicate7, genderPredicate3, genderPredicate16, genderPredicate10]  # working set 2
    # set = [genderPredicate18,genderPredicate3,genderPredicate16,genderPredicate10]

    print timeBudget
    outsideObjects = []

    # blockList = [4000]
    blockList = [800]

    executionPerformed = 0
    thinkTimeList = []
    executionTimeList = []
    # for percent in thinkPercentList:
    realF1List = []
    for block in blockList:
        # totalAllowedExecution = 1000
        executionPerformed = 0
        # The dictionary currentProbability stores the information about the output of previously ran classifiers. Suppose image 20 has gone through c2 and c3 and the output probability was 0.5 and 0.6.
        # Then the hashmap element of image 20 will be as follows:  20: [0,0.5,0.6,0].
        currentProbability1 = {}
        currentProbability2 = {}
        for i in range(len(dl)):
            key = i
            value = [-1, -1, -1, -1]
            if key in currentProbability1:
                currentProbability1[key].append(value)
                currentProbability2[key].append(value)
            else:
                currentProbability1[key] = [value]
                currentProbability2[key] = [value]
        # print currentProbability

        # The dictionary prevClassifier stores the information about previously ran classifiers. Suppose image 20 has gone through c2 and c3. Then the hashmap element of image 20 will be as follows:  20: [0,1,1,0].
        # The bit vector corresponding to classifier 2 and classifier 3 are set.
        prevClassifier1 = {}
        prevClassifier2 = {}


        for i in range(len(dl)):
            key = i
            value = [0, 0, 0, 0]
            if key in prevClassifier1:
                prevClassifier1[key].append(value)
                prevClassifier2[key].append(value)
            else:
                prevClassifier1[key] = [value]
                prevClassifier2[key] = [value]

        # print prevClassifier

        # currentUncertainty list stores the information of current uncertainty of all the images.

        currentUncertainty1 = [0.99] * len(dl)
        currentUncertainty2 = [0.99] * len(dl)
        currentF1measure = 0
        # print currentUncertainty
        operator = set[0]
        count = 0
        totalExecutionTime = 0
        totalThinkTime = 0
        topKIndexes = [0]
        timeElapsed = 0
        timeList = []
        f1List = []
        blockSize = 800
        executionTime = 0

        stepSize = epoch  # step size of 20 seconds. After every 20 seconds evaluate the quality
        currentTimeBound = epoch

        t11 = 0
        t12 = 0

        while True:
            # t11 = time.time()
            # for i in topKIndexes:

            if count == 0:

                operator = set[0]

                for i1 in range(len(dl)):
                    t1 = time.time()
                    probValues = operator(dl[i1])
                    # print>>f1,probValues
                    indexClf = set.index(operator)

                    # Adding the probability value of the object.
                    tempProb = currentProbability1[i1][0]

                    if probValues > 0.5:
                        probValues = probValues + 0.2
                    else:
                        probValues = 0.1

                    # print tempProb
                    # print probValues
                    tempProb[indexClf] = probValues

                    # setting the bit for the corresponding classifier
                    tempClf = prevClassifier1[i1][0]
                    tempClf[indexClf] = 1

                    tempClf = prevClassifier2[i1][0]
                    tempClf[indexClf] = 1
                    # print prevClassifier[i][0]

                    # calculating the current cobined probability
                    combinedProbability = combineProbability(currentProbability[i1])

                    # using the combined probability value to calculate uncertainty
                    uncertainty = -combinedProbability * np.log2(combinedProbability) - (
                                1 - combinedProbability) * np.log2(1 - combinedProbability)
                    currentUncertainty1[i1] = uncertainty

                    currentUncertainty2[i1] = uncertainty

                qualityOfAnswer1 = findQuality(currentProbability1)
                qualityOfAnswer2 = findQuality(currentProbability2)
                # print 'size of answer set : %d'%(len(qualityOfAnswer[3]))
                # print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
                if len(qualityOfAnswer1[3]) > 0 and len(qualityOfAnswer2[3]) > 0:
                    realF1 = findRealF1(qualityOfAnswer1[3]) * findRealF1(qualityOfAnswer2[3])
                else:
                    realF1 = 0
                # print 'f1 measure after initial classifier:%f'%(realF1)
                # print>>f1,'real F1 : %f'%(realF1)
                # f1measure = qualityOfAnswer[0]
                f1measure = realF1
                timeList.append(0)
                f1List.append(f1measure)
                # currentTimeBound = currentTimeBound + stepSize

                # print>>f1,'time bound completed:%d'%(currentTimeBound)

                #print 'after running one classifier'

            # KNN,RF,LR,SVM
            # 1,6,7,10

            # t11 = time.time()
            if count > 0:
                for w in range(4):
                    tempClfList = ['KNN', 'RF', 'ET', 'SVM']
                    # print>>f1,"w = %d"%(w)
                    # print>>f1, tempClfList[w]
                    imageIndex = [item for item in topKIndexes if nextBestClassifier1[item] == tempClfList[w]]
                    if w != 4:
                        operator = set[w]
                    # print>>f1,operator
                    images = [dl[k] for k in imageIndex]

                    # print>>f1,"images to be run with this operator : {} ".format(imageIndex)
                    if len(imageIndex) > 0:
                        # probValues = operator(images)
                        ######## Executing the function on all the objects ###########
                        for i1 in range(len(imageIndex)):
                            t11 = time.time()

                            probValues = operator(images[i1])
                            rocProb = probValues

                            indexClf = w
                            tempProb = currentProbability1[imageIndex[i1]][0]
                            tempProb[indexClf] = rocProb
                            # print tempProb

                            # setting the bit for the corresponding classifier
                            tempClfList = prevClassifier1[imageIndex[i1]][0]
                            tempClfList[indexClf] = 1
                            # print tempClf

                            # calculating the current cobined probability
                            combinedProbability = combineProbability(currentProbability1[imageIndex[i1]])

                            # using the combined probability value to calculate uncertainty
                            uncertainty = -combinedProbability * np.log2(combinedProbability) - (
                                        1 - combinedProbability) * np.log2(1 - combinedProbability)
                            currentUncertainty1[imageIndex[i1]] = uncertainty

                            t12 = time.time()
                            totalExecutionTime = totalExecutionTime + (t12 - t11)
                            timeElapsed = timeElapsed + (t12 - t11)

                            if timeElapsed > currentTimeBound:
                                qualityOfAnswer1 = findQuality(currentProbability1)
                                qualityOfAnswer2 = findQuality(currentProbability2)

                                if len(qualityOfAnswer[3]) > 0:
                                    realF1 = findRealF1(qualityOfAnswer1[3]) * findRealF1(qualityOfAnswer2[3])
                                else:
                                    realF1 = 0
                                # print>>f1,'real F1 : %f'%(realF1)
                                # f1measure = qualityOfAnswer[0]
                                f1measure = realF1
                                timeList.append(timeElapsed)
                                f1List.append(f1measure)
                                print 'real f1:%f' % (realF1)

                                # print 'time bound completed:%d'%(currentTimeBound)
                                print>> f1, 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f ' % (
                                f1measure, qualityOfAnswer[1], qualityOfAnswer[2], totalExecutionTime, totalThinkTime,
                                timeElapsed)
                                # print 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)

                                currentTimeBound = currentTimeBound + stepSize
                            # break

                            if timeElapsed > timeBudget:
                                break

                    imageIndex[:] = []
                    images[:] = []
                # print>>f1,"Outside of inner for loop"
                # continue

            executionTimeList.append(totalExecutionTime)

            nextBestClassifier1 = [-1] * len(dl)
            deltaUncertainty1 = [0] * len(dl)
            nextBestClassifier2 = [-1] * len(dl)
            deltaUncertainty2 = [0] * len(dl)

            benefitArray = [float(-10)] * 2* len(dl)
            topKIndexes = [0] * len(dl)  # initial block size.

            # currentTempProbability = copy.deepcopy(currentProbability)
            newUncertaintyValue = 0  # initializing

            #### Think phase starts
            # calculating benefit of each objects. Benefit is measured in terms of improvement in f1 measure.
            t21 = time.time()
            # first determining the objects which are not in answer set
            qualityOfAnswer = findQuality(currentProbability)
            currentAnswerSet = qualityOfAnswer[3]
            allObjects = list(range(0, len(dl)))

            qualityOfAnswer1 = findQuality(currentProbability1)
            qualityOfAnswer2 = findQuality(currentProbability2)

            currentAnswerSet.append(qualityOfAnswer1[3])
            currentAnswerSet.append(qualityOfAnswer2[3])

            # outsideObjects = allObjects

            ### Uncomment this part for choosing objects from outside of the answer set.
            if count != 0:
                outsideObjects = [x for x in allObjects if x not in currentAnswerSet]

            # print>>f1,"inside objects : {} ".format(currentAnswerSet)
            # print>>f1,"length of inside objects : %f"%len(currentAnswerSet)
            stateListInside = []
            stateListInside = findStates(currentAnswerSet, prevClassifier)
            # print>>f1,"state of inside objects: {}".format(stateListInside)

            # print>>f1,"outsideObjects : {} ".format(outsideObjects)
            # print>>f1,"length of outsideObjects : %f"%len(outsideObjects)
            stateListOutside = []
            stateListOutside = findStates(outsideObjects, prevClassifier)
            # print>>f1,"state of outside objects: {}".format(stateListOutside)

            if (len(outsideObjects) == 0 and count != 0):
                break

            # print>>f1,"outsideObjects : {} ".format(outsideObjects)
            # print 'count=%d'%(count)

            for j in range(len(outsideObjects)):
                # print>>f1,'deciding for object %d'%(outsideObjects[j])
                # print>>f1,"currentUncertainty: {}".format(currentUncertainty)
                [nextBestClassifier1[outsideObjects[j]], deltaUncertainty1[outsideObjects[j]]] = chooseNextBest(
                    prevClassifier1.get(outsideObjects[j])[0], currentUncertainty1[outsideObjects[j]])
                newUncertaintyValue = currentUncertainty1[outsideObjects[j]] + float(deltaUncertainty1[outsideObjects1[j]])
                newProbabilityValue1 = convertEntropyToProb(newUncertaintyValue)
                # print 'newUncertaintyValue:%f'%(newUncertaintyValue)

                # finding index of classifier
                # indexTempProbClf = set.index(nextBestClassifier[j])
                if outsideObjects[j] < len(prevClassifier1):
                    if nextBestClassifier1[outsideObjects[j]] == 'KNN':
                        # nextBestClassifier[outsideObjects[j]] = 'NA'
                        indexTempProbClf = 0
                    if nextBestClassifier1[outsideObjects[j]] == 'RF':
                        indexTempProbClf = 1
                    if nextBestClassifier1[outsideObjects[j]] == 'ET':
                        indexTempProbClf = 2
                    if nextBestClassifier1[outsideObjects[j]] == 'SVM':
                        indexTempProbClf = 3
                else:
                    if nextBestClassifier2[outsideObjects[j]] == 'KNN':
                        # nextBestClassifier[outsideObjects[j]] = 'NA'
                        indexTempProbClf = 0
                    if nextBestClassifier2[outsideObjects[j]] == 'RF':
                        indexTempProbClf = 1
                    if nextBestClassifier2[outsideObjects[j]] == 'ET':
                        indexTempProbClf = 2
                    if nextBestClassifier2[outsideObjects[j]] == 'SVM':
                        indexTempProbClf = 3

                # higher probability value
                newProbabilityValue2 = 1 - newProbabilityValue1

                # benefit is  (pi * pi_new)/cost(i)
                probability_i = combineProbability(currentProbability[outsideObjects[j]])
                # print 'probability_i: %f, new probability : %f, cost : %f'%(probability_i,newProbabilityValue2,cost(nextBestClassifier[j]))

                if outsideObjects[j] < len(prevClassifier1):
                    if cost(nextBestClassifier1[outsideObjects[j]]) != 0:
                        benefit = float(
                            (probability_i * newProbabilityValue2 * len(qualityOfAnswer1[3])) / float(cost(nextBestClassifier[outsideObjects[j]])))
                        benefitArray[outsideObjects[j]] = benefit
                    else:
                        benefitArray[outsideObjects[j]] = -1
                else:
                    if cost(nextBestClassifier2[outsideObjects[j] - len(prevClassifier1) ]) != 0:
                        benefit = float(
                            (probability_i * newProbabilityValue2 * len(qualityOfAnswer1[3])) / float(cost(nextBestClassifier[outsideObjects[j]])))
                        benefitArray[outsideObjects[j] - len(prevClassifier1)] = benefit
                    else:
                        benefitArray[outsideObjects[j] - len(prevClassifier1)] = -1

            # seq = sorted(benefitArray)
            # print benefitArray
            # Ordering the objects based on deltaUncertainty Value
            # order = [seq.index(v) for v in benefitArray]
            # topIndex= benefitArray.index(max(benefitArray))

            # topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)

            if len(outsideObjects) < blockSize:
                topKIndexes = outsideObjects
            # topKIndexes = heapq.nlargest(len(outsideObjects), range(len(outsideObjects)), benefitArray.__getitem__)
            else:
                topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)

            '''
            if len(outsideObjects) > (blockSize/4):
                topKIndexes = [x for x in topKIndexes if x not in currentAnswerSet]
            else:
                topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
            '''
            # print 'top index:%d'%(topIndex)

            t22 = time.time()
            totalThinkTime = totalThinkTime + (t22 - t21)
            thinkTimeList.append(t22 - t21)

            '''
            if(all(element==0 or element==-1 for element in benefitArray) and count >20):
                break
            '''
            # i=topIndex #next image to be run
            t2 = time.time()

            # timeElapsed = totalExecutionTime + totalThinkTime
            timeElapsed = timeElapsed + totalThinkTime

            classifierSet = [nextBestClassifier[item2] for item2 in topKIndexes]
            '''
            if(all(element=='NA' for element in classifierSet) and count > 10):
                break
            '''
            allClassifierSet = [nextBestClassifier[item3] for item3 in allObjects]
            if (all(element == 'NA' for element in classifierSet) and count > 10):
                topKIndexes = allObjects
            # break
            if (all(element == 'NA' for element in allClassifierSet) and count > 10):
                break
            # print>>f1,'classifier set: {}'.format(classifierSet)
            benefitArray[:] = []
            classifierSet[:] = []

            # print 'round %d completed'%(count)
            # print 'time taken %f'%(timeElapsed)

            # block size is determined in this part.
            if count == 0:
                blockSize = block
                topKIndexes[:] = []
                print 'blockSize: %d' % (blockSize)

            ###### Time check########

            if timeElapsed > currentTimeBound:
                qualityOfAnswer = findQuality(currentProbability)

                if len(qualityOfAnswer[3]) > 0:
                    realF1 = findRealF1(qualityOfAnswer[3])
                else:
                    realF1 = 0
                # print>>f1,'real F1 : %f'%(realF1)
                # f1measure = qualityOfAnswer[0]
                f1measure = realF1
                timeList.append(timeElapsed)
                f1List.append(f1measure)

                # f1measure = qualityOfAnswer[0]
                # timeList.append(timeElapsed)
                # f1List.append(f1measure)

                # print 'time bound completed:%d'%(currentTimeBound)
                print>> f1, 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f ' % (
                f1measure, qualityOfAnswer[1], qualityOfAnswer[2], totalExecutionTime, totalThinkTime, timeElapsed)
                # print 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)

                currentTimeBound = currentTimeBound + stepSize

            if timeElapsed > timeBudget:
                break

            count = count + 1

        plt.title('Quality vs Time Value')
        # print>>f1,'percent : %f'%(percent)
        # print>>f1,'block size : %f'%(block)
        # print>>f1,"f1 measures : {} ".format(realF1List)
        # print>>f1,'total think time :%f'%(totalThinkTime)
        # print>>f1,'total execution time :%f'%(totalExecutionTime)
        plt.ylabel('Quality')
        plt.xlabel('time')
        xValue = timeList
        yValue = f1List
        # print>>f1,"x value : {} ".format(xValue)
        # print>>f1,"y value : {} ".format(yValue)
        # print "x value : {} ".format(xValue)
        # print "y value : {} ".format(yValue)

        '''
        # uncomment this part for plotting
        plt.plot(xValue, yValue,'b')
        plt.ylim([0, 1])
        plt.legend(loc="upper left")
        plt.savefig('plotQualityAdaptive8'+str(block)+'.eps',format = 'eps')
        plt.title('Quality vs Time for block size = '+str(block))
        #plt.show()
        plt.close()
        '''
    # xValue = timeList
    # yValue = f1List

    
    return [timeList, f1List]




def calculateProgressiveScore(t1_list, q1_list):
	length = len(t1_list)
	weight_decrement = 1.0/(length-1)
	
	#print t1_list
	#print q1_list
	#print weight_decrement
	
	prog_score = 0.0
	weight = 1
	
	for i1 in range(1, len(q1_list)):
		improvement = q1_list[i1] - q1_list[i1-1]
		prog_score += (weight * improvement)
		weight = weight - weight_decrement 
	
		#print 'prog_score=%f'%(prog_score)
		#print 'weight = %f'%(weight)
	#print>>f1,"epoch_list = {} ".format(epoch_list)
	#print>>f1,"score_list = {} ".format(score_list)	
	return prog_score



def baseline1(budget, epochSize):
################
#### This approach chooses the set of function and the set of objects in a random fashion. 
#### In each epoch, it chooses a set of object ids and for each of them a set of functions from the remaining functions.
################## 
	
	
	f1 = open('QueryResultBaseline1SentimentTweet.txt','w+')
	#gnb,et,rf,svm
	
	totalTime = 0 
	totalQuality= 0
	timeList =[]
	f1List = []
	
	executionTime = 0
	stepSize = epochSize  #step size of 4 seconds. After every 4 seconds evaluate the quality
	currentTimeBound = epochSize
		
	
	set = [genderPredicate7,genderPredicate3,genderPredicate16,genderPredicate10] 
	aucSet = [0.67,0.66,0.66,0.65]
	#costSet = [ 0.003995, 0.020180, 0.018, 0.000659]
	costSet = [0.003995,0.020180, 0.018, 0.038]
	
	
	
	####################### Initializing the data structures ##################
	round = 1 
	count = 0 
	currentUncertainty = [1]*len(dl)
	currentProbability = {}
	for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]	
				
	prevClassifier = {}
	for i in range(len(dl)):
		key = i
		value = [0,0,0,0]
		if key in prevClassifier:
			prevClassifier[key].append(value)
		else:
			prevClassifier[key] = [value]
				
	#############################################################################
	t1 = time.time()
	

	if count ==0:
		operator = set[0]			
		
				
		for i in range(len(dl)):
			probValues = operator(dl[i])
			#print>>f1,probValues
			indexClf = set.index(operator)
			tempProb = currentProbability[i][0]
			if probValues>0.5:
				probValues = probValues + 0.2
			else:
				probValues = 0.1
			tempProb[indexClf] = probValues
			tempClf = prevClassifier[i][0]
			tempClf[indexClf] = 1
		
	t2 = time.time()
	executionTime = executionTime + (t2- t1)
	set.remove(genderPredicate7)
	
	qualityOfAnswer = findQuality(currentProbability)
	print 'returned images'
	print qualityOfAnswer[3]
	print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
	if len(qualityOfAnswer[3]) > 0:
		realF1 = findRealF1(qualityOfAnswer[3])
	else: 
		realF1 = 0
	print>>f1,'real F1 : %f'%(realF1)
	
	f1measure = realF1
	timeList.append(0)
	f1List.append(f1measure)
	
	aucSet = [0.66,0.66,0.65]
	costSet = [  0.020180, 0.0018, 0.038]
	
	
	
	total_object_length = len(dl)-1
	init_object_list = np.arange(total_object_length)
	
	
	timeElapsed = 0
	
	while timeElapsed < budget:
		total_object_length = len(dl)-1
		t1 = time.time()		
		object_set = random.sample(init_object_list, int(total_object_length/4))
		print object_set
		t2 = time.time()
		timeElapsed += (t2 - t1)
		
		for j in range(len(object_set)):
			workflow = np.random.permutation(4) 
			
			for i in range(len(workflow)):
				operator = set[workflow[i]-1]
				t11 = time.time()
				imageProb = operator(dl[object_set[j]])
				
				rocProb = imageProb
				
					
				#index of classifier
				indexClf = set.index(operator)
				tempProb = currentProbability[object_set[j]][0]
				tempProb[indexClf+1] = rocProb
				t12 = time.time()
				
				
				
				timeElapsed = timeElapsed + (t12- t11)
				if timeElapsed > currentTimeBound:
					qualityOfAnswer = findQuality(currentProbability)
				
					if len(qualityOfAnswer[3]) > 0:
						realF1 = findRealF1(qualityOfAnswer[3])
					else:
						realF1 = 0
					f1measure = realF1
					timeList.append(currentTimeBound)
					f1List.append(f1measure)
					currentTimeBound = currentTimeBound + stepSize
					
				if timeElapsed > budget:
					break
			
			
	
		
		
		
	print>>f1,"Workflow : {} ".format(workflow)
	print>>f1,'Time taken: %f, f1 measure of the answer set: %f, precision:%f, recall:%f'%(timeElapsed,qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2])
	#return f1measure
	return [timeList,f1List]



	
	
def baseline2():  
	'''
	For this algorithm, classifiers are ordered based on (AUC)/Cost value.
	'''
	f1 = open('QueryExecutionResultMuctBaseline2Gender.txt','w+')
	
	
	#Initialization step. 
	currentProbability = {}
	
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbability:
			currentProbability[key].append(value)
		else:
			currentProbability[key] = [value]	
			
	t1 = time.time()
	
	#gnb,et,rf,svm
	set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
	aucSet = [0.85,0.92,0.92,0.89]
	#costSet = [0.063052,0.014482,0.015253,1.567327]
	costSet = [0.029360,0.018030,0.020180,0.790850]
	
	benefitSet = [ float(aucSet[i])/costSet[i] for i in range(len(aucSet))]
	print benefitSet
	workflow =[x for y, x in sorted(zip(benefitSet, set),reverse=True)]
	print workflow
	round = 1 
	
	
	for i in range(len(workflow)):
		operator = workflow[i]
		probValues = operator(dl)
		
		for j in range(len(dl)):
			imageProb = probValues[j]
			rocProb = imageProb
			averageProbability = 0;
			#print 'image:%d'%(j)
			#print("Roc Prob : {} ".format(rocProb))
				
			#index of classifier
			indexClf = set.index(operator)
			tempProb = currentProbability[j][0]
			tempProb[indexClf] = rocProb

		print 'round %d completed'%(round)
		set.remove(operator)
		round = round + 1
		
			
	t2 = time.time()
	timeElapsed = t2-t1
	qualityOfAnswer = findQuality(currentProbability)
	print>>f1,'Time taken: %f, f1 measure of the answer set: %f, precision:%f, recall:%f'%(timeElapsed,qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2])
	print("Workflow : {} ".format(workflow))


def baseline3(budget,epochSize):  
	'''
	For this algorithm, classifiers are chosen based on auc/cost value. But for one classifier, we try to run it on all the images.
	'''
	f1 = open('QueryResultBaseline3SentimentTweet.txt','w+')
	#gnb,et,rf,svm
	
	totalTime = 0 
	totalQuality= 0
	timeList =[]
	f1List = []
	
	executionTime = 0
	stepSize = epochSize #step size of 20 seconds. After every 20 seconds evaluate the quality
	currentTimeBound = epochSize
	
	#knn,rf,lr,svm
	
	set = [genderPredicate7,genderPredicate3,genderPredicate16,genderPredicate10] # working set 2
	#set = [genderPredicate18,genderPredicate3,genderPredicate16,genderPredicate10]
	aucSet = [0.67,0.66,0.66,0.65]
	#costSet = [ 0.003995, 0.020180, 0.018, 0.000659]
	costSet = [0.003995,0.020180, 0.018, 0.038]
	
	
	
	benefitSet = [ float(aucSet[i])/costSet[i] for i in range(len(aucSet))]
	#print benefitSet
	workflow =[x for y, x in sorted(zip(benefitSet, set),reverse=True)]
	print 'in baseline3'
	print workflow
	round = 1 
	count = 0 
	currentUncertainty = [1]*len(dl)
	currentProbability = {}
	for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]	
				
	prevClassifier = {}
	for i in range(len(dl)):
		key = i
		value = [0,0,0,0]
		if key in prevClassifier:
			prevClassifier[key].append(value)
		else:
			prevClassifier[key] = [value]
				
				
	t1 = time.time()
	if count ==0:
		operator = set[0]			
		
				
		for i in range(len(dl)):
			probValues = operator(dl[i])
			#print>>f1,probValues
			indexClf = set.index(operator)
			tempProb = currentProbability[i][0]
			if probValues>0.5:
				probValues = probValues + 0.2
			else:
				probValues = 0.1
			#print probValues
			tempProb[indexClf] = probValues
			#print probValues
			#print>>f1,"temp prob : {} ".format(tempProb)
					
			# setting the bit for the corresponding classifier
			tempClf = prevClassifier[i][0]
			tempClf[indexClf] = 1
					
			
	
	t2 = time.time()
	executionTime = executionTime + (t2- t1)
	set.remove(genderPredicate7)
	#set.remove(genderPredicate18)
	#set.remove(genderPredicate4)
	
	qualityOfAnswer = findQuality(currentProbability)
	#print 'returned images'
	#print qualityOfAnswer[3]
	#print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
	if len(qualityOfAnswer[3]) > 0:
		realF1 = findRealF1(qualityOfAnswer[3])
	else: 
		realF1 = 0
	#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
	print 'real f1:%f'%(realF1)
	f1measure = realF1
	timeList.append(0)
	f1List.append(f1measure)
	#currentTimeBound = currentTimeBound + stepSize
	#print>>f1,'time bound completed:%d'%(currentTimeBound)
	
	
	#aucSet = [0.66,0.65]
	#costSet = [  0.020180, 0.000659]
	#7,3,16,10
	
	aucSet = [0.66,0.66,0.65]
	#costSet = [  0.020180, 0.000236, 0.00012]
	costSet = [  0.020180, 0.0018, 0.038]
	
	#set = [genderPredicate7,genderPredicate3,genderPredicate16,genderPredicate10]
	
	
	#print 'size of the dataset:%d'%(len(dl))
	#print 'budget:%d'%(budget)
	
	benefitSet = [ float(aucSet[i])/costSet[i] for i in range(len(aucSet))]
	#print benefitSet
	workflow =[x for y, x in sorted(zip(benefitSet, set),reverse=True)]
	print 'in baseline3 after initial step'
	print workflow
	round = 1
	probabilitySet = []
	probDictionary = {}
 
	
	
	for k in range(0,1):  # number of times this algorithm will be executed
		#Initialization step. 
		
				
		t1 = time.time()
	
		#print("Workflow : {} ".format(workflow))
		
		for i in range(len(workflow)):
			operator = workflow[i]	
			for j in range(len(dl)):
			#for key in sorted_x:
				#imageProb = probValues[j]
				
				t11 = time.time()
				#imageProb = operator(dl[key[0]])
				#imageProb = operator(dl[len(dl) - j -1])
				imageProb = operator(dl[j])
				
				
				rocProb = imageProb
				averageProbability = 0;

				#print 'image:%d'%(j)
				#print("Roc Prob : {} ".format(rocProb))
					
				#index of classifier
				indexClf = set.index(operator)
				#tempProb = currentProbability[key[0]][0]
				#tempProb = currentProbability[len(dl) -j-1][0]
				
				tempProb = currentProbability[j][0]
				tempProb[indexClf+1] = rocProb
				#print>>f1,"temp prob : {} ".format(tempProb)
				
				t12 = time.time()
				executionTime = executionTime + (t12- t11)
				
				
			
			
			
				if executionTime > currentTimeBound:
					qualityOfAnswer = findQuality(currentProbability)
					#print 'returned images'
					#print qualityOfAnswer[3]
					#print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
					if len(qualityOfAnswer[3]) > 0:
						realF1 = findRealF1(qualityOfAnswer[3])
					else: 
						realF1 = 0
					#print>>f1,'real F1 : %f'%(realF1)
					print 'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
					f1measure = realF1
					timeList.append(currentTimeBound)
					f1List.append(f1measure)
					currentTimeBound = currentTimeBound + stepSize
					#print>>f1,'time bound completed:%d'%(currentTimeBound)
				if executionTime > budget:
					break
				
				
			#print 'round %d completed'%(round)
			
			
			
			round = round + 1
			
				
		t2 = time.time()
		timeElapsed = t2-t1
		qualityOfAnswer = findQuality(currentProbability)
		f1measure = qualityOfAnswer[0]
		
		# store the time values and F1 values
		#print>>f1,"budget values : {} ".format(timeList)
		#print>>f1,"f1 measures : {} ".format(f1List)
			
		#plot quality vs time 
		#timeList = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280] 
		#f1List = [0.6667605490515276, 0.666838374853555, 0.6669078673934243, 0.6670447162904274, 0.6671836282556838, 0.6671978153931125, 0.6671986297399674, 0.6671819086745975, 0.6671767559957542, 0.6672143236162872, 0.6673124457328257, 0.6674186908487334, 0.6674062673780302, 0.6674207092762636] 
		
		'''
		plt.title('Quality vs Time Value for BaseLine 3')
		xValue = timeList
		yValue = f1List
		plt.plot(xValue, yValue)
		plt.ylim([0, 1])
		plt.ylabel('Quality')
		plt.xlabel('Time')	
		plt.savefig('QualityBaseLine3SentimentTwitter.eps', format='eps')
		#plt.show()
		plt.close()
		'''
		
		
	#print>>f1,"Workflow : {} ".format(workflow)
	#print>>f1,'Time taken: %f, f1 measure of the answer set: %f, precision:%f, recall:%f'%(timeElapsed,qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2])
	#return f1measure
	return [timeList,f1List]


def baseline4(budget,epochSize):  
	'''
	For this algorithm, classifiers are ordered based on auc/cost value. But for each images, we try to run all the classifiers before going to another image.
	'''
	
	f1 = open('QueryResultBaseline4SentimentTweet.txt','w+')
	#gnb,et,rf,svm
	
	totalTime = 0 
	totalQuality= 0
	print 'Query completion time:%f'%(budget)
	
	timeList =[]
	f1List = []
	executionTime = 0
	stepSize = epochSize   
	currentTimeBound = epochSize
	
	
	#DT,GNB,RF,KNN
	#LDA,GNB,RF,KNN
	set = [genderPredicate7,genderPredicate3,genderPredicate16,genderPredicate10]
	#set = [genderPredicate18,genderPredicate3,genderPredicate16,genderPredicate10]
	aucSet = [0.67,0.66,0.66,0.65]
	#costSet = [ 0.003995, 0.020180, 0.000236, 0.000659]
	costSet = [ 0.003995, 0.020180, 0.000236, 0.038]
	
	
	
	currentUncertainty = [1]*len(dl)
	count = 0
	t1 = time.time()
	currentProbability = {}
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbability:
			currentProbability[key].append(value)
		else:
			currentProbability[key] = [value]
			
	prevClassifier = {}
	for i in range(len(dl)):
		key = i
		value = [0,0,0,0]
		if key in prevClassifier:
			prevClassifier[key].append(value)
		else:
			prevClassifier[key] = [value]
	
	operator = set[0]			
		
		
	for i in range(len(dl)):
		probValues = operator(dl[i])
		#print>>f1,probValues
		#indexClf = set.index(operator)
		tempProb = currentProbability[i][0]
		
		
		if probValues>0.5:
			probValues = probValues + 0.2
		else:
			probValues = 0.1
		tempProb[0] = probValues
		
		#print>>f1,"temp prob : {} ".format(tempProb)
					
		# setting the bit for the corresponding classifier
		tempClf = prevClassifier[i][0]
		tempClf[0] = 1
					
					
			
	t2 = time.time()
	executionTime = executionTime + (t2- t1)
	
	print 'After initialization'
	
	
	set.remove(genderPredicate7)
	#set.remove(genderPredicate18)
	
	qualityOfAnswer = findQuality(currentProbability)
	#print 'returned images'
	#print qualityOfAnswer[3]
	#print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
	if len(qualityOfAnswer[3]) > 0:
		realF1 = findRealF1(qualityOfAnswer[3])
	else: 
		realF1 = 0
	#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
	f1measure = realF1
	timeList.append(0)
	f1List.append(f1measure)
	#currentTimeBound = currentTimeBound + stepSize
	#print>>f1,'time bound completed:%d'%(currentTimeBound)
	
	
	#set.remove(genderPredicate1)
	
	
	
	aucSet = [0.66,0.66,0.65]
	#costSet = [ 0.020180, 0.000236, 0.000659]
	costSet = [ 0.020180, 0.000236, 0.038]
	
	
	
	
	#print 'size of the dataset:%d'%(len(dl))
	#print 'budget:%d'%(budget)
	
	benefitSet = [ float(aucSet[i])/costSet[i] for i in range(len(aucSet))]
	#print benefitSet
	workflow =[x for y, x in sorted(zip(benefitSet, set),reverse=True)]
	#print workflow
	round = 1 
	
	
	for k in range(0,1):  # number of times this algorithm will be executed
		#Initialization step. 
		
				
		t1 = time.time()
	
		#print("Workflow : {} ".format(workflow))
		

		for j in range(len(dl)):
				#imageProb = probValues[j]
			for i in range(len(workflow)):
				operator = workflow[i]
				t11 = time.time()
				imageProb = operator(dl[j])
				t12 = time.time()
				rocProb = imageProb
				
				#print 'image:%d'%(j)
				#print("Roc Prob : {} ".format(rocProb))
					
				#index of classifier
				indexClf = set.index(operator)
				tempProb = currentProbability[j][0]
				tempProb[indexClf+1] = rocProb
				
				
				
				
				executionTime = executionTime + (t12- t11)
				
				
				if executionTime > budget:
					break
				
				
				if executionTime > currentTimeBound:
					qualityOfAnswer = findQuality(currentProbability)
					#print 'returned images'
					#print qualityOfAnswer[3]
					if len(qualityOfAnswer[3]) > 0:
						realF1 = findRealF1(qualityOfAnswer[3])
					else:
						realF1 = 0
					#print 'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
					f1measure = realF1
					#f1measure = qualityOfAnswer[0]
					timeList.append(currentTimeBound)
					f1List.append(f1measure)
					currentTimeBound = currentTimeBound + stepSize
					#print 'time bound completed:%d'%(currentTimeBound)	
					
			if executionTime > budget:
				break
			
			round = round + 1
			
				
		t2 = time.time()
		timeElapsed = t2-t1
		qualityOfAnswer = findQuality(currentProbability)
		f1measure = qualityOfAnswer[0]
		
		# store the time values and F1 values
		#print>>f1,"budget values : {} ".format(timeList)
		#print>>f1,"f1 measures : {} ".format(f1List)
			
		#plot quality vs time 
		#timeList = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280] 
		#f1List = [0.6667605490515276, 0.666838374853555, 0.6669078673934243, 0.6670447162904274, 0.6671836282556838, 0.6671978153931125, 0.6671986297399674, 0.6671819086745975, 0.6671767559957542, 0.6672143236162872, 0.6673124457328257, 0.6674186908487334, 0.6674062673780302, 0.6674207092762636] 
		
		'''
		plt.title('Quality vs Time Value for BaseLine 4')
		xValue = timeList
		yValue = f1List
		plt.plot(xValue, yValue)
		plt.ylabel('Quality')
		plt.xlabel('Time')	
		plt.ylim([0, 1])
		
		plt.savefig('QualityBaseLine4SentimentTwitter.eps', format='eps')
		#plt.show()
		plt.close()
		'''
	
	
	print>>f1,"Workflow : {} ".format(workflow)
	print>>f1,'Time taken: %f, f1 measure of the answer set: %f, precision:%f, recall:%f'%(timeElapsed,qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2])
	#return f1measure
	return [timeList,f1List]
	
def compareCost():
	f1 = open('DecisionTableTweet\Results\NewAlgorithm\CostCompare.txt','w+')
	operatorList = [genderPredicate1,genderPredicate2, genderPredicate3,genderPredicate4]
	objects = [dl[k] for k in range(100)]
	for i in range(len(operatorList)):
		operator = operatorList[i]
		t11 = time.time()
		print>>f1,operator
		for j in range(100):
		    imageProb = operator(dl[j])
		t12 = time.time()
		print 'individual time:%f'%(t12-t11)
		print>>f1,'individual time:%f'%(t12-t11)
		print>>f1,'individual average time:%f'%((t12-t11)/100)
		
		
		t21 = time.time()	
		prob2 = operator(objects)
		t22 = time.time()
		print 'aggregated time:%f'%(t22-t21)
		print>>f1,'aggregated time:%f'%(t22-t21)
		print>>f1,'aggregated average time:%f'%((t22-t21)/100)
		







def generateMultipleExecutionResult():
	#createSample()
	#createRandomSample()
	
	#q1_all,q2_all,q3_all = [],[],[]
	f1 = open('QueryExecutionTweetResultsSnfCompare.txt','w+')
	t1_all,q1_all,t2_all,q2_all,t3_all,q3_all,t4_all,q4_all=[],[],[],[],[],[],[],[]
	
	t_init_list = []
	
	# Total number of times the queries will executed.
	number_of_runs = 1
	
	
	for i in range(number_of_runs):
		global dl,nl 
		t_load_obj_start = time.time()
		imageIndex = np.random.choice(len(dl2), 1000, replace=False)
		
		imageIndex.sort()
		
		dl_test = [dl2[i1] for i1 in  imageIndex]
		t_load_obj_end = time.time()
		t_init = (t_load_obj_end - t_load_obj_start)
		print>>f1,'total object load time :%f'%(t_init)
		t_init_list.append(t_init)
		
		nl_test = [nl2[i1] for i1 in imageIndex]
		
		t_exec_1 = time.time()
		test_predict_proba = [sentiment_dt.prob_classify(t).prob('1') for t in dl_test]
		t_exec_2 = time.time()
		print>>f1,'initial function evaluation time :%f'%(t_exec_2 - t_exec_1 )
		print (t_exec_2 - t_exec_1 )
		
		
		
		dl = np.array(dl_test)
		nl = np. array(nl_test)
		
		# First parameter is the query completion time and the second parameter is the epoch size. 
		[t4,q4]=baseline1(10,2)
		[t3,q3] =adaptiveOrder10(10,2)		
		[t2,q2] =baseline4(10,2)
		[t1,q1]=baseline3(10,2)
		
				
		
		t1_all.append(t1)
		t2_all.append(t2)
		t3_all.append(t3)
		t4_all.append(t4)
		q1_all.append(q1)
		q2_all.append(q2)
		q3_all.append(q3)
		q4_all.append(q4)
		
		
		print>>f1,'sameple id : %d'%(i)
		print>>f1,"t1 = {} ".format(t1)
		print>>f1,"q1 = {} ".format(q1)
		print>>f1,"t2 = {} ".format(t2)
		print>>f1,"q2 = {} ".format(q2)
		print>>f1,"t3 = {} ".format(t3)
		print>>f1,"q3 = {} ".format(q3)
		print>>f1,"t4 = {} ".format(t4)
		print>>f1,"q4 = {} ".format(q4)
		
		
		print 'iteration :%d completed'%(i)
	
	print>>f1,'average initial predicate evaluation time :%f'%(np.mean(t_init_list))
	q1 = [sum(e)/len(e) for e in zip(*q1_all)]
	q2 = [sum(e)/len(e) for e in zip(*q2_all)]
	q3 = [sum(e)/len(e) for e in zip(*q3_all)]
	q4 = [sum(e)/len(e) for e in zip(*q4_all)]
	plt.plot(t1, q1,lw=2,color='green',marker='o',  label='Baseline1 (Function Based Approach)')
	plt.plot(t2, q2,lw=2,color='orange',marker='^',  label='Baseline2 (Object Based Approach)')
	plt.plot(t3, q3,lw=2,color='blue',marker='d', label='Iterative Approach') ##2,000
	plt.plot(t4, q4,lw=2,color='black',marker ='s', label='Random')


	print>>f1,'for progressive score, storing all time values and quality values'
	print>>f1,"t1 = {} ".format(t1)
	print>>f1,"q1 = {} ".format(q1)
	print>>f1,"t2 = {} ".format(t2)
	print>>f1,"q2 = {} ".format(q2)
	print>>f1,"t3 = {} ".format(t3)
	print>>f1,"q3 = {} ".format(q3)
	print>>f1,"t4 = {} ".format(t4)
	print>>f1,"q4 = {} ".format(q4)
	
	## Generating progressiveness score and storing it.
	score1 =  calculateProgressiveScore(t1,q1)
	score2 =  calculateProgressiveScore(t2,q2)
	score3 =  calculateProgressiveScore(t3,q3)
	score4 =  calculateProgressiveScore(t4,q4)
	
	print>>f1,'score of baseline1 = %f, score of baseline2 = %f, '%(score1,score2)
	print>>f1,'score of our approach = %f, score of random approach = %f, '%(score3,score4)
	
	
	
	plt.ylim([0, 1])
	plt.xlim([0, max(max(t1),max(t2),max(t3))])
	plt.title('Quality vs Cost')
	plt.legend(loc="lower left",fontsize='medium')
	plt.ylabel('F1-measure')
	plt.xlabel('Cost')	
	plt.savefig('PlotTweet_F1-measure.png', format='png')
	plt.savefig('PlotTweet_F1-measure.eps', format='eps')
	plt.close()
	
	
	q1_new = np.asarray(q1)
	q2_new = np.asarray(q2)
	q3_new = np.asarray(q3)
	q4_new = np.asarray(q4)
	
	t1_new = [sum(e)/len(e) for e in zip(*t1_all)]
	t2_new = [sum(e)/len(e) for e in zip(*t2_all)]
	t3_new = [sum(e)/len(e) for e in zip(*t3_all)]
	t4_new = [sum(e)/len(e) for e in zip(*t4_all)]
	
	
	
	
	min_val = min(min(q1_new),min(q2_new),min(q3_new), min(q4_new))
	max_val = max(max(q1_new),max(q2_new),max(q3_new), max(q4_new))
	
	q1_norm = (q1_new-min_val)/(max_val - min_val)
	q2_norm = (q2_new-min_val)/(max_val - min_val)
	q3_norm = (q3_new-min_val)/(max_val - min_val)
	q4_norm = (q4_new-min_val)/(max_val - min_val)
	
	# Removing less than zero values
	q1_norm = [0 if i < 0 else i for i in q1_norm]
	q2_norm = [0 if i < 0 else i for i in q2_norm]
	q3_norm = [0 if i < 0 else i for i in q3_norm]
	q4_norm = [0 if i < 0 else i for i in q4_norm]
	
	
	plt.plot(t1_new, q1_norm,lw=2,color='green', marker='o', label='Baseline1 (Function Based Approach)')
	plt.plot(t2_new, q2_norm,lw=2,color='orange',marker='^',  label='Baseline2 (Object Based Approach)')
	plt.plot(t3_new, q3_norm,lw=2,color='blue',marker ='d', label='Iterative Approach') ##2,000
	plt.plot(t4_new, q4_norm,lw=2,color='black',marker ='s', label='Random') ##2,000
	
	
	## Generating progressiveness score and storing it.
	score1 =  calculateProgressiveScore(t1_new,q1_norm)
	score2 =  calculateProgressiveScore(t2_new,q2_norm)
	score3 =  calculateProgressiveScore(t3_new,q3_norm)
	score4 =  calculateProgressiveScore(t4_new,q4_norm)
	
	print>>f1,'Gain score of baseline1 = %f, score of baseline2 = %f, '%(score1,score2)
	print>>f1,'score of our approach = %f, score of random approach = %f, '%(score3,score4)
	
	
	

	
	plt.title('Quality vs Cost')
	#plt.legend(loc="upper left",fontsize='medium')
	plt.ylabel('Gain')
	plt.xlabel('Cost')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	#plt.ylim([0, 1])
	plt.xlim([0, max(max(t1_new),max(t2_new),max(t3_new))])	
	plt.savefig('PlotTweet_Gain.png', format='png')
	plt.savefig('PlotTweet_Gain.eps', format='eps')
		#plt.show()
	plt.close()
	

def generateOptimalEpoch():
	#createSample()
	#createRandomSample()
	
	#q1_all,q2_all,q3_all = [],[],[]
	epoch_list = [1,2,3,4,5,6,7,8,9,10]
	t1_all,q1_all,t2_all,q2_all,t3_all,q3_all=[],[],[],[],[],[]
	t4_all,q4_all,t5_all,q5_all,t6_all,q6_all=[],[],[],[],[],[]
	t7_all,q7_all,t8_all,q8_all,t9_all,q9_all=[],[],[],[],[],[]	
	t10_all,q10_all=[],[]
	
	f1 = open('OptimalEpochSizeVariation.txt','w+')
	budget = 160
	
	for i in range(10):
		global dl,nl 

		imageIndex = [i1 for i1 in sorted(random.sample(xrange(len(dl2)), 6400 ))]
		dl_test = [dl2[i1] for i1 in  imageIndex]
		nl_test = [nl2[i1] for i1 in imageIndex]
	
		
		
		dl = np.array(dl_test)
		nl = np. array(nl_test)
		
		[t1,q1]=adaptiveOrder10(budget,float((1/budget)*100))
		[t2,q2]=adaptiveOrder10(budget,float((2/budget)*100))
		[t3,q3]=adaptiveOrder10(budget,float((3/budget)*100))
		[t4,q4]=adaptiveOrder10(budget,float((4/budget)*100))
		[t5,q5]=adaptiveOrder10(budget,float((5/budget)*100))
		[t6,q6]=adaptiveOrder10(budget,float((6/budget)*100))
		[t7,q7]=adaptiveOrder10(budget,float((7/budget)*100))
		[t8,q8]=adaptiveOrder10(budget,float((8/budget)*100))
		[t9,q9]=adaptiveOrder10(budget,float((9/budget)*100))
		[t10,q10]=adaptiveOrder10(budget,float((10/budget)*100))
	
	
		
		t1_all.append(t1)
		t2_all.append(t2)
		t3_all.append(t3)
		t4_all.append(t4)
		t5_all.append(t5)
		t6_all.append(t6)
		t7_all.append(t7)
		t8_all.append(t8)
		t9_all.append(t9)
		t10_all.append(t10)
		
		q1_all.append(q1)
		q2_all.append(q2)
		q3_all.append(q3)		
		q4_all.append(q4)
		q5_all.append(q5)
		q6_all.append(q6)
		q7_all.append(q7)
		q8_all.append(q8)
		q9_all.append(q9)
		q10_all.append(q10)
		


		print 'iteration :%d completed'%(i)
	q1 = [sum(e)/len(e) for e in zip(*q1_all)]
	q2 = [sum(e)/len(e) for e in zip(*q2_all)]
	q3 = [sum(e)/len(e) for e in zip(*q3_all)]
	q4 = [sum(e)/len(e) for e in zip(*q4_all)]
	q5 = [sum(e)/len(e) for e in zip(*q5_all)]
	q6 = [sum(e)/len(e) for e in zip(*q6_all)]
	q7 = [sum(e)/len(e) for e in zip(*q7_all)]
	q8 = [sum(e)/len(e) for e in zip(*q8_all)]
	q9 = [sum(e)/len(e) for e in zip(*q9_all)]
	q10 = [sum(e)/len(e) for e in zip(*q10_all)]
	
	
	plt.plot(t1, q1,lw=2,color='blue',marker='o',  label='Iterative Approach(epoch=1)')
	plt.plot(t2, q2,lw=2,color='green',marker='^',  label='Iterative Approach(epoch=2)')
	plt.plot(t3, q3,lw=2,color='orange',marker ='d', label='Iterative Approach(epoch=3)') ##2,000
	plt.plot(t4, q4,lw=2,color='yellow',marker='o',  label='Iterative Approach(epoch=4)')
	plt.plot(t5, q5,lw=2,color='black',marker='^',  label='Iterative Approach(epoch=5)')
	plt.plot(t6, q6,lw=2,color='cyan',marker ='d', label='Iterative Approach(epoch=6)') ##2,000
	
	
	
	
	plt.ylim([0, 1])
	plt.xlim([0, max(max(t1),max(t2),max(t3))])
	plt.title('Quality vs Cost')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .11), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='x-small')
	plt.ylabel('F1-measure')
	#plt.ylabel('Gain')
	plt.xlabel('Cost')	
	plt.savefig('Twitter_epoch_size_variation_gender_1600_obj_epoch_iter20.png', format='png')
	plt.savefig('Twitter_epoch_size_variation_gender_1600_obj_epoch_iter20.eps', format='eps')
		#plt.show()
	plt.close()
	
	
	q1_new = np.asarray(q1)
	q2_new = np.asarray(q2)
	q3_new = np.asarray(q3)
	q4_new = np.asarray(q4)
	q5_new = np.asarray(q5)
	q6_new = np.asarray(q6)
	q7_new = np.asarray(q7)
	q8_new = np.asarray(q8)
	q9_new = np.asarray(q9)
	q10_new = np.asarray(q10)
	
	t1_new = [sum(e)/len(e) for e in zip(*t1_all)]
	t2_new = [sum(e)/len(e) for e in zip(*t2_all)]
	t3_new = [sum(e)/len(e) for e in zip(*t3_all)]
	t4_new = [sum(e)/len(e) for e in zip(*t4_all)]
	t5_new = [sum(e)/len(e) for e in zip(*t5_all)]
	t6_new = [sum(e)/len(e) for e in zip(*t6_all)]
	t7_new = [sum(e)/len(e) for e in zip(*t7_all)]
	t8_new = [sum(e)/len(e) for e in zip(*t8_all)]
	t9_new = [sum(e)/len(e) for e in zip(*t9_all)]
	t10_new = [sum(e)/len(e) for e in zip(*t10_all)]
	
	

	t1_list = [t1_new,t2_new,t3_new,t4_new,t5_new,t6_new,t7_new,t8_new,t9_new,t10_new]
	q1_list = [q1_new,q2_new,q3_new,q4_new,q5_new,q6_new,q7_new,q8_new,q9_new,q10_new]
	epoch_list = [1,2,3,4,5,6,7,8,9,10] # percent list
	score_list = []
	
	
	for i1 in range(len(t1_list)):
		t1_2 = t1_list[i1]
		t1_2 = t1_2[1:]
		q1_2 = q1_list[i1]
		weight_t1 = [max(1-float(element - 1)/budget,0) for element in t1_2]
		improv_q1 = [x - q1_2[i - 1] for i, x in enumerate(q1_2) if i > 0]
		print weight_t1
		print improv_q1
		a1 = np.dot(weight_t1,improv_q1)
		print a1
		score_list.append(a1)
	print>>f1,"epoch_list = {} ".format(epoch_list)
	print>>f1,"score_list = {} ".format(score_list)	
	plt.plot(epoch_list, score_list,lw=2,color='blue',marker='o',  label='Iterative Approach')
	
	
	#plt.ylim([0, 1])
	plt.xlim([0, max(epoch_list)])
	plt.title('AUC Score vs Epoch Size')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .11), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	plt.ylabel('AUC Score',fontsize='large')
	plt.xlabel('Percentage of time spent in plan generation phase',fontsize='large')	
	plt.savefig('Twitter_EpochSize_AUC_Plot_1600_obj_list.png', format='png')
	plt.savefig('Twitter_EpochSize_AUC_Plot_1600_obj_list.eps', format='eps')
		#plt.show()
	plt.close()	
	
	##### Plotting with setting the ylim #######
	plt.plot(epoch_list, score_list,lw=2,color='blue',marker='o',  label='Iterative Approach')
	
	
	plt.ylim([0, 1])
	plt.xlim([0, max(epoch_list)])
	plt.title('AUC Score vs Epoch Size')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .11), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	plt.ylabel('AUC Score',fontsize='large')
	#plt.ylabel('Gain')
	#plt.xlabel('Epoch Size')	
	plt.xlabel('Percentage of time spent in plan generation phase',fontsize='large')	
	plt.savefig('TwitterEpochSize_AUC_Plot_ylim_100_iter_1600obj_list.png', format='png')
	plt.savefig('TwitterEpochSize_AUC_Plot_ylim_100_iter_1600obj_list.eps', format='eps')
		#plt.show()
	plt.close()	
	


if __name__ == '__main__':
	t1 = time.time()
	setup()
	f1 = open('TweetSnfCandidateSelection.txt','w+')
	generateMultipleExecutionResult()
	
	