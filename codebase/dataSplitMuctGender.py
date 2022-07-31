import sys,re
import os,pickle
import numpy as np
import csv
from skimage.feature import hog
from skimage import io
from skimage import data,color,exposure

'''
def getMuctTrainData():
	testX = []
	testY = []
	#yaleDirectory = 'E:\Codes\Codes\ExtendedYaleB\ExtendedYaleB\TestImageSmall'
	dfp = 'Muct_Training2.pkl'
	rd = pickle.load(open(dfp,'rb'))
	
	
	for fu in rd:
		#print fu[0]
		#print fu[1]
		#print fu[2]
		
		namep = fu[0]
		
		if("-m" in namep):
			testX.append(fu[2])
			testY.append(1)
		else:
			testX.append(fu[2])
			testY.append(0)
		
		
		
	# return dl
	print(testX)
	print(testY)
	pickle.dump([testX,testY],open('MuctTrainGender2_XY.p','wb'))
    
def getMuctValidationData():
	testX = []
	testY = []
	#yaleDirectory = 'E:\Codes\Codes\ExtendedYaleB\ExtendedYaleB\TestImageSmall'
	dfp = 'Muct_Validation2.pkl'
	rd = pickle.load(open(dfp,'rb'))

	for fu in rd:
		#print fu[0]
		#print fu[1]
		#print fu[2]
		
		namep = fu[0]
		
		if("-m" in namep):
			testX.append(fu[2])
			testY.append(1)
		else:
			testX.append(fu[2])
			testY.append(0)
		
		
		
	# return dl
	print(testX)
	print(testY)
	pickle.dump([testX,testY],open('MuctValidationGender2_XY.p','wb'))
	

    
def getMuctTestData():
	testX = []
	testY = []
	#yaleDirectory = 'E:\Codes\Codes\ExtendedYaleB\ExtendedYaleB\TestImageSmall'
	dfp = 'Muct_Testing2.pkl'
	rd = pickle.load(open(dfp,'rb'))

	for fu in rd:
		#print fu[0]
		#print fu[1]
		#print fu[2]
		
		namep = fu[0]
		
		if("-m" in namep):
			testX.append(fu[2])
			testY.append(1)
		else:
			testX.append(fu[2])
			testY.append(0)
		
		
		
	# return dl
	print(testX)
	print(testY)
	pickle.dump([testX,testY],open('MuctTestGender2_XY.p','wb'))
	'''
############################################################################
# For feature glass
############################################################################	
def getMuctTrainData():
	testX = []
	testY = []
	#yaleDirectory = 'E:\Codes\Codes\ExtendedYaleB\ExtendedYaleB\TestImageSmall'
	dfp = 'Muct_Training2.pkl'
	rd = pickle.load(open(dfp,'rb'))
	
	
	for fu in rd:
		#print fu[0]
		#print fu[1]
		#print fu[2]
		
		namep = fu[0]
		print namep
		if("g." in namep):
		#if("-g" in namep):
			testX.append(fu[2])
			testY.append(1)
		else:
			testX.append(fu[2])
			testY.append(0)
		
		
		
	# return dl
	print(testX)
	print(testY)
	pickle.dump([testX,testY],open('MuctTrainGlass2_XY.p','wb'))
    
def getMuctValidationData():
	testX = []
	testY = []
	#yaleDirectory = 'E:\Codes\Codes\ExtendedYaleB\ExtendedYaleB\TestImageSmall'
	dfp = 'Muct_Validation2.pkl'
	rd = pickle.load(open(dfp,'rb'))

	for fu in rd:
		#print fu[0]
		#print fu[1]
		#print fu[2]
		
		namep = fu[0]
		
		if("g." in namep):
		#if("-m" in namep):
			testX.append(fu[2])
			testY.append(1)
		else:
			testX.append(fu[2])
			testY.append(0)
		
		
		
	# return dl
	print(testX)
	print(testY)
	pickle.dump([testX,testY],open('MuctValidationGlass2_XY.p','wb'))
	

    
def getMuctTestData():
	testX = []
	testY = []
	#yaleDirectory = 'E:\Codes\Codes\ExtendedYaleB\ExtendedYaleB\TestImageSmall'
	dfp = 'Muct_Testing2.pkl'
	rd = pickle.load(open(dfp,'rb'))

	for fu in rd:
		#print fu[0]
		#print fu[1]
		#print fu[2]
		
		namep = fu[0]
		
		if("g." in namep):
			testX.append(fu[2])
			testY.append(1)
		else:
			testX.append(fu[2])
			testY.append(0)
		
		
		
	# return dl
	print(testX)
	print(testY)
	pickle.dump([testX,testY],open('MuctTestGlass2_XY.p','wb'))



if __name__ =='__main__':
	#getMuctTrainData()
	#getMuctValidationData()
	getMuctTestData()
