##|################################################
##| naive_bayes.py
##|
##| Use: Calculate Bayesian Classifiers for data
##|################################################
##| MIT License
##|################################################
##| Author: Seth Jaksik
##| Version: 1.0.0
##| Email: seth.jaksik@mavs.uta.edu
##|################################################

import sys
import numpy as np
import math

# train = "pendigits_training.txt"
test = "yeast_test.txt"
train = "yeast_training.txt"

datafile = open(train, 'r')
lines = datafile.readlines()
trainData = []
for row in lines:
    trainData.append(row.split())
    
datafile = open(train, 'r')
lines = datafile.readlines()
testData = []
for row in lines:
    testData.append(row.split()) 

#TRAINING
totalTrain = len(trainData)
# seperate based on class
classes = dict()
for i in range(len(trainData)):
    vector = trainData[i]
    class_value = int(vector[-1])
    if (class_value not in classes):
        classes[class_value] = list()
    classes[class_value].append(vector[:-1])
    
# convert arrays to floats
for i in classes:
    for j in range(len(classes[i])):
        classes[i][j] = np.array(classes[i][j]).astype(np.float64)

# prep mean and std matrices
numClass = len(classes)
numAttr = len(next(iter(classes.values()))[0])
trainMean = dict()
trainSTD = dict()


# calc mean and std for each class and attr
for i in classes: # each class
    tempMean = tempSTD = [0]*numAttr
    
    #calc mean
    for j in range(len(classes[i])):
        tempMean += classes[i][j]
    tempMean /= len(classes[i])
    trainMean[i] = tempMean
    
    #calc std
    for j in range(len(classes[i])):
        tempSTD += ((classes[i][j] - trainMean[i])**2)
    tempSTD = (tempSTD/(len(classes[i])-1))**(1/2)    
    trainSTD[i] = tempSTD

# set to 0.01 if smaller
for i in trainSTD:
    for j in range(len(trainSTD[i])):
        if trainSTD[i][j] < 0.01:
            trainSTD[i][j] = 0.01

# calc class probabilities
classProb = dict()
for i in classes:
    classProb[i] = len(classes[i])/totalTrain

# print train stats
for i in sorted(trainMean.keys()):
    for j in range(len(trainMean[i])):
        print("Class {:d}, attribute {:d}, mean = {:0.2f}, std = {:0.2f}".format(i,j+1,trainMean[i][j], trainSTD[i][j]))

#CLASSIFICATION
def gaussian(x):
    return (1/())