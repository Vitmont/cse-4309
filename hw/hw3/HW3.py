#!/usr/bin/env python
# coding: utf-8

# In[315]:


import numpy as np
import sys
import math
import os


# In[316]:


trainFile = "yeast_training.txt"
testFile = "yeast_test.txt"


# In[317]:


# Funcs
def phiFunc(x,degree):
    phi = [1]
    for i in range(len(x)):
        for j in range(1,degree+1):
            phi.append(x[i]**j)
    return phi

def trainWeights(phi, target,lambdaVal):
    output = []
    weights = np.dot(np.dot(np.linalg.pinv(np.dot(phi.T, phi) + np.dot(lambdaVal,np.identity(len(phi[0])))), phi.T), target)
    for i in range(len(weights)):
        output.append([weights[i]])
    return output


# In[318]:


degree = 2
lambdaVal = 0

inFile = open(trainFile, 'r')
lines = inFile.readlines()
trainData = []
for row in lines:
    temp = row.split()
    trainData.append(np.array(temp).astype(np.float64))

inFile = open(testFile, 'r')
lines = inFile.readlines()
testData = []
for row in lines:
    temp = row.split()
    testData.append(np.array(temp).astype(np.float64))


# In[320]:


# TRAIN PHASE
phiVals = []
targetOut = []
for i in range(len(trainData)):
    phiVals.append(phiFunc(trainData[i][:-1],degree))
    targetOut.append(trainData[i][-1])
phiVals = np.array(phiVals)
targetOut = np.array(targetOut)
weights = np.array(trainWeights(phiVals, targetOut, lambdaVal))
for i in range(len(weights)):
    print("w{:d}={:.4f}".format(i, weights[i][0]))


# In[321]:


# TEST PHASE
for i in range(len(testData)):
    predict = np.dot(weights.T,phiFunc(testData[i][:-1],degree))
    error = (predict[0] - testData[i][-1])**2
    print("ID={:5d}, output={:14.4f}, target value = {:10.4f}, squared error = {:0.4f}".format(i+1,predict[0],testData[i][-1],error))


# In[ ]:




