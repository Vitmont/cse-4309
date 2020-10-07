##|################################################
# | Author: Seth Jaksik
# | ID: 1001541359
# | Email: seth.jaksik@mavs.uta.edu
##|################################################
# | linear_regression.py
# |
# | Use: Calculate linear regression of data
##|################################################
# | Licence: MIT License
##|################################################

##|################################################
# | Imports
##|################################################
import sys
import numpy as np
import math

##|################################################
# | Functions
##|################################################
# Calculate the phi vals of vector
def phiFunc(x,degree):
    phi = [1]
    for i in range(len(x)):
        for j in range(1,degree+1):
            phi.append(x[i]**j)
    return phi

# Get weights from training data
def trainWeights(phi, target,lambdaVal):
    output = []
    weights = np.dot(np.dot(np.linalg.pinv(np.dot(phi.T, phi) + np.dot(lambdaVal,np.identity(len(phi[0])))), phi.T), target)
    for i in range(len(weights)):
        output.append([weights[i]])
    return output

##|################################################
# | Data Prep
##|################################################
if len(sys.argv) < 5:
    print("Not enough arguments")
    exit(0)

degree = int(sys.argv[3])
lambdaVal = float(sys.argv[4])

# Read in files
inFile = open(sys.argv[1], 'r')
lines = inFile.readlines()
trainData = []
for row in lines:
    temp = row.split()
    trainData.append(np.array(temp).astype(np.float64))

inFile = open(sys.argv[2], 'r')
lines = inFile.readlines()
testData = []
for row in lines:
    temp = row.split()
    testData.append(np.array(temp).astype(np.float64))

##|################################################
# | Training Phase
##|################################################
phiVals = []
targetOut = []
for i in range(len(trainData)):
    # get phi function and target values
    phiVals.append(phiFunc(trainData[i][:-1],degree))
    targetOut.append(trainData[i][-1])
phiVals = np.array(phiVals)
targetOut = np.array(targetOut)

# train the weights
weights = np.array(trainWeights(phiVals, targetOut, lambdaVal))
for i in range(len(weights)):
    print("w{:d}={:.4f}".format(i, weights[i][0]))

##|################################################
# | Testing Phase
##|################################################
for i in range(len(testData)):
    predict = np.dot(weights.T,phiFunc(testData[i][:-1],degree))
    error = (predict[0] - testData[i][-1])**2
    print("ID={:5d}, output={:14.4f}, target value = {:10.4f}, squared error = {:0.4f}".format(i+1,predict[0],testData[i][-1],error))