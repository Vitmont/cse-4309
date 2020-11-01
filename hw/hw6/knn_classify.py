##|################################################
# | Author: Seth Jaksik
# | ID: 1001541359
# | Email: seth.jaksik@mavs.uta.edu
##|################################################
# | decision_tree.py
# |
# | Use: Decision tree to classify data sets
##|################################################
# | Licence: MIT License
##|################################################

##|################################################
# | Imports
##|################################################
import sys
import numpy as np
import math
import random
from scipy import stats

##|################################################
# | Functions
##|################################################


def normalize():
    means = np.mean(trainData, axis=0)
    stds = np.std(trainData, axis=0, ddof=1)
    for i in range(len(stds)-1):
        if stds[i] == 0:
            stds[i] = 1
    for i in range(len(trainData)):
        for j in range(len(means)-1):
            trainData[i][j] = (trainData[i][j] - means[j])/stds[j]
    for i in range(len(testData)):
        for j in range(len(means)-1):
            testData[i][j] = (testData[i][j] - means[j])/stds[j]

def L_2(v1,v2):
    summation = 0
    for i in range(len(v1)-1):
        summation += (v1[i]- v2[i])**2
    return math.sqrt(summation)


#|################################################
# | Data Prep
#|################################################
# if len(sys.argv) < 4:
#     print("Not enough arguments")
#     exit(0)

# k = sys.argv[3]


# #Read in files
# inFile = open(sys.argv[1], 'r')
# lines = inFile.readlines()
# trainData = []
# for row in lines:
#     temp = row.split()
#     trainData.append(np.array(temp).astype(np.float64))

# inFile = open(sys.argv[2], 'r')
# lines = inFile.readlines()
# testData = []
# for row in lines:
#     temp = row.split()
#     testData.append(np.array(temp).astype(np.float64))
k = 5

classes = []
#Read in files
inFile = open("yeast_training.txt", 'r')
lines = inFile.readlines()
trainData = []
for row in lines:
    temp = row.split()
    trainData.append(np.array(temp).astype(np.float64))
    if int(temp[-1]) not in classes:
        classes.append(int(temp[-1]))

inFile = open("yeast_test.txt", 'r')
lines = inFile.readlines()
testData = []
for row in lines:
    temp = row.split()
    testData.append(np.array(temp).astype(np.float64))

#|################################################
# | Data Prep
#|################################################
normalize()
#|################################################
# | Classification
#|################################################
totalAcc = 0
for n in range(len(testData)):
    k_nearest_val = np.full(k,9999)
    k_nearest_class = np.full(k,-1)
    for i in range(len(trainData)):
        distance = L_2(trainData[i], testData[n])
        max_val = np.argmax(k_nearest_val)
        if distance < k_nearest_val[max_val]:
            k_nearest_val[max_val] = distance
            k_nearest_class[max_val] = trainData[i][-1]
    max_count = 0
    predictClass = [0]
    for i in classes:
        counts = np.count_nonzero(k_nearest_class == i)
        if counts > max_count:
            max_count = counts
            predictClass.clear()
            predictClass.append(i)
        elif counts == max_count:
            predictClass.append(i)
    accuracy = 1/len(predictClass) if int(testData[n][-1] in predictClass) else 0
    totalAcc += accuracy
    print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format(n+1, int(predictClass[0]), int(testData[n][-1]), accuracy))

print("classification accuracy={:6.4f}".format(totalAcc/len(testData)))