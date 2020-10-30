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
import random

##|################################################
# | Functions
##|################################################


def normalize(train, test):
    max = 0
    for i in range(len(train)):
        for j in range(len(train[i])-1):
            if train[i][j] > max:
                max = train[i][j]
    for i in range(len(train)):
        for j in range(len(train[i])-1):
            train[i][j] /= max
    for i in range(len(test)):
        for j in range(len(test[i])-1):
            test[i][j] /= max
    return train, test


def findClasses(data):
    classes = dict()
    for i in range(len(data)):
        if data[i][-1] not in classes:
            classes[data[i][-1]] = []
    counter = 0
    for i in sorted(classes.keys()):
        temp = np.zeros(len(classes))
        temp[counter] = 1
        counter += 1
        classes[i] = temp
    return classes


def initWeights(weights):
    for x in np.nditer(weights, op_flags=['readwrite']):
        x[...] = random.uniform(-0.05, 0.05)
    return weights


def initJ():
    J = np.zeros(layers, dtype=int)
    for i in range(len(J)):
        J[i] = unitsPerLayer
    J[0] = int(D)
    J[-1] = int(K)
    return J


def initT(data):
    t = np.zeros((len(data), K))
    for i in range(len(data)):
        t[i] = classes[int(data[i][-1])]
    return t


def calcLearningRate(round):
    return 0.98**(round-1)


def sigmoid(x):
    return 1/(1+math.exp(-x))
##|################################################
# | Data Prep
##|################################################
if len(sys.argv) < 6:
    print("Not enough arguments")
    exit(0)

layers = int(sys.argv[3])
unitsPerLayer = int(sys.argv[4])
rounds = int(sys.argv[5])


#Read in files
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

# normalize the training data
trainData, testData = normalize(trainData, testData)
# find classes
classes = findClasses(trainData)
K = len(classes)
D = len(trainData[0])-1
J = initJ()
t = initT(trainData)

# initialize the weights to uniform distribution
maxVal = max(K, D, unitsPerLayer)
b = np.zeros((layers, maxVal))
w = np.zeros((layers, maxVal, maxVal))
b = initWeights(b)
w = initWeights(w)

for round in range(rounds):  # num of rounds
    for n in range(len(trainData)):  # for each input in a round
        # Step 1: Init input layer
        z = np.zeros((layers,), dtype=np.ndarray)
        z[0] = trainData[n][:-1]  # set input layer

        # Step 2: Compute outputs
        a = np.ndarray((layers,), dtype=np.ndarray)
        for l in range(1, layers):
            a[l] = np.zeros(J[l])
            z[l] = np.zeros(J[l])
            for i in range(J[l]):
                temp = 0
                for j in range(J[l-1]):
                    temp += (w[l][i][j] * z[l-1][j])
                a[l][i] = b[l][i] + temp
                z[l][i] = sigmoid(a[l][i])

        # Step 3: Compute New delta Vals
        delta = np.zeros((layers,), dtype=np.ndarray)
        for i in range(layers):
            delta[i] = np.zeros(J[i])
        for i in range(J[-1]):
            delta[layers-1][i] = (z[layers-1][i] - t[n][i]) * \
                z[layers-1][i] * (1-z[layers-1][i])
        for l in range(layers-2, 0, -1):
            delta[l] = np.zeros(J[l])
            for i in range(J[l]):
                temp = 0
                for k in range(J[l+1]):
                    temp += delta[l+1][k] * w[l+1][k][i]
                delta[l][i] = temp * z[l][i] * (1-z[l][i])

        # Step 4: Update Weights
        for l in range(1, layers):
            for i in range(J[l]):
                b[l][i] = b[l][i] - calcLearningRate(round) * delta[l][i]
                for j in range(J[l-1]):
                    w[l][i][j] = w[l][i][j] - \
                        (calcLearningRate(round) * delta[l][i] * z[l-1][j])

##|################################################
# | Classification Phase
##|################################################
totalAcc = 0
t_test = initT(testData)
for n in range(len(testData)):
    # Step 1: Init input layer
    z = np.zeros((layers,), dtype=np.ndarray)
    z[0] = testData[n][:-1]  # set input layer

    # Step 2: Compute outputs
    a = np.ndarray((layers,), dtype=np.ndarray)
    for l in range(1, layers):
        a[l] = np.zeros(J[l])
        z[l] = np.zeros(J[l])
        for i in range(J[l]):
            temp = 0
            for j in range(J[l-1]):
                temp += w[l][i][j] * z[l-1][j]
            a[l][i] = b[l][i] + temp
            z[l][i] = sigmoid(a[l][i])
    predictClass = []
    predictVal = 0
    accuracy = 0
    for i in range(J[-1]):
        if z[layers-1][i] == predictVal:
            predictClass.append(int(sorted(classes.keys())[i]))
        if z[layers-1][i] > predictVal:
            predictVal = z[layers-1][i]
            predictClass.clear()
            predictClass.append(int(sorted(classes.keys())[i]))
    accuracy = 1 if (int(testData[n][-1]) in predictClass) else 0
    totalAcc += accuracy
    print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format(
        n+1, predictClass[0], int(testData[n][-1]), accuracy))
print("classification accuracy={:6.4f}".format(totalAcc/len(testData)))
