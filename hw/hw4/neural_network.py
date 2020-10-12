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

# notes
# l = layer
# i = index of layer
# w(lij) = weight of layer l-1 unit j to l unit i

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


def normalize(data):
    max = 0
    for i in range(len(data)):
        for j in range(len(data[i])-1):
            if data[i][j] > max:
                max = data[i][j]
    for i in range(len(data)):
        for j in range(len(data[i])-1):
            data[i][j] /= max
    return data


def findClasses(data):
    classes = []
    for i in range(len(data)):
        if data[i][-1] not in classes:
            classes.append(data[i][-1])
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


def initT():
    t = np.zeros((len(trainData), K))
    for i in range(len(trainData)):
        t[i][int(trainData[i][-1])] = 1
    return t


def calcLearningRate(round):
    return 0.98**(round-1)


def sigmoid(x):
    return 1/(1+math.exp(-x))
##|################################################
# | Data Prep
##|################################################


# REMOVE THIS AFTER TESTING and uncomment below
layers = int(3)
unitsPerLayer = int(20)
rounds = int(20)
inFile = open("pendigits_training.txt", 'r')
lines = inFile.readlines()
trainData = []
for row in lines:
    temp = row.split()
    trainData.append(np.array(temp).astype(np.float64))

inFile = open("pendigits_test.txt", 'r')
lines = inFile.readlines()
testData = []
for row in lines:
    temp = row.split()
    testData.append(np.array(temp).astype(np.float64))
#################################
# if len(sys.argv) < 6:
#     print("Not enough arguments")
#     exit(0)

# layers = int(sys.argv[3])
# unitsPerLayer = int(sys.argv[4])
# rounds = int(sys.argv[5])


# Read in files
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

##|################################################
# | Training Phase
##|################################################

# normalize the training data
trainData = normalize(trainData)

# find classes
classes = findClasses(trainData)
K = len(classes)
D = len(trainData[0])-1
J = initJ()
t = initT()

# initialize the weights to uniform distribution
b = np.zeros((layers, unitsPerLayer))
w = np.zeros((layers, unitsPerLayer, unitsPerLayer))
b = initWeights(b)
w = initWeights(w)

last_error = 0
err = 0
for round in range(rounds):  # num of rounds
    print(round)
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
        #delta[layers-1] = np.zeros(K)
        for i in range(J[-1]):
            delta[layers-1][i] = (z[layers-1][i] - t[n][i]) * \
                z[layers-1][i] * (1-z[layers-1][i])
        for l in range(layers-2, 1, -1):
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
                    w[l][i][j] = w[l][i][j] - (calcLearningRate(round) * delta[l][i] * z[l-1][j])

##|################################################
# | Classification Phase
##|################################################
totalAcc = 0
for n in range(len(testData)):
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
                temp += w[l][i][j] * z[l-1][j]
            a[l][i] = b[l][i] + temp
            z[l][i] = sigmoid(a[l][i])
    predict = (0, 0)
    accuracy = 1
    for i in range(len(z[layers-1])):
        if z[layers-1][i] == predict[0]:
            accuracy += 1
        if z[layers-1][i] > predict[0]:
            predict = z[layers-1][i], i
            accuracy = 1
    accuracy = 1 if (int(predict[0]) == int(testData[n][-1])) else 0
    totalAcc += accuracy
    print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format(
        n+1, predict[1], int(testData[n][-1]), accuracy))

print("classification accuracy={:6.4f}".format(totalAcc/len(testData)))
