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
def normalize(data):
    max = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j]>max:
                max = data[i][j]
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] /= max
    return data           
##|################################################
# | Data Prep
##|################################################
if len(sys.argv) < 6:
    print("Not enough arguments")
    exit(0)

layers = int(sys.argv[3])
unitsPerLayer = int(sys.argv[4])
rounds = int(sys.argv[5])

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

trainData = normalize(trainData)