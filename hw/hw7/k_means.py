##|################################################
# | Author: Seth Jaksik
# | ID: 1001541359
# | Email: seth.jaksik@mavs.uta.edu
##|################################################
# | k_means.py
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

##|################################################
# | Functions
##|################################################
def L_2(v1, v2):
    summation = 0
    for i in range(len(v1)):
        summation += (v1[i] - v2[i])**2
    return math.sqrt(summation)

def group(vals):
    values = set(map(lambda x: x[1], vals))
    return [[y[0] for y in vals if y[1] == x] for x in values]

#|################################################
# | Data Prep
#|################################################
if len(sys.argv) < 4:
    print("Not enough arguments")
    exit(0)

k = int(sys.argv[2])
initialization = sys.argv[3]

#Read in files
inFile = open(sys.argv[1], 'r')
lines = inFile.readlines()
data = []
for row in lines:
    temp = row.split()
    temp = [float(x) for x in temp]
    data.append([temp, -1])
    
#|################################################
# | Initialization
#|################################################
if initialization == "round-robin":
    clusterNum = 0
    for i in range(len(data)):
        data[i][1] = clusterNum
        clusterNum = (clusterNum+1) % k
elif initialization == "random":
    for i in range(len(data)):
        data[i][1] = random.randint(0, k-1)
else:
    print("Unknown initialization type...")
    exit(1)

grouped = group(data)
clusterMeans = []
for i in range(k):
    clusterMeans.append(np.mean(grouped[i], axis=0))
#|################################################
# | Classification
#|################################################
while(True):
    # recluster based of means
    newCluster = []
    for i in range(len(data)):
        smallestDistance = 99999
        clusterNum = -1
        for x in range(len(clusterMeans)):
            if L_2(data[i][0], clusterMeans[x]) < smallestDistance:
                clusterNum = x
                smallestDistance = L_2(data[i][0], clusterMeans[x])
        newCluster.append([data[i][0], clusterNum])

    # check if clusters are same
    same = True
    for i in range(len(data)):
        if data[i][1] != newCluster[i][1]:
            same = False
            break
    if same == True:
        break

    data = newCluster.copy()
    grouped = group(data)
    
    # find new means
    for i in range(k):
        clusterMeans[i] = np.mean(grouped[i], axis=0)

# print final results
for i in range(len(data)):
    if len(data[i][0]) == 1:
        print("{:10.4f} --> cluster {:d}".format(data[i][0][0], data[i][1]+1))
    else:
        print("({:10.4f}, {:10.4f}) --> cluster {:d}".format(
            data[i][0][0], data[i][0][1], data[i][1]+1))