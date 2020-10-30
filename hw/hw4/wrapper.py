import os
import datetime
print(datetime.datetime.now())
print("train, test, layers, units, rounds, accuracy")
training_file = ["pendigits_training.txt","satellite_training.txt","yeast_training.txt"]
test_file = ["pendigits_test.txt","satellite_test.txt","yeast_test.txt"]
layers = 0
units_per_layer = 0
rounds = 0
counter = 1
for i in range(len(training_file)):
    for layers in range(5,11):
        for units_per_layer in range(20,31):
            for rounds in range(10,21):
                os.system("C:/Users/JaksikSethJoseph/anaconda3/python.exe e:/Code/cse-4309/hw/hw4/neural_network_test.py {} {} {} {} {}".format(training_file[i], test_file[i], layers, units_per_layer, rounds))
print(datetime.datetime.now())