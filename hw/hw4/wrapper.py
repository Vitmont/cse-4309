import os

training_file = ["pendigits_training.txt","satellite_training.txt","yeast_training.txt"]
test_file = ["pendigits_test.txt","satellite_test.txt","yeast_test.txt"]
layers = 0
units_per_layer = 0
rounds = 0

counter = 0
for i in range(len(training_file)):
    for layers in range(2,6):
        for units_per_layer in range(15,31):
            for rounds in range(10,20):
                print("neural_networks.py {} {} {} {} {} {}".format(training_file[i], test_file[i], layers, units_per_layer, rounds, rounds/(counter+1)))
                counter+= 1
                
print(counter)