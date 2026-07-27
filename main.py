import numpy as np
import json
import imgtoarray as ita
import trainer as tr
import asdlkfj as deez

print(deez.nuts())

def dingus(input_layer, weights1, bias1, weights2, bias2):
    l1 = np.dot(input_layer, weights1) + bias1
    l1_outputs = tr.relu(l1)
    l2 = np.dot(l1_outputs, weights2) + bias2
    l2_outputs = tr.softmax(l2)
    return l2_outputs

elements = 144
input_faces = ita.read_training_data(elements)
desired_training_outputs = [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],
                            [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
                            [0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],
                            [1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],
                            [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
                            [0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],
                            ]
#list_desired_outputs = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
#desired_outputs = np.array([list_desired_outputs]).T

#read gains
with open("training_data/weights1.txt", "r") as file:
    weights1 = json.load(file)
with open("training_data/bias1.txt", "r") as file:
    bias1 = json.load(file)
with open("training_data/weights2.txt", "r") as file:
    weights2 = json.load(file)
with open("training_data/bias2.txt", "r") as file:
    bias2 = json.load(file)
'''
training_data = tr.train(input_faces, desired_training_outputs)
weights1 = training_data[0]
bias1 = training_data[1]
weights2 = training_data[2]
bias2 = training_data[3]'''
print("Type \"Help\" for a list of commands")
while True:
    command = input("> ")
    if command == "help":
        print("train, banana")
    elif command == "train":
        tr.train(input_faces, desired_training_outputs)
    elif command == "banana":
        path = input("Enter image filename: ")
        input_image = ita.test(path)
        prediction = np.round(dingus(input_image, weights1, bias1, weights2, bias2), decimals=3)
        print(prediction)
        if prediction[0][1] >= .5:
            print("Mid")
        elif prediction[0][0] >= .5:
            print("Good")
        elif prediction[0][2] >= .5:
            print("Bad")
    else:
        print("Invalid command")
        
    
    
