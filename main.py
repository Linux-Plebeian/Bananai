import numpy as np
import imgtoarray as ita
import trainer as tr

def dingus(input_layer, weights1, bias1, weights2, bias2):
    l1 = np.dot(input_layer, weights1) + bias1
    l1_outputs = tr.relu(l1)
    l2 = np.dot(l1_outputs, weights2) + bias2
    l2_outputs = tr.softmax(l2)
    return l2_outputs

categories = 2
elements = 10
input_faces = ita.read_training_data(categories,elements)
list_desired_outputs = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1], #bad
                        [1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],] #good

#list_desired_outputs = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
#desired_outputs = np.array([list_desired_outputs]).T


training_data = tr.train(input_faces, list_desired_outputs)
weights1 = training_data[0]
bias1 = training_data[1]
weights2 = training_data[2]
bias2 = training_data[3]
while True:
    path = input("Enter image filename: ")
    input_image = ita.test(path)
    prediction = np.round(dingus(input_image, weights1, bias1, weights2, bias2), decimals=3)
    print(prediction)
    
    