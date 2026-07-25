import numpy as np
import imgtoarray as ita
import trainer as tr

def dingus(input_layer, weights, bias):
    return tr.sigmoid(np.dot(input_layer, weights) + bias)

categories = 2
elements = 9
input_faces = ita.read_training_data(categories,elements)
list_desired_outputs = []
for i in range(0,categories*elements):
    if i<elements:
        list_desired_outputs.append(1)
    else:
        list_desired_outputs.append(0)
desired_outputs = np.array([list_desired_outputs]).T


training_data = tr.train(input_faces, desired_outputs)
weights = training_data[0]
bias = training_data[1]
outputs = training_data[2]
while True:
    path = input("Enter image filename: ")
    input_image = ita.test(path)
    print(input_image)
    if dingus(input_image, weights, bias)  > .5:
        print(":]")
    else:
        print(":[")
    