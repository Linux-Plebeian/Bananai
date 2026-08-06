import numpy as np
import json

#activation
def relu(x):
    return np.maximum(0, x)
def relu_derivative(x):
    return (x > 0).astype(float)
def softmax(x): #exaggerates differences by exponentiation
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def forward_prop(input_layer, weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4, weights5, bias5):
    l1 = np.dot(input_layer, weights1) + bias1
    l1_outputs = relu(l1)
    l2 = np.dot(l1_outputs, weights2) + bias2
    l2_outputs = relu(l2)
    l3 = np.dot(l2_outputs, weights3) + bias3
    l3_outputs = relu(l3)
    l4 = np.dot(l3_outputs, weights4) + bias4
    l4_outputs = relu(l4)
    l5 = np.dot(l4_outputs, weights5) + bias5
    l5_outputs = softmax(l5)
    return l5_outputs

with open("training_data/weights1.txt", "r") as file:
    weights1 = json.load(file)
with open("training_data/bias1.txt", "r") as file:
    bias1 = json.load(file)
with open("training_data/weights2.txt", "r") as file:
    weights2 = json.load(file)
with open("training_data/bias2.txt", "r") as file:
    bias2 = json.load(file)
with open("training_data/weights3.txt", "r") as file:
    weights3 = json.load(file)
with open("training_data/bias3.txt", "r") as file:
    bias3 = json.load(file)
with open("training_data/weights4.txt", "r") as file:
    weights4 = json.load(file)
with open("training_data/bias4.txt", "r") as file:
    bias4 = json.load(file)
with open("training_data/weights5.txt", "r") as file:
    weights5 = json.load(file)
with open("training_data/bias5.txt", "r") as file:
    bias5 = json.load(file)