import numpy as np
import json
#const
image_lw = 784
hidden_n = 1024
hidden2_n = 784
hidden3_n = 196
hidden4_n = 32
n_outputs = 3
learning_rate = 0.00001
#activation
def relu(x):
    return np.maximum(0, x)
def relu_derivative(x):
    return (x > 0).astype(float)
def softmax(x): #exaggerates differences by exponentiation
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True) #prevents overflow (?)
    
