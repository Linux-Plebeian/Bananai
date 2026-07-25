import numpy as np
# Constants
image_lw = 16

# Activation Function
def sigmoid(x):
    r = 1 / (1 + np.exp(-x))
    #print (r)
    return r

def sigmoid_derivative(x):
    return x * (1 - x)

def train(training_inputs, training_outputs):
    # Initialize Parameters
    np.random.seed(1)
    weights = 2 * np.random.random((image_lw, 1)) - 1
    bias = 0
    #print("Initial Weights:\n", weights)
    learning_rate = 1
    for train_loop in range(1000):
        # Forward Pass
        input_layer = training_inputs
        outputs = sigmoid(np.dot(input_layer, weights) + bias)
        error = training_outputs - outputs
        # Backpropagation
        adjustments = error * sigmoid_derivative(outputs)
        # Update Parameters
        weights += np.dot(input_layer.T, adjustments) * learning_rate
        bias += np.sum(adjustments) * learning_rate
        if train_loop % 1000 == 1:
            print(np.round(outputs, decimals=2))
    #print("\nWeights After Training:\n", weights)
    #print("\nOutputs After Training:\n", np.round(outputs,1))
    return weights, bias, outputs

    
