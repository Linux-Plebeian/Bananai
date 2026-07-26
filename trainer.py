import numpy as np
#const
image_lw = 256
hidden_n = 32
n_outputs = 2
#activation
def relu(x):
    return np.maximum(0, x)
def relu_derivative(x):
    return (x > 0).astype(float)
def softmax(x): #exaggerates differences by exponentiation
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True) #prevents overflow

def train(training_inputs, training_outputs):
    #init params
    np.random.seed(1)
    weights1 = 2 * np.random.random((image_lw, hidden_n)) - 1
    bias1 = 0
    weights2 = 2 * np.random.random((hidden_n, n_outputs)) - 1
    bias2 = 0
    #print("Initial Weights:\n", weights)
    learning_rate = 0.001
    for train_loop in range(10000):
        #forward
        input_layer = training_inputs
        l1 = np.dot(input_layer, weights1) + bias1
        l1_outputs = relu(l1)
        l2 = np.dot(l1_outputs, weights2) + bias2
        l2_outputs = softmax(l2)
        #backprop (output)
        grad_l2 = l2_outputs - training_outputs          # softmax+cross-entropy shortcut
        grad_weights2 = np.dot(l1_outputs.T, grad_l2)
        grad_bias2 = np.sum(grad_l2, axis=0, keepdims=True)

        #backprop (hidden layer)
        grad_l1_outputs = np.dot(grad_l2, weights2.T)
        grad_l1 = grad_l1_outputs * relu_derivative(l1)
        grad_weights1 = np.dot(input_layer.T, grad_l1)
        grad_bias1 = np.sum(grad_l1, axis=0, keepdims=True)

        #update params
        weights2 -= learning_rate * grad_weights2
        bias2 -= learning_rate * grad_bias2
        weights1 -= learning_rate * grad_weights1
        bias1 -= learning_rate * grad_bias1
        if train_loop % 1000 == 1:
            print(np.round(l2_outputs, decimals=2))
            loss = -np.mean(np.sum(np.array(training_outputs) * np.log(l2_outputs + 1e-8), axis=1))
            print(f"Loop {train_loop}, loss: {loss:.4f}")
    #print("\nWeights After Training:\n", weights)
    #print("\nOutputs After Training:\n", np.round(outputs,1))
    return weights1, bias1, weights2, bias2

    
