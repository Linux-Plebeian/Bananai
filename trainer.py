import numpy as np
import json
#const
image_lw = 784
hidden_n = 128
hidden2_n = 32
n_outputs = 3
learning_rate = 0.0001
#activation
def relu(x):
    return np.maximum(0, x)
def relu_derivative(x):
    return (x > 0).astype(float)
def softmax(x): #exaggerates differences by exponentiation
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True) #prevents overflow (?)

def train(training_inputs, training_outputs):
    epochs=int(input("Epochs? "))
    print("training...")
    #init params
    np.random.seed(1)
    weights1 = 2 * np.random.random((image_lw, hidden_n)) - 1
    bias1 = 0
    weights2 =  2 * np.random.random((hidden_n, hidden2_n)) - 1
    bias2 = 0
    weights3 =  2 * np.random.random((hidden2_n, n_outputs))
    bias3 = 0
    #print("Initial Weights:\n", weights)
    
    for epoch in range(epochs):
        #forward
        input_layer = training_inputs
        l1 = np.dot(input_layer, weights1) + bias1
        l1_outputs = relu(l1)
        l2 = np.dot(l1_outputs, weights2) + bias2
        l2_outputs = relu(l2)
        l3 = np.dot(l2_outputs, weights3) + bias3
        l3_outputs = softmax(l3)
        #backprop (output)
        grad_l3 = l3_outputs - training_outputs          # softmax+cross-entropy shortcut
        grad_weights3 = np.dot(l2_outputs.T, grad_l3)
        grad_bias3 = np.sum(grad_l3, axis=0, keepdims=True)

        #backprop (hidden layer 2??????)
        grad_l2_outputs = np.dot(grad_l3, weights3.T)
        grad_l2 = grad_l2_outputs * relu_derivative(l2)
        grad_weights2 = np.dot(l1.T, grad_l2)
        grad_bias2 = np.sum(grad_l2, axis=0, keepdims=True)
        
        #backprop (hidden layer 1)
        grad_l1_outputs = np.dot(grad_l2, weights2.T)
        grad_l1 = grad_l1_outputs * relu_derivative(l1)
        grad_weights1 = np.dot(input_layer.T, grad_l1)
        grad_bias1 = np.sum(grad_l1, axis=0, keepdims=True)

        #update params
        weights3 -= learning_rate * grad_weights3
        bias3 -= learning_rate * grad_bias3
        weights2 -= learning_rate * grad_weights2
        bias2 -= learning_rate * grad_bias2
        weights1 -= learning_rate * grad_weights1
        bias1 -= learning_rate * grad_bias1
        if epoch % 100 == 1:
            #print(np.round(l2_outputs, decimals=2))
            #loss = -np.mean(np.sum(np.array(training_outputs) * np.log(l2_outputs + 1e-8), axis=1))
            #print(f"Loop {train_loop}, loss: {loss:.4f}")
            print("Epoch: ", epoch, "/", epochs)
        if epoch == epochs-1:
            print("Epoch: ", epochs, "/", epochs)
    
    #store training data to avoid unnessecary gym sessions
    with open("training_data/l3_outputs.txt", "w") as file:
        json.dump(l3_outputs.tolist(), file)
    with open("training_data/weights1.txt", "w") as file:
        json.dump(weights1.tolist(), file)
    with open("training_data/bias1.txt", "w") as file:
        json.dump(bias1.tolist(), file)
    with open("training_data/weights2.txt", "w") as file:
        json.dump(weights2.tolist(), file)
    with open("training_data/bias2.txt", "w") as file:
        json.dump(bias2.tolist(), file)
    with open("training_data/weights3.txt", "w") as file:
        json.dump(weights3.tolist(), file)
    with open("training_data/bias3.txt", "w") as file:
        json.dump(bias3.tolist(), file)

    return weights1, bias1, weights2, bias2, weights3, bias3

    
