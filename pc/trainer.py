import numpy as np
import json
from pathlib import Path
import neural_network as nn

#const
image_lw = 12288 #64x64 images with r, g, and b data
hidden_n = 512 #reduce to prevent inductive bias and enforce generalization
hidden2_n = 256
hidden3_n = 128
hidden4_n = 64
n_outputs = 3
learning_rate = 0.001




def dump_corrections(training_data_paths):
    desired_training_outputs = []
    file_ct  = len([f for f in Path(training_data_paths[0]).iterdir() if f.is_file()])
    for files in range(0,file_ct):
        print(files)
        desired_training_outputs.append([1,0,0]) 
    file_ct  = len([f for f in Path(training_data_paths[1]).iterdir() if f.is_file()])
    for files in range(0,file_ct):
        print(files)
        desired_training_outputs.append([0,1,0]) 
    file_ct  = len([f for f in Path(training_data_paths[2]).iterdir() if f.is_file()])
    for files in range(0,file_ct):
        print(files)
        desired_training_outputs.append([0,0,1]) 
    '''file_ct  = len([f for f in Path(training_data_paths[3]).iterdir() if f.is_file()])
    for files in range(0,file_ct):
        print(files)
        desired_training_outputs.append([0,0,0,1])''' 
    print(desired_training_outputs)
    with open("training_data/desired_training_outputs.txt", "w") as file:
        json.dump(desired_training_outputs, file)

def train(training_inputs, training_outputs):

    
    kerfuffle = np.random.permutation(training_inputs.shape[0]) #prevent AI from learning the order of data over the image data
    training_inputs = training_inputs[kerfuffle]
    training_outputs = training_outputs[kerfuffle]

    #training_inputs = training_inputs[:2000]
    #training_outputs = training_outputs[:2000]

    epochs=int(input("Epochs? "))
    print("training...")
    #init params
    np.random.seed(1)
    weights1 = np.random.randn(image_lw, hidden_n) * np.sqrt(2.0 / image_lw)
    bias1 = np.zeros((1, hidden_n))
    weights2 = np.random.randn(hidden_n, hidden2_n) * np.sqrt(2.0 / hidden_n)
    bias2 = np.zeros((1, hidden2_n))
    weights3 = np.random.randn(hidden2_n, hidden3_n) * np.sqrt(2.0 / hidden2_n)
    bias3 = np.zeros((1, hidden3_n))
    weights4 = np.random.randn(hidden3_n, hidden4_n) * np.sqrt(2.0 / hidden3_n)
    bias4 = np.zeros((1, hidden4_n))
    weights5 = np.random.randn(hidden4_n, n_outputs) * np.sqrt(2.0 / hidden4_n)
    bias5 = np.zeros((1, n_outputs))
    
    #print("Initial Weights:\n", weights)
    
    for epoch in range(epochs):
        kerfuffle = np.random.permutation(training_inputs.shape[0]) #prevent AI from learning the order of data over the image data
        training_inputs = training_inputs[kerfuffle]
        training_outputs = training_outputs[kerfuffle]
        #forward
        input_layer = training_inputs
        l1 = np.dot(input_layer, weights1) + bias1
        l1_outputs = nn.relu(l1)
        l2 = np.dot(l1_outputs, weights2) + bias2
        l2_outputs = nn.relu(l2)
        l3 = np.dot(l2_outputs, weights3) + bias3
        l3_outputs = nn.relu(l3)
        l4 = np.dot(l3_outputs, weights4) + bias4
        l4_outputs = nn.relu(l4)
        l5 = np.dot(l4_outputs, weights5) + bias5
        l5_outputs = nn.softmax(l5)


        #backprop (output)
        grad_l5 = (l5_outputs - training_outputs) #softmax+cross-entropy shortcut
        grad_weights5 = np.dot(l4_outputs.T, grad_l5)
        grad_bias5 = np.sum(grad_l5, axis=0, keepdims=True)
        
        #backprop (hl4 what the fuck is this shit)
        grad_l4_outputs = np.dot(grad_l5, weights5.T)
        grad_l4 = grad_l4_outputs * nn.relu_derivative(l4)
        grad_weights4 = np.dot(l3_outputs.T, grad_l4)
        grad_bias4 = np.sum(grad_l4, axis=0, keepdims=True)
        
        #backprop (hl3 send help)
        grad_l3_outputs = np.dot(grad_l4, weights4.T)
        grad_l3 = grad_l3_outputs * nn.relu_derivative(l3)
        grad_weights3 = np.dot(l2_outputs.T, grad_l3)
        grad_bias3 = np.sum(grad_l3, axis=0, keepdims=True)

        #backprop (hidden layer 2??????)
        grad_l2_outputs = np.dot(grad_l3, weights3.T)
        grad_l2 = grad_l2_outputs * nn.relu_derivative(l2)
        grad_weights2 = np.dot(l1_outputs.T, grad_l2)
        grad_bias2 = np.sum(grad_l2, axis=0, keepdims=True)
        
        #backprop (hidden layer 1)
        grad_l1_outputs = np.dot(grad_l2, weights2.T)
        grad_l1 = grad_l1_outputs * nn.relu_derivative(l1)
        grad_weights1 = np.dot(input_layer.T, grad_l1)
        grad_bias1 = np.sum(grad_l1, axis=0, keepdims=True)

        #fix chonk values
        batch_size = training_inputs.shape[0]
        grad_weights5 /= batch_size
        grad_bias5 /= batch_size
        grad_weights4 /= batch_size
        grad_bias4 /= batch_size
        grad_weights3 /= batch_size
        grad_bias3 /= batch_size
        grad_weights2 /= batch_size
        grad_bias2 /= batch_size
        grad_weights1 /= batch_size
        grad_bias1 /= batch_size

        #update params
        weights5 -= learning_rate * grad_weights5
        bias5 -= learning_rate * grad_bias5
        weights4 -= learning_rate * grad_weights4
        bias4 -= learning_rate * grad_bias4
        weights3 -= learning_rate * grad_weights3
        bias3 -= learning_rate * grad_bias3
        weights2 -= learning_rate * grad_weights2
        bias2 -= learning_rate * grad_bias2
        weights1 -= learning_rate * grad_weights1
        bias1 -= learning_rate * grad_bias1
        #if epoch % 100 == 1:
            #print(np.round(l2_outputs, decimals=2))
            #loss = -np.mean(np.sum(np.array(training_outputs) * np.log(l2_outputs + 1e-8), axis=1))
            #print(f"Loop {train_loop}, loss: {loss:.4f}")
        print("Epoch: ", epoch, "/", epochs)
        loss = -np.mean(np.sum(np.array(training_outputs) * np.log(l5_outputs + 1e-8), axis=1))
        if epoch % 10 == 1:
            print(
                np.mean(l1_outputs == 0),
                np.mean(l2_outputs == 0),
                np.mean(l3_outputs == 0),
                np.mean(l4_outputs == 0)
            )
            pred = np.argmax(l5_outputs, axis=1)
            actual = np.argmax(training_outputs, axis=1)
            print("Predicted:")
            print(np.bincount(pred))

            print("Actual:")
            print(np.bincount(actual))
        print(f"Loop {epoch}, loss: {loss:.4f}")
        if epoch == epochs-1:
            print("Epoch: ", epochs, "/", epochs)
            
    
    #store training data to avoid unnessecary gym sessions
    with open("training_data/l5_outputs.txt", "w") as file:
        json.dump(l5_outputs.tolist(), file)
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
    with open("training_data/weights4.txt", "w") as file:
        json.dump(weights4.tolist(), file)
    with open("training_data/bias4.txt", "w") as file:
        json.dump(bias4.tolist(), file)
    with open("training_data/weights5.txt", "w") as file:
        json.dump(weights5.tolist(), file)
    with open("training_data/bias5.txt", "w") as file:
        json.dump(bias5.tolist(), file)
         

    return weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4, weights5, bias5

    
