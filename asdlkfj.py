import numpy as np
# Activation functions
def relu(x):
    return np.maximum(0, x)
def softmax(x): #exaggerates differences by exponentiation
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True) #prevents overflow
# Hyperparameters
input_size = 100   # 10x10 pixels flattened
hidden_size = 32   # Number of hidden neurons
output_size = 5    # 5 classification outputs
# Initialize weights and biases randomly
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))
# Forward pass function
def forward_propagation(image_10x10):
    # Flatten 10x10 image to (1, 100)
    x = image_10x10.reshape(1, -1)
    
    # Hidden layer
    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)
    
    # Output layer (5 outputs)
    z2 = np.dot(a1, W2) + b2
    output_probs = softmax(z2)
    
    return output_probs
# Test with a random 10x10 image
sample_image = np.random.rand(10, 10)
probabilities = forward_propagation(sample_image)
print("Output probabilities for 5 classes:", probabilities)
print("Predicted class:", np.argmax(probabilities))