import numpy as np
import mnist
from sklearn.preprocessing import OneHotEncoder

# Load MNIST data
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten images
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

# One-hot encode labels
encoder = OneHotEncoder(sparse=False)
train_labels = encoder.fit_transform(train_labels.reshape(-1, 1))
test_labels = encoder.transform(test_labels.reshape(-1, 1))

# Neural Network parameters
input_size = train_images.shape[1]
hidden_size = 64
output_size = 10
learning_rate = 0.1
epochs = 10

# Initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.randn(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_output = np.zeros((1, output_size))

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(train_images, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    final_layer_output = sigmoid(final_layer_input)
    
    # Backward pass
    error = final_layer_output - train_labels
    d_output = error * sigmoid_derivative(final_layer_output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    weights_hidden_output -= hidden_layer_output.T.dot(d_output) * learning_rate
    bias_output -= np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden -= train_images.T.dot(d_hidden_layer) * learning_rate
    bias_hidden -= np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    
    # Print loss
    loss = np.mean(np.square(error))
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')

# Evaluate the model
def predict(images):
    hidden_layer_input = np.dot(images, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    final_layer_output = sigmoid(final_layer_input)
    return np.argmax(final_layer_output, axis=1)

predictions = predict(test_images)
accuracy = np.mean(predictions == np.argmax(test_labels, axis=1))
print(f'Accuracy: {accuracy:.2f}')

# Example usage
new_image = test_images[0]  # Test with the first test image
predicted_label = predict(new_image.reshape(1, -1))
print(f'Predicted Label: {predicted_label[0]}')
