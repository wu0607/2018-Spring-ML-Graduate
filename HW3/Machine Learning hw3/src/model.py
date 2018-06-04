from __future__ import division
import numpy as np

class NN_two_layer():
    def __init__(self, inputSize, h1_Size, outputSize):
        self.W1 = np.random.randn(inputSize, h1_Size) / np.sqrt(inputSize)
        self.W2 = np.random.randn(h1_Size, outputSize) / np.sqrt(h1_Size)

    def fully_connected(self, x, W):
        return np.dot(x, W)

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def sigmoid_Prime(self, x):
        return x * (1. - x)

    def softmax(self, x):
        exp_scores = np.exp(x)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def cross_entropy(self, y_hat, y):
        data_num = y.shape[0]
        log_likelihood = -np.log(y_hat[range(data_num),y])
        loss = np.sum(log_likelihood) / data_num
        return loss

    def cross_entropy_Prime(self, y_hat, y): # Actually y_hat is softmax
        data_num = y.shape[0]
        grad = y_hat 
        grad[range(data_num),y] -= 1
        return grad

    def forward(self, X):
        self.a1 = self.fully_connected(X, self.W1)
        self.z1 = self.sigmoid(self.a1)
        self.a2 = self.fully_connected(self.z1, self.W2)
        output = self.softmax(self.a2)
        return output

    def backward(self, X, y, o):
        learning_rate = 1e-3

        delta3 = self.cross_entropy_Prime(o, y)
        dW2 = (self.z1.T).dot(delta3)
        delta2 = delta3.dot(self.W2.T) * (self.sigmoid_Prime(self.z1))
        dW1 = np.dot(X.T, delta2)

        self.W1 += -learning_rate * dW1
        self.W2 += -learning_rate * dW2

    def train(self, X, y):
        output = self.forward(X)
        self.backward(X, y, output)

    def predict(self, X):
        predict = self.forward(X)
        pred = np.argmax(predict, axis=1)
        return pred

class NN_three_layer():
    def __init__(self, inputSize, h1_Size, h2_Size, outputSize, activation='sigmoid'):
        self.W1 = np.random.randn(inputSize, h1_Size) / np.sqrt(inputSize)
        self.W2 = np.random.randn(h1_Size, h2_Size) / np.sqrt(h1_Size)
        self.W3 = np.random.randn(h2_Size, outputSize) / np.sqrt(h2_Size)
        self.activation = activation

    def fully_connected(self, x, W):
        return np.dot(x, W)

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def sigmoid_Prime(self, x):
        return x * (1. - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_Prime(self, x):
        return 1. * (x > 0)

    def softmax(self, x):
        exp_scores = np.exp(x)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def cross_entropy(self, y_hat, y):
        data_num = y.shape[0]
        log_likelihood = -np.log(y_hat[range(data_num),y])
        loss = np.sum(log_likelihood) / data_num
        return loss

    def cross_entropy_Prime(self, y_hat, y): # Actually y_hat is softmax
        data_num = y.shape[0]
        grad = y_hat 
        grad[range(data_num),y] -= 1
        return grad

    def forward(self, X):
        self.a1 = self.fully_connected(X, self.W1)
        self.z1 = self.sigmoid(self.a1) if self.activation == 'sigmoid' else self.relu(self.a1)
        self.a2 = self.fully_connected(self.z1, self.W2)
        self.z2 = self.sigmoid(self.a2) if self.activation == 'sigmoid' else self.relu(self.a2)
        self.a3 = self.fully_connected(self.z2, self.W3)
        output = self.softmax(self.a3)
        return output

    def backward(self, X, y, o):
        learning_rate = 1e-3 if self.activation == 'sigmoid' else 1e-2
        
        delta3 = self.cross_entropy_Prime(o, y)
        dW3 = np.dot(self.z2.T, delta3)

        delta2 = delta3.dot(self.W3.T) * (self.sigmoid_Prime(self.z2)) if self.activation == 'sigmoid' \
                 else delta3.dot(self.W3.T) * (self.relu_Prime(self.a2))
        dW2 = np.dot(self.z1.T, delta2)
        
        delta1 = delta2.dot(self.W2.T) * (self.sigmoid_Prime(self.z1)) if self.activation == 'sigmoid' \
                 else delta2.dot(self.W2.T) * (self.relu_Prime(self.a1))
        dW1 = np.dot(X.T, delta1)

        self.W1 += -learning_rate * dW1
        self.W2 += -learning_rate * dW2
        self.W3 += -learning_rate * dW3

    def train(self, X, y):
        output = self.forward(X)
        self.backward(X, y, output)

    def predict(self, X):
        predict = self.forward(X)
        pred = np.argmax(predict, axis=1)
        return pred

# Cross Entropy reference: https://deepnotes.io/softmax-crossentropy
# backward reference: https://github.com/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb