from __future__ import division
import random, os, math
from sklearn.decomposition import PCA
from sklearn import metrics
import numpy as np
from model import NN_two_layer, NN_three_layer
from util import normalize_preliminary, normalize_dataset, load_dataset, decision_boundary

CLASS_NUM = 3
FILE_NUM = 1000
TOTAL_FILE = CLASS_NUM * FILE_NUM
SPLIT_RATIO = 0.2

train_iteration = 3000
batch_size = 32

np.random.seed(1) # for random consistent

print("Loading dataset...")
dataset = load_dataset()

print("Shuffling dataset...")
dataset = np.random.permutation(dataset)

print("Spliting dataset into {:d} training, {:d} testing...".format( \
        int(TOTAL_FILE*(1-SPLIT_RATIO)), int(TOTAL_FILE*(SPLIT_RATIO))))
training = dataset[:-int(TOTAL_FILE*SPLIT_RATIO)]
testing = dataset[-int(TOTAL_FILE*SPLIT_RATIO):]

X_train, y_train = training[:, 0], training[:, 1]
X_test, y_test = testing[:, 0], testing[:, 1]
X_train = np.array([np.array(feature) for feature in X_train])
X_test = np.array([np.array(feature) for feature in X_test])
y_train = np.array([np.array(label) for label in y_train])
y_test = np.array([np.array(label) for label in y_test])

print("X_train shape:", X_train.shape,"X_test shape", X_test.shape)
print("y_train shape:", y_train.shape,"y_test shape", y_test.shape)

print("Normalizing dataset...")
x_max, x_min = normalize_preliminary(X_train)
X_train = normalize_dataset(X_train, x_max, x_min, 1.0)
X_test = normalize_dataset(X_test, x_max, x_min, 1.0)

print("Dimension reducing by PCA...")
PCA_model = PCA(n_components=2)
X_train_PCA = PCA_model.fit_transform(X_train)
X_test_PCA = PCA_model.transform(X_test)

X = X_train_PCA.copy() # for decision boundary

print("Concatenating bias...")
X_train_PCA = np.concatenate((X_train_PCA, np.ones((X_train_PCA.shape[0], 1))), axis=1)
X_test_PCA = np.concatenate((X_test_PCA, np.ones((X_test_PCA.shape[0], 1))), axis=1)
print("X_train_PCA shape:", X_train_PCA.shape,"X_test_PCA shape", X_test_PCA.shape)


print("Begin training...")
ceiling = int(math.ceil(X_train_PCA.shape[0] / batch_size))

# Uncommand if using Sigmoid(Part A)
# model = NN_two_layer(inputSize=3, h1_Size=5, outputSize=3)
# model_name = "Neural Network (2-layer)"

# Uncommand if using Relu(Part B)
# model = NN_three_layer(inputSize=3, h1_Size=5, h2_Size=5, outputSize=3)
# model_name = "Neural Network (3-layer)"

# Uncommand if using Relu(Part C)
model = NN_three_layer(inputSize=3, h1_Size=5, h2_Size=5, outputSize=3, activation='relu')
model_name = "Neural Network (3-layer)"


for i in range(train_iteration):
    batch_begin = (i % ceiling) * batch_size
    batch_end = ((i+1) % ceiling) * batch_size
    # meet end of each epoch
    if ((i+1) % ceiling) == 0 and i != 0:
        x_batch, y_batch = X_train_PCA[batch_begin:], y_train[batch_begin:]
    else:
        x_batch, y_batch = X_train_PCA[batch_begin: batch_end], \
                           y_train[batch_begin: batch_end]
    model.train(x_batch, y_batch)

    if i % 250 == 0:
        predict = model.forward(x_batch)
        pred = np.argmax(predict, axis=1)
        accuracy = metrics.accuracy_score(y_batch, pred)
        loss = model.cross_entropy(predict, y_batch)
        print("[iter {:4d}] loss:{:3.3f} accuracy:{:3.2f}".format(i, loss, accuracy))

print("Evaluation with testing data...")
predict = model.forward(X_test_PCA)
pred = np.argmax(predict, axis=1)
accuracy = metrics.accuracy_score(y_test, pred)
print("testing accuracy:{:3.2f}".format(accuracy))

print("Decision boundary...")
decision_boundary(model=model, X=X, model_name=model_name)