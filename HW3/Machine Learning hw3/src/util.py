from __future__ import division
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_dataset():
    CLASS_NUM = 3
    FILE_NUM = 1000

    dataset = list()
    for itr_class in range(CLASS_NUM):
        file_dir = "./data/Data_Train/Class{:d}/".format(itr_class + 1)
    
        for idx in range(FILE_NUM):
            file_name = "faceTrain{:d}_{:d}.bmp".format(itr_class + 1, idx + 1)
            # load image from directory and transfer into 1-d vector (30, 30) -> (900)
            tmp_img = np.array(Image.open(file_dir + file_name))
            tmp_img = tmp_img.reshape(tmp_img.shape[0]*tmp_img.shape[1])
            label = itr_class
            dataset.append((tmp_img, label))

    return dataset

def normalize_preliminary(data):
    dimension = data.shape[1]
    x_max = data.max(axis=0)
    x_min = data.min(axis=0)
    return x_max, x_min

def normalize_dataset(data, x_max, x_min, scaling):
    normalised_data = (data - x_min) / (x_max - x_min) * scaling
    return normalised_data

def decision_boundary(model, X, model_name):
    # create a mesh to plot in
    h = 0.2
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    feature = np.c_[xx.ravel(), yy.ravel()]
    feature = np.concatenate((feature, np.ones((feature.shape[0], 1))), axis=1)
    Z = model.predict(feature)
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Set3)
    ax.axis('off')
    
    #Plot also the training points
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)  
    prediction = model.predict(X)
    for i in range(0, X.shape[0]):
        if(prediction[i] == 0):
            plt.scatter(X[i][0], X[i][1], c='r', label='0', s=3)
        elif(prediction[i] == 1):
            plt.scatter(X[i][0], X[i][1], c='g', label='1', s=3)
        else:
            plt.scatter(X[i][0], X[i][1], c='b', label='2', s=3)
    
    ax.set_title(model_name + " decision boundary")
    plt.show()

def one_hot(a):
    a = np.array(a) # ensure type is numpy array
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b