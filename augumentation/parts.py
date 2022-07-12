
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy
from scipy import ndimage
from sklearn.decomposition import PCA
import matplotlib.image as mpimg


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def mnist_number(number):
    if number == 0:
        return 1
    elif number == 1:
        return 3
    elif number == 2:
        return 5
    elif number == 3:
        return 7
    elif number == 4:
        return 2
    elif number == 5:
        return 0
    elif number == 6:
        return 18
    elif number == 7:
        return 15
    elif number == 8:
        return 17
    elif number == 9:
        return 4


def print_pca(value, output_flatsize, value_size, dataset_number, augumentation, end, output_dimention_num):
    value = value.reshape(value_size, output_flatsize)
    pca = PCA(n_components=2)
    pca.fit(value)
    print(pca.explained_variance_ratio_)
    value = pca.transform(value)

    x = np.linspace(0, value_size, value_size)
    plt.scatter(
        value[:, 0],
        value[:, 1],
        c=x,
        alpha=0.3,
        cmap="viridis",
    )
    plt.colorbar()
    plt.title(augumentation +
              f' Number {dataset_number} dimention {output_dimention_num} of layer {end}')
    plt.xlabel('0D')
    plt.ylabel('1D')
    plt.show()


def print_pca_3d(value, output_flatsize, value_size, dataset_number, augumentation, end, output_dimention_num):
    value = value.reshape(value_size, output_flatsize)
    pca = PCA(n_components=3)
    pca.fit(value)
    print(pca.explained_variance_ratio_)
    value = pca.transform(value)

    x = np.linspace(0, value_size, value_size)
    ax = plt.axes(projection='3d')
    ax.scatter(
        value[:, 0],
        value[:, 1],
        value[:, 2],
        c=x,
        alpha=0.3,
        cmap="viridis",
    )
    plt.title(augumentation +
              f' Number {dataset_number} dimention {output_dimention_num} of layer {end}')
    ax.set_xlabel('0D')
    ax.set_ylabel('1D')
    ax.set_zlabel('2D')
    plt.show()


def dimention_of_layer(layer):
    if layer in (0, 1, 2):
        return (24, 24)
    elif layer == 3:
        return (12, 12)
    elif layer in (4, 5, 6):
        return (8, 8)
    elif layer == 7:
        return (4, 4)
    elif layer == 8:
        return (256,)
    elif layer in (9, 10):
        return (120,)
    elif layer in (11, 12):
        return (84,)


def number_of_layer(layer):
    if layer in (0, 1, 2, 3):
        return 6
    elif layer in (4, 5, 6, 7):
        return 16
    elif layer in (8, 9, 10, 11, 12):
        return -1
