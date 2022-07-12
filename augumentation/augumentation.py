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
from parts import imshow, mnist_number, print_pca, dimention_of_layer, print_pca_3d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])

dataset = "cifar10"

if dataset == "mnist":
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
elif dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )


def image_augumentation(
    net,
    dataset_number,
    end,
    output_dimention_num,
    augumentation,
    dataset
):
    if dataset == "cifar10":
        datasize = 32
        data_dimention = 3
        axes = (2, 1)
    elif dataset == "mnist":
        datasize = 28
        data_dimention = 1
        axes = (1, 0)

    output_size = dimention_of_layer(end)

    if augumentation in ("shift_x", "shift_y"):
        value_size = datasize * 2 + 1
    elif augumentation == "rotate":
        value_size = 360
    elif augumentation == "shift_x_y":
        value_size = 3249
    elif augumentation in ("shuffle", "change_color"):
        value_size = 300

    output_flatsize = 1
    for i in output_size:
        output_flatsize *= i

    output_size = (value_size,) + output_size

    value = torch.empty(output_size, dtype=torch.float)

    for degrees in range(0, value_size):
        if dataset == "mnist":
            image = trainset[mnist_number(dataset_number)][0].reshape(
                datasize, datasize
            )
        elif dataset == "cifar10":
            image = trainset[mnist_number(dataset_number)][0].reshape(
                data_dimention, datasize, datasize
            )

        if augumentation == "shift_x":
            if dataset == "cifar10":
                image = ndimage.shift(image, (0, 0, (degrees - datasize)))
            elif dataset == "mnist":
                image = ndimage.shift(image, (0, (degrees - datasize)))
        elif augumentation == "shift_y":
            if dataset == "cifar10":
                image = ndimage.shift(image, (0, (degrees - datasize), 0))
            elif dataset == "mnist":
                image = ndimage.shift(image, ((degrees - datasize), 0))
        elif augumentation == "rotate":
            image = ndimage.rotate(
                image, degrees, axes=axes, mode="reflect", reshape=False)
        elif augumentation == "shift_x_y":
            if dataset == "cifar10":
                image = ndimage.shift(
                    image, (0, (degrees//57 - datasize), (degrees % 57 - datasize)))
            elif dataset == "mnist":
                image = ndimage.shift(
                    image, ((degrees//57 - datasize), (degrees % 57 - datasize)))
        elif augumentation == "shuffle":
            image_shuffle(image, 0, 0, 32, 32)
        elif augumentation == "change_color":
            image *= 0.01 * degrees

        if not augumentation in ("shuffle", "change_color"):
            image = torch.from_numpy(image)

        # imshow(torchvision.utils.make_grid(image))
        image = image.reshape(1, data_dimention, datasize, datasize).to(device)

        if output_dimention_num >= 0:
            outputs = net(image, end)
            value[degrees] = outputs[0][output_dimention_num].cpu()
        else:
            outputs = net(image, end)
            value[degrees] = outputs[0].cpu()

    # print_pca(value, output_flatsize, value_size,dataset_number,augumentation,end,output_dimention_num)

    # print_pca_3d(value, output_flatsize, value_size,dataset_number,augumentation,end,output_dimention_num)


def image_shuffle(image, sx, sy, dx, dy):
    # 座標ごとのshuffle,ある特定区間のshuffleも作ってみたい
    idx = torch.randperm(image.nelement())
    print(image.nelement())
    image = image.view(-1)[idx].view(image.size())
