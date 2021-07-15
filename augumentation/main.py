# %%
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
from augumentation import image_augumentation
from parts import imshow, number_of_layer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(31415926535)

transform_withoutrotate = transforms.Compose([transforms.ToTensor()])

batch_size = 4

dataset = "mnist"

if dataset == "mnist":

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(-20, 20)),
            transforms.RandomCrop(size=(28, 28), padding=(4, 4)),
        ]
    )

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0
    )

elif dataset == "cifar10":

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(-20, 20)),
            transforms.RandomCrop(size=(32, 32), padding=(4, 4)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0
    )


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight.data, 0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 0.001)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.constant_(m.weight.data, 0.001)
        nn.init.constant_(m.bias.data, 0)

# add to batch normalization


class Net(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        if dataset == "cifar10":
            first_dim = 3
            first_padding = 9
        elif dataset == "mnist":
            first_dim = 1
            first_padding = 5
        self.conv1 = nn.Conv2d(first_dim, 6, first_padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, end=None):
        layer = 0
        x = self.conv1(x)
        if end == layer:
            return x
        layer += 1
        x = F.relu(x)
        if end == layer:
            return x
        layer += 1
        x = self.bn1(x)
        if end == layer:
            return x
        layer += 1
        x = self.pool(x)
        if end == layer:
            return x
        layer += 1
        x = self.conv2(x)
        if end == layer:
            return x
        layer += 1
        x = F.relu(x)
        if end == layer:
            return x
        layer += 1
        x = self.bn2(x)
        if end == layer:
            return x
        layer += 1
        x = self.pool(x)
        if end == layer:
            return x
        layer += 1
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        if end == layer:
            return x
        layer += 1
        x = self.fc1(x)
        if end == layer:
            return x
        layer += 1
        x = F.relu(x)
        if end == layer:
            return x
        layer += 1
        x = self.fc2(x)
        if end == layer:
            return x
        layer += 1
        x = F.relu(x)
        if end == layer:
            return x
        layer += 1
        x = self.fc3(x)
        return x


class FC_Net(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.fc1 = nn.Linear(784, 252)
        self.fc2 = nn.Linear(252, 10)

    def forward(self, x, end=None):
        layer = 0
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        if end == layer:
            return x
        layer += 1
        x = self.fc1(x)
        if end == layer:
            return x
        layer += 1
        x = F.relu(x)
        if end == layer:
            return x
        x = self.fc2(x)
        return x


net = FC_Net(dataset)
net.to(device)
# net.apply(initialize_weights)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


with torch.no_grad():

    # for dim in range(0, 13):
    #     number = number_of_layer(dim)
    #     if number == -1:
    #         image_augumentation(net, 4, dim, number, "rotate", dataset)
    #     else:
    #         for i in range(0, number):
    #             image_augumentation(net, 4, dim, i, "rotate", dataset)

    for i in range(0, 10):
        image_augumentation(net, i, 0, -1, "shift_x", dataset)
        image_augumentation(net, i, 1, -1, "shift_x", dataset)
        image_augumentation(net, i, 2, -1, "shift_x", dataset)

for epoch in range(3):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        # imshow(torchvision.utils.make_grid(inputs))

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print("[%d, %5d] loss: %.3f" %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(
                "Accuracy of the network on the 10000 test images: %f %%"
                % (100 * correct / total)
            )

with torch.no_grad():

    # for dim in range(0, 13):
    #     number = number_of_layer(dim)
    #     if number == -1:
    #         image_augumentation(net, 4, dim, number, "rotate", dataset)
    #     else:
    #         for i in range(0, number):
    #             image_augumentation(net, 4, dim, i, "rotate", dataset)

    for i in range(0, 10):
        image_augumentation(net, i, 0, -1, "shift_x", dataset)
        image_augumentation(net, i, 1, -1, "shift_x", dataset)
        image_augumentation(net, i, 2, -1, "shift_x", dataset)


print("Finished Training")

PATH = "./CIFARtrainset_" + dataset + \
    " + dataset = torchvision.datasets." + dataset + "_net.pth"
torch.save(net.state_dict(), PATH)

# %%
