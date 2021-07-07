
#%%

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
from image_rotate import image_rotate, image_rotate_dim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(31415926535)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.RandomRotation(degrees=(0, 360))]
)

transform_withoutrotate = transforms.Compose([transforms.ToTensor()])

batch_size = 4

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

trainset_withoutrotate = torchvision.datasets.CIFARtrainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform_withoutrotate
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=0
)

testset = torchvision.datasets.CIFARtrainset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=0
)

# def initialize_weights(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.constant_(m.weight.data, 0.01)
#         if m.bias is not None:
#             nn.init.constant_(m.bias.data, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight.data, 0.01)
#         nn.init.constant_(m.bias.data, 0)
#     elif isinstance(m, nn.Linear):
#         nn.init.constant_(m.weight.data, 0.01)
#         nn.init.constant_(m.bias.data, 0)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(
        self,
        x,
        first_layer_flag=False,
        second_layer_flag=False,
        third_layer_flag=False,
        fourth_layer_flag=False,
        fifth_layer_flag=False,
        last_layer_flag=False,
    ):
        x = self.conv1(x)
        if first_layer_flag:
            return x
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        if second_layer_flag:
            return x
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        if third_layer_flag:
            return x
        x = self.fc1(x)
        if fourth_layer_flag:
            return x
        x = F.relu(x)
        if fifth_layer_flag:
            return x
        x = self.fc2(x)
        x = F.relu(x)
        if last_layer_flag:
            return x
        x = self.fc3(x)
        return x


net = Net()
net.to(device)
# net.apply(initialize_weights)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


with torch.no_grad():

    for i in range(0, 6):
        image_rotate_dim(
            net, trainset_withoutrotate, 2, (360, 24, 24), 576, i, first_layer_flag=True
        )

    for i in range(0, 16):
        image_rotate_dim(
            net, trainset_withoutrotate, 2, (360, 8, 8), 64, i, second_layer_flag=True
        )

    image_rotate(net, trainset_withoutrotate, 2, (360, 256), 256, third_layer_flag=True)

    image_rotate(
        net, trainset_withoutrotate, 2, (360, 120), 120, fourth_layer_flag=True
    )

    image_rotate(
        net, trainset_withoutrotate, 2, (360, 120), 120, fourth_layer_flag=True
    )

    image_rotate(net, trainset_withoutrotate, 2, (360, 84), 84, last_layer_flag=True)
# want to code : image_number(number) -> image of number

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    # calculate outputs by running images through the network
                    outputs = net(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(
                "Accuracy of the network on the 10000 test images: %f %%"
                % (100 * correct / total)
            )


with torch.no_grad():

    for i in range(0, 6):
        image_rotate_dim(
            net, trainset_withoutrotate, 2, (360, 24, 24), 576, i, first_layer_flag=True
        )

    for i in range(0, 16):
        image_rotate_dim(
            net, trainset_withoutrotate, 2, (360, 8, 8), 64, i, second_layer_flag=True
        )

    image_rotate(net, trainset_withoutrotate, 2, (360, 256), 256, third_layer_flag=True)

    image_rotate(
        net, trainset_withoutrotate, 2, (360, 120), 120, fourth_layer_flag=True
    )

    image_rotate(
        net, trainset_withoutrotate, 2, (360, 120), 120, fourth_layer_flag=True
    )

    image_rotate(net, trainset_withoutrotate, 2, (360, 84), 84, last_layer_flag=True)


print("Finished Training")

PATH = "./CIFARtrainset = torchvision.datasets.MNIST_net.pth"
torch.save(net.state_dict(), PATH)

# %%