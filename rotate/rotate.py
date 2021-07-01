
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
from image_rotate import image_rotate,image_rotate_dim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(31415926535)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomRotation(degrees=(0, 360))]
)

transform_withoutrotate = transforms.Compose(
    [transforms.ToTensor()]
)

batch_size = 4

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

trainset_withoutrotate = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform_withoutrotate
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x,first_layer_flag = False,second_layer_flag = False,last_layer_flag = False):
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
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        if last_layer_flag:
            return x
        x = self.fc3(x)
        return x


net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
value_before_training = torch.empty(360, 84, dtype=torch.float)
value_after_training = torch.empty(360, 84, dtype=torch.float)
value_first_before_training = torch.empty(360, 24,24 ,dtype=torch.float)
value_first_after_training = torch.empty(360, 24,24, dtype=torch.float)

with torch.no_grad():
    image_rotate_dim(net,trainset_withoutrotate,2,(360,24,24),576,0,first_layer_flag = True)

with torch.no_grad():
    image_rotate_dim(net,trainset_withoutrotate,2,(360,24,24),576,1,first_layer_flag = True)

with torch.no_grad():
    image_rotate_dim(net,trainset_withoutrotate,2,(360,24,24),576,2,first_layer_flag = True)

with torch.no_grad():
    image_rotate_dim(net,trainset_withoutrotate,2,(360,24,24),576,3,first_layer_flag = True)
    
with torch.no_grad():
    image_rotate_dim(net,trainset_withoutrotate,2,(360,24,24),576,4,first_layer_flag = True)
    
with torch.no_grad():
    image_rotate_dim(net,trainset_withoutrotate,2,(360,24,24),576,5,first_layer_flag = True)

with torch.no_grad():
    image_rotate(net,trainset_withoutrotate,2,(360,84),84,last_layer_flag = True)
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

            print('Accuracy of the network on the 10000 test images: %f %%' % (
                100 * correct / total))
            
            
with torch.no_grad():
    image_rotate_dim(net,trainset_withoutrotate,2,(360,24,24),576,1,first_layer_flag = True)
    # want to code : image_number(number) -> image of number
    
    
with torch.no_grad():
    image_rotate(net,trainset_withoutrotate,2,(360,84),84,last_layer_flag = True)


print("Finished Training")

PATH = "./mnist_net.pth"
torch.save(net.state_dict(), PATH)

