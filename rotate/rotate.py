
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(31415926535)

transform = transforms.Compose(
    [transforms.ToTensor()]
)

batch_size = 4

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
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

    def forward(self, x,last_layer_flag = False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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

with torch.no_grad():
    for angle_in_degrees in range(0,360):
        image = trainset[0][0].reshape(28,28)
        # how to rotate image? ->complete
        image = ndimage.rotate(image, angle_in_degrees, reshape=False)
        image = torch.from_numpy(image)
        # print(angle_in_degrees)
        # imshow(torchvision.utils.make_grid(image))
        image = image.reshape(1,1,28,28).to(device)
        
        outputs = net(image,last_layer_flag = True)
        # how to get values of immedeate layer ? -> complete
        value_before_training[angle_in_degrees] = outputs
    pca = PCA(n_components=2)
    pca.fit(value_before_training)
    value_before_training = pca.transform(value_before_training)
    
    # value_before_training_numpy = value_before_training.numpy()
    rng = np.random.RandomState(0)
    x = np.linspace(0, 360, 360)
    colors = np.arange(360) / 180
    sizes = 1000 * rng.rand(100)
    print(value_before_training[:,0])
    print(value_before_training[:,1])
    plt.scatter(value_before_training[:,0], value_before_training[:,1],c = x, alpha=0.3,
                cmap='viridis')
    plt.colorbar();  # show color scale
    plt.legend(numpoints=1)
    plt.xlim(-0.1, 0.1);
    plt.ylim(-0.1, 0.1);
    plt.show()

for epoch in range(2):  # loop over the dataset multiple times

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
    for angle_in_degrees in range(0,360):
        image = trainset[0][0].reshape(28,28)
        # how to rotate image? ->complete
        image = ndimage.rotate(image, angle_in_degrees, reshape=False)
        image = torch.from_numpy(image)
        # print(angle_in_degrees)
        # imshow(torchvision.utils.make_grid(image))
        image = image.reshape(1,1,28,28).to(device)
        
        outputs = net(image,last_layer_flag = True)
        # how to get values of immedeate layer ? -> complete
        print("check")
        value_after_training[angle_in_degrees] = outputs.cpu()
        print("check")
        
    pca = PCA(n_components=2)
    pca.fit(value_after_training)
    value_after_training = pca.transform(value_after_training)

    # value_after_training_numpy = value_after_training.numpy()
    rng = np.random.RandomState(0)
    x = np.linspace(0, 360, 360)
    colors = np.arange(360) / 180
    sizes = 1000 * rng.rand(100)
    print(value_after_training[:,0])
    print(value_after_training[:,1])
    plt.scatter(value_after_training[:,0], value_after_training[:,1],c = x, alpha=0.3,
                cmap='viridis')
    plt.colorbar();  # show color scale
    plt.legend(numpoints=1)
    plt.xlim(-10, 10);
    plt.ylim(-10, 10);
    plt.show()
            

print("Finished Training")

PATH = "./mnist_net.pth"
torch.save(net.state_dict(), PATH)

