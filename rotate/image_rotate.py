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

def image_rotate(
    net,
    dataset,
    dataset_number,
    output_size,
    output_flatsize,
    first_layer_flag = False,
    second_layer_flag = False,
    last_layer_flag = False
    ):
    value_first_before_training = torch.empty(output_size,dtype=torch.float)
    for angle_in_degrees in range(0,360):
        image = dataset[dataset_number][0].reshape(28,28)
        image = ndimage.rotate(image, angle_in_degrees, reshape=False)
        image = torch.from_numpy(image)
        # imshow(torchvision.utils.make_grid(image))
        image = image.reshape(1,1,28,28).to(device)
        
        outputs = net(image,first_layer_flag = first_layer_flag,second_layer_flag = second_layer_flag,last_layer_flag = last_layer_flag)
        value_first_before_training[angle_in_degrees] = outputs.cpu()
        # only applied fot first layer
        
    
    value_first_before_training = value_first_before_training.reshape(360,output_flatsize)
    pca = PCA(n_components=2)
    pca.fit(value_first_before_training)
    value_first_before_training = pca.transform(value_first_before_training)
    
    x = np.linspace(0, 360, 360)
    plt.scatter(value_first_before_training[:,0], value_first_before_training[:,1],c = x, alpha=0.3,
                cmap='viridis')
    plt.colorbar();  # show color scale
    plt.legend(numpoints=1)
    plt.show()
    
def image_rotate_dim(
    net,
    dataset,
    dataset_number,
    output_size,
    output_flatsize,
    output_dimention_num,
    first_layer_flag = False,
    second_layer_flag = False,
    last_layer_flag = False
    ):
    value_first_before_training = torch.empty(output_size,dtype=torch.float)
    for angle_in_degrees in range(0,360):
        image = dataset[dataset_number][0].reshape(28,28)
        image = ndimage.rotate(image, angle_in_degrees, reshape=False)
        image = torch.from_numpy(image)
        # imshow(torchvision.utils.make_grid(image))
        image = image.reshape(1,1,28,28).to(device)
        
        outputs = net(image,first_layer_flag = first_layer_flag,second_layer_flag = second_layer_flag,last_layer_flag = last_layer_flag)
        value_first_before_training[angle_in_degrees] = outputs[0][output_dimention_num].cpu()
        # only applied fot first layer
        
    
    value_first_before_training = value_first_before_training.reshape(360,output_flatsize)
    pca = PCA(n_components=2)
    pca.fit(value_first_before_training)
    value_first_before_training = pca.transform(value_first_before_training)
    
    x = np.linspace(0, 360, 360)
    plt.scatter(value_first_before_training[:,0], value_first_before_training[:,1],c = x, alpha=0.3,
                cmap='viridis')
    plt.colorbar();  # show color scale
    plt.legend(numpoints=1)
    plt.show()
