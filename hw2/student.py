#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import tensorflow.keras as keras

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """

    # Transforms should do things like cropping, padding, normalizing etc.\

    # give ourselves more data
    
    if mode == 'train':
        # This happens at batch level, so it should increase the amount of data over multiple epochs
        # maybe should use randomApply    
        return transforms.Compose([
            transforms.RandomChoice([ # chooses ONE of the below transformations (1-1/(transforms+1))% of the time. Should give an equal amount of transforms and normal data 
                transforms.RandomHorizontalFlip(p = 1), # sometimes flips horizontally
                transforms.RandomVerticalFlip(p = 1), # sometimes flips vertically
                transforms.ColorJitter(brightness = 0.5), # changing the brightness a little bit
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)), # A bit of blur
                # transforms.Resize(80),
                transforms.RandomRotation(degrees=(0, 360)), # randoms rotates some images
                transforms.RandomAdjustSharpness(sharpness_factor=2), # randomly adjusts sharpness of image,
                transforms.RandomAutocontrast(), 
                # transforms.RandomResizedCrop(size = (60,60)), # not sure if this is the right size !!!!!!
                transforms.ToTensor()
            ], p = 0.88)
        ])
    elif mode == 'test':
        return transforms.ToTensor()


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):
    def __init__(self):
        # from assignment 1
        super(Network, self).__init__()
        self.conv1=nn.Conv2d(in_channels = 3, out_channels = 96,kernel_size = 11, stride = 4) 
        self.batchNorm1 = nn.BatchNorm2d(96)
        self.max_pool=nn.MaxPool2d(kernel_size = (3,3), stride = (2,2)) 
        self.conv2=nn.Conv2d(in_channels =96, out_channels = 256, kernel_size = 5, stride = (1,1), padding = 5) 
        self.batchNorm2 = nn.BatchNorm2d(256)
        self.conv3=nn.Conv2d(in_channels =256, out_channels = 384, kernel_size = (3,3), stride = (1,1), padding = 5) 
        self.batchNorm3 = nn.BatchNorm2d(384)
        self.drop = nn.Dropout(0.5)
        self.fc_layer_1=nn.Linear(75264,4096)
        self.fc_layer_2=nn.Linear(4096,8)

        
    def forward(self, input):
        input = F.relu(self.conv1(input))
        batchNorm = self.batchNorm1
        input = batchNorm(input)
        input = self.max_pool(input)
        input = F.relu(self.conv2(input))
        batchNorm = self.batchNorm2
        input = batchNorm(input)
        input = self.max_pool(input)
        input = self.conv3(input)
        batchNorm = self.batchNorm3
        input = batchNorm(input)
        input = input.view(input.size(0), -1)  #flattening the inputs. 
        input = self.drop(input)
        input = F.relu(self.fc_layer_1(input))
        input = self.drop(input)
        input = self.fc_layer_2(input)
        input = F.log_softmax(input, dim=1)  
        return input 
  

net = Network()
# Need to convert to pytorch check this link for how to do it
# https://discuss.pytorch.org/t/pytorch-equivalent-of-keras/29412
# keras.models.Sequential([
#     keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(4096, activation='relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(4096, activation='relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(10, activation='softmax')
# ])
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

def loss_def(input, target):
    return F.cross_entropy(input, target)

loss_func = loss_def


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    return

scheduler = None

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data/data"
train_val_split = 0.8
batch_size = 200
epochs = 100
