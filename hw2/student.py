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

    # Transforms should do things like cropping, padding, normalizing etc.
    if mode == 'train':
        return transforms.ToTensor()
    elif mode == 'test':
        return transforms.ToTensor()


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):
    def __init__(self):
        # from assignment 1
        super(Network, self).__init__()
        self.conv1=nn.Conv2d(in_channels = 3, out_channels = 64,kernel_size = 5, padding=2) 
        self.max_pool=nn.MaxPool2d(2,2) 
        self.conv2=nn.Conv2d(in_channels =64,out_channels = 24,kernel_size = 5, padding=2) 
        self.fc_layer_1=nn.Linear(9600,230)
        self.fc_layer_2=nn.Linear(230,8)
        
    def forward(self, input):
        input = F.relu(self.conv1(input))
        input = self.max_pool(input)
        input = F.relu(self.conv2(input))
        input = self.max_pool(input)
        input = input.view(input.size(0), -1)  #flattening the inputs. 
        input = F.relu(self.fc_layer_1(input))
        input = self.fc_layer_2(input)
        input = F.log_softmax(input, dim=1)  
        return input 
  

    # def forward(self, input):
    #     input = input[-2]
    #     print(input.shape)
    #     self.hid1 = torch.tanh(self.lay1(input))
    #     layer2Input = torch.cat([input, self.hid1], dim = 1)
    #     self.hid2 = torch.tanh(self.lay2(layer2Input))
    #     outInput = torch.cat([self.hid2, self.hid1, input], dim = 1)
    #     return torch.sigmoid(self.lay3(outInput))

net = Network()
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)

def loss_def(input, target):
    return F.nll_loss(input, target, reduction='sum')

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
epochs = 10
