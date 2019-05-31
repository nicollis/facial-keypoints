import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net224(nn.Module):

    def __init__(self):
        super(Net224, self).__init__()
       
        # Output Tensor Size (32, 221, 221)
        # After Max Pool Tensor Size (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5, 1) # (224+1-5)/1 + 1 = 221
        
        # Output Tensor Size (64, 109, 109)
        # After Max Pool Tensor Size (128, 54, 54)
        self.conv2 = nn.Conv2d(32, 96, 3, 1) # (110+1-3)/1 + 1 = 109
        
        # Output Tensor Size (128, 50, 50)
        # After Max Pool Tensor Size (128, 25, 25)
#         self.conv3 = nn.Conv2d(64, 128, 5, 1) # (53+1-5)/1 + 1 = 50
        
        self.pool = nn.MaxPool2d(2,2)
        
        # Dropout layer will be used to prevent overfitting in the network. Rate increase with each level
        # Starting with droping with a rate of 0.1 we incrase with n*0.1 with each level.
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        
        # Defining the Dense or Fully Connected layers of the CNN.
        # Final layer outputs 136 values, 2 for each of the 68 keypoint (x,y) pairs
        self.fc1 = nn.Linear(96*54*54, 2000)
        self.fc2 = nn.Linear(2000, 136)
        
        # Initalize Dense Layers using Glorot Uniform Initalization
        for x in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(x.weight)
            x.bias.data.fill_(0.01)
        
    def forward(self, x):
        # Convolution layer => ELU activation => Max Pooling => Dropout
        x = self.drop3(self.pool(F.elu(self.conv1(x))))
        x = self.drop3(self.pool(F.elu(self.conv2(x))))
#         x = self.drop4(self.pool(F.elu(self.conv3(x))))
        
        # Flatten for Dense Layers
        x = x.view(x.size(0), -1)
        
        # Fully connected Dense Layer => ELU Activation => Dropout
        x = self.drop5(F.elu(self.fc1(x)))
        
        # Final Dense Layer, No activation
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
class Net224m(nn.Module):

    def __init__(self):
        super(Net224m, self).__init__()
       
        # Output Tensor Size (32, 220, 220)
        # After Max Pool Tensor Size (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 64, 3) # (226-5)/1 + 1 = 222
        
        self.pool = nn.MaxPool2d(2,2)
        
        # Dropout layer will be used to prevent overfitting in the network. Rate increase with each level
        # Starting with droping with a rate of 0.1 we incrase with n*0.1 with each level.
        self.drop1 = nn.Dropout(p=0.5)
        
        # Defining the Dense or Fully Connected layers of the CNN.
        # Final layer outputs 136 values, 2 for each of the 68 keypoint (x,y) pairs
        self.fc1 = nn.Linear(788544, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 136)
        
        # Initalize Dense Layers using Glorot Uniform Initalization
        for x in [self.fc1, self.conv1]:
            nn.init.xavier_uniform_(x.weight)
            x.bias.data.fill_(0.01)
        
    def forward(self, x):
        # Convolution layer => ELU activation => Max Pooling => Dropout
        x = self.drop1(self.pool(F.elu(self.conv1(x))))
        
        # Flatten for Dense Layers
        x = x.view(x.size(0), -1)
        
        # Fully connected Dense Layer => ELU Activation => Dropout
        x = self.drop1(F.elu(self.fc1(x)))
        x = self.drop1(F.elu(self.fc2(x)))
        
        # Final Dense Layer, No activation
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
class Net96(nn.Module):

    def __init__(self):
        super(Net96, self).__init__()
        
        # Defining 4 types of convoluitonal layers. Example Below
        # 1 input image channel (grayscale), 32 output channels/feature maps, 4x4 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        # Defining the type of pooling layers to use in the NN. This will be resued after each Conv Layer
        # Will fine the max value in each kernel. Each kernel is 2x2 with a non-overlapping stride of 2
        self.pool = nn.MaxPool2d(2,2)
        
        # Dropout layer will be used to prevent overfitting in the network. Rate increase with each level
        # Starting with droping with a rate of 0.1 we incrase with n*0.1 with each level.
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.6)
        
        # Defining the Dense or Fully Connected layers of the CNN.
        # Final layer outputs 136 values, 2 for each of the 68 keypoint (x,y) pairs
        self.fc1 = nn.Linear(6400, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
        # Initalize weights
        
        # Initalize Convolutional Layers with random numbers drawn from uniform distribution
        for x in [self.conv1, self.conv2, self.conv3, self.conv4]:
            nn.init.xavier_uniform_(x.weight)
            x.bias.data.fill_(0.01)
        
        # Initalize Dense Layers using Glorot Uniform Initalization
        for x in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(x.weight)
            x.bias.data.fill_(0.01)
        
    def forward(self, x):
        # Convolution layer => ELU activation => Max Pooling => Dropout
        x = self.drop1(self.pool(F.elu(self.conv1(x))))
        x = self.drop2(self.pool(F.elu(self.conv2(x))))
        x = self.drop3(self.pool(F.elu(self.conv3(x))))
        x = self.drop4(self.pool(F.elu(self.conv4(x))))
        
        # Flatten for Dense Layers
        x = x.view(x.size(0), -1)
        
        # Fully connected Dense Layer => ELU Activation => Dropout
        x = self.drop5(F.elu(self.fc1(x)))
        x = self.drop6(F.elu(self.fc2(x)))
        
        # Final Dense Layer, No activation
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    