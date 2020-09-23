## TODO: define the convolutional neural network architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

'''
---------------------------------------------------------------
Layer Number	  Layer Name     	    原Layer Shape    |  现Layer Shape
	1 			Input1 					(1, 96, 96)        | ( 1, 224, 224)
	2  (4,4)卷积Convolution2d1 			(32, 93, 93)      | ( 32, 221, 221)  224-4+1=221
	3 			Activation1 			(32, 93, 93)       | ( 32, 221, 221)
	4			Maxpooling2d1 池化层		(32, 46, 46)     | ( 32, 110, 110)  221/2=110.5
	5 	  p=0.1	Dropout1 				(32, 46, 46)       | ( 32, 110, 110)
	6  (3,3)卷积Convolution2d2 			(64, 44, 44)      | ( 64, 108, 108)
	7 			Activation2 			(64, 44, 44)       | ( 64, 108, 108)
	8 			Maxpooling2d2 池化层		(64, 22, 22)     | ( 64, 54, 54)
	9 	  p=0.2	Dropout2 				(64, 22, 22)       | ( 64, 54, 54)
	10 (2,2)卷积Convolution2d3 			(128, 21, 21)     | ( 128, 53, 53)
	11 			Activation3 			(128, 21, 21)      |  ( 128, 53, 53)
	12 			Maxpooling2d3 池化层		(128, 10, 10)    |  ( 128, 26, 26)
	13	  p=0.3	Dropout3 				(128, 10, 10)      |  ( 128, 26, 26)
	14 (1,1)卷积Convolution2d4 			(256, 10, 10)     | ( 256, 26, 26)
	15 			Activation4 			(256, 10, 10)      |  ( 256, 26, 26)
	16 			Maxpooling2d4 池化层		(256, 5, 5)   	  | ( 256, 13, 13)
	17	  p=0.4	Dropout4 				(256, 5, 5)   	   |  ( 256, 13, 13)
	18 			Flatten1 				(6400)			    |  (43264)
	19 			Dense1 					(1000)			    |  (1000)
	20 			Activation5 			(1000)			    |  (1000)
	21	  p=0.5	Dropout5 				(1000)			    |  (1000)
	22 			Dense2 					(1000)			    |  (600)
	23 			Activation6 线性激活函数	(1000)			   | (600)
	24	  p=0.6	Dropout6 				(1000)			    |  (600)
	25 			Dense3 					(2)			       | (136)
------------------------------------------------------------
'''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()       
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 4x4 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        self.bn2 = nn.BatchNorm2d(num_features=64, eps=1e-05)
        self.bn3 = nn.BatchNorm2d(num_features=128, eps=1e-05)
        self.bn4 = nn.BatchNorm2d(num_features=256, eps=1e-05)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
        
        self.fc1 = nn.Linear(in_features=43264, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=600)
        self.fc3 = nn.Linear(in_features=600, out_features=136)
               
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.dropout3(x)
        x = self.pool(self.bn4(F.relu(self.conv4(x))))
        x = self.dropout4(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
