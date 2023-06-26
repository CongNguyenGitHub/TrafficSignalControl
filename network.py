import torch.nn as nn
import torch
class QNetwork(nn.Module):
    def __init__(self,num_actions):
        super(QNetwork,self).__init__()
        #Convolutional layers
        self.conv1=nn.Conv2d(3,16,kernel_size=8,stride=4)
        self.conv2=nn.Conv2d(16,32,kernel_size=4,stride=2)
        #Fully connected layers
        self.fc=nn.Linear(32*14*14,128)
        #Output layers
        self.value=nn.Linear(128,num_actions)
    def forward(self,x):
        x=nn.functional.relu(self.conv1(x))
        x=nn.functional.relu(self.conv2(x))
        x=x.reshape(x.size(0),-1)
        x=nn.functional.relu(self.fc(x))
        value=self.value(x)
        return value
class PGNetwork(nn.Module):
    def  __init__(self,num_actions):
        super(PGNetwork,self).__init__()
        #Convolutional layers
        self.conv1=nn.Conv2d(3,16,kernel_size=8,stride=4)
        self.conv2=nn.Conv2d(16,32,kernel_size=4,stride=2)
        #Fully connected layers
        self.fc1=nn.Linear(32*14*14,128)
        #Output layers
        self.policy=nn.Linear(128,num_actions)
    def forward(self,x):
        x=nn.functional.relu(self.conv1(x))
        x=nn.functional.relu(self.conv2(x))
        x=x.reshape(x.size(0),-1)
        x=nn.functional.relu(self.fc1(x))
        #Compute policy distribution overs actions
        policy=nn.functional.softmax(self.policy(x),dim=1)
        return policy
class VNetwork(nn.Module):
    def __init__(self):
        super(VNetwork,self).__init__()
        self.conv1=nn.Conv2d(3,16,kernel_size=8,stride=4)
        self.conv2=nn.Conv2d(16,32,kernel_size=4,stride=2)
        self.fc1=nn.Linear(32*14*14,128)
        self.value=nn.Linear(128,1)
    def forward(self,x):
        x=nn.functional.relu(self.conv1(x))
        x=nn.functional.relu(self.conv2(x))
        x=x.reshape(x.size(0),-1)
        x=nn.functional.relu(self.fc1(x))
        value=self.value(x)
        return value

