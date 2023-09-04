#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

from pytorchtools_CNN_paper import EarlyStopping

from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

import scipy.stats as st

from matplotlib.offsetbox import AnchoredText


# In[2]:


#GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")


# In[3]:


class CNN(nn.Module):

    def __init__(self, input_size, kernel_size, p, p_cnn, neg_slope):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
        
        nn.Conv2d(input_size, 32, kernel_size, dtype = torch.float, padding = 'same'),
        nn.ReLU(),
        nn.Dropout2d(p = p_cnn),
        nn.MaxPool2d(5, stride = 1),
        nn.BatchNorm2d(32, dtype = torch.float),

        nn.Conv2d(32, 32, kernel_size, dtype = torch.float, padding = 'same'),
        nn.ReLU(),
        nn.Dropout2d(p = p_cnn),
        nn.MaxPool2d(5, stride = 1),
        nn.BatchNorm2d(32, dtype = torch.float),
        
        nn.Conv2d(32, 64, kernel_size, dtype = torch.float, padding = 'same'),
        nn.ReLU(),
        nn.Dropout2d(p = p_cnn),
        nn.MaxPool2d(5, stride = 1),
        nn.BatchNorm2d(64, dtype = torch.float),
        
        )

        self.fcff = nn.Sequential(
        
        nn.Linear(16384, 1000, dtype = torch.float),
        nn.Dropout1d(p = p),
        nn.BatchNorm1d(1000, dtype = torch.float),
        nn.LeakyReLU(neg_slope),

        nn.Linear(1000, 400, dtype = torch.float),
        nn.Dropout1d(p = p),
        nn.BatchNorm1d(400, dtype = torch.float),
        nn.LeakyReLU(neg_slope),

        nn.Linear(400, 200, dtype = torch.float),
        nn.Dropout1d(p = p),
        nn.BatchNorm1d(200, dtype = torch.float),
        nn.LeakyReLU(neg_slope),
        
        nn.Linear(200, 2, dtype = torch.float),

        )
        
        #self._init_weights(self.conv)
        #self._init_weights(self.fcff)
        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        
        if isinstance(module, nn.Conv2d):
            torch.nn.init.dirac_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.08)
                
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, 0, 0.1)
            if module.bias is not None:
                module.bias.data.fill_(0.08)

        
    def forward(self, x1, x2):

        x1 = self.conv(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.fcff(x1)
        
        x2 = self.conv(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fcff(x2)
        
        return torch.stack([x1, x2])


# In[4]:


class MyLoss (nn.Sequential):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, W, E):
        
        xw, yw = torch.nonzero(W, as_tuple = True)
        
        W = W[xw,yw]
        E = E[xw,yw]

        # loss = torch.mean(torch.abs(E-W)/W).to(device)
        loss = torch.mean((torch.abs(E-W)/W) + ((torch.abs(E-W)/(E + 7e-3))))

        return loss


# In[5]:


def Reg_Loss(loss, LAMBDA):
    
    l2_reg = 0
    for W in model.parameters():
        l2_reg = l2_reg + torch.norm(W, 1)
    
    loss_reg = loss + LAMBDA*l2_reg
    
    return loss_reg


# In[6]:


def Dist(W, E):
    
    x, y = torch.nonzero(W, as_tuple = True)
    
    W = W[x,y]
    E = E[x,y]
    
    dist = E/W
    
    return dist


# In[ ]:


#Dataloder

train, _ = torch.load('/data/gabrieleb/data/MNIST/train_eq_s=3000', map_location=device)
validation, _ = torch.load('/data/gabrieleb/data/MNIST/validation_eq_s=1000', map_location=device)

train = train.to(torch.float)
validation = validation.to(torch.float)

W_dist_train, _ = torch.load('/data/gabrieleb/data/MNIST/Wasserstein_dist_train_eq_s=3000', map_location=device)
W_dist_validation, _ = torch.load('/data/gabrieleb/data/MNIST/Wasserstein_dist_validation_eq_s=1000', map_location=device)

W_dist_train = W_dist_train.to(torch.float)
W_dist_validation = W_dist_validation.to(torch.float)


#parameters

n_sample_train = train.size(0)
n_sample_validation = validation.size(0)

batch_size = 200

n_batches_train = int(n_sample_train/batch_size)
n_batches_validation = int(n_sample_validation/batch_size)

train = torch.stack(torch.chunk(train, n_batches_train, dim = 0), dim = 0).view(n_batches_train, batch_size, 1, 28, 28)
W_dist_train = torch.stack(torch.chunk(torch.stack(torch.chunk(W_dist_train, n_batches_train, dim = -1), dim = 0), n_batches_train, dim = 1), dim = 0)

validation = validation.view(validation.size(0), 1,validation.size(1), validation.size(2))

input_size = 1
kernel_size = 5

#Train
#-----------------------------------------------------------------------------------------------------------SETTING
learning_rate = 1e-2

criterion = MyLoss()
#criterion = nn.MSELoss()

LAMBDA = 5e-5


patience = 30

iteration = 1000

counter_train = 0 
counter_validation = 0

neg_slope = 1e-1

p = 0.1
p_cnn = 0.2

model = CNN(input_size, kernel_size, p, p_cnn, neg_slope).to(device)

optimizer = optim.Adam(model.parameters(),lr = learning_rate)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.5)

early_stopping = EarlyStopping(patience = patience, verbose = True)

#---------------------------------------------------------------------------------NN
for ite in range (iteration):
    #Train
    #---------------------------------------------------------------------------TRAIN
    model.train()
    for batch_x in range (n_batches_train):
        for batch_y in range (n_batches_train):
            if torch.count_nonzero(W_dist_train[batch_x][batch_y]) == 0:continue
            else:
                #embedding
                sample_emb_train = model(train[batch_x], train[batch_y]).to(device)

                E_dist_train = torch.cdist(sample_emb_train[0], sample_emb_train[1], p = 2).to(device) 

                loss = criterion(W_dist_train[batch_x][batch_y], E_dist_train)
                # loss_reg = Reg_Loss(loss, LAMBDA)

                optimizer.zero_grad()
                loss.requires_grad_()
                loss.backward(retain_graph=True)

                optimizer.step()
#----------------------------------------------------------------------------------------------------------------------VALIDATION
    #Validation
    model.eval()
    #embedding
    sample_emb_validation = model(validation, validation).to(device)

    E_dist_validation = torch.cdist(sample_emb_validation[0], sample_emb_validation[0], p = 2).to(device)

    valid_loss = criterion(W_dist_validation, E_dist_validation)
#------------------------------------------------------------------------------------------PLOT
    early_stopping(valid_loss, model, p_cnn)
    scheduler.step()


    if early_stopping.early_stop:
            print("Early stopping")
            break
