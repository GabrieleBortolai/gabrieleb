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

from pytorchtools_Transformer_vision import EarlyStopping

from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

import scipy.stats as st

from matplotlib.offsetbox import AnchoredText
import math


# In[2]:


#GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")


# In[3]:


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# In[4]:


class Transformer(nn.Module):
    def __init__(self, d_model, num_layers, p, p_cnn, neg_slope, n_head, pos_dropout, kernel_size):
        super(Transformer, self).__init__()

        self.dmodel = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = n_head, dim_feedforward = 2048, dropout = 0.2, batch_first = False, dtype = torch.float)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer = self.encoder_layer, num_layers = num_layers)
        self.pos_encoder = PositionalEncoding(d_model = d_model, dropout = pos_dropout)
        
        self.linear = nn.Linear(1024, d_model, dtype = torch.float)
        
        self.conv = nn.Sequential(
        
            nn.Conv2d(1, 32, kernel_size, dtype = torch.float, padding = 'same'),
            nn.ReLU(),
            nn.Dropout2d(p = p_cnn),
            nn.MaxPool2d(kernel_size, stride = 1),
            nn.BatchNorm2d(32, dtype = torch.float),

            nn.Conv2d(32, 32, kernel_size, dtype = torch.float, padding = 'same'),
            nn.ReLU(),
            nn.Dropout2d(p = p_cnn),
            nn.MaxPool2d(kernel_size, stride = 1),
            nn.BatchNorm2d(32, dtype = torch.float),

            nn.Conv2d(32, 64, kernel_size, dtype = torch.float, padding = 'same'),
            nn.ReLU(),
            nn.Dropout2d(p = p_cnn),
            nn.MaxPool2d(kernel_size, stride = 1),
            nn.BatchNorm2d(64, dtype = torch.float),
        
        )
        
        self.fcff = nn.Sequential(
          
            nn.Linear(512, 1200, device = device),
            nn.BatchNorm1d(1200),
            nn.Dropout1d(p = p),
            nn.LeakyReLU(neg_slope),
            
            nn.Linear(1200, 450, device = device),
            nn.BatchNorm1d(450),
            nn.Dropout1d(p = p),
            nn.LeakyReLU(neg_slope),

            nn.Linear(450, 30, device = device),
            nn.BatchNorm1d(30),
            nn.Dropout1d(p = p),
            nn.LeakyReLU(neg_slope),

            nn.Linear(30, 3,  device = device),
        )
        
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
        
    def img_patcher(self, x, patch_size = 4):
        
        H_patch = x.size(2)/patch_size
        W_patch = x.size(3)/patch_size
        
        x = x.view(x.size(0), x.size(1), int(H_patch), patch_size, int(W_patch), patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.flatten(1, 2)
        x = x.flatten(2, 4)
        
        return x

        
        
    def forward(self, x1, x2):

        x1 = self.conv(x1)
        x1 = self.img_patcher(x1)
        # x1 = x1.reshape(-1, 256,4)
        # # # x1 = torch.flatten(x1, 1)
        x1 = self.linear(x1)
        x1 = x1.permute(1, 0, 2)
        x1 = self.pos_encoder(x1 * torch.sqrt(torch.tensor(self.dmodel)))
        x1 = self.transformer_encoder(x1)
        x1 = x1.permute(1, 0, 2)
        x1 = torch.flatten(x1, 1)
        x1 = self.fcff(x1)

        x2 = self.conv(x2)
        x2 = self.img_patcher(x2)
        # x2 = x2.reshape(-1, 256,4)
        # # # x2 = torch.flatten(x2, 1)
        x2 = self.linear(x2)
        x2 = x2.permute(1, 0, 2)
        x2 = self.pos_encoder(x2 * torch.sqrt(torch.tensor(self.dmodel)))
        x2 = self.transformer_encoder(x2)
        x2 = x2.permute(1, 0, 2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fcff(x2)
        
        return torch.stack([x1, x2])


# In[5]:


class MyLoss (nn.Sequential):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, W, E):
        
        xw, yw = torch.nonzero(W, as_tuple = True)
        
        W = W[xw,yw]
        E = E[xw,yw]

        loss = torch.mean(torch.abs(E-W)/W).to(device)

        return loss


# In[6]:


def Reg_Loss(loss, LAMBDA):
    
    l2_reg = 0
    for W in model.parameters():
        l2_reg = l2_reg + torch.norm(W, 1)
    
    loss_reg = loss + LAMBDA*l2_reg
    
    return loss_reg


# In[7]:


def Dist(W, E):
    
    x, y = torch.nonzero(W, as_tuple = True)
    
    W = W[x,y]
    E = E[x,y]
    
    dist = E/W
    
    return dist


# In[ ]:


#Dataloder

train, targets_train = torch.load('/home/gbortolai/Thesis/data/MNIST/train_eq_s=3000', map_location=device)
validation, targets_validation = torch.load('/home/gbortolai/Thesis/data/MNIST/validation_eq_s=1000', map_location=device)

W_dist_train, targets_train = torch.load('/home/gbortolai/Thesis/data/MNIST/Wasserstein_dist_train_eq_s=3000', map_location=device)
W_dist_validation, targets_validation = torch.load('/home/gbortolai/Thesis/data/MNIST/Wasserstein_dist_validation_eq_s=1000', map_location=device)

validation = validation.view(W_dist_validation.size(0), 1, validation.size(1), validation.size(2)).double()

del targets_train, targets_validation

#parameters

n_sample_train = train.size(0)
n_sample_validation = validation.size(0)

batch_size = 200

n_batches_train = int(n_sample_train/batch_size)
n_batches_validation = int(n_sample_validation/batch_size)

train = torch.stack(torch.chunk(train, n_batches_train, dim = 0), dim = 0).view(n_batches_train, batch_size, 1, 28, 28)
W_dist_train = torch.stack(torch.chunk(torch.stack(torch.chunk(W_dist_train, n_batches_train, dim = -1), dim = 0), n_batches_train, dim = 1), dim = 0)

train = train.to(torch.float)
validation = validation.to(torch.float)


d_model = 32
num_layers = 2
n_head = 4
pos_dropout = 0.2
kernel_size = 5

#Train
#-----------------------------------------------------------------------------------------------------------SETTING
learning_rate = 1e-3

criterion = MyLoss()
#criterion = nn.MSELoss()

LAMBDA = 1e-6


patience = 30

iteration = 1000

counter_train = 0 
counter_validation = 0

neg_slope = 1e-1

p = 0.1
p_cnn = 0.2

model = Transformer(d_model, num_layers, p, p_cnn, neg_slope, n_head, pos_dropout, kernel_size).to(device)#modificare

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
            if torch.count_nonzero(W_dist_train[batch_x][batch_y]) != 0:
                #embedding
                sample_emb_train = model(train[batch_x], train[batch_y]).to(device)
                E_dist_train = torch.cdist(sample_emb_train[0], sample_emb_train[1], p = 2).to(device)
                
                loss = criterion(W_dist_train[batch_x][batch_y], E_dist_train)
                loss_reg = Reg_Loss(loss, LAMBDA)

                optimizer.zero_grad()
                loss_reg.requires_grad_()
                loss_reg.backward(retain_graph=True)

                optimizer.step()
#----------------------------------------------------------------------------------------------------------------------VALIDATION 
    #Validation
    model.eval()
    #embedding
    sample_emb_validation = model(validation, validation).to(device)


    E_dist_validation = torch.cdist(sample_emb_validation[0], sample_emb_validation[0], p = 2).to(device)

    valid_loss = criterion(W_dist_validation, E_dist_validation)
#------------------------------------------------------------------------------------------PLOT
    early_stopping(valid_loss, model)
    scheduler.step()


    if early_stopping.early_stop:
            print("Early stopping")
            break
