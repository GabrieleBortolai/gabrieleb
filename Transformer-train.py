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

from pytorchtools_Transformer_simple import EarlyStopping

from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

import scipy.stats as st

from matplotlib.offsetbox import AnchoredText
import math
import itertools


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
    def __init__(self, d_model, num_layers, p, neg_slope, n_head, pos_dropout):
        super(Transformer, self).__init__()

        self.dmodel = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = n_head, dim_feedforward = 2048, dropout = 0.25, batch_first = False, dtype = torch.float)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer = self.encoder_layer, num_layers = num_layers)
        self.pos_encoder = PositionalEncoding(d_model = d_model, dropout = pos_dropout)
        
        self.linear = nn.Linear(3, d_model)
        
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
        
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, 0, 0.1)
            if module.bias is not None:
                module.bias.data.fill_(0.08)
        
    
    def forward(self, x1, x2):

        x1 = x1.permute(1, 0, 2)
        x1 = self.linear(x1)
        x1 = self.pos_encoder(x1 * torch.sqrt(torch.tensor(self.dmodel)))
        x1 = self.transformer_encoder(x1)
        x1 = x1.permute(1, 0, 2)
        x1 = torch.flatten(x1, 1)
        x1 = self.fcff(x1)

        x2 = x2.permute(1, 0, 2)
        x2 = self.linear(x2)
        x2 = self.pos_encoder(x2 * torch.sqrt(torch.tensor(self.dmodel)))
        x2 = self.transformer_encoder(x2)
        x2 = x2.permute(1, 0, 2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fcff(x2)
        
        return torch.stack([x1, x2])


# In[8]:


# class MyLoss (nn.Sequential):
#     def __init__(self):
#         super(MyLoss, self).__init__()

#     def forward(self, W, E):
        
#         xw, yw = torch.nonzero(W, as_tuple = True)
        
#         W = W[xw,yw]
#         E = E[xw,yw]

#         loss = torch.mean(torch.abs(E-W)/W).to(device)

#         return loss

class MyLoss (nn.Sequential):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, W, E):
        
        xw, yw = torch.nonzero(W, as_tuple = True)
        
        W = W[xw,yw]
        E = E[xw,yw]

        # loss = torch.mean((torch.abs(E-W)/W)+SIGMA/(E + 7e-3))
        loss = torch.mean((torch.abs(E-W)/W) + ((torch.abs(E-W)/(E + 7e-3))))
        # loss = torch.mean((E-W)**2)

        return loss


def classes_reg(dist, targets_x, targets_y):
    
    #reg = 0
    
    for clas in [1,2,3,4]:
        ind_x = torch.nonzero(targets_x == clas)[:,0]
        ind_y = torch.nonzero(targets_y == clas)[:,0]
        ind = torch.tensor(list(itertools.combinations_with_replacement(torch.cat((ind_x, ind_y)) ,2)))
        dist = dist.to('cpu')
        reg = reg + dist[ind[:,0],ind[:,1]]
                
    return reg


def Reg_Loss(loss, LAMBDA, SIGMA, dist):
    
    l_reg = 0
    for W in model.parameters():
        l_reg = l_reg + torch.norm(W, 1)
    
    loss_reg = loss + LAMBDA*l_reg + SIGMA*torch.mean(1/dist)
    
    return loss_reg


# In[11]:


def Dist(W, E):
    
    x, y = torch.nonzero(W, as_tuple = True)
    
    W = W[x,y]
    E = E[x,y]
    
    dist = E/W
    
    return dist


# In[37]:


def Train(train, W_dist_train):
    
    for batch_x in range (n_batches_train):
        for batch_y in range (n_batches_train):
            if torch.count_nonzero(W_dist_train[batch_x][batch_y]) != 0:
                
                sample_emb_train = model(train[batch_x], train[batch_y]).to(device)
                E_dist_train = torch.cdist(sample_emb_train[0], sample_emb_train[1], p = 2).to(device) 

                loss = criterion(W_dist_train[batch_x][batch_y], E_dist_train)
                # loss = Reg_Loss(loss, LAMBDA, SIGMA, E_dist_train).to('cpu')

                optimizer.zero_grad()
                loss.requires_grad_()
                loss = loss.to(torch.float)
                # print(loss.dtype)
                loss.backward(retain_graph=True)
                # print(loss.dtype)

                optimizer.step()


# In[16]:


def Validation(validaiton, W_dist_validation):
    
    sample_emb_validation = model(validation, validation).to(device)
    E_dist_validation = torch.cdist(sample_emb_validation[0], sample_emb_validation[0], p = 2).to(device)
    valid_loss = criterion(W_dist_validation, E_dist_validation)
    
    return valid_loss


# In[ ]:


#Dataloder
device = 'cuda'
train, _ = torch.load('/data/gabrieleb/data/Jets/dataset_train', map_location=device)
validation, _  = torch.load('/data/gabrieleb/data/Jets/dataset_validation', map_location=device)

W_dist_train, _ = torch.load('/data/gabrieleb/data/Jets/Wasserstein_dist_train_s=12000', map_location=device)
W_dist_validation, _ = torch.load('/data/gabrieleb/data/Jets/Wasserstein_dist_validation_s=2400', map_location=device)

train = train.to(torch.float)
validation = validation.to(torch.float)

#parameters

n_sample_train = train.size(0)
batch_size = 300
n_batches_train = int(n_sample_train/batch_size)

train = torch.stack(torch.chunk(train, n_batches_train, dim = 0), dim = 0)
W_dist_train = torch.stack(torch.chunk(torch.stack(torch.chunk(W_dist_train, n_batches_train, dim = -1), dim = 0), n_batches_train, dim = 1), dim = 0)

d_model = 32
num_layers = 2
n_head = 4
pos_dropout = 0.1

#Train
#-----------------------------------------------------------------------------------------------------------SETTING
learning_rate = 1e-3

criterion = MyLoss()
# criterion = nn.MSELoss()

LAMBDA = 0
# SIGMA = 1e-5#1e-4

patience = 30
iteration = 1000
neg_slope = 1e-2
p = 0.1

model = Transformer(d_model, num_layers, p, neg_slope, n_head, pos_dropout).to(device)
optimizer = optim.Adam(model.parameters(),lr = learning_rate, weight_decay = 0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.5)
early_stopping = EarlyStopping(patience = patience, verbose = True, notifica = False)

for ite in range (iteration):
    #Train
    model.train()
    SIGMA = 1e-4
    Train(train, W_dist_train)

    #Validation
    model.eval()
    SIGMA = 0
    valid_loss = Validation(validation, W_dist_validation)

    early_stopping(valid_loss, model)
    scheduler.step()

    if early_stopping.early_stop:
            print("Early stopping")
            break