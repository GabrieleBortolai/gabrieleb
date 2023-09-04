#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

from pytorchtools_Transformer_coppie import EarlyStopping

from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

import scipy.stats as st

from matplotlib.offsetbox import AnchoredText
import math


# In[22]:


#GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")


# In[23]:


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


# In[24]:


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

            nn.Linear(30, 2,  device = device),
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


# In[25]:


class MyLoss (nn.Sequential):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, W, E):
        
        xw = torch.nonzero(W, as_tuple = False)[:,0]
        
        W = W[xw]
        E = E[xw]

        loss = torch.mean(torch.abs(E-W)/W).to(device)

        return loss


# In[26]:


def Reg_Loss(loss, LAMBDA):
    
    l2_reg = 0
    for W in model.parameters():
        l2_reg = l2_reg + torch.norm(W, 1)
    
    loss_reg = loss + LAMBDA*l2_reg
    
    return loss_reg


# In[27]:


def Dist(W, E):
    
    x = torch.nonzero(W, as_tuple = False)[:,0]
    
    W = W[x]
    E = E[x]
    
    return E/W


# In[28]:


def datasetmaker(data, val_max):
    return torch.stack([data[:int(val_max/2)], data[int(val_max/2):val_max]])


# In[29]:


def w_dist_excavator(W, val_min, val_max):
    return W[torch.arange(val_min), torch.arange(val_min, val_max)]


# In[30]:


def batchesmaker(data, n_batch, dim):#dim = 1 data, dim = 0 W
    return torch.stack(torch.chunk(data, n_batch, dim = dim), dim = dim)


# In[14]:


#Dataloder

train, targets_train = torch.load('/home/gbortolai/Thesis/data/Jets/dataset_train-coppie', map_location=device)
validation, targets_validation  = torch.load('/home/gbortolai/Thesis/data/Jets/dataset_validation-coppie', map_location=device)

W_dist_train, targets_train = torch.load('/home/gbortolai/Thesis/data/Jets/Wasserstein_dist_train-coppie_s=100000', map_location=device)
W_dist_validation, targets_validation = torch.load('/home/gbortolai/Thesis/data/Jets/Wasserstein_dist_validation-coppie_s=20000', map_location=device)

#parameters

n_sample_train = train.size(0)
n_sample_validation = validation.size(0)

batch_size = 100

n_batches_train = int(n_sample_train/batch_size)

train = datasetmaker(train, n_sample_train)
train = batchesmaker(train, n_batches_train, 1)
# W_dist_train = w_dist_excavator(W_dist_train, 6000, 120)
W_dist_train = batchesmaker(W_dist_train, n_batches_train, 0)

validation = datasetmaker(validation, validation.size(0))
# W_dist_validation = w_dist_excavator(W_dist_validation, 600, 1200)

train = train.to(torch.float)
validation = validation.to(torch.float)


d_model = 32
num_layers = 2
n_head = 4
pos_dropout = 0.1

#Train
#-----------------------------------------------------------------------------------------------------------SETTING
learning_rate = 1e-3

criterion = MyLoss()
#criterion = nn.MSELoss()

LAMBDA = 0


patience = 30

iteration = 1000

counter_train = 0 
counter_validation = 0

neg_slope = 0

p = 0.1

model = Transformer(d_model, num_layers, p, neg_slope, n_head, pos_dropout).to(device)#modificare

optimizer = optim.Adam(model.parameters(),lr = learning_rate, weight_decay = 0)#, betas=(0.9, 0.99), eps=1e-5 ,weight_decay=0)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.5)

pdist = nn.PairwiseDistance(p=2, eps = 0)

early_stopping = EarlyStopping(patience = patience, verbose = True)

#---------------------------------------------------------------------------------NN
for ite in range (iteration):
    #Train
    #---------------------------------------------------------------------------TRAIN
    model.train()
    for batch in range (n_batches_train):
        #embedding
        sample_emb_train = model(train[0][batch], train[1][batch]).to(device)

        E_dist_train = pdist(sample_emb_train[0], sample_emb_train[1]).to(device) 

        loss = criterion(W_dist_train[batch], E_dist_train)
        # dist = Dist(W_dist_train[batch], E_dist_train)
        loss_reg = Reg_Loss(loss, LAMBDA)

        optimizer.zero_grad()
        loss_reg.requires_grad_()
        loss_reg.backward(retain_graph=True)

        optimizer.step()
#----------------------------------------------------------------------------------------------------------------------VALIDATION
    #Validation
    model.eval()
    #embedding
    sample_emb_validation = model(validation[0], validation[1]).to(device)

    E_dist_validation = pdist(sample_emb_validation[0], sample_emb_validation[1]).to(device)

    valid_loss = criterion(W_dist_validation, E_dist_validation)
    # valid_dist = Dist(W_dist_validation, E_dist_validation)

#------------------------------------------------------------------------------------------PLOT
    early_stopping(valid_loss, model)
    scheduler.step()


    if early_stopping.early_stop:
            print("Early stopping")
            break
