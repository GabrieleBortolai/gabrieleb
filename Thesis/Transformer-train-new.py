#!/usr/bin/env python
# coding: utf-8

# In[26]:


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


# In[27]:


#GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")


# In[28]:


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


# In[29]:


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


# In[30]:


class MyLoss (nn.Sequential):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, W, E):
        
        xw, yw = torch.nonzero(W, as_tuple = True)
        
        W = W[xw,yw]
        E = E[xw,yw]

        loss = torch.mean(torch.abs(E-W)/W).to(device)

        return loss


# In[31]:


def Reg_Loss(loss, LAMBDA):
    
    l2_reg = 0
    for W in model.parameters():
        l2_reg = l2_reg + torch.norm(W, 1)
    
    loss_reg = loss + LAMBDA*l2_reg
    
    return loss_reg


# In[32]:


def Dist(W, E):
    
    x, y = torch.nonzero(W, as_tuple = True)
    
    W = W[x,y]
    E = E[x,y]
    
    dist = E/W
    
    return dist


# In[36]:


def Train(train_tot, train_targets, W_dist_train_tot, batch_size, prongx, prongy, LAMBDA):
    
    train = torch.zeros(torch.count_nonzero(train_targets == prongx).item(), 16, 3)
    W_dist_train = torch.zeros(torch.count_nonzero(train_targets == prongx).item(), torch.count_nonzero(train_targets == prongx).item())

    
    l = 0
    for i in torch.nonzero(train_targets == prongx)[:,0]:
        train[l] = train_tot[l]
        m = 0
        for j in torch.nonzero(train_targets == prongy)[:,0]:
            W_dist_train[l][m] = W_dist_train_tot[i][j]
            m = m+1
        l = l+1
    
    size = train.size(0)
    loss_reg = torch.zeros(int(size/batch_size)**2)
    train = torch.stack(torch.chunk(train, int(size/batch_size), dim = 0), dim = 0)
    W_dist_train = torch.stack(torch.chunk(torch.stack(torch.chunk(W_dist_train, int(size/batch_size), dim = -1), dim = 0), int(size/batch_size), dim = 1), dim = 0)

    k=0
    for batch_x in range (int(train.size(0)/batch_size)):
        for batch_y in range (int(train.size(0)/batch_size)):
            if torch.count_nonzero(W_dist_train[batch_x][batch_y]) != 0:
                #embedding
                sample_emb_train = model(train[batch_x], train[batch_y]).to(device)

                E_dist_train = torch.cdist(sample_emb_train[0], sample_emb_train[1], p = 2).to(device) 

                loss = criterion(W_dist_train[batch_x][batch_y], E_dist_train)
                loss_reg[k] = Reg_Loss(loss, LAMBDA)
                k = k+1
    
    return loss_reg


# In[ ]:


#Dataloder

train, train_targets = torch.load('/home/gbortolai/Thesis/data/Jets/dataset_train', map_location=device)
validation, _  = torch.load('/home/gbortolai/Thesis/data/Jets/dataset_validation', map_location=device)

W_dist_train, _ = torch.load('/home/gbortolai/Thesis/data/Jets/Wasserstein_dist_train_s=12000', map_location=device)
W_dist_validation, _ = torch.load('/home/gbortolai/Thesis/data/Jets/Wasserstein_dist_validation_s=2400', map_location=device)

train = train.to(torch.float)
validation = validation.to(torch.float)

#parameters
batch_size = 300

d_model = 32
num_layers = 2
n_head = 4
pos_dropout = 0.1

#Train
#-----------------------------------------------------------------------------------------------------------SETTING
learning_rate = 1e-3

criterion = MyLoss()

LAMBDA = 0

patience = 30

iteration = 1000

neg_slope = 1e-1

p = 0.1

model = Transformer(d_model, num_layers, p, neg_slope, n_head, pos_dropout).to(device)#modificare

optimizer = optim.Adam(model.parameters(),lr = learning_rate, weight_decay = 0)#, betas=(0.9, 0.99), eps=1e-5 ,weight_decay=0)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.7)

early_stopping = EarlyStopping(patience = patience, verbose = True)

#---------------------------------------------------------------------------------NN
for ite in range (iteration):
    #Train
    #---------------------------------------------------------------------------TRAIN
    model.train()
    
    loss1 = Train(train, train_targets, W_dist_train, batch_size, 1, 1, LAMBDA)
    loss2 = Train(train, train_targets, W_dist_train, batch_size, 1, 2, LAMBDA)
    loss3 = Train(train, train_targets, W_dist_train, batch_size, 1, 3, LAMBDA)
    loss4 = Train(train, train_targets, W_dist_train, batch_size, 1, 4, LAMBDA)
    loss5 = Train(train, train_targets, W_dist_train, batch_size, 2, 1, LAMBDA)
    loss6 = Train(train, train_targets, W_dist_train, batch_size, 2, 2, LAMBDA)
    loss7 = Train(train, train_targets, W_dist_train, batch_size, 2, 3, LAMBDA)
    loss8 = Train(train, train_targets, W_dist_train, batch_size, 2, 4, LAMBDA)
    loss9 = Train(train, train_targets, W_dist_train, batch_size, 3, 1, LAMBDA)
    loss10 = Train(train, train_targets, W_dist_train, batch_size, 3, 2, LAMBDA)
    loss11 = Train(train, train_targets, W_dist_train, batch_size, 3, 3, LAMBDA)
    loss12 = Train(train, train_targets, W_dist_train, batch_size, 3, 4, LAMBDA)
    loss13 = Train(train, train_targets, W_dist_train, batch_size, 4, 1, LAMBDA)
    loss14 = Train(train, train_targets, W_dist_train, batch_size, 4, 2, LAMBDA)
    loss15 = Train(train, train_targets, W_dist_train, batch_size, 4, 3, LAMBDA)
    loss16 = Train(train, train_targets, W_dist_train, batch_size, 4, 4, LAMBDA)
    
    loss = torch.cat((torch.tensor(loss1), torch.tensor(loss2), torch.tensor(loss3), torch.tensor(loss4), torch.tensor(loss5), torch.tensor(loss6), torch.tensor(loss7),torch.tensor(loss4),torch.tensor(loss9),torch.tensor(loss10), torch.tensor(loss11),torch.tensor(loss12),torch.tensor(loss13),torch.tensor(loss14),torch.tensor(loss15),torch.tensor(loss16)))
    loss_reg = torch.mean(loss)
    print(loss_reg.size())

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

