#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/home/gbortolai/Thesis/')


# In[1]:


import torch
import numpy as np
import ot


# In[2]:


#GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
print(f"Using {device} device")


# In[3]:


def wasserstein_dist(data_source, data_target):#1-Wasserstein
    
    source = data_source[:,0]/torch.sum(data_source[:,0], dtype = torch.double)#.to(device)
    target = data_target[:,0]/torch.sum(data_target[:,0], dtype = torch.double)#.to(device)
    
    source = source.to(device)
    target = target.to(device)
    
    M = ot.dist(torch.stack([data_source[:,1], data_source[:,2]], dim = -1), torch.stack([data_target[:,1], data_target[:,2]], dim = -1), metric = 'euclidean').to(device)
    
    T = ot.emd(source, target, M).to(device)
    W = torch.sum(T*M).to(device)
    
    return W


# In[8]:


def datasetmaker(data, val_max):
    return torch.stack([data[:int(val_max/2)], data[int(val_max/2):val_max]])


# In[9]:


jets, targets = torch.load('data/Jets/dataset_test-coppie', map_location = device)
jets = jets.to(torch.double)
jets = datasetmaker(jets, jets.size(0))


#Wasserstein distance 

size = jets.size(1)

Wasserstein_dist=torch.zeros(size, dtype = torch.double).to(device)

for i in range (size):
    Wasserstein_dist[i] = wasserstein_dist(jets[0][i], jets[1][i]).to(device)
    print('riga n:',(i/size)*100)
#Wasserstein_dist = Wasserstein_dist/torch.max(Wasserstein_dist)


# In[38]:


torch.save([Wasserstein_dist, targets],'data/Jets/Wasserstein_dist_test-coppie_s='+str(size))

