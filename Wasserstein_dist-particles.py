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
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# In[3]:


def wasserstein_dist(data_source, data_target):#1-Wasserstein

#     ind1 = torch.stack([data_source[:,1], data_source[:,2]], dim = -1)#aggiunto ora
#     ind2 = torch.stack([data_target[:,1], data_target[:,2]], dim = -1)#aggiunto ora
    
#     ind1 = ind1.double()
#     ind2 = ind2.double()
    
    source = data_source[:,0]/torch.sum(data_source[:,0], dtype = torch.double)#.to(device)
    target = data_target[:,0]/torch.sum(data_target[:,0], dtype = torch.double)#.to(device)
    
    source = source.to(device)
    target = target.to(device)
    
    M = ot.dist(torch.stack([data_source[:,1], data_source[:,2]], dim = -1), torch.stack([data_target[:,1], data_target[:,2]], dim = -1), metric = 'euclidean').to(device)
    
    T = ot.emd(source, target, M).to(device)
    W = torch.sum(T*M).to(device)
    
    return W


# In[4]:


jets, targets = torch.load('data/Jets/dataset_train_real', map_location = device)
jets = jets.to(torch.double)


# In[5]:


#Wasserstein distance 

size = jets.size(0)

Wasserstein_dist=torch.zeros(size, size, dtype = torch.double).to(device)

for i in range (size):
    for j in filter(lambda h: h>i, range (size)):
        Wasserstein_dist[i][j] = wasserstein_dist(jets[i], jets[j]).to(device)
    print('riga n:',(i/size)*100)
#Wasserstein_dist = Wasserstein_dist/torch.max(Wasserstein_dist)


# In[38]:


torch.save([Wasserstein_dist, targets],'data/Jets/Wasserstein_dist_train_real_s='+str(size))

