#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import ot


# In[6]:


#GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")


# In[18]:


def wasserstein_dist(a,b,metric, d):
    
    source = a[torch.nonzero(a, as_tuple = False)[:,0], torch.nonzero(a, as_tuple = False)[:,1]].view(torch.nonzero(a, as_tuple = False).size(0))
    target = b[torch.nonzero(b, as_tuple = False)[:,0], torch.nonzero(b, as_tuple = False)[:,1]].view(torch.nonzero(b, as_tuple = False).size(0))
    
    source = source.to(device)
    target = target.to(device)
    
    source = source/torch.sum(source, dtype = torch.double)
    target = target/torch.sum(target, dtype = torch.double)
    
    M = ot.dist(torch.nonzero(a, as_tuple = False).double(), torch.nonzero(b, as_tuple = False).double(), metric = metric).to(device)
    M = M/d
    
    T = ot.emd(source, target, M).to(device)
    W = torch.sum(T*M).to(device)
    
    W = torch.sqrt(W)
    
    return W


# In[4]:


data, targets = torch.load('data/MNIST/test_eq_s=1500', map_location = device)
d = 2*28*28


# In[13]:


#Wasserstein distance 

metric='sqeuclidean'

size = data.size(0)

Wasserstein_dist=torch.zeros(size, size, dtype = torch.double).to(device)

for i in range (size):
    for j in filter(lambda h: h>i, range (size)):
        Wasserstein_dist[i][j] = wasserstein_dist(data[i], data[j], metric, d).to(device)
    print('Row n:',(i/size)*100)


# In[ ]:


torch.save([Wasserstein_dist, targets],'data/MNIST/Wasserstein_dist_test_eq_s='+str(size))
