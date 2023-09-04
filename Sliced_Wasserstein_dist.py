#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/home/gbortolai/Thesis/')


# In[1]:


import torch
import numpy as np
import ot
from multiprocessing import Pool
import time


# In[2]:
device = "cpu"
print(f"Using {device} device")


# In[3]:


def sliced_wasserstein_dist(a,b):
    
    source = a[torch.nonzero(a, as_tuple = False)[:,0], torch.nonzero(a, as_tuple = False)[:,1]].view(torch.nonzero(a, as_tuple = False).size(0)).float()
    target = b[torch.nonzero(b, as_tuple = False)[:,0], torch.nonzero(b, as_tuple = False)[:,1]].view(torch.nonzero(b, as_tuple = False).size(0)).float()
    
    source = torch.unsqueeze(source, dim = 1)
    target = torch.unsqueeze(target, dim = 1)
    
    source = source.to(device)
    target = target.to(device)
    
    W = ot.sliced.sliced_wasserstein_distance(source, target, p = 2).to(device)
    
    return W


data, targets = torch.load('data/Jets/dataset_validation')
data = data.to(device)

size = data.size(0)

Sliced_Wasserstein_dist=torch.zeros(size, size, dtype = torch.float).to(device)

# st = time.time()
for i in range (size):
    for j in filter(lambda h: h>i, range (size)):
        Sliced_Wasserstein_dist[i][j] = sliced_wasserstein_dist(data[i], data[j]).to(device)
    if i%5 == 0:
        print(f'Rows done: {(i/size)*100} %')
# et = time.time()
# print(et-st)

torch.save([Sliced_Wasserstein_dist, targets],'data/Jets/Sliced_Wasserstein_dist_validation_simple_s='+str(size))

