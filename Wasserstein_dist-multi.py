#!/usr/bin/env python
# coding: utf-8


import torch
import numpy as np
import ot
from multiprocessing import Pool


# In[2]:


#GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")


# In[3]:


def wasserstein_dist(a,b,metric, d):
    
    source = a[torch.nonzero(a, as_tuple = False)[:,0], torch.nonzero(a, as_tuple = False)[:,1]].view(torch.nonzero(a, as_tuple = False).size(0))
    target = b[torch.nonzero(b, as_tuple = False)[:,0], torch.nonzero(b, as_tuple = False)[:,1]].view(torch.nonzero(b, as_tuple = False).size(0))
    
    source = source.to(device)
    target = target.to(device)
    
    source = source/torch.sum(source, dtype = torch.double)
    target = target/torch.sum(target, dtype = torch.double)
    
    M = ot.dist(torch.nonzero(a, as_tuple = False), torch.nonzero(b, as_tuple = False), metric = metric).to(device)
    M = M/d
    
    T = ot.emd(source, target, M).to(device)
    W = torch.sum(T*M).to(device).to(device)
    
    W = torch.sqrt(W)
    
    return W


# In[4]:


def fun(data_source, data_target, metric, d, ind1, ind2):
    
    W = torch.zeros(data_source.size(0), data_target.size(0))
    
    if ind1 == ind2:
        for i in range(data_source.size(0)):
            for j in filter(lambda h: h>i, range(data_target.size(0))):

                W[i][j] = wasserstein_dist(data_source[i], data_target[j], metric, d)
                
    elif ind2 > ind1:
        
        for i in range(data_source.size(0)):
            for j in range(data_target.size(0)):
                
                W[i][j] = wasserstein_dist(data_source[i], data_target[j], metric, d)
            
    return W
            


# In[5]:


def iterable(data, processes, metric, d):
    
    data = torch.stack(torch.chunk(data, processes, dim = 0), dim = 0)
    
    for i in range(data.size(0)):
        for j in range(data.size(0)):
            
            yield([data[i], data[j], metric, d, i, j])


# In[6]:


data, targest = torch.load('data/MNIST/train_eq_s=3000')
data = data.to(device)
d = 2*28*28

# In[ ]:


# import time
#Wasserstein distance 
# if __name__ == '__int__':

# size = data.size(0)

size = 100
metric = 'sqeuclidean'
W_dist = torch.zeros(size, size)
processes = 1
ite = iterable(data[:size], processes, metric, d)


# st = time.time()
with Pool(processes = processes) as p:

    W_dist = torch.cat(torch.chunk(torch.cat(p.starmap(fun, ite), dim = 1), processes, dim = 1), dim = 0)
    

# et = time.time()
# print(W_dist)
# print(et - st)


# torch.save([Wasserstein_dist, targets],'data/MNIST/Wasserstein_dist_train_s='+str(size))

# funzione per estrarre valori: scipy.sqrform
