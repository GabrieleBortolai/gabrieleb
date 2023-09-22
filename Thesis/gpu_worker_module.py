import torch
import ot

def fun(data_source, data_target, metric, d, ind1, ind2, device):
    
    # torch.cuda.set_device(device)
    
    W = torch.zeros(data_source.size(0), data_target.size(0), dtype = torch.float)
    
    if ind1 == ind2:
        for i in range(data_source.size(0)):
            for j in filter(lambda h: h>i, range(data_target.size(0))):

                W[i][j] = wasserstein_dist(data_source[i], data_target[j], metric, d).to(device)
                
    elif ind2 > ind1:
        
        for i in range(data_source.size(0)):
            for j in range(data_target.size(0)):
                
                W[i][j] = wasserstein_dist(data_source[i], data_target[j], metric, d).to(device)
            
    return W
            
def wasserstein_dist(a,b,metric, d):
    
    source = a[torch.nonzero(a, as_tuple = False)[:,0], torch.nonzero(a, as_tuple = False)[:,1]].view(torch.nonzero(a, as_tuple = False).size(0))
    target = b[torch.nonzero(b, as_tuple = False)[:,0], torch.nonzero(b, as_tuple = False)[:,1]].view(torch.nonzero(b, as_tuple = False).size(0))
    
    source = source.to(torch.float)
    target = target.to(torch.float)
    
    source = source/torch.sum(source, dtype = torch.float)
    target = target/torch.sum(target, dtype = torch.float)
    
    M = ot.dist(torch.nonzero(a, as_tuple = False).float(), torch.nonzero(b, as_tuple = False).float(), metric = metric)#.to(device)
    M = M/d
    
    T = ot.emd(source, target, M)#.to(device)
    W = torch.sum(T*M)#.to(device)
    
    W = torch.sqrt(W)
    
    return W