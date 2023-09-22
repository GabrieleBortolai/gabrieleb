import torch
import ot
from multiprocessing import Pool


def wasserstein_loss(W,E):

    x, y = torch.nonzero(W, as_tuple = True)

    W = W[x,y]
    E = E[x,y]

    W = W.to(torch.double)
    E = E.to(torch.double)

    W = W/torch.sum(W, dtype = torch.double)
    E = E/torch.sum(E, dtype = torch.double)

    loss = ot.emd2_1d(W.t(), E.t(), metric = 'euclidean')

    return loss