#!/usr/bin/env python
# coding: utf-8


import numpy as np

from data_gen_utils import NP2_gen, plot_training_data


def Sig(sig_type, NS_p, rng):
    
    if sig_type=='NP1':
        sig = rng.normal(loc=6.4, scale=0.16, size=(NS_p,1))
    elif sig_type=='NP2':
        sig = np.expand_dims(NP2_gen(NS_p, rng), axis=1)*8
    elif sig_type=='NP3':
        sig = rng.normal(loc=1.6, scale=0.16, size=(NS_p,1))
    else: 
        sig = rng.exponential(scale=1, size=(NS_p,1))

    return sig


def Generator(seed, NS, NR, N_R, sig_type, Pois_ON):
    
    rng = np.random.default_rng(seed)
    
    if Pois_ON:
        NS_p = rng.poisson(lam = NS, size = 1).item()
        NR_p = rng.poisson(lam = NR, size = 1).item()
        ND = NS_p + NR_p # size of data sample (this sum should be fluctuating around 2000 if isPoi==True)
        N = ND + N_R # total size
    else:
        NS_p = NS
        NR_p = NR
    
    ref = rng.exponential(scale=1, size=(N_R,1))

    bkg = rng.exponential(scale=1, size=(NR_p,1))
    sig = Sig(sig_type, NS_p, rng)

    data = np.concatenate((bkg, sig), axis = 0)

    if sig_type=='NP4':
        mask_idx = np.where((data >= 5.07))[0]
        data = np.delete(data, mask_idx, axis=0)

    return [ref, data]
