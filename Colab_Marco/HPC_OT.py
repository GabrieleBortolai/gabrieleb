#!/usr/bin/env python
# coding: utf-8


import ot
import torch
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.offsetbox import AnchoredText
from data_gen_1D import Generator

import time 


def wasserstein_distance(a, b, wa, wb):
        
    W_dist = ot.wasserstein_1d(a, b, wa, wb, p = 1)

    return W_dist


def Z_score(H0, t1):
    
    if t1 <= torch.max(H0):
        p_value = (torch.count_nonzero(H0 >= t1) + 1)/(torch.tensor(H0.size()) + 1)

    if t1 > torch.max(H0):
        p_value = 1/(torch.tensor(H0.size()) + 1)
        
    return torch.tensor(norm.ppf(1-p_value))


def Plotter(W_dist_calibration, W_dist, color, label, save):

    fig, ax = plt.subplots(1, 1, figsize = (7, 7))

    ax.set_title('Wasserstein distance', fontsize = 20)
    ax.set_xlabel('$W_{1}$', fontsize = 15)
    ax.set_ylabel('Density', fontsize = 15)

    z_score = Z_score(W_dist_calibration, torch.median(W_dist))
    anchored_text_test = AnchoredText('$Z_{score}$:'+str('%.3f' % z_score.item()), bbox_to_anchor = (1, 0.85), bbox_transform = ax.transAxes, loc = 'right')

    ax.hist(W_dist_calibration, bins = 'auto', color = color[0], density = True, label = 'ref-'+str(label[0]), alpha = 0.5)
    ax.hist(W_dist, bins = 'auto', color = color[1], density = True, label = 'ref-'+str(label[1]), alpha = 0.5)
    ax.legend()

    ax.add_artist(anchored_text_test)

    if save:

        fig.savefig(f'./HPC_Plot/W_dist_{label}.pdf')


def fun(iteration, sig_type, value, save):

    Pois_ON = False

    NS = 0
    NR = 2000

    N_R = 200000

    W_dist_calibration = torch.zeros(iteration)
    W_dist = torch.zeros(iteration)
    
    for seed in range(iteration):

        #calibration
        ref, data = Generator(seed, NS, NR, N_R, 'ref', Pois_ON)
        W_dist_calibration[seed] = wasserstein_distance(torch.squeeze(ref), torch.squeeze(data), torch.tensor(1/ref.size(0)), torch.tensor(1/data.size(0)))

        #classes
        ref, data_true  = Generator(seed + int(1e6), value[0], value[1], N_R, sig_type, Pois_ON)
        W_dist[seed] = wasserstein_distance(torch.squeeze(ref), torch.squeeze(data_true), torch.tensor(1/ref.size(0)), torch.tensor(1/data_true.size(0)))
    
    Plotter(W_dist_calibration, W_dist, ['#ff7f0e', '#2ca02c'], ['ref', sig_type], save)


def iterable(classes, iteration, values, save):

    for i in range(len(classes)):

        yield([iteration, classes[i], values[i], save])

iteration = 1000
save = True

classes = ['NP1', 'NP2', 'NP3', 'NP4']
values = [[10, 1990], [110, 1890], [80, 1920], [0, 2000]]

ite = iterable(classes, iteration, values, save)

with Pool (processes = 4) as p:
    st = time.time()

    W_dist = p.starmap(fun, ite)

    et = time.time()
    print(f'Computational time: {et-st}')

