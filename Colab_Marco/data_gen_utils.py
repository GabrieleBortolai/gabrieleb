import numpy as np

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


def plot_training_data(data, weight_data, ref, weight_ref, feature_labels, bins_code, xlabel_code, ymax_code={},
                       save=False, save_path='', file_name=''):
    '''
    Plot distributions of the input variables for the training samples.
    
    data:            (numpy array, shape (None, n_dimensions)) data training sample (label=1)
    weight_data:     (numpy array, shape (None,)) weights of the data sample (default ones)
    ref:             (numpy array, shape (None, n_dimensions)) reference training sample (label=0)
    weight_ref:      (numpy array, shape (None,)) weights of the reference sample
    feature_labels:  (list of string) list of names of the training variables
    bins_code:       (dict) dictionary of bins edge for each training variable (bins_code.keys()=feature_labels)
    xlabel_code:     (dict) dictionary of xlabel for each training variable (xlabel.keys()=feature_labels)
    ymax_code:       (dict) dictionary of maximum value for the y axis in the ratio panel for each training variable (ymax_code.keys()=feature_labels)
    '''
    plt_i = 0
    for key in feature_labels:
        bins = bins_code[key]
        plt.rcParams["font.family"] = "serif"
        plt.style.use('classic')
        fig = plt.figure(figsize=(10, 10)) 
        fig.patch.set_facecolor('white')  
        ax1= fig.add_axes([0.1, 0.43, 0.8, 0.5])        
        hD = plt.hist(data[:, plt_i],weights=weight_data, bins=bins, label='DATA', color='black', lw=1.5, histtype='step', zorder=4)
        hR = plt.hist(ref[:, plt_i], weights=weight_ref, color='#a6cee3', ec='#1f78b4', bins=bins, lw=1, label='REFERENCE')
        plt.errorbar(0.5*(bins[1:]+bins[:-1]), hD[0], yerr= np.sqrt(hD[0]), color='black', ls='', marker='o', ms=5, zorder=3)
        font = font_manager.FontProperties(family='serif', size=16)
        l    = plt.legend(fontsize=18, prop=font, ncol=2)
        font = font_manager.FontProperties(family='serif', size=18) 
        plt.tick_params(axis='x', which='both',    labelbottom=False)
        plt.yticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        plt.ylabel("events", fontsize=22, fontname='serif')
        plt.yscale('log')
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.3]) 
        x   = 0.5*(bins[1:]+bins[:-1])
        plt.errorbar(x, hD[0]/hR[0], yerr=np.sqrt(hD[0])/hR[0], ls='', marker='o', label ='DATA/REF', color='black')
        font = font_manager.FontProperties(family='serif', size=16)
        plt.legend(fontsize=18, prop=font)
        plt.xlabel(xlabel_code[key], fontsize=22, fontname='serif')
        plt.ylabel("ratio", fontsize=22, fontname='serif')
        if key in list(ymax_code.keys()):
            plt.ylim(0., ymax_code[key])
        plt.yticks(fontsize=16, fontname='serif')
        plt.xticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        plt.grid()
        if save:
            if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
            else:
                if file_name=='': file_name = 'InputVariable_%s'%(key)
                else: file_name += '_InputVariable_%s'%(key)
                fig.savefig(save_path+file_name+'.pdf')
        plt.show()
        plt.close()
        plt_i+=1
    return


def NP2_gen(size, rng):
    if size>10000:
        raise Warning('Sample size is grater than 10000: Generator will not approximate the tale well')
    sample = np.array([])
    #normalization factor                                                                                                                                    
    Norm = 256.*0.25*0.25*np.exp(-2)
    while(len(sample)<size):
        x = rng.uniform(0,1) #assuming not to generate more than 10 000 events                                                                         
        p = rng.uniform(0, Norm)
        if p<= 256.*x*x*np.exp(-8.*x):
            sample = np.append(sample, x)
    return sample