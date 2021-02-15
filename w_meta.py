# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 12:16:47 2020

This file executes w_runner a desired number of times. It calls the plotting 
functions such that each trial can be plotted but also the averages.

@author: Lieke Ceton
"""

import numpy as np
import matplotlib.pyplot as plt
from inputs import get_obs, tmat, maxtime
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools 

def get_spars_DMS(dms_stim, obs_func = get_obs):
    #This function performs a sparsity analysis on DMS stimuli
    #A way of measuring the sparsity is the cosine similarity
    #Cosine similarity == normalised dot product

    #sparsity of time vectors (maxtime = 10, imported from inputs)
    dms_time = tmat
    t_corr = np.zeros((maxtime,maxtime))
    for i, j in itertools.product(range(maxtime), repeat = 2):
        vec_norm = np.linalg.norm(dms_time[i])*np.linalg.norm(dms_time[j])
        t_corr[i][j] = np.dot(dms_time[i],dms_time[j])/vec_norm
    
    #sparsity of letters 
    extra_stim = np.array(['p']) #add small p for fixation
    all_stim = np.concatenate((dms_stim,extra_stim))
    #all_stim = dms_stim
    stim_bin = np.vstack(list(map(obs_func, all_stim)))[:,10:] #convert to binary strings and only select letter part
    s = len(stim_bin)

    stim_corr = np.zeros((s,s))
    for i, j in itertools.product(range(s), repeat = 2):
        vec_norm = np.linalg.norm(stim_bin[i])*np.linalg.norm(stim_bin[j])
        stim_corr[i][j] = np.dot(stim_bin[i],stim_bin[j])/vec_norm

    #Convert to latent layer
    wl, wh = -.5, .5
    nl = 15
    nx = 7
    print(len(stim_bin))
    W_lx = np.random.sample((nl, nx))*(wh-wl) + wl
    
    l_in = np.zeros((len(stim_bin),nl))
    for ii in range(len(all_stim)):
        l_in[ii] = W_lx.dot(stim_bin[ii])
    
    print(np.shape(l_in))
    sigmoid_offset = 2.5
    transfer = lambda x: 1 / ( 1. + np.exp(sigmoid_offset - x) )
    l_sens = transfer(l_in)
    print(np.shape(l_sens))
    
    sparse_corr = np.zeros((s,s))
    for i, j in itertools.product(range(s), repeat = 2):
        vec_norm = np.linalg.norm(l_sens[i])*np.linalg.norm(l_sens[j])
        sparse_corr[i][j] = np.dot(l_sens[i],l_sens[j])/vec_norm

    return sparse_corr    

def plot_sparse(t_corr, stim_corr, s, all_stim):    
    #Assess the level of sparsity between inputs
    print('Cosine similarity of time vector t=3 compared to t=1:10')
    print(np.around(t_corr[3],3))
    avg_corr = np.mean(np.mean(stim_corr)) #Average similarity between letters
    print('Average overlap between normalized stimuli')
    print("%.3f" % avg_corr)

    #Plot thresholded images 
    threshold = 0.5 #(20% overlap)
    stim_corr_thres = (stim_corr < threshold).astype(int)
    ratio = sum(sum(stim_corr_thres))/(s*s)
    print('Percentage of stimuli-pairs with < %.2f overlap' % threshold)
    print("%.3f" % ratio)

    #plot the time-focused overlap matrix
    fig, ax = plt.subplots()
    plt.xlabel('Stimulus at time t')
    plt.ylabel('Stimulus at time t')
    plt.title('Cosine similarity of time-part DSM stimuli')
    im = plt.imshow(t_corr.T, interpolation='nearest', cmap='seismic', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(tmat)))
    ax.set_yticks(np.arange(len(tmat)))
    fig.colorbar(im, ax=ax)
    im.set_clim(-1.0, 1.0)
    plt.show()
    
    #plot the letter-focused overlap matrix
    fig, ax = plt.subplots(figsize=(10,5))
    plt.xlabel('Stimulus')
    plt.ylabel('Stimulus')
    plt.title('Cosine similarity of DSM stimuli')
    im = plt.imshow(stim_corr.T, interpolation='nearest', cmap='seismic', vmin=-1, vmax=1)
    
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = list(all_stim)
    ax.set_xticks(np.arange(s))
    ax.set_yticks(np.arange(s))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax)
    im.set_clim(-0.4, 1.0)
    plt.show()
    
    #plot the thresholded letter-focused overlap matrix
    fig, ax = plt.subplots(figsize=(10,5))
    plt.xlabel('Stimulus')
    plt.ylabel('Stimulus')
    plt.title('Thresholded cosine similarity (T = %.2f)' % threshold)
    im = plt.imshow(stim_corr_thres.T, interpolation='nearest', cmap='gray', vmin=0, vmax=1)
    
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = list(all_stim)
    ax.set_xticks(np.arange(s))
    ax.set_yticks(np.arange(s))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.show()
    
    return 
    

def plot_W_xh(agent): 
    W_hx_start = agent.W_hx_start #at trial 1
    W_hx_end = agent.W_hx 
    max1 = W_hx_end.max()
    min1 = W_hx_end.min()
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout(pad = 3.0)
    fig.suptitle('Weights W_xh')
    im1 = ax1.imshow(W_hx_start.T, cmap='hot', interpolation='nearest')
    ax1.set(xlabel="nh", ylabel="nx")
    ax1.set_xticks(range(len(W_hx_start)))
    ax1.set_yticks(range(len(W_hx_start.T)))
    ax1.set_title('Before learning')
    im2 = ax2.imshow(W_hx_end.T, cmap='hot', interpolation='nearest')
    ax2.set(xlabel="nh", ylabel="nx")
    ax2.set_xticks(range(len(W_hx_end)))
    ax2.set_yticks(range(len(W_hx_end.T)))
    ax2.set_title('After learning')
    
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    fig.colorbar(im1, cax=cax1)
    im1.set_clim(min1, max1)
    fig.colorbar(im2, cax=cax)
    im2.set_clim(min1, max1)
    plt.show()
    return

def plot_W_xS(agent):
    #save fixed weights W_xS #activity into memory
    W_xS = agent.W_Sx
    plt.figure(2)
    plt.imshow(W_xS.T, cmap='hot', interpolation='nearest')
    plt.xlabel('nS')
    plt.ylabel('nx')
    plt.xticks(range(len(W_xS)))
    plt.yticks(range(len(W_xS.T)))
    plt.title('Weights W_xS')
    plt.show()
    return
