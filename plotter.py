# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:44:41 2020
@author: Lieke Ceton
"""
import matplotlib.pyplot as plt
import numpy as np
import tictoc

def plot_ggsa(saved_i,saved_p):
    #print pro/antisaccade performance in a graph
    plt.plot(saved_i,saved_p[:,0], label='pro')
    plt.plot(saved_i,saved_p[:,1], label='anti')
    plt.plot(saved_i,saved_p[:,2], label='total')
    plt.xlabel('Trials')
    plt.ylabel('Performance')
    plt.title('Pro/Antisaccade task')
    plt.legend(loc = 'upper left')
    plt.show
    return

def plot_dms(trial_print,saved_p,iswi,n_switches,conv_crit):
    #print DMS performance in a graph
    arr = [i*trial_print for i in range(0,len(saved_p))]
    #get the markers on the line
    a = (np.floor(iswi/trial_print)).astype(int)    #find point closest below
    b = (np.ceil(iswi/trial_print)).astype(int)     #find point closest above
    
    iswi_p = np.zeros(len(iswi))
    for x in range(len(iswi)):                      #for all but the last one..
        if b[x] == len(saved_p):
            b[x] = b[x]                          #if its the last trial, set b[x] back to floor
        diff = saved_p[a[x] - 1] - saved_p[b[x] - 1]        #find difference in performance
        if diff < 0:                                #is b is bigger than a
           iswi_p[x] = b[x] - 1                          #choose b
        else:
           iswi_p[x] = a[x] - 1                         #if a is bigger choose a
    
    iswi_p = iswi_p.astype(int)
    
    plt.plot(iswi-trial_print,saved_p[iswi_p],'r*', markersize = 10)
    plt.plot(arr,saved_p,'r')
    plt.xlabel('Trials')
    plt.ylabel('Performance')
    plt.title('DMS task')
    plt.show
    return

def plot_abba(saved_i,saved_p,abba_i):
    #print ABBA performance in a graph
    plt.plot(saved_i,saved_p)
    plt.plot(abba_i,np.ones(len(abba_i)),'b*')
    plt.xlabel('Trials')
    plt.ylabel('Performance')
    plt.title('ABBA task')
    plt.show
    return
    
def plot_otax(trial_print, lesson_perf_buff,saved_p):
    arr = [i*trial_print for i in range(0,len(saved_p))]
    plt.plot(arr,saved_p,'r')
    plt.axvspan(0, lesson_perf_buff[0][0], facecolor='xkcd:grey', alpha=0.5)
    plt.axvspan(sum(lesson_perf_buff[0][0:2]), sum(lesson_perf_buff[0][0:3]), facecolor='xkcd:grey', alpha=0.5)
    plt.axvspan(sum(lesson_perf_buff[0][0:4]), sum(lesson_perf_buff[0][0:5]), facecolor='xkcd:grey', alpha=0.5)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('Trials')
    plt.ylabel('Performance')
    plt.title('AX12 task')
    plt.show
    return

def visual_weights(W_hx_start, W_hx_end, weight_label = 'test', agent_label = 'test2'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
        fig.suptitle('Weights ' + weight_label + ' of agent ' + agent_label)
        sub1 = ax1.imshow(W_hx_start.T, cmap='hot', interpolation='nearest')
        ax1.set(xlabel="nh", ylabel="nx")
        ax1.set_xticks(range(len(W_hx_start)))
        ax1.set_yticks(range(len(W_hx_start.T)))
        ax1.set_title('Before learning')
        sub2 = ax2.imshow(W_hx_end.T, cmap='hot', interpolation='nearest')
        ax2.set(xlabel="nh", ylabel="nx")
        ax2.set_xticks(range(len(W_hx_start)))
        ax2.set_yticks(range(len(W_hx_start.T)))
        ax2.set_title('After learning')
        fig.colorbar(sub1, ax=ax1)
        fig.colorbar(sub2, ax=ax2)
        plt.show()
        
def visual_weights_fixed(W_Sx, weight_label = 'test', agent_label = 'test2'):
        #save fixed weights W_xS #activity into memory
        #W_xS2 = agent2.W_Sx
        plt.figure(2)
        plt.imshow(W_Sx.T, cmap='hot', interpolation='nearest')
        plt.xlabel('nS')
        plt.ylabel('nx')
        plt.xticks(range(len(W_Sx)))
        plt.yticks(range(len(W_Sx.T)))
        plt.title('Weights W_xS2')
        plt.show()
    