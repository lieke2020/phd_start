# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:18:37 2021
@author: Lieke Ceton
"""

import numpy as np
import matplotlib.pyplot as plt
import random

import premate_LJ01
import pretask

trial_print = 150 #printing of trials

def plot_class(trial_print, saved_p, conv_crit):
    #print DMS performance in a graph
    arr = [i*trial_print for i in range(0,len(saved_p))]
    plt.plot(arr,saved_p,'r')
    plt.xlabel('Trials')
    plt.ylabel('Performance')
    plt.title('DMS-stimuli classification task')
    plt.show
    return
    
def _run_trial(agent, env, log_q=False, **kwargs):
    """
    Generic function to run a trial:
    agent   : the agent being run
    env     : the environment implementing the task
    env     : if log_q=True, function returns both the reward and the output
                values in the q-layer

    Additional kwargs can be passed to be set in the environment (not used here)
    """
    if not kwargs is None:
        env.__dict__.update(kwargs)
    # init trial results:
    totrew = 0.0
    qlog = []

    # step until obs == 'RESET', collect reward and q-values
    while True:
        totrew += agent.step()
        if log_q:
            print((agent.x))
            qlog.append (agent.q)
        if agent.obs == 'RESET':
            break
    # format result:

    retval = (totrew, np.array(qlog)) if log_q else totrew
    return retval

n_switches = 5 # i.e. 6 stimulus sets, 3 stimuli per set
n_stim = 3
conv_crit = 0.85

def run_class(viz_converged=True, agent_type = 'W', print_all = 'on'):
    plt.close('all')
    #makes the numbers predictable!
    #random.seed(1) 
    #n_switches = 5 # i.e. 6 stimulus sets, 3 stimuli per set
    dms = pretask.DMS_classify(n_stim=n_stim, n_switches=n_switches)
    
    # init the agent
    agent = premate_LJ01.PreMATe(dms)
    print('Pretraining agent')
        
    ### Initialize training:
    # buffer to store performance
    saved_p = np.zeros(1)
    # buffer for last 500 trials:
    total_buff = np.zeros(500) 
    # buffer for performance immediately after a switch
    
    # counters
    i = 0 

    aa = True
    while aa:
        # run trial, get performance
        r = _run_trial(agent, dms)
        # increase i
        i += 1
        # was the trial correct?
        corr = (r >= dms.reward)
        total_buff[0] = corr
        total_buff = np.roll(total_buff, 1)

        # if the past 100 trials were 85% correct, set is 'learned'
        if np.mean(total_buff[:100]) >= conv_crit:
            print('Convergence at {}!'.format(i))
            break

        # print progress:
        if i % trial_print == 0:
            #print performance in a list
            if print_all == 'on':
                print(i, '\t', np.mean(total_buff)) 
            saved_p = np.append(saved_p,np.mean(total_buff))
            
        if i >= 100000:
            aa = False
    print('Loop ended')
    
    plot_class(trial_print, saved_p, conv_crit)
    return (dms, saved_p, agent)

