#!/usr/bin/env python
"""
This file is part of the WorkMATe source code, provided as supplemental material
for the article:
    "Flexible Working Memory through Selective Gating and Attentional Tagging"
Wouter Kruijne, Sander M Bohte, Pieter R Roelfsema, Christian N L Olivers

Please see the README file for additional details.
Questions or comments regarding this file can be addressed to w.kruijne@vu.nl
-------------------
This file implements four functions for individual runs for the four tasks
tasks. The random seeds that each run get here yield 'illustrative'
convergence times.
"""
import numpy as np
import matplotlib.pyplot as plt

#from multiprocessing import Pool
#from IPython import get_ipython

from plotter import plot_ggsa, plot_abba, plot_dms
import tasks
import workmate
import workmate_PG
import workmate_LJ
import workmate_LJ01
import workmate_LJ03
import workmate_LJ02
import workmate_LJ02_match
#import workmate_LJ04
import workmate_LJ07
import workmate_LJ07b
import workmate_LJ06
import workmate_sparse 

#np.random.seed(2187682668) # a randnum [0- 1e10]
ncore = 4
trial_print = 150 #printing of trials

def _run_trial(agent, env,log_q=False, **kwargs):
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

def run_dms(viz_converged=True, agent_type = 'W', print_all = 'off', beta2 = 1, beta3 = 1, set_seed = 1):
    plt.close('all')
    #makes the numbers predictable!
    seed = set_seed
    np.random.seed(seed) 
    #n_switches = 5 # i.e. 6 stimulus sets, 3 stimuli per set
    
    dms = tasks.DMS(n_stim=n_stim, n_switches=n_switches)
    
    # init the agent
    #enable the option of choosing another agent to execute the tasks
    if agent_type == 'W2':
        agent = workmate.WorkMATe(dms, nblocks=2, nhidden = 30, block_size = 30)
        print('Wouters agent with blocksize 30')
    elif agent_type == 'W40':
        agent = workmate.WorkMATe(dms, nblocks=2, nhidden = 40, block_size = 40)
        print('Wouters agent with blocksize 40')
    elif agent_type == 'W2S':
        agent = workmate_sparse.WorkMATe(dms, nblocks=2, block_size = 30)
        print('Wouters agent with sparse inputs')
        #print('Wouters agent with sparse inputs and blocksize 30')
    elif agent_type == 'L40': 
        agent = workmate_LJ02_match.WorkMATe(dms, nblocks=2)
        print('Agent with 40 layer, sparse inputs, different match')
    elif agent_type == 'L01': 
        agent = workmate_LJ01.WorkMATe(dms, nblocks=2)
        print('Agent with random layer L and 1-on-1 S projection')
    elif agent_type == 'L03': 
        agent = workmate_LJ03.WorkMATe(dms, nblocks=2)
        print('Agent with random layer L and S')
    elif agent_type == 'L02': 
        agent = workmate_LJ02.WorkMATe(dms, nblocks=2)
        print('Agent with random layer L and 1-on-1 S projection')
    elif agent_type == 'L07': 
        agent = workmate_LJ07b.WorkMATe(dms, nblocks=2, beta2 = beta2, beta3 = beta3)
        print('Agent with extra layer l that acts as a hidden layer!!!!')
    elif agent_type == 'L06': 
        agent = workmate_LJ06.WorkMATe(dms, nblocks=2)
        print('L06: Agent with learned extra layer and random S projection')
    elif agent_type == 'L': 
        agent = workmate_LJ.WorkMATe(dms, nblocks=2)
        print('Agent with extra layer')
    else:
        agent = workmate.WorkMATe(dms, nblocks=2, beta2 = beta2, beta3 = beta3)
        print('Wouters agent')
        
    ### Initialize training:
    # buffer to store 'the moment of switching' (=convergence)
    iswi = np.zeros((n_switches+1)) 
    # buffer for last 500 trials:
    saved_p = np.zeros(1)
    total_buff = np.zeros(500) 
    # buffer for performance immediately after a switch
    swiperf = np.nan * np.ones((n_switches+1, total_buff.size))
    
    # counters
    i = 0 
    i_ = 0

    aa = True
    while aa:
        # run trial, get performance
        r = _run_trial(agent, dms)
        # increase i
        i += 1
        # was the trial correct?
        corr = ( r >= dms.bigreward )
        total_buff[0] = corr
        total_buff = np.roll(total_buff, 1)

        # if the past 100 trials were 85% correct, set is 'learned'
        if np.mean(total_buff[:100]) >= conv_crit:
            print('Convergence at {}\tSwitch to set {}'.format(i,dms.setnr+1))
            iswi[dms.setnr] = i
            # if criterion reached in less than 500 trials,
            # 'performanc post-switch' hasn't been logged yet -- do that now,
            # using only the trials with this set:
            if i < i_ + 500:
                swiperf[dms.setnr, :(i- i_)] = total_buff[:(i - i_)] # leaves nans for the rest of performance

            if dms.setnr == 5:
                break

            dms.switch_set()
            total_buff *= 0 # reset performance buffer
            i_ = i

        # @ iswi + 500: store post-switch performance:
        if i == i_ + 500:
            swiperf[dms.setnr, :] = total_buff

        # print progress:
        if i % trial_print == 0:
            #print performance in a list
            if print_all == 'on':
                print(i, '\t', np.mean(total_buff)) 
            saved_p = np.append(saved_p,np.mean(total_buff))
            
        if i >= 30000:
            aa = False
    print('Loop ended')
    
    plot_dms(trial_print,saved_p,iswi,n_switches,conv_crit)
    return (saved_p, iswi, agent)

# TODO: variant 2?

def run_otax(agent_type = 'W'):
#def run_otax(seed=1):
    """
    12-AX, trial-based with curriculum learning
    """
    #np.random.seed(seed)
    otax = tasks.OneTwoAX()
    
    if agent_type == 'L': 
        agent = workmate_sparse.WorkMATe(otax)
        print('Liekes agent sparse')
    else:
        agent = workmate.WorkMATe(otax)
        print('Wouters agent')
    
    #agent = workmate.WorkMATe(otax)
    
    total_buff      = np.zeros(100)
    saved_p = np.zeros(1)
    lesson_perf_buff = np.zeros((2, 5))

    i = 0  # how many trials were run?
    k = 0  # how many trials were run at the current level?

    aa = True
    while aa:
        # run trial, get performance
        r = _run_trial(agent, otax)
        # increase i
        i += 1
        k += (1 if otax.trial_at_highest_level else 0)
        # store 'correct' in buffer(s)
        corr = ( r >= otax.bigreward )
        total_buff[0] = corr
        total_buff = np.roll(total_buff, 1)

        # check whether crit is met:
        if np.mean(total_buff) > .85:
            print('Converged at difficulty {}'.format(otax.difficulty))
            print('after {} /// {} trials'.format(i, k))
            lesson_perf_buff[:, otax.difficulty] = i,k
            print(lesson_perf_buff)

            if otax.difficulty == 4:
                #otax.difficulty == 4 #check convergence
                break
            otax.difficulty += 1
            total_buff *= 0 
            i = 0
            k = 0
        # print progress.
        if (i>0) and (i % trial_print == 0):
            print(i, '\t', np.mean(total_buff))
            saved_p = np.append(saved_p,np.mean(total_buff))
            
        if i >= 40000:
            aa = False
    #print('Loop ended')
            
    return (lesson_perf_buff, saved_p, agent)

"""
ABAB ordered recognition task:
"""
def run_abba(agent_type = 'W'):
    #np.random.seed()
    # create abba environment
    abba = tasks.ABBA_recog()
    # create agent:
    nhidden = 30 #number of hidden units
    
    # init the agent
    #enable the option of choosing another agent to execute the tasks
    if agent_type == 'L': 
        agent = workmate_LJ.WorkMATe(abba, nhidden=nhidden)
        print('Liekes agent')
    else:
        agent = workmate.WorkMATe(abba, nhidden=nhidden)
        print('Wouters agent')
    
    # buffers; total & per trial type
    saved_p         = np.zeros(4)
    saved_i         = np.zeros(1)
    total_buff      = np.zeros(100)
    trtype_buff     = np.zeros((4, total_buff.size))
    res = []
    i = 0 

    abba_i = np.zeros(5)
    aa = True
    while aa:
        r = _run_trial(agent, abba, p = 1-np.mean(trtype_buff, axis=1))
        i += 1

        # store whether it was correct
        tp = abba.trial_type
        corr = ( r >= abba.bigreward )
        total_buff[0] = corr
        total_buff = np.roll(total_buff, 1)
        trtype_buff[tp, 0] = corr
        trtype_buff[tp, :] = np.roll( trtype_buff[tp,:], 1 )

        # 'convergence' on individual trial types:
        x = np.mean(trtype_buff[tp, :])
        if abba_i[tp] == 0 and x > 0.75:
            print("Done with ", tp)
            abba_i[tp] = i

        # criterion for full convergence
        if np.all(np.mean(trtype_buff, axis=1) > .75) and np.mean(total_buff) > .85:
            print('Done.', i)
            abba_i[-1] = i
            break

        # Uncomment this for dynamic condition:
        #if i % 3000== 0:
           #abba.switch_set()

        # print progress.
        abba_print = 1000
        if i % abba_print == 0:
            print(i, '\t'.join([str(v) for v in np.mean(trtype_buff, axis=1)]), end=' ') 
            saved_p = np.vstack((saved_p,np.mean(trtype_buff, axis=1)))
            saved_i = np.append(saved_i,i)
            print('\t',np.mean(total_buff))
            step_arr = np.r_[np.mean(total_buff), np.mean(trtype_buff, axis=1)]
            res += [step_arr]
    # return np.array(res) 
    
        if i >= 200000:
            aa = False
    print('Loop ended')
    
    plot_abba(saved_i,saved_p,abba_i)
    return agent

"""
Pro-/Antisaccade task
"""
def run_ggsa(seed=1,prefixed_gates=False, agent_type = 'W'):
    np.random.seed(seed)
    # create ggsa
    ggsa= tasks.GGSA()
    # create agent
    #enable the option of choosing another agent to execute the tasks
    if agent_type == 'L': 
        agent = workmate_LJ.WorkMATe(ggsa)
        print('Liekes agent')
    else:
        agent = workmate.WorkMATe(ggsa)
        print('Wouters agent')
    
    if prefixed_gates:
        agent = workmate_PG.WorkMATePG(ggsa)
    else:
        agent = workmate.WorkMATe(ggsa)

    # 2 buffers: overall & per trial-type
    saved_p         = np.zeros(3)
    saved_i         = np.zeros(1)
    total_buff      = np.zeros(100)
    trtype_buff     = np.zeros((2, total_buff.size))
    i = 0  
    
    aa = True
    while aa:
        r = _run_trial(agent, ggsa)
        i += 1

        # store 'correct' in buffer(s)
        tp = ggsa.trial_type
        corr = ( r >= ggsa.bigreward )
        total_buff[0] = corr
        total_buff = np.roll(total_buff, 1)
        trtype_buff[tp, 0] = corr
        trtype_buff[tp, :] = np.roll( trtype_buff[tp,:], 1 )

        # check whether crit is met:
        separate_perf = np.mean(trtype_buff, axis=1)
        total_perf    = np.mean(total_buff)
        if (np.all(separate_perf) > .75) and (total_perf > .85):
            print('Converged after {}'.format(i))
            print(i, '\t'.join([str(v) for v in np.mean(trtype_buff, axis=1)]))
            print("=========")
            break
        # print progress.
        if i == 250:
            print('\033[1m'+'Trial','\t','Perf1','\t','Perf2','\t','TotalPerf'+'\033[0m')
        if i % 250 == 0:
            print(i,'\t','\t'.join([str(v) for v in np.mean(trtype_buff, axis=1)]), end=' ') #print trial number, the mean performance on both trial types
            perf = np.append(np.mean(trtype_buff, axis=1),np.mean(total_buff))
            saved_p = np.vstack((saved_p,perf))
            saved_i = np.append(saved_i,i)
            print('\t',np.mean(total_buff)) #print total performance
    
        if i >= 20000:
            aa = False
    print('Loop ended')
    
    plot_ggsa(saved_i,saved_p)
    return agent

#if __name__ == '__main__':
    #run_dms(seed=1)
    #run_otax(seed=4)
    #run_abba(seed=5)
    #run_ggsa(2)
    #run_ggsa(2,prefixed_gates=True)