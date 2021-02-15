# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:56:22 2021

This is the main file used as a script to run the agents

@author: Lieke Ceton
"""

from w_runner import trial_print, n_switches, conv_crit, run_dms, run_otax
import numpy as np
import matplotlib.pyplot as plt
from plotter import plot_dms, plot_otax, visual_weights, visual_weights_fixed
import tasks
from w_meta import get_spars_DMS
from inputs import get_obs, get_obs2, get_obs3
import tictoc
import w_pre_runner
import random
#import itertools

# %% Visualing weights per layer/ multi-plots of tasks and sparsity analysis
#For a specific input
dms = tasks.DMS(n_stim = 3, n_switches = 6)
dms_stim = np.ndarray.flatten(dms.stimset)
all_stim = np.concatenate((dms_stim,np.array(['p'])))
#stim_bin_1 = np.vstack(list(map(obs_func, all_stim)))
#stim_bin_sparse = 
#Find activation by dot product with weights

def multi_dms(n_agents = 5, agent_type = 'W'):
        iswix = np.zeros((n_agents,n_switches+1))
        time = np.zeros(n_agents)
        for i in range(n_agents):
            tictoc.tic()
            (saved_p,iswi,agent) = run_dms(agent_type = agent_type)
            iswix[i,:] = iswi
            time[i] = tictoc.toc()
        
        #iswix_med = np.median(iswix, axis = 0)
        iswix_avg = iswix.mean(axis = 0)
        iswix_err = iswix.std(axis = 0)
        time_avg = np.mean(time)
        print(time_avg)
        plt.figure(0)
        plot_dms(trial_print,saved_p,iswi,n_switches,conv_crit) #print last agent
        return (iswix_avg, iswix_err, time_avg)

#plot average number of trials needed for level convergence
def multi_dms_plot(iswix_avg, iswix_err, label, n_agents):
        plt.errorbar((range(1,n_switches+2)),iswix_avg,fmt='-o',ecolor = 'black', yerr=iswix_err, markersize = 8, capsize = 10, elinewidth=2,
    markeredgewidth=2, label = label)
        plt.xlabel('DMS curriculum learning levels')
        plt.ylabel('Trials until convergence')
        plt.title('Average DMS task performance of %i agents' %n_agents)
        plt.legend(loc="upper left", title = "Agent")

def layer_run():
    #Intermediate agent 1: Extra random layer
    #(saved_p, iswi, agent) = run_dms(agent_type = 'L03', print_all = 'on')
    #plot_dms(trial_print,saved_p,iswi,n_switches,conv_crit)
    
    #Intermediate agent 2: 1-on-1 COPY mapping onto memory 
    #(saved_p, iswi, agent) = run_dms(agent_type = 'L02', print_all = 'on')
    #plot_dms(trial_print,saved_p,iswi,n_switches,conv_crit)
    #print(agent.W_Sl)
    #print(agent.S)
    #print(agent.x)
    #visual_weights(agent.W_lx_start, agent.W_lx, 'W_lx' ,'L02')
    #visual_weights(agent.W_hl_start, agent.W_hl, 'W_hl' ,'L02')
    #visual_weights_fixed(agent.W_Sl, 'W_Sl', 'L02')
    
    #Intermediate agent 7: Learned layer L with x in S
    (saved_p, iswi, agent) = run_dms(agent_type = 'L07', print_all = 'on')
    plot_dms(trial_print,saved_p,iswi,n_switches,conv_crit)
    print(agent.W_Sx)
    print(agent.S)
    print(agent.x)
    visual_weights(agent.W_lx_start, agent.W_lx, 'W_lx' ,'L07')
    visual_weights(agent.W_hl_start, agent.W_hl, 'W_hl' ,'L07')
    visual_weights_fixed(agent.W_Sx, 'W_Sl', 'L07')
    
    #Intermediate agent 6: Learned layer L and random projection to S
    #(saved_p, iswi, agent) = run_dms(agent_type = 'L06', print_all = 'on')
    #plot_dms(trial_print,saved_p,iswi,n_switches,conv_crit)
    #print(agent.W_Sl)
    #print(agent.S)
    #print(agent.x)
    #visual_weights(agent.W_lx_start, agent.W_lx, 'W_lx' ,'L06')
    #visual_weights(agent.W_hl_start, agent.W_hl, 'W_hl' ,'L06')
    #visual_weights_fixed(agent.W_Sl, 'W_Sl', 'L06')
    
    #(avg1, err1, time1) = multi_dms(n_agents = 1, agent_type = 'L03')
    #(avg2, err2, time2) = multi_dms(n_agents = 1, agent_type = 'W')
    
    #Multi_run plot of LJ01, LJ02, WorkMATe
    #(avg1, err1, time1) = multi_dms(n_agents = 10, agent_type = 'L03')
    #(avg2, err2, time2) = multi_dms(n_agents = 10, agent_type = 'W')
    #(avg3, err3, time3) = multi_dms(n_agents = 10, agent_type = 'L02')
    
    #plt.figure(1)
    #multi_dms_plot(avg1, err1, 'L03')
    #multi_dms_plot(avg2, err2, 'W2S')
    #multi_dms_plot(avg3, err3, 'L02')
    #plt.show()

def run6():    
    
    (saved_p2, iswi2, agent2) = run_dms(agent_type = 'W2S') #for sparse inputs
    #visual_weights(agent2.W_hx_start, agent2.W_hx, 'W_hx' ,'W2S')
    
    #Print weights for agent L!
    (saved_p, iswi, agent) = run_dms(agent_type = 'L', print_all = 'on') #for sparse inputs
    visual_weights(agent.W_lx_start, agent.W_lx, 'W_lx' ,'L')
    visual_weights(agent.W_hl_start, agent.W_hl, 'W_hl' ,'L')
    
    (saved_p3, iswi3, agent3) = run_dms(agent_type = 'L05', print_all = 'on') #for sparse inputs
    visual_weights(agent3.W_lx_start, agent3.W_lx, 'W_lx' ,'L05')
    visual_weights(agent3.W_hl_start, agent3.W_hl, 'W_hl' ,'L05')
    #What do we see? Something is off..
    #Change weights for agent L. This does not matter.
    
    #Print all feedabck to get an intuition of what we are working with
    #print(agent.fbh)
    #print(agent.feedbackh)
    print(agent.fbh_transfer)
    print(agent.W_hl_transpose)
    print(agent.W_hl)
    print(agent.W_hl[:,-3])
    print(agent.fhl) #scalar?
    #print(agent.feedbackl) #VERY ALIKE (0.016 - 0.20)
    
    print(agent2.fbh)
    print(agent2.feedback)    
    #Run intermediate agents for more information
    
    #W_hx_start2 = agent2.W_hx_start
    #W_hx_end2 = agent2.W_hx 
    
    #find total activity and compare (should be quite similar)
    #average over all possible stimuli
    dms = tasks.DMS(n_stim = 3, n_switches = 6)
    dms_stim = np.ndarray.flatten(dms.stimset)
    #for each old stimulus in the set
    #act = get_obs(dms_stim).dot(W_xh_start)
    #for each new stimulus in the set
    #act2 = get_obs2(dms_stim).dot(W_xh_start_sparse)
    
    #This W_xh before and after can also be used to visualise 
    #the weights in the agent with the extra layer
    #What can we learn and what do we expect to happen?
    #Make heatmaps before and after > even add animation?
    return agent.fhl

def sparse_latent():
    #set seed and number of trials to 1?
    (saved_p, iswi, agent) = run_dms(agent_type = 'L06', print_all = 'on')
    (saved_p2, iswi2, agent2) = run_dms(agent_type = 'W', print_all = 'on')
    (saved_p3, iswi3, agent3) = run_dms(agent_type = 'W2S', print_all = 'on')
    
    dms = tasks.DMS(n_stim = 3, n_switches = 6)
    dms_stim = np.ndarray.flatten(dms.stimset)
    F = get_spars_DMS(dms_stim, get_obs)
    
    #plot the letter-focused overlap matrix
    fig, ax = plt.subplots(figsize=(10,5))
    plt.xlabel('Stimulus')
    plt.ylabel('Stimulus')
    plt.title('Cosine similarity of DSM stimuli')
    im = plt.imshow(F.T, interpolation='nearest', cmap='seismic', vmin=-1, vmax=1)
    
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = list(all_stim)
    ax.set_xticks(np.arange(len(all_stim)))
    ax.set_yticks(np.arange(len(all_stim)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax)
    im.set_clim(0, 1.0)
    plt.show()
    
    avg_corr = np.mean(np.mean(F)) #Average similarity between letters
    print('Average overlap between normalized stimuli')
    print("%.3f" % avg_corr)
    
    return 


def spars():
    #Defining the DMS stimuli per agent and their sparsity
    #(saved_p, iswi, agent) = run_dms()
    dms = tasks.DMS(n_stim = 3, n_switches = 6)
    dms_stim = np.ndarray.flatten(dms.stimset)
    
    #A function that prints the sparsity analysis of DMS inputs
    get_spars_DMS(dms_stim, get_obs)
    #get_spars_DMS(dms_stim, get_obs3)
    #Within inputs a new function get_obs2 for a sparser stimulus dictionary
    get_spars_DMS(dms_stim, get_obs2)
    
    #check for all possible stimuli
    #stimuli = string.ascii_lowercase + string.ascii_uppercase + string.digits
    #get_spars_DMS(stimuli, get_obs)
    #get_spars_DMS(stimuli, get_obs2)
    
    #check for all used stimuli
    #used_stimuli = string.ascii_uppercase + '012glarp'
    #get_spars_DMS(used_stimuli, get_obs)
    #get_spars_DMS(used_stimuli, get_obs2)
    
    #Workmate_sparse (W2S) uses this function get_obs2
    #Nothing changes about the stimset, only the way it is encoded as a binary vector
    #run_dms(print_all = 'on')
    #run with this new stimdict
    #run_dms(agent_type = 'W2S', print_all = 'on')
    
    #apply this to the Workmate agent
    #run multi_dms (5x, does it help anything?)
    return

#The DMS task ran multiple times for multiple agent (meta-run)
def run():
    
    #Run agents a number of times
    (avg1, err1, t1) = multi_dms(n_agents = 10, agent_type = 'W2S') 
    (avg2, err2, t2) = multi_dms(n_agents = 10, agent_type = 'W')
    #(avg1, err1, t1) = multi_dms(n_agents = 10, agent_type = 'W') 
    #(avg3, err3) = multi_dms(n_agents = 5, agent_type = 'W2')
    #(avg4, err4) = multi_dms(n_agents = 5, agent_type = 'W40')

    n_agents = 10            #number of agents 

    def multi_dms_plot_median(iswix_med, iswix_err, label):   
        plt.errorbar((range(1,n_switches+2)),iswix_med,fmt='-o',ecolor = 'black', yerr=iswix_err, markersize = 8, capsize = 10, elinewidth=2,
    markeredgewidth=2, label = label)
        plt.xlabel('DMS curriculum learning levels')
        plt.ylabel('Trials until convergence')
        plt.title('Average DMS task performance over %i agents'% n_agents)
        plt.legend(loc="upper left", title = "Agent")
        
    plt.figure(1)
    multi_dms_plot(avg2, err2, label = "W", n_agents = 10)
    multi_dms_plot(avg1, err1, label = "W2S", n_agents = 10)
    #multi_dms_plot(avg3, err3, label = "Wh30")
    #multi_dms_plot(avg2, err2, label = "Wsparse")
    #multi_dms_plot(avg4, err4, label = "Wh40")
    plt.show()
    
    #plt.figure(1)
    #multi_dms_plot(avg2, err2, label = "W")
    #multi_dms_plot(avg1, err1, label = "W2S")
    #multi_dms_plot(avg3, err3, label = "Wh30")
    #multi_dms_plot(avg2, err2, label = "Wsparse")
    #multi_dms_plot(avg4, err4, label = "Wh40")
    #plt.show()
       
    print( "Average elapsed time W2S: %f seconds.\n" %t1 )
    print( "Average elapsed time W: %f seconds.\n" %t2 )
    
    return
    #print(iswix.mean(axis=0))
    #return (iswix_avg, iswix_err)

#Multirun for AX12 task (where timesteps matter)
def run1():
    #make an AX12 multirun!
    nr_levels = 5
    def multi_otax(n_agents = 5, agent_type = 'W'):
        multi_buff = np.zeros((n_agents, nr_levels + 1))
        cum_multi_buff = np.zeros((n_agents, nr_levels + 1))
        levels = range(0,nr_levels+1)
        time = np.zeros(n_agents)
        
        for i in range(n_agents):
            tictoc.tic()
            (lesson_perf_buff, saved_p, agent) = run_otax(agent_type = agent_type) 
            multi_buff[i,1:] = lesson_perf_buff[0,:]
            cum_multi_buff[i,1:] = np.cumsum(lesson_perf_buff[0,:])
            time[i] = tictoc.toc()
            
        time_avg = np.mean(time)
        buff_avg = cum_multi_buff.mean(axis = 0)
        buff_err = cum_multi_buff.std(axis = 0)
        #define plot with average performance and error bar
        return (buff_avg, buff_err, time_avg)
    
    def multi_otax_avg_plot(buff_avg, buff_err, label):
        plt.errorbar((range(0,nr_levels+1)),buff_avg,fmt='-o',ecolor = 'black', yerr=buff_err, markersize = 8, capsize = 10, elinewidth=2,
    markeredgewidth=2, label = label)
        plt.xlabel('AX12 curriculum learning levels')
        plt.ylabel('Trials until convergence')
        plt.title('Average AX12 task performance over %i agents'% n_agents)
        plt.legend(loc="upper left", title = "Agent")
    
    n_agents = 2
    (buff1, err1, time1) = multi_otax(n_agents = 5, agent_type = 'W')
    (buff2, err2, time2) = multi_otax(n_agents = 5, agent_type = 'L')
    
    plt.figure(1)
    multi_otax_avg_plot(buff1, err1, 'W')
    multi_otax_avg_plot(buff2, err2, 'L')
    plt.show()
    
    def nulti_otax_plot(n_agents = 5, agent_type = 'W'):
        multi_buff = np.zeros((n_agents, nr_levels + 1))
        cum_multi_buff = np.zeros((n_agents, nr_levels + 1))
        levels = range(0,nr_levels+1)
        multi_levels = [levels,]*n_agents
        
        for i in range(n_agents):
            (lesson_perf_buff, saved_p, agent) = run_otax() 
            multi_buff[i,1:] = lesson_perf_buff[0,:]
            cum_multi_buff[i,1:] = np.cumsum(lesson_perf_buff[0,:])
            
        fig, ax = plt.subplots()    
        plt.xticks(levels)
        plt.xlabel('AX12 curriculum learning levels')
        plt.ylabel('Trials until convergence')
        plt.title('AX12 convergence for multiple agents')
        plt.plot(np.array(multi_levels).T,np.array(cum_multi_buff).T,'-o') #transpose to plot each row separately
        plt.show()
        
        fig, ax = plt.subplots()    
        plt.xticks(levels)
        plt.xlabel('AX12 curriculum learning levels')
        plt.ylabel('Trials until convergence per level')
        plt.title('AX12 convergence for multiple agents')
        plt.plot(np.array(multi_levels)[:,1:].T,np.array(multi_buff)[:,1:].T,'-o') #transpose to plot each row separately
        plt.show()
        #It does not learn level 5
        #Up to level 4 the agent converges
        return
    
    return

# %% Sigmoid/tanh offset

#seed = 1
offset = np.arange(0.5,4.5,0.5)
iswix = np.zeros((len(offset),n_switches+1))
for count, item in enumerate(offset):
    (saved_p, iswi, agent) = run_dms(print_all = 'on', tan_off = item)
    iswix[count] = iswi
    print(agent.tanh_offset)
levels = range(1,n_switches + 2)
multi_levels = [levels,]*len(offset)

fig, ax = plt.subplots()    
plt.xticks(levels)
plt.xlabel('DMS curriculum learning levels')
plt.ylabel('Trials until convergence')
plt.plot(np.array(multi_levels).T,np.array(iswix).T,'-o') #transpose to plot each row separately
plt.legend(offset, bbox_to_anchor=(1.05, 1.0), loc='upper left', title = "tanh_offset")
plt.show()
#result: sigmoid offset 2.5 is optimal, lower does not cause horrible 
#changes (between 0 and 2.5 seems fine), higher does

# %% WorkMATe with tanh

tanh_offset = 2.5
transfer_t = lambda x: np.tanh(x - tanh_offset)
dtransfer_t = lambda x: 1-x**2

(saved_p, iswi, agent) = run_dms(print_all = 'on')
print(agent.tanh_offset)
#it runs when tanh is hightened with + 1
#with normal tanh, qvec contains NaN > division by 0 in softmaxnorm?
#ran the offset plot > 0,3.5 and 4 do not converge before 30000 trials
#offset 2.0 works fastest (between 1 and 3 seems fine)

# %% WorkMATe with ReLu

(saved_p, iswi, agent) = run_dms(print_all = 'on')
print(agent.relu_offset)
#relu with offset 1.6 (which is closest to sigmoid/tanh does not converge at all)
#I tried He initialisation but it does not add a lot
#ReLu without offset does not converge either
#ReLu with offset -1 is horror > almost always 0

# %% WorkMATe with a regularizer (to force sparse representations)

(saved_p, iswi, agent) = run_dms(agent_type = 'L07', print_all = 'on')
print(agent)
print(agent.beta)
print(agent.beta2)
print(agent.beta3)
#print(agent.L1)
#change sigmoid offset > does that help?
#No real changes when sigmoid_offset is set to zero
#add match2 > dot product

# %% WorkMATe with different learning speeds

x = np.array([0.15, 0.10, 0.05, 0.025, 0.001])
#only the last five converged somewhat >> learning rate should stay small!
#b2 = 0.001, b3 = 0.1, 0.05

x2 = np.array([0.002, 0.001, 0.0005, 0.0001, 0.00001])
x3 = np.array([0.15, 0.10, 0.05, 0.025, 0.015])

iswix = np.zeros((5*len(x),n_switches+1))
test = np.transpose([np.arange(0,25)+1,]*(n_switches+1))

from itertools import product

for idx, (i, j) in enumerate(product(x2,x3), 1):
    (saved_p, iswi, agent) = run_dms(agent_type = 'L07', print_all = 'on', beta2 = i, beta3 = j)
    iswix[idx-1] = iswi
    print(iswi)
    print(iswix[idx-1])
    print(agent.beta2)
    print(agent.beta3)
    print(idx, i, j)

fig, ax = plt.subplots()    
plt.xticks(levels)
plt.xlabel('DMS curriculum learning levels')
plt.ylabel('Trials until convergence')
plt.plot(np.array(multi_levels).T,np.array(iswix).T,'-o') #transpose to plot each row separately
plt.legend(np.arange(25), bbox_to_anchor=(1.05, 1.0), loc='upper left', title = "beta2, beta3")
plt.show()

# %% Optimal learning combinations
arr = np.array([8,10,13,14,15,17,18,19,20,23,25]) - 1
print(iswix[arr, :])

for idx, (i, j) in enumerate(product(x2,x3), 1):
    if idx in (arr + 1):
        print(idx, i, j)
        
(saved_p, iswi, agent) = run_dms(agent_type = 'L07', print_all = 'on', beta2 = 0.001, beta3 = 0.05)

# %%
arr_beta2 = np.array([0.001, 0.001, 0.00001, 0.00001, 0, 0])
arr_beta3 = np.array([0.05, 0.015, 0.05, 0.015, 0.05, 0.015])

iswix2 = np.zeros((len(arr_beta2) + 1, n_switches+1))
for idx, i in enumerate(arr_beta2):
    (saved_p, iswi, agent) = run_dms(agent_type = 'L07', print_all = 'on', beta2 = i, beta3 = arr_beta3[idx])
    iswix2[idx] = iswi

(saved_p, iswi, agent) = run_dms(agent_type = 'W', print_all = 'on', beta2 = 0, beta3 = 0)
iswix2[6] = iswi
    
#%%
levels = range(1,n_switches + 2)
legend = ['0.001 / 0.05', '0.001 / 0.015', '0.00001 / 0.05', '0.00001 / 0.015' , 'fixed / 0.05', 'fixed / 0.015', 'workmate']
multi_levels = [levels,]*len(iswix2)
fig, ax = plt.subplots()    
plt.xticks(levels)
plt.title('DMS task performance with extra layer L')
plt.xlabel('DMS curriculum learning levels')
plt.ylabel('Trials until convergence')
plt.plot(np.array(multi_levels).T,np.array(iswix2).T,'-o') #transpose to plot each row separately
plt.legend(legend, bbox_to_anchor=(1.05, 1.0), loc='upper left', title = "beta2, beta3")
plt.show()    
    
#%% Try with different seeds (0.001 / 0.05)

set_seed_arr = np.array([1,2,3,4,5])
iswix3 = np.zeros((len(set_seed_arr), n_switches+1))
for idx, i in enumerate(set_seed_arr):
    (saved_p, iswi, agent) = run_dms(agent_type = 'L07', print_all = 'on', beta2 = 0.001, beta3 = 0.05, set_seed = i)
    iswix3[idx] = iswi
    print(set_seed_arr[i])
    agent1 = agent

levels = range(1,n_switches + 2)
legend = ['1','2','3','4','5']
multi_levels = [levels,]*len(iswix3)
fig, ax = plt.subplots()    
plt.xticks(levels)
plt.title('DMS task performance with extra layer L')
plt.xlabel('DMS curriculum learning levels')
plt.ylabel('Trials until convergence')
plt.plot(np.array(multi_levels).T,np.array(iswix3).T,'-o') #transpose to plot each row separately
plt.legend(legend, bbox_to_anchor=(1.05, 1.0), loc='upper left', title = "seed")
plt.show()  

# %% Visualise weights of agent seed = 1

(saved_p, iswi, agent) = run_dms(agent_type = 'L07', print_all = 'on', beta2 = 0.001, beta3 = 0.05, set_seed = 1)
visual_weights(agent.W_lx_start, agent.W_lx, 'W_lx' ,'L07b')
visual_weights(agent.W_hl_start, agent.W_hl, 'W_hl' ,'L07b')    
#Not a lot changes in lx due to the very small learning param beta2

# %% Pretraining agent - PreMATe
random.seed(1)
(dms, saved_p, agent) = w_pre_runner.run_class()
visual_weights(agent.W_lx_start, agent.W_lx, 'W_lx' ,'premate')
visual_weights(agent.W_hl_start, agent.W_hl, 'W_hl' ,'premate') 
visual_weights(agent.W_qh_start, agent.W_qh, 'W_qh' ,'premate') 

print(agent.L1)
print(dms.t)
print(dms.sample_stim)
print(dms.stims[dms.target])

# %% Combine PreMATe and WorkMATe-7b

#Try with beta3 = 0.001, beta3 = 0 and beta3 = higher than 0.001
#HAs the space become any sparser?
#Should I change the layer size of preMATe?

(dms, saved_p, agent, W_xl) = w_pre_runner.run_class()
w_runner.run_dms(W_xl = W_xl)
run!

#Get preliminary results








