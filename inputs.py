#!/usr/bin/env python
'''
This file is part of the WorkMATe source code, provided as supplemental material
for the article:
    "Flexible Working Memory through Selective Gating and Attentional Tagging"
Wouter Kruijne, Sander M Bohte, Pieter R Roelfsema, Christian N L Olivers

Please see the README file for additional details.
Questions or comments regarding this file can be addressed to w.kruijne@vu.nl
-------------------
This file implements a 'get_obs()' function, which is used by the agents
to convert a symbolic 'observation' at a time point t into an input vector 
code, combining an input representation for the time cells and an input 
representation for the stimulus offered by the environment.

If this file is run independently, it will plot two example input vectors 
for an entire Pro/antisaccade trial and for a 12-AX trial
'''

import numpy as np
import string
from random import sample
from itertools import combinations

##################### time cell coding #####################
maxtime  = 10
# Time vectors are created by convolving a response vector 
# with an identity matrix, yielding [maxtime] rows of time cell responses,
# each peaking at a unique, consecutive time.
z = [0.1, 0.25, 0.5, 1, 0.5, 0.25, 0.1]
crop = int((len(z)-1)/2) # the '3'-cropping here removes edge artefacts from convolution; 
# Time cell 0 (at row 0) peaks at the first moment in time (column 0).
tmat = np.vstack([np.convolve(z, t)[crop:maxtime + crop] for t in np.eye(maxtime)])

##################### stimulus coding  #####################
def get_obs(obs='A', t=0):
    stimbits = 7 # (= (256 unique stimuli)) >> 128?? (LJC)
    # construct binary stim_repres
    binstr = '0{}b'.format(stimbits)
    binstrings = [format(i, binstr) for i in range(2**stimbits)]
    tobinarr = lambda s : np.array([float(c) for c in s])
    Dx = np.vstack(  [tobinarr(i) for i in binstrings]  )
    
    # Dx now is a matrix of 128 x 7 bits. 'stimbits' is a dict that will order the 
    # first 52 of these in a lookup table, #why not choose 2**6 when you only use the first 52? (LJC)
    chars = string.ascii_lowercase + string.ascii_uppercase
    stimdict = dict( list(zip( chars, Dx )) )
    
    # Stimuli with these 5 letters are used in prosaccade/antisaccade, and here made
    # linearly separable, cf. Rombouts et al., 2015
    stimdict['g'] = np.zeros(stimbits)
    stimdict['p'] = np.eye(stimbits)[0]
    stimdict['a'] = np.eye(stimbits)[1]
    stimdict['l'] = np.eye(stimbits)[2]
    stimdict['r'] = np.eye(stimbits)[3] #why? this ruins the neat dictionary that you just made.. (LJC)
    
    # digits, used in 12-AX, are added to the stimdict in a similar manner
    digdict = dict( 
        [(d,Dx[i + 2**(stimbits-1) ]) for i,d in enumerate(string.digits) ])
    stimdict.update(digdict)
    # return time-stim vec:
    return np.r_[ tmat[t], stimdict[obs] ]

def get_obs3(obs='A', t=0):
    stimbits = 7 # (= (256 unique stimuli)) >> 128?? (LJC)
    # construct binary stim_repres
    binstr = '0{}b'.format(stimbits)
    binstrings = [format(i, binstr) for i in range(2**stimbits)]
    tobinarr = lambda s : np.array([float(c) for c in s])
    Dx = np.vstack(  [tobinarr(i) for i in binstrings]  )
    
    #shuffle as a test
    shuffle = sample(range(len(Dx)),len(Dx)) #shuffle the rows randomly 
    Dx = Dx[shuffle,:] #randomly sample from the x the length of chars
    
    # Dx now is a matrix of 128 x 7 bits. 'stimbits' is a dict that will order the 
    # first 52 of these in a lookup table, #why not choose 2**6 when you only use the first 52? (LJC)
    chars = string.ascii_lowercase + string.ascii_uppercase
    stimdict = dict( list(zip( chars, Dx )) )
    
    # Stimuli with these 5 letters are used in prosaccade/antisaccade, and here made
    # linearly separable, cf. Rombouts et al., 2015
    stimdict['g'] = np.zeros(stimbits)
    stimdict['p'] = np.eye(stimbits)[0]
    stimdict['a'] = np.eye(stimbits)[1]
    stimdict['l'] = np.eye(stimbits)[2]
    stimdict['r'] = np.eye(stimbits)[3] #why? this ruins the neat dictionary that you just made.. (LJC)
    
    # digits, used in 12-AX, are added to the stimdict in a similar manner
    digdict = dict( 
        [(d,Dx[i + 2**(stimbits-1) ]) for i,d in enumerate(string.digits) ])
    stimdict.update(digdict)
    # return time-stim vec:
    return np.r_[ tmat[t], stimdict[obs] ]

#function for creating binary vectors with a n number of ones
def place_ones(size, count):
        for positions in combinations(range(size), count):
            p = [0] * size

            for i in positions:
                p[i] = 1

            yield p

#11 bit binary vectors with 2 or 3 ones > average overlap 24%
#This seems like a desired sparsity
#Orginial dict: 50% of stimuli >50% overlap, now: 12%
stimbits = 11
z = np.array(list(place_ones(stimbits,3)))
y = np.array(list(place_ones(stimbits,2)))
x = np.vstack((z,y))
shuffle = sample(range(len(x)),len(x)) #shuffle the rows randomly 
Dx = x[shuffle,:] #randomly sample from the x the length of chars
#Dx = x #TAKE OUT THE SHUFFLE AS A TEST
#the stimulus dictionary consists of all lower and upper case letters and digits
stimuli = string.ascii_lowercase + string.ascii_uppercase + string.digits
stimdict2 = dict(list(zip(stimuli, Dx))) #assign stimdict

#update for prosaccade
stimdict2['g'] = np.zeros(stimbits)
stimdict2['p'] = np.eye(stimbits)[0]
stimdict2['a'] = np.eye(stimbits)[1]
stimdict2['l'] = np.eye(stimbits)[2]
stimdict2['r'] = np.eye(stimbits)[3]

#Alternative stimulus coding: Sparsity analysis
def get_obs2(obs='A', t=0):
    #return time-stim vec of a differently shaped stimdict
    #Transform the existing dictionary to something more sparse
    stimdict = stimdict2
    
    #print(stimdict2)
    return np.r_[ tmat[t], stimdict[obs] ]

##################### visualization of stimuli  #####################
if __name__ == '__main__':
    import matplotlib.pyplot as plt
# =============================================================================
#     ################ Example sequnece of Pro/Antisaccade  ###########
#     ggsa_seq = 'plppg'
#     O =  np.vstack([get_obs(s, t) for t,s in enumerate(ggsa_seq)])
#     plt.imshow(O.T, interpolation='nearest', cmap='seismic', vmin=-1, vmax=1)
#     plt.show()
# =============================================================================
    
    ################ Example sequence of 12-AX  ###########
    axseq = '1RGY2SZABY'
    O =  np.vstack([get_obs(s, t) for t,s in enumerate(axseq)])
    F = np.vstack([get_obs2(s, t) for t,s in enumerate(axseq)])
    
    #labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = list(axseq)
    
    fig, ax = plt.subplots()
    fig.canvas.draw()    
    ax.set_xticks(np.arange(0, len(axseq)).tolist())
    ax.set_xticklabels(labels)
    plt.imshow(O.T, interpolation='nearest', cmap='seismic', vmin=-1, vmax=1)
    
    fig, ax = plt.subplots()
    fig.canvas.draw()        
    ax.set_xticks(np.arange(0, len(axseq)).tolist())
    ax.set_xticklabels(labels)
    plt.imshow(F.T, interpolation='nearest', cmap='seismic', vmin=-1, vmax=1)
    
    plt.show()





