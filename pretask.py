# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 11:06:46 2020

@author: Lieke Ceton
"""
import random
import numpy as np
import string
from inputs import maxtime

#Learn to discriminate all the letters within the stimulus set of DMS and 
#the small letter p that is used to indicate fixation
class DMS_classify:     
    def __init__(self, n_stim = 3, n_switches = 5):
    #A constructor that at least defines the n_actions for the task
        #what do the inputs look like?    
        n_blocks = n_switches + 1
        stims = np.random.choice(list(string.ascii_uppercase),n_stim * n_blocks, replace=False) 
        
        self.stims = np.append(stims, 'p') #the string of letters that need to be discriminated   
        self.n_actions = len(self.stims) #each stimulus can be chosen
        self.reward = 1 #only rewarded when correct
        self.minireward = 0.1 
        self.reset()
        #self.t = 0 #For now, set t to zero, change to random.choice(tmat) later
    #Do we need a reset function here? Or is the learning one trial?
    def reset(self):
    #A function to initialize a new trial 
        new_stim = random.sample(list(enumerate(self.stims)),1)    
        self.sample_stim = new_stim[0][1]  
        self.target = new_stim[0][0]
        self.t = np.random.choice(maxtime)
        #self.t = 0
       
    #A function that evaluates the agent's action, and returns a new 
    #observation and a reward in return. If this observation is 'RESET', 
    #the agent assumes the trial has ended and will reset its tags.
    def step(self,action):
        if action == -1:
            self.reset()
            newobs = self.sample_stim
            return newobs, 0.0
        else:
            newobs, reward = self._go(action)
        return newobs, reward
    
    def _go(self, action):
        # was it on the target?
        if action == self.target:
            return 'RESET', self.reward
        else:
            return 'RESET', -1 * self.minireward #this is a kind of supervised learning?? 
            #Or no punishment > return 'RESET', 0.0
            
#write a task
#input is action of the agent (a specific choice of q)
#output is a reward if correct (+1) and no reward if incorrect (0)
#output is also a change of input to the next randomly chosen letter

#write an agent
#input is reward/no reward 
#output is choice of input letter

