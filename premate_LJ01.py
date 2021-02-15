# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 17:32:19 2020

This agent is used to classify all possible incoming stimuli of the 
tasks that WorkMATe can be trained on. This can be seen as a type of 
pretraining and the weights from x to l will be copied to the WorkMATE agent.

@author: Lieke Ceton
"""

#The agent only consists of a feedforward-type network and does not have
#any memory module. It consists of an input layer x, a latent layer l, 
#hidden layer h and an action layer q.

import numpy as np 
import inputs
import tasks

dms = tasks.DMS(n_stim = 3, n_switches = 6)
dms_stim = np.ndarray.flatten(dms.stimset)
all_stim = np.concatenate((dms_stim,np.array(['p'])))

class PreMATe(object):
    """
    Architecture schematic:
    x --> l --> h --> q_ext
    """
    
    def __init__(self, env=None, nhidden=20):
        super(PreMATe, self).__init__()
        assert env is not None
        self.env=env

        ## learning params (adopted from Rombouts et al., 2015)
        self.beta = 0.4
        self.gamma = 0.90
        self.L = 0.8
        # exploration rate
        self.epsilon = 0.025
        self.bias = 1
        
        ## member lambda functions:
        # sigmoid transfer function, offset at 2.5
        sigmoid_offset = 2.5
        self.transfer = lambda x: 1 / ( 1. + np.exp(sigmoid_offset - x) )
        self.dtransfer= lambda x: x * (1. - x) # derivative
        
        # softmax normalization; for action selection - boltzmann controller
        self.softmaxnorm = lambda x: (
            np.exp( x - x.max() ) / np.exp( x - x.max() ) .sum() )

        ## init network architecture -- inputs and output shape from env
        # input and hidden
        nx = inputs.get_obs('a').size
        nl = nhidden
        nh = nhidden
        # output -- q layer consisting of as many nodes as inputs that need to be discriminated
        nq = np.sum(len(all_stim))

        ## init network layers (activations 0)
        # (x will be constructed when processing 'new_obs')
        self.l = np.zeros( nl )
        self.h = np.zeros( nh ) 
        self.q = np.zeros( nq ) 

        ## init weights, tags traces, (+1 indicates projection from bias node)
        # ALL PLASTIC CONNECTIONS
        wl, wh = -.5, .5
        # Input projection with bias node
        # connections x -> l
        self.W_lx = np.random.sample((nl, nx + 1))*(wh-wl) + wl
        self.W_lx_start = np.copy(self.W_lx)
         
        # connections l -> h; nl + match nodes + bias
        self.W_hl  = np.random.sample( (nh, nl + 1) ) * (wh-wl) + wl
        self.W_hl_start = np.copy(self.W_hl)

        # connections h->q:
        self.W_qh  = np.random.sample( (nq, nh + 1) ) * (wh-wl) + wl
        self.W_qh_start  = np.copy(self.W_qh)
        
        # tags are shaped like weights but initialized at 0:
        zeros_ = np.zeros_like
        self.Tag_W_lx, self.Trace_W_lx = zeros_(self.W_lx), zeros_(self.W_lx)
        self.Tag_W_hl, self.Trace_W_hl = zeros_(self.W_hl), zeros_(self.W_hl)
        self.Tag_W_qh, self.Trace_W_qh = zeros_(self.W_qh), zeros_(self.W_qh)
        
        # Init action state
        self.action = -1
        # (prev) predicted reward:
        self.qat_1 = self.qat = None
        self.t   = 0 
        return

    def _intertrial_reset(self):
        """
        Reset time, memory, tags and traces
        """
        self.t = 0 
        # previous action = zeros 
        self.z *= 0
        # reset tags/traces for each Wmat
        zeros_ = np.zeros_like
        self.Tag_W_lx, self.Trace_W_lx = zeros_(self.W_lx), zeros_(self.W_lx)
        self.Tag_W_hl, self.Trace_W_hl = zeros_(self.W_hl), zeros_(self.W_hl)
        self.Tag_W_qh, self.Trace_W_qh = zeros_(self.W_qh), zeros_(self.W_qh)
        
        # reset 'current action', and the like
        self.action = -1
        self.qat = None
        return
    
    def step(self):
        # get observation and reward from env
        self.obs, self.r = self.env.step(self.action) 
        # do feedforward:
        self._feedforward()
        # learn from the obtained reward
        self._learn()
        # end of trial?
        if 'RESET' in self.obs:
            self._intertrial_reset()
            return self.r
        # do feedback (tag placement)
        self._feedback()
        # act (internal, external)
        self._act()
        self.t += 1
        return self.r

    def _feedforward(self):
        # shift previous action
        self.qat_1 = self.qat
        if 'RESET' in self.obs:
            # no meaningful feedforward sweep:  qat is not computed
            self.qat = None
            return
        # else:
        # compute input, hidden, output:
        self.construct_input()
        self.compute_latent()
        self.compute_hidden()
        self.compute_output()
        # determine z from q (action selection)
        self.action_selection()
        # determine new qat
        self.qat = (self.z * self.q).sum()
        return

    def _learn(self):
        """
        Learn from the reward; compute RPE and update weights
        general form form delta = r + gamma * qat - qat_1
        ...but there are edge cases
        """
        r = self.r
        if self.qat and self.qat_1: # regular
            delta = r + (self.gamma * self.qat) - self.qat_1
        elif self.qat_1 is None: # first step
            delta = r + (self.gamma * self.qat) - self.qat
        else: # self.qa(t) is None (final step):
            delta = r - self.qat_1
        self.delta = delta
        self.update_weights()
        return

    def _feedback(self):
        # updates traces and tags, based on action selection:
        # traces and tags
        self.update_traces()
        self.update_tags()
        return

    def _act(self):
        # only an external action:
        z =  self.z
        self.action = np.argmax(z)
        return

    def construct_input(self):
        """
        Turn obs into a vector; uses coding defined in 'inputs.py'
        """
        # input consists of: observation and time t
        self.x_sens = inputs.get_obs(self.obs, self.t)
        self.x = np.r_ [self.x_sens, self.bias] #only bias included 
        return
    
    def compute_latent(self):
        # x -> l 
        self.l_in = self.W_lx.dot(self.x) #compute l weights
        self.l_sens = self.transfer(self.l_in) #compute transfer
        self.l_out = np.r_[self.l_sens, 1.0] #add bias
        return
    
    def compute_hidden(self):
        # l -> h
        h_in = self.W_hl.dot(self.l_out) #compute h weights
        self.h_out  = np.r_[ self.transfer(h_in), self.bias] #transfer + bias
        return

    def compute_output(self):
        # hidden output (has bias added)
        self.q = self.W_qh.dot(self.h_out)
        # (no transfer, q nodes are linear)
        return

    def action_selection(self):
        # using q, determine z (based on argmax or exploration)
        self.z = np.zeros_like(self.q)
        # action selection for both modules separately
        qvec = self.q # get the module's qvalues 
        # check exploration; if not just take argmax:
        if ( np.random.sample() >= self.epsilon ):
            action = np.argmax(qvec)
        else: # compute softmax over Q and explore:
            pvec = self.softmaxnorm(qvec)
            action = np.random.choice( list(range(qvec.size)), p = pvec) 
        # set zvec: 1-hot code of actions:            
        zvec = np.zeros_like(qvec)
        zvec[action] = 1.0
        # place zvec into z at the right indices:
        self.z = zvec
        #Is this equal to self.z[action] = 1.0?
        return

    def update_weights(self):
        """
        all Weight-Trace pairs are updated with the same rule:
        w += beta * delta * tag
        """
        wt_pairs = (
                     ( self.W_hl, self.Tag_W_hl ), 
                     ( self.W_qh, self.Tag_W_qh )) #added by LJC
        for W, Tag in wt_pairs:
            W += self.beta * self.delta * Tag
        
        #Add L1 regularisation term/weight decay
        #( self.W_lx, self.Tag_W_lx ),
        self.L1 = 10**-5
        self.W_lx += self.beta * self.delta * self.Tag_W_lx #- self.L1 * np.sign(self.W_lx)
        return
    
    def update_traces(self):
        """
        Traces are the intermediate layers' markers
        The Traces are a relic from old AuGMEnT code, have no 'meaning' here
        """
        # Regulars, are replaced by new input:
        self.Trace_W_lx *= 0.0
        self.Trace_W_hl *= 0.0
        # add 1 x X vec to H x X matrix yields H copies of 1 x X vec
        self.Trace_W_lx += self.x.reshape(1, self.x.size) # this includes trace for bias
        self.Trace_W_hl += self.l_out.reshape(1, self.l_out.size) # this includes trace for bias
        return

    def update_tags(self):
        # 1. old tag decay:
        alltags = ( self.Tag_W_hl, self.Tag_W_qh)
        for Tag in alltags:
            Tag *= (self.L * self.gamma)

        # 2. form new tags:
        # tags onto output units: selected action.
        self.Tag_W_qh[self.z.astype('bool'), :] += self.h_out

        # feedback to hidden
        dh = self.dtransfer(self.h_out[:-1])
        self.fbh = self.W_qh[self.z.astype('bool'), :-1] # excluding the bias node
        self.fbh = self.fbh.sum(axis = 0 ) # summed contribution of all actions
        self.feedbackh = self.fbh * dh
        #update tag hl
        self.Tag_W_hl +=  np.expand_dims(self.feedbackh, 1) *  self.Trace_W_hl 
        
        #feedback to latent
        dl = self.dtransfer(self.l_sens)
        self.fbh_transfer = self.dtransfer(self.fbh) #the feedback onto h is transferred through the hidden layer
        W_hl_no_m = self.W_hl[:,:-1] #no bias
        self.W_hl_transpose = W_hl_no_m.T
        self.fhl = self.W_hl_transpose.dot(self.fbh_transfer) #for each node in l, all active h units are summed
        self.feedbackl = self.fhl * dl
        self.Tag_W_lx += np.expand_dims(self.feedbackl,1) * self.Trace_W_lx
        return

#input
#translate the sample_stim to an observation
#include time? How does this work?
#input the observation into the x layer

#transform to l
#transform to h
#transform to q
#output the softmax chosen option

#learn through feedback from the tags

#reset?