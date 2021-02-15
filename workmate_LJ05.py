# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:19:31 2020

This agent is to test the implementation of the learning of the latent layer onto LJ02.
The latent layer before was a fixed random projection and the memory 
projection is now set 1on1 with the size of one block. The nput is saved in 
the memory blocks and encoded during the matching. Such, the weights are
the same for the new and old inputs.

@author: Lieke Ceton
"""

import numpy as np 
from activations import relu, drelu, tanh, dtanh
#import scipy.spatial.distance
#from IPython import embed
import inputs

class WorkMATe(object):
    """
    Architecture schematic:
        x -- S
          \/
          h
          |
    [q_int|q_ext]
    """
    def __init__(self, env=None, nhidden=15, nblocks=2, block_size=15):
        super(WorkMATe, self).__init__()
        assert env is not None
        self.env=env

        ## learning params (adopted from Rombouts et al., 2015)
        self.beta = 0.15
        self.gamma = 0.90
        self.L = 0.8
        # exploration:
        self.epsilon = 0.025
        
        ## member lambda functions:
        # sigmoid transfer function, offset at 2.5
        sigmoid_offset = 2.5
        self.transfer = lambda x: 1 / ( 1. + np.exp(sigmoid_offset - x) )
        self.dtransfer= lambda x: x * (1. - x) # derivative
        # softmax normalization; for action selection - boltzmann controller
        self.softmaxnorm = lambda x: (
            np.exp( x - x.max() ) / np.exp( x - x.max() ) .sum() )

        ## init network architecture -- inputs and output shape from env
        # input, latent and hidden
        self.nx = nx = inputs.get_obs('a').size
        nl = block_size #latent layer has size memory block
        nh = nhidden
        # memory cell properties:
        self.nblocks = nblocks
        self.block_size = block_size
        nS = nblocks * nx #the memory blocks have input size
        
        # output -- q layer consisting of 2 modules
        # module for n external actions, internal actions for nblocks + 1 (null) 
        mod_sz = env.n_actions, nblocks + 1
        nq = np.sum(mod_sz)
        # indices of module for each node:
        self.zmods = np.hstack( [ [i] * sz for i, sz in enumerate( mod_sz ) ] )

        ## init network layers (activations 0)
        # (x will be constructed when processing 'new_obs')
        self.l = np.zeros( nl )
        self.S = np.zeros( nS )
        self.h = np.zeros( nh ) 
        self.q = np.zeros( nq ) 

        ## init weights, tags traces, (+1 indicates projection from bias node)        
    
        # Memory projection (x > S)
        # Note that time and sensory input cells are not separated in memory
        # this projection is not random but a fixed one-on-one mapping
        self.W_Sx = np.identity(nx) #Connection from input directly to S

        # Memory projection (l > S)        
        self.W_Sl = np.identity(nl)

        # PLASTIC CONNECTIONS (all except memory projection)
        wl, wh = -1, 1
        #Input projection (x > l); nx + bias
        self.W_lx = np.random.sample((nl, nx)) * (wh-wl) + wl
        
        wl, wh = -.25, .25
        # connections x->h; nx + match nodes + bias
        self.W_hl  = np.random.sample( (nh, nl + nblocks + 1) ) * (wh-wl) + wl
        # connections S->h;
        self.W_hS  = np.random.sample( (nh, nS    ) ) * (wh-wl) + wl
        # connections h->q;
        self.W_qh  = np.random.sample( (nq, nh + 1) ) * (wh-wl) + wl
        # tags are shaped like weights but initialized at 0:
        zeros_ = np.zeros_like
        self.Tag_W_lx, self.Trace_W_lx = zeros_(self.W_lx), zeros_(self.W_lx)
        self.Tag_W_hl, self.Trace_W_hl = zeros_(self.W_hl), zeros_(self.W_hl)
        self.Tag_W_hS, self.Trace_W_hS = zeros_(self.W_hS), zeros_(self.W_hS)
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
        self.S *= 0

        # previous action = zeros 
        self.z *= 0
        # reset tags/traces for each Wmat
        zeros_ = np.zeros_like
        self.Tag_W_lx, self.Trace_W_lx = zeros_(self.W_lx), zeros_(self.W_lx)
        self.Tag_W_hl, self.Trace_W_hl = zeros_(self.W_hl), zeros_(self.W_hl)
        self.Tag_W_hS, self.Trace_W_hS = zeros_(self.W_hS), zeros_(self.W_hS)
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
        # external and internal actions:
        zext =  self.z[self.zmods == 0]
        zint =  self.z[self.zmods == 1]
        self.action = np.argmax(zext)
        self.update_memory( zint )
        return

    @staticmethod
    #this might be something fun to look into! (LJC)
    def match(x,y):
        ons = np.maximum(y-x, 0.0 )
        off = np.maximum(x-y, 0.0 )
        return (ons+off).sum()/x.size

    def construct_input(self):
        """
        Turn obs into a vector; uses coding defined in 'inputs.py'
        """
        # input consists of: observation and time t
        self.x_sens = inputs.get_obs(self.obs, self.t)
        #self.x = np.append(self.x_sens,1.0) #add bias unit
        self.x = self.x_sens
        
        # sensory input is mapped onto a latent variable
        # x -> l (sigmoid layer)
        l_in = self.W_lx.dot(self.x)
        self.l_sens = self.transfer(l_in) #apply sigmoid transformation
        #self.l_sens = relu(l_in)
        #self.l_sens = l_in

        # Compute match value - S equal in each block!
        Snew = self.W_Sl.dot(self.l_sens)
        
        #In this agent, Sold is the saved x inputs (with x.size)
        #Now again encode using the x -> l sigmoid layer
        S_ = self.S.reshape((self.nblocks,self.x.size))
        Sold = np.zeros((self.nblocks,self.block_size))
        m = np.zeros(self.nblocks)
        for aa in range(self.nblocks):
            #Sold[aa,:] = self.W_lx.dot(S_[aa,:])
            #Sold[aa,:] = relu(self.W_lx.dot(S_[aa,:]))
            Sold[aa,:] = self.transfer(self.W_lx.dot(S_[aa,:]))
            m[aa] = 1 - WorkMATe.match(Snew, Sold[aa,:])

        # add match nodes + bias to latent input vector
        self.l = np.r_ [self.l_sens, m, 1.0]
        return

    def compute_hidden(self):
        # x->h  +  S->h
        self.S_out  = self.S
        # Compute Ha and h
        ha = self.W_hl.dot(self.l) + self.W_hS.dot(self.S_out)
        self.h_out  = np.r_[ self.transfer(ha), 1.0 ] #bias added
        return

    def compute_output(self):
        # hidden output (has bias added)
        self.q = self.W_qh.dot(self.h_out)
        # (no transfer, q nodes are linear)
        return

    def action_selection(self):
        # using q, per module, determine z (based on argmax or exploration)
        self.z = np.zeros_like(self.q)
        # action selection for both modules separately
        for mod_idx in np.unique(self.zmods):
            qvec = self.q[self.zmods == mod_idx] # get the module's qvalues 
            # check exploration; if not just take argmax:
            if ( np.random.sample() >= self.epsilon ):
                action = np.argmax(qvec)
            else: # compute softmax over Q and explore:
                pvec = self.softmaxnorm(qvec)
                action = np.random.choice( list(range(qvec.size)), p = pvec) 
            # set zvec: 1-hot code of actions:            
            zvec = np.zeros_like(qvec)
            zvec[ action ] = 1.0
            # place zvec into z at the right indices:
            self.z[self.zmods == mod_idx] = zvec
        return

    def update_weights(self):
        """
        all Weight-Trace pairs are updated with the same rule:
        w += beta * delta * tag
        """
        #wt_pairs = ( ( self.W_hl, self.Tag_W_hl ), 
             #        ( self.W_hS, self.Tag_W_hS ),
             #        ( self.W_qh, self.Tag_W_qh ))
        wt_pairs = ((  self.W_lx, self.Tag_W_lx ),
                     ( self.W_hl, self.Tag_W_hl ), 
                     ( self.W_hS, self.Tag_W_hS ),
                     ( self.W_qh, self.Tag_W_qh ))
        for W, Tag in wt_pairs:
            W += self.beta * self.delta * Tag
        return

    def update_traces(self):
        """
        Traces are the intermediate layers' markers
        The Traces are a relic from old AuGMEnT code, have no 'meaning' here
        """
        # Regulars, are replaced by new input:
        self.Trace_W_lx *= 0.0
        self.Trace_W_hl *= 0.0
        self.Trace_W_hS *= 0.0
        # add 1 x X vec to H x X matrix yields H copies of 1 x X vec
        self.Trace_W_lx += self.x.reshape(1, self.x.size) # this excludes trace for bias
        self.Trace_W_hl += self.l.reshape(1, self.l.size) # this includes trace for bias
        self.Trace_W_hS += self.S_out.reshape(1, self.S.size)
        return

    def update_tags(self):
        # 1. old tag decay:
        alltags = ( self.Tag_W_lx, self.Tag_W_hl, self.Tag_W_hS, self.Tag_W_qh)
        for Tag in alltags:
            Tag *= (self.L * self.gamma)

        # 2. form new tags:
        # tags onto output units: selected action.
        self.Tag_W_qh[self.z.astype('bool'), :] += self.h_out

        # feedback from hidden
        self.dh = self.dtransfer(self.h_out[:-1])
        self.fbh= self.W_qh[self.z.astype('bool'), :-1] # excluding the bias node
        self.fbh = self.fbh.sum(axis = 0 ) # summed contribution of all actions
        self.feedback = self.fbh * self.dh
        self.Tag_W_hl +=  np.expand_dims(self.feedback, 1) *  self.Trace_W_hl 
        self.Tag_W_hS +=  np.expand_dims(self.feedback, 1) *  self.Trace_W_hS
        
        #feedback from latent
        dl = self.dtransfer(self.l_sens) #no transfer function at this point
        #dl = self.l_sens
        #dl = drelu(self.l_sens)
        fbh_transfer = self.dtransfer(self.fbh) #feedback transferred through hidden layer
        W_hl_no_m = self.W_hl[:,:-3] #no match no bias
        W_hl_transpose = W_hl_no_m.T
        fhl = W_hl_transpose.dot(fbh_transfer) #for each node l, all active
        #h units are transferred
        feedbackl = fhl * dl
        self.Tag_W_lx += np.expand_dims(feedbackl,1) * self.Trace_W_lx
        return

    def update_memory(self, zvec):
        # final z is 'do not gate'; nothing happens then
        if not zvec[-1] == 1:
            # else:
            gate_idx = np.argmax(zvec)
            x_sens = self.x_sens     #this excludes bias- and  match
            S_ = self.S.reshape((self.nblocks, self.nx))
            #W_ = self.W_Sx.reshape( S_.shape +  (x.size, ) ) 
            # project x->S (encode)
            Sproj = self.W_Sx.dot(x_sens)
            # store @ gated 'stripe'
            S_[gate_idx,:] = Sproj
            # transform S back to its flat representation
            self.S = S_.reshape(self.S.shape)
        return

