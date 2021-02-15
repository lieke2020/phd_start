# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:27:12 2020

@author: Lieke
"""

import matplotlib.pyplot as plt
import numpy as np
block_size = 15
nblocks = 2

W_Sl = np.identity(block_size)
for i in range(nblocks - 1):
    W_Sl = np.vstack((W_Sl,np.identity(block_size)))

l = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
#print(W_Sl.dot(l).reshape((nblocks,block_size)))
#print(W_Sl)

'''
import numpy as np
def oneonemap(nblocks = 2, block_size = 15):
    header = []
    for i in range(nblocks):
        header[i] = np.identity(block_size)
    return header

f = np.ones((4,1))
g = np.array([1,3,4,5])
print(f,g,f*g)
'''

sigmoid_offset = 2.5
transfer = lambda x: 1 / ( 1. + np.exp(sigmoid_offset - x) )
x = np.linspace(-3,7,100)
plt.figure(0)
plt.axvline(x=-0.25, color = 'r')
plt.axvline(x=0.75, color = 'r')
plt.plot(x,transfer(x))
plt.show()




