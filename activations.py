# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 13:36:05 2020

@author: Lieke Ceton
"""
import numpy as np
import matplotlib.pyplot as plt

relu_offset = 1.6
def relu(x):
    return np.maximum(x - relu_offset,0)

def drelu(x):
    y = x.copy()
    y[(x - relu_offset) <=0] = 0
    y[(x - relu_offset) >0] = 1
    return y

def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=1-t**2
    return t, dt

tanh_offset = 2.5
transfer_t = lambda x: 1 + np.tanh(x - tanh_offset)
dtransfer_t = lambda x: 1-(x - 1)**2

sigmoid_offset = 2.5
transfer = lambda x: 1 / ( 1. + np.exp(sigmoid_offset - x) )
dtransfer= lambda x: x * (1. - x) # derivative

x = np.linspace(-10,10,1000)
t, dt = tanh(x)

plt.plot(x, relu(x), label = 'relu')
plt.plot(x, transfer(x), label = 'sigmoid')
plt.plot(x, transfer_t(x), label = 'tanh')
plt.legend(loc="upper left")
plt.show()

plt.plot(x, drelu(x), label = 'relu')
plt.plot(x, dtransfer(transfer(x)), label = 'sigmoid')
plt.plot(x, dtransfer_t(transfer_t(x)), label = 'tanh')
plt.legend(loc="upper left")
plt.show()

#plt.plot(x, dtransfer(transfer(x)))
#plt.show()