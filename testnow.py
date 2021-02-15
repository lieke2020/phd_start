# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:07:24 2020

# =============================================================================
# @author: Lieke
# """
import numpy as np
# import matplotlib.pyplot as plt
# 
# def student(firstname = 'Lieke', lastname ='Ceton', standard ='Fifth'):
#     if firstname == 'Lieke':
#         print('YEH')
#     else:
#         print('whatever')
#     print(firstname, lastname, 'studies in', standard, 'Standard')
# 
# def run_dms2(seed=1, viz_converged=True, agent_type = 'W'):
#     #enable the option of choosing another agent to execute the tasks
#     if agent_type == 'L': 
#         print('Liekes agent')
#     else:
#         print('Wouters agent')
# 
# student()
# run_dms2()
# 
# #My first ever plot!!
# def letsgo(x = 1):
#     buffer = np.zeros(x)
#     for i in range (0,6):
#         buffer = np.append(buffer,1)
#         print(buffer)
# 
# time_step = 300
# def double(lst):
#     return [i*time_step for i in lst]
# 
# trials = np.random.rand(88)
# arr = [i*time_step for i in range(0,len(trials))]
# #print(arr)
# plt.plot(arr,trials,'r')
# 
# =============================================================================

import matplotlib.pyplot as plt

egg = list("foobar")
fig, ax = plt.subplots()
# We need to draw the canvas, otherwise the labels won't be positioned and 
# won't have values yet.
a = fig.canvas.draw()
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = egg
ax.set_xticklabels(labels)

b = plt.figure(2)
x = np.arange(0, len(egg)/2, 0.5).tolist()
y = [90,40,65,33,15,8]
labels = egg
plt.plot(x,y,'r')
plt.xticks(x, labels)

b.show()