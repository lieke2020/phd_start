# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 12:38:22 2021

@author: Lieke Ceton
"""
# %% Exercise 1
import numpy as np
import matplotlib.pyplot as plt

#define variables
x = np.arange(0,1.5,0.01)
f = x**x

#find minimum
y_min = f[0]
for count, item in enumerate(f):
    #print(item)
    if item < y_min:
        y_min = item
        x_min = x[count]

#print text
#print(y_min)
#print(x_min)
print(str(x_min), str(round(y_min,2)))

#plot function
plt.plot(x,f,'b-')
plt.text(0.1, 1.2, "(xmin, ymin) = ( %1.2f" %x_min + ", %1.2f)" %y_min, fontsize = 15)
plt.plot(x_min, y_min, 'ro')
plt.show()

#%%
#Exercise 2
height = int(input("Hello, how height would you like ur pyramid to be? height: "))

#condition input
while int(height) < 0 or int(height) > 23:
    height = int(input("Hello, how height would you like ur pyramid to be? height: "))
    
for i in range(0,height,1):                            #outerloop  (rows: 1 >> height)
    for j in range(height, 0, -1):                     #innerloop  (columns: height >> 1 )  (these form a grid)
        if j >= i + 1:                                  #statement that needs to be satisfied.
            print(" ", end=' ')
        else:
            print("#", end=' ')
    print("#")                                         #extra "#" for identical picture, not too fancy.
    
#%%
import numpy as np
import matplotlib as plot
import matplotlib.pyplot as plt
import math

#define lists
x_list = []
y_list = []

#function
def roots(a,b,c):
    for x in np.arange(-6,6.01,0.01):
        #run the polynominal over x in specific range
        y = (a * (x**2)) + (b * x) + c
        #add values to lists
        x_list.append(np.round(x, 2))
        y_list.append(np.round(y, 2))
    #find roots
    num1 = -b - math.sqrt((b**2) - (4 * a * c))
    num2 = -b + math.sqrt((b**2) - (4 * a * c))
    den = 2 * a
    x1 = float(np.round(num1/den, 2))
    x2 = float(np.round(num2/den, 2))
    index_x1 = x_list.index(x1)
    index_x2 = x_list.index(x2)
    print(index_x1)
    print(index_x2)
    y1 = y_list[index_x1]
    y2 = y_list[index_x2]
    return x1, x2, y1, y2

result = roots(1,2,-10)
print(result)

plt.plot(x_list,y_list,'b-')
#plt.text(0.1, 1.2, "(xmin, ymin) = ( %1.2f" %x_min + ", %1.2f)" %y_min, fontsize = 15)
plt.show()
    
    
    
    