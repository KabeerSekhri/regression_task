import numpy as np 
import pandas as pd 
import data_preprocessing

#Equation of line = w0 + w1x1 + w2x2 + w3x3 

# L = learning rate
# n = interations
def linear_regress(x,y,L,n):
  y = y.flatten() 

  w = np.zeros((x.shape[1],1)) # Initialize W to all zeros

  cost_list=[]
  
  for i in range(n):
    y_pred = np.dot(x,w)
    cost = (1/(2*len(Y)))*(np.sum(np.square(y-y_pred)))
    gd = 1/len(Y) * np.dot(x.T,(y_pred-y))
    w = w - L*gd

    if i%10==0:
      cost_list.append(cost)
  return w,cost_list


