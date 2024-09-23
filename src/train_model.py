import numpy as np 
import pandas as pd 
import data_preprocessing

#Equation of line = w0 + w1x1 + w2x2 + w3x3 
# L = learning rate
# n = interations

#Cost Function
def cost_func(x,y):
  y_pred = np.random.randint(0,1,size=(len(y),1))
  w = np.random.randint(0,1,size=(x.shape[1],1))
  MSE = 0
  for i in range(len(y)):
    y_pred[i] = np.dot(x[i],w)
    MSE += np.square(y[i] - y_pred[i])
  MSE = MSE/(2*len(y))
  return MSE 


#Implementation with Matrix
def linear_regress(x,y,L,n):

  w = np.zeros((x.shape[1],1)) # Initialize theta to all zeros

  cost_list=[] #To check if costs are actually reducing

  for i in range(n):
    y_pred = np.dot(x,w) # Predicted value of Y
    cost = (1/(2*len(y)))*(np.sum(np.square(y-y_pred))) # MSE
    gd = 1/len(y) * np.dot(x.T,(y_pred-y)) # Gradiant descent
    w = w - L*gd # Update theta

    if i%1000==0:
      cost_list.append(cost)
  return w, cost_list



