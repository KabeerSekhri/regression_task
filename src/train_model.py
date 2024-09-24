import numpy as np 
import pandas as pd 
import data_preprocessing

#Equation of line = w0 + w1x1 + w2x2 + w3x3 
# 3 Feautures, 1 Target
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

W = np.random.randint(0,1,size=(X.shape[1],1)).astype(np.float64) # Initialise W to 0s

#Gradiant Descent
def gradiant_descent(x,y,L,w):
    gd0=0;gd1=0;gd2=0;gd3=0
    for i in range(len(y)):
      gd0 += -(1/len(y)) * (y[i] - (w[0]*(x[i][0]) + w[1] + w[2] +w[3]))
      gd1 += -(1/len(y)) * (y[i] - (w[0]*(x[i][1]) + w[1] + w[2] +w[3]))*x[i][1]
      gd2 += -(1/len(y)) * (y[i] - (w[0]*(x[i][2]) + w[1] + w[2] +w[3]))*x[i][2]
      gd3 += -(1/len(y)) * (y[i] - (w[0]*(x[i][3]) + w[1] + w[2] +w[3]))*x[i][3]

    w[0]-= L*gd0
    w[1]-= L*gd1
    w[2]-= L*gd2
    w[3]-= L*gd3

    return w
    

#Implementation with Matrix
def linear_regress(x,y,L,n):

  w = np.zeros((x.shape[1],1)) # Initialize theta to all zeros

  cost_list=[] #To check costs

  for i in range(n):
    y_pred = np.dot(x,w) # Predicted value of Y
    cost = (1/(2*len(y)))*(np.sum(np.square(y-y_pred))) # MSE
    gd = 1/len(y) * np.dot(x.T,(y_pred-y)) # Gradiant descent
    w = w - L*gd # Update theta

    if i%1000==0:
      cost_list.append(cost)
  return w, cost_list



