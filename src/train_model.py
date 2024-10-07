import numpy as np 
import pandas as pd 

#Equation of line = w0 + w1x1 + w2x2 + w3x3 
# 3 Feautures, 1 Target
# L = learning rate
# n = interations

#Cost Function
def cost_func(x,y,w):  #for given thetas (w)
  y_pred = np.random.randint(0,1,size=(len(y),1)) # Initialise column with 0s
  MSE = 0
  for i in range(len(y)):
    y_pred[i] = np.dot(x[i],w) # Predict y for row i
    MSE += np.square(y[i] - y_pred[i]) # MSE for row i
  MSE = MSE/(2*len(y)) # MSE for all rows
  return MSE

#Gradiant Descent 
def gradiant_descent(x,y,L,w): 

    gd0=0;gd1=0;gd2=0;gd3=0
    for i in range(len(y)): # For each row in Y
      y_pred = np.dot(x[i],w) # Y_pred = X.W
      gd0 += -(1/len(y)) * (y[i] - y_pred) # Derivative wrt w0
      gd1 += -(1/len(y)) * (y[i] - y_pred)*x[i][1] # Derivative wrt w1
      gd2 += -(1/len(y)) * (y[i] - y_pred)*x[i][2] # Derivative wrt w2
      gd3 += -(1/len(y)) * (y[i] - y_pred)*x[i][3] # Derivative wrt w3

    #Updating Ws
    w[0]-= L*gd0 
    w[1]-= L*gd1
    w[2]-= L*gd2
    w[3]-= L*gd3

    return w


#Implementation with Matrix (Faster)
def linear_regress(x,y,L,n):

  w = np.zeros((x.shape[1],1)) # Initialize theta to all zeros

  cost_list=[] #To check costs

  for i in range(n):
    y_pred = np.dot(x,w) # Predicted value of Y
    cost = (1/(2*len(y)))*(np.sum(np.square(y-y_pred))) # MSE
    gd = 1/len(y) * np.dot(x.T,(y_pred-y)) # Gradiant descent
    w = w - L*gd # Update theta

    if i%100==0:
      cost_list.append(cost)
  return w, cost_list



