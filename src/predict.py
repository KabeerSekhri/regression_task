import pickle
import numpy as np
import pandas as pd
from train_model import cost_func, gradiant_descent, linear_regress
from data_preprocessing import X,Y


n= 1000
L = 0.00001

# Basic model without vetorization
cost_list = []
W = np.random.randint(0,1,size=(X.shape[1],1)).astype(np.float64) # Initialise W to 0s

for i in range(n):
  W = gradiant_descent(X,Y,L,W)
  if i%100==0:
    cost_list.append(cost_func(X,Y,W))
final_cost = cost_list[-1]

with open('../models/regression_model1.pkl','wb') as file:
   pickle.dump(W, file)

with open('../models/regression_model1.pkl', 'rb') as file:
    loaded_w = pickle.load(file)

y_pred_b = np.dot(X, loaded_w) 

# Faster model with vectorization (matrix)
w, cost_list = linear_regress(X,Y,L,n)
final_cost = cost_list[-1]

with open('../models/regression_model_final.pkl', 'wb') as file:
    pickle.dump(w,file)

with open('../models/regression_model_final.pkl', 'rb') as file:
    loaded_w = pickle.load(file)

y_pred_v = np.dot(X, loaded_w) 

