from data_preprocessing import X,Y
from train_model import linear_regress

L = 0.0000001
n = 100000
w,cost_list = linear_regress(X,Y,L,n)
print(w) # Final W values
print(cost_list[-1]) # Cost of final W