from data_preprocessing import X,Y
from train_model import linear_regress

L = 0.00001
n = 10000
w,cost_list = linear_regress(X,Y,L,n)
print(w)
print(cost_list)