import numpy as np 
import pandas as pd 

dataset = pd.read_csv("/Users/hemantsekhri/Documents/IML/fuel_train.csv")

dataset.columns = dataset.columns.str.strip()  # Removes leading and trailing spaces in titles
#dataset.isnull().sum() # Check for missing values
#dataset.duplicated().sum() # Check for duplicate values

dataset.dropna(inplace=True)
dataset.drop_duplicates(inplace=True)

#dataset.corr()['FUEL CONSUMPTION'] # To find correlation

X = dataset[['ENGINE SIZE','CYLINDERS','COEMISSIONS']] # Features (independent)
Y = dataset['FUEL CONSUMPTION'] # Target (dependent)
X = np.c_[np.ones(X.shape[0]), X] # Adding a column of 1s for constant in matrix multiplication

split_size = int(0.75*len(dataset)) # Split X and Y into training and testing set in 8:2
X_train, Y_train = [X[:split_size]], [Y[:split_size]]
X_test, Y_test = [X[split_size:]], [Y[split_size:]]
