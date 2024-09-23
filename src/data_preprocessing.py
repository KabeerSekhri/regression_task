import numpy as np 
import pandas as pd 

dataset = pd.read_csv("/Users/hemantsekhri/Documents/IML/fuel_train.csv")

dataset.columns = dataset.columns.str.strip()  # Removes leading and trailing spaces in titles
dataset.isnull().sum() # Check for missing values
dataset.duplicated().sum() # Check for duplicate values

dataset.dropna(inplace=True)
dataset.drop_duplicates(inplace=True)

