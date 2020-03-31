#Multiple linear regression

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
  #import datasets
dataset = pd.read_csv('Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values 

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X=onehotencoder.fit_transform(X).toarray()

#splitting data into dataset and training set
from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state = 0)  

#Fit Multiple linear regression to training set
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
regressor = LinearRegression()   
regressor.fit(X_train,Y_train)

# Predicting Test Set Results

y_pred = regressor.predict(X_test)
print(r2_score(Y_test,y_pred))




