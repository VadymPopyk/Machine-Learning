# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:40:05 2020

@author: SAI AJAY
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

#Fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Fitting Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg =  PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

#now fitting the X_ploy into the regression to know the result of polynomial regression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

#Visualising for Linear Regression
plt.scatter(X , Y, color = 'red')
plt.plot(X , lin_reg.predict(X), color = 'blue')
plt.title("Linear Regression")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#Visualising Polynomial regression
plt.scatter(X , Y, color = 'red')
plt.plot(X , lin_reg_2.predict( poly_reg.fit_transform(X)), color = 'blue')
plt.title("Polynomial Regression")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
