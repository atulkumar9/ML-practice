#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 23:13:04 2018

@author: atul
"""

#Polynomial Regression 

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
'''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting linear rergression model to dataset
from sklearn.linear_model import LinearRegression 
Lin_reg = LinearRegression()
Lin_reg.fit(X,Y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures 
Poly_reg = PolynomialFeatures(degree = 10)
X_poly = Poly_reg.fit_transform(X)
Lin_reg2 = LinearRegression()
Lin_reg2.fit(X_poly, Y)

#Visualising the Linear Regression result
plt.scatter(X, Y, color = 'red')
plt.plot(X, Lin_reg.predict(X), color = 'blue')
plt.title("Truth or Bluff(Linear Regression model)")
plt.xlabel("Position Label")
plt.ylabel("Salary")
plt.show()

#Visualising the Polynominal Regression result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, Lin_reg2.predict(Poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title("Truth or Bluff(Polynomial Regression model)")
plt.xlabel("Position Label")
plt.ylabel("Salary")
plt.show()

#predicting a new result with Linear Regression 
Lin_reg.predict(6.5)

#predicting a new result with Polynolial Regression Model
Lin_reg2.predict(Poly_reg.fit_transform(6.5))
