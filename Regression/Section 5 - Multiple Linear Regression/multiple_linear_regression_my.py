#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 21:10:41 2018

@author: atul
"""


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

#encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3]) 
onehotEncoder = OneHotEncoder(categorical_features = [3])
X = onehotEncoder.fit_transform(X).toarray()

#Avoid the dummy variable trap, (it is taken care off by the libraries but this is how it is done)
X = X[:, 1:]

from sklearn.cross_validation import train_test_split 
X_train , X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elemination

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) ## adding the bo value in the slope formula

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
'''
#Naive way
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#endog -> 1-d endogenous response variable. The dependent Variable
#exog -> a nobs x k array 
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() 
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() 
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() 
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() 
regressor_OLS.summary()
'''

'''
Using function with p_values and Adjusted R-Squared
'''
def backwardElimination(x, sl):
    numVars = len(x[0])
    temp = np.zeros((50, 6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVars = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVars > sl:
            for j in range(0, numVars - i):
                if regressor_OLS.pvalues[j].astype(float) == maxVars:
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(Y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if adjR_before >= adjR_after :
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

X_train_mod, X_test_mod, Y_train_mod, Y_test_mod = train_test_split(X_Modeled, Y, test_size = 0.2, random_state = 0)
regressor = LinearRegression()
regressor.fit(X_train_mod, Y_train)

y_pred_mod = regressor.predict(X_test_mod)
