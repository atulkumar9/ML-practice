#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 20:28:37 2018

@author: atul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pt

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LE = LabelEncoder()
X[:, 3] = LE.fit_transform(X[:, 3])
OHE = OneHotEncoder(categorical_features = [3])
X = OHE.fit_transform(X).toarray()
X = X[:, 1:]

import statsmodels.formula.api as sm
X = np.append(np.ones((50, 1)).astype(int), X, 1)

def backwardElimination (x, sl):
    length = len(x[0])
    temp = np.zeros((50, 6)).astype(int)
    for i in range(0, length):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxPval = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxPval > sl :
            for j in range(0, length - i):
                if regressor_OLS.pvalues[j].astype(float) == maxPval:
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(Y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if adjR_before > adjR_after:
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue 
    print(regressor_OLS.summary())
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
    
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_Modeled, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)

