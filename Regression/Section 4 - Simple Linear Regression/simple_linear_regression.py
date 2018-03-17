
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values #matrix of features
Y = dataset.iloc[:, 1].values #vector of labels

#Dividing the dataset into training data and testing data
from sklearn.cross_validation import train_test_split 
X_train , X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

'''
Feature Scaling is already taken care by the simple linear regression
'''

#Fitting Simple Linear Regression to the Training Set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the test set result 
y_pred = regressor.predict(X_test)


#Checking the accuracy score

meanerror = 0
error= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for x in range(0, 9) :
    error[x] = (Y_test[x] - y_pred[x])/Y_test[x]
    if error[x] < 0 : 
        error[x] = error[x]*-1
    meanerror += error[x]
meanerror = meanerror/10;
print("accuracy = ", 100 - (meanerror*100))

#Visualising the traing set result

    # X-> no of years of experience, Y-> Salary
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

#Visualising the test set result
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()
