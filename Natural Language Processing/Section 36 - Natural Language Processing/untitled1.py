#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 20:54:53 2018

@author: atul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the libraries
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #quoting = 3 reffers to ignoring double quotes

#Cleaning the text
import re
import nltk
corpus = [0]*1000 #corpus is a collection of text, favourite variable name used in NLP
nltk.download('stopwords') # downloadd the stopwords which are irrelevent words like a, the, and etc. 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus[i] = review

#Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray() #matrix of independent variable
y = dataset.iloc[:, 1].values #vector of dependent variable

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


 
