# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 14:18:36 2025

@author: Arijit
"""

import pandas as pd

#reading the dataset 
netflix_df = pd.read_csv("C:\\Users\\Arijit\\Downloads\\netflix_titles.csv")

#checking the attributes and checking for null values
netflix_df.info()
netflix_df.head()

#Data Cleaning and preprocessing- removing the null values 
netflix_df = netflix_df.dropna(subset=['type', 'release_year', 'rating', 'duration', 'listed_in', 'country'])


#Using label-coding we convert the categorical variables into numerical ones
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
netflix_df['rating'] = le.fit_transform(netflix_df['rating'])
netflix_df['listed_in'] = le.fit_transform(netflix_df['listed_in'])
netflix_df['country'] = le.fit_transform(netflix_df['country'])

#extracting the total duration from movie seasons and minutes
netflix_df['duration'] = netflix_df['duration'].str.extract('(\d+)').astype(float)


#Selecting the independent variables
X = netflix_df[['release_year', 'rating', 'duration', 'listed_in', 'country']]

#Selecting the dependent variable
Y = netflix_df['type'].map({'Movie': 0, 'TV Show': 1})

#Classifying the train set and the test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Standardizing the features to optimize the performance of the model
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Using Logistic Regression Model:

from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(X_train, Y_train)
Y_pred_lr = model_lr.predict(X_test)

#Using KNN Model:

from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, Y_train)
Y_pred_knn = model_knn.predict(X_test)

#Using Naive Bayes Model:

from sklearn.naive_bayes import GaussianNB
model_nb = GaussianNB()
model_nb.fit(X_train, Y_train)
Y_pred_nb = model_nb.predict(X_test)

#Evaluating the models:

from sklearn.metrics import classification_report
print("Logistic Regression:\n", classification_report(Y_test, Y_pred_lr))
print("kNN:\n", classification_report(Y_test, Y_pred_knn))
print("Naïve Bayes:\n", classification_report(Y_test, Y_pred_nb))









