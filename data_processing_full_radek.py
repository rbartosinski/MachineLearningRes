#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:31:59 2018

@author: radek

Pełny formularz przygotowania danych do dalszej pracy z nimi ML
"""

#wczytanie bibliotek
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#wczytanie danych
dataset = pd.read_csv('your_data_file.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values


#rozdzielenie na sety training i test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 0
        )

#wypełnienie brakujących danych
#from sklearn.preprocessing import Imputer
#mputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3])


#zamiana danych tekstowych i dummy variables
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)


#standaryzacja i/lub normalizacja danych
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
