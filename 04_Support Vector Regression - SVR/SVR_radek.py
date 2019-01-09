#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:04:28 2018

@author: radek
"""

#wczytanie bibliotek
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#wczytanie danych
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# skalowanie do matrycy
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#dopasowanie SVR do setu
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

#przewidywanie nowego wyniku
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#wizualizacja SVR
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Prawda czy fałsz (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# wizualizacja SVR (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Prawda czy fałsz (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
