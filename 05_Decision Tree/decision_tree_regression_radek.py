#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:25:19 2019

@author: radek

model interesujący w więcej niż jednym wymiarze
"""

#wczytanie bibliotek
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#wczytanie danych
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#dopasowanie Decision Tree do setu
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

#przewidywanie nowego wyniku
y_pred = regressor.predict(6.5)

"""
#wizualizacja Decision Tree (błędna dla tego modelu regresji)
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Prawda czy fałsz (Decision Tree)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"""

# wizualizacja Decision Tree (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Prawda czy fałsz (Decision Tree)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()