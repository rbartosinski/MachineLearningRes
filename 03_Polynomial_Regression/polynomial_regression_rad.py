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
y = dataset.iloc[:, 2].values

#dopasowanie LR do setu
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#dopasowanie Polynomial Regr. do setu
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#wizualizacja LR
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Position level vs. Salary (Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#wizulizacja PR
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Position level vs. Salary (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#wynik z LR
lin_reg.predict(6.5)

#wynik z PR
lin_reg_2.predict(poly_reg.fit_transform(6.5))