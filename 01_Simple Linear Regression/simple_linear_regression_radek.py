#wczytanie bibliotek
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#wczytanie danych
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#podzielenie na set_training i set_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 1/3, random_state = 0
        )

#dopasowanie Simple Linear Regression do train_setów
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#przewidywanie wyników test_setów
y_pred = regressor.predict(X_test)

#wizualizacja wyników training_setów
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Expirience (training set)')
plt.xlabel('Years of expirience')
plt.ylabel('Salary')
plt.show()

#wizualizacja wyników test_setów
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Expirience (test set)')
plt.xlabel('Years of expirience')
plt.ylabel('Salary')
plt.show()
